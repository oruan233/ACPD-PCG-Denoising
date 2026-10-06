import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, find_peaks, correlate


# ============================================================
# 1) 包络提取
# ============================================================
def smooth_envelope(x, sr, lp_hz=8.0):
    """
    Hilbert 包络 + 低通平滑
    """
    x = np.asarray(x, dtype=float)
    x -= np.mean(x)

    env = np.abs(hilbert(x))
    b, a = butter(3, lp_hz / (sr / 2), btype="low")
    env = filtfilt(b, a, env)

    return np.maximum(env, 0.0)


# ============================================================
# 2) 包络峰检测（无任何 S1/S2 语义）
# ============================================================
def detect_envelope_peaks(
    env,
    sr,
    height_q=0.3,
    prom_q=0.3,
    min_dist_sec=0.08
):
    """
    只检测包络峰
    """
    env = np.asarray(env, dtype=float)

    height_thr = np.quantile(env, height_q)
    prom_thr = np.quantile(env, 0.9) * prom_q
    min_dist = int(min_dist_sec * sr)

    peaks, _ = find_peaks(
        env,
        height=height_thr,
        prominence=prom_thr,
        distance=min_dist
    )

    return peaks


# ============================================================
# 3) 心动周期估计（ACF）
# ============================================================
def estimate_cycle_length_from_acf(env, sr, hr_range=(40, 180)):
    """
    基于去均值、能量归一化的线性包络自相关估计心动周期 T0。

    这里的归一化按每个候选滞后分别计算，避免长滞后因重叠样本变少而
    被未归一化相关量系统性压低，也避免信号幅值变化影响周期估计。
    """
    env = np.asarray(env, dtype=float)
    n = len(env)
    if n < 4 or sr <= 0:
        return None

    hr_min, hr_max = hr_range
    T_min = 60.0 / hr_max
    T_max = 60.0 / hr_min

    lag_min = max(1, int(round(T_min * sr)))
    lag_max = min(n - 1, int(round(T_max * sr)))

    if lag_max <= lag_min + 2:
        return None

    x = env - np.mean(env)
    if not np.any(np.isfinite(x)):
        return None
    if np.sqrt(np.mean(x * x)) < 1e-12:
        return None

    acf = correlate(x, x, mode="full", method="fft")
    acf = acf[n - 1:n + lag_max]  # acf[lag] = <x[:-lag], x[lag:]>

    sq_prefix = np.concatenate(([0.0], np.cumsum(x * x)))
    lags = np.arange(lag_min, lag_max + 1)
    left_energy = sq_prefix[n - lags] - sq_prefix[0]
    right_energy = sq_prefix[n] - sq_prefix[lags]
    denom = np.sqrt(np.maximum(left_energy * right_energy, 0.0)) + 1e-12

    rho = acf[lags] / denom
    rho = np.nan_to_num(rho, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    if not np.any(np.isfinite(rho)):
        return None

    k = int(lags[int(np.argmax(rho))])

    return float(k / sr)


# ============================================================
# 4) S1 检测（纯节律锁定）
# ============================================================
def detect_s1_by_phase_lock(
    env,
    peaks,
    sr,
    T0,
    win_ratio=0.15,
    relax_ratio=0.30,
    min_gap_ratio=0.5
):
    """
    仅使用 envelope peaks + phase-lock 的 S1 检测（增强兜底版）

    规则：
    1) 第一个 S1 = peaks[0]
    2) 后续 S1：
        a) 在 T0 ± win 中找 envelope peaks
        b) 若没有 → 放宽窗口
        c) 若仍没有 → 在该周期窗口内直接从 env 重找最大峰
    """
    if len(peaks) == 0 or T0 is None:
        return np.array([], dtype=int)

    env = np.asarray(env, dtype=float)
    peaks = np.asarray(peaks, dtype=int)

    T0s = int(round(T0 * sr))
    win = int(win_ratio * T0s)
    relax_win = int(relax_ratio * T0s)
    min_gap = int(min_gap_ratio * T0s)

    s1 = [int(peaks[0])]
    cur = int(peaks[0])

    while True:
        t_pred = cur + T0s
        if t_pred >= len(env):
            break

        # ===============================
        # 1️⃣ 严格窗口：只用已有 envelope peaks
        # ===============================
        candidates = peaks[
            (peaks >= t_pred - win) &
            (peaks <= t_pred + win)
        ]

        # ===============================
        # 2️⃣ 放宽窗口
        # ===============================
        if len(candidates) == 0:
            candidates = peaks[
                (peaks >= t_pred - relax_win) &
                (peaks <= t_pred + relax_win)
            ]

        # ===============================
        # 3️⃣ 若仍无候选 → 从 env 中补一个峰（关键兜底）
        # ===============================
        if len(candidates) == 0:
            L = max(0, t_pred - relax_win)
            R = min(len(env), t_pred + relax_win)

            if R <= L + 3:
                break

            # 直接取该窗口内包络最大值
            nxt = L + np.argmax(env[L:R])
        else:
            # ===============================
            # 综合评分：时间优先 + 幅度辅助
            # ===============================
            time_score = -np.abs(candidates - t_pred) / T0s
            amp_score = env[candidates] / (np.max(env[candidates]) + 1e-9)
            score = 0.7 * time_score + 0.3 * amp_score
            nxt = candidates[np.argmax(score)]

        # ===============================
        # 防止一个周期内重复
        # ===============================
        if nxt - cur < min_gap:
            break

        s1.append(int(nxt))
        cur = int(nxt)

    return np.array(s1, dtype=int)


def detect_s2_between_s1(
    env,
    sr,
    s1,
    peaks,
    T0,
    Ts_ref=None,
    min_ratio=0.25
):
    """
    S2 检测（左右双边距离约束，生理安全版）

    S2 必须满足：
        S1[i] + min_ratio*T0 < S2 < S1[i+1] - min_ratio*T0
    """
    env = np.asarray(env, dtype=float)
    s1 = np.asarray(s1, dtype=int)
    peaks = np.asarray(peaks, dtype=int)

    T0s = int(round(T0 * sr))
    min_dist = int(min_ratio * T0s)

    Ts_ref_samp = None
    if Ts_ref is not None:
        Ts_ref_samp = int(Ts_ref * sr)

    s2 = []

    for i in range(len(s1) - 1):
        L = s1[i] + min_dist
        R = s1[i + 1] - min_dist

        if R <= L + 3:
            continue

        # ---------- 1️⃣ 优先使用原始 envelope peaks ----------
        cand = peaks[(peaks > L) & (peaks < R)]

        if len(cand) > 0:
            if Ts_ref_samp is not None:
                # 优先接近 Ts_ref
                score = -np.abs((cand - s1[i]) - Ts_ref_samp)
                idx = np.argmax(score)
            else:
                # 退化：选 envelope 最大
                idx = np.argmax(env[cand])

            s2.append(int(cand[idx]))
            continue

        # ---------- 2️⃣ 兜底：区间内包络最大 ----------
        idx = L + np.argmax(env[L:R])
        s2.append(int(idx))

    return np.array(s2, dtype=int)


def odd_even_orientation(peaks):
    """
    判断：奇峰是 S1 还是偶峰是 S1
    返回：
        +1 : odd = S1, even = S2
        -1 : even = S1, odd = S2
        0  : 不可靠
    """
    peaks = np.sort(np.asarray(peaks, dtype=int))
    M = len(peaks)
    if M < 6:
        return 0

    odd = peaks[0::2]
    even = peaks[1::2]

    L = min(len(odd), len(even)) - 1
    if L <= 1:
        return 0

    D_oe = even[:L] - odd[:L]
    D_eo = odd[1:L+1] - even[:L]

    MD_oe = np.mean(D_oe)
    MD_eo = np.mean(D_eo)

    if MD_oe < MD_eo:
        return +1   # odd = S1
    else:
        return -1   # even = S1


def merge_s1_s2_to_peaks(s1, s2):
    """
    把 S1 和 S2 合并成时间有序的事件峰序列
    """
    s1 = np.asarray(s1, dtype=int)
    s2 = np.asarray(s2, dtype=int)

    all_peaks = np.concatenate([s1, s2])
    all_peaks = np.sort(all_peaks)

    return all_peaks


# ============================================================
# 5) 一键接口
# ============================================================
def detect_s1_s2(pcg, sr):
    env = smooth_envelope(pcg, sr)
    peaks = detect_envelope_peaks(env, sr)
    T0 = estimate_cycle_length_from_acf(env, sr)

    if T0 is None:
        raise RuntimeError("T0 estimation failed")

    # ---------- S1 ----------
    s1 = detect_s1_by_phase_lock(env, peaks, sr, T0)

    # ---------- Ts 参考（用中位数更稳） ----------
    if len(s1) >= 2:
        Ts_ref = np.median(np.diff(s1)) / 2 / sr
    else:
        Ts_ref = None

    # ---------- S2 ----------
    s2 = detect_s2_between_s1(
        env=env,
        sr=sr,
        s1=s1,
        peaks=peaks,
        T0=T0,
        Ts_ref=Ts_ref,  # 可选
        min_ratio=0.25  # ⭐ 核心参数
    )

    # 已经有：
    # s1, s2 ← 来自 phase-lock + interval rule

    merged_peaks = merge_s1_s2_to_peaks(s1, s2)

    orient = odd_even_orientation(merged_peaks)

    if orient != 0:
        # orient = +1 → odd = S1
        # orient = -1 → even = S1

        odd = merged_peaks[0::2]
        even = merged_peaks[1::2]

        # 判断当前 s1 更接近 odd 还是 even
        def mean_min_dist(A, B):
            return np.mean([np.min(np.abs(A - b)) for b in B])

        d_odd = mean_min_dist(s1, odd)
        d_even = mean_min_dist(s1, even)

        phase_s1_is_odd = d_odd < d_even

        # 如果不一致 → 整体翻转
        if (orient == +1 and not phase_s1_is_odd) or \
                (orient == -1 and phase_s1_is_odd):
            s1, s2 = s2, s1

    return env, peaks, s1, s2, T0


# ============================================================
# 6) 可视化
# ============================================================
def plot_s1_s2(pcg, sr, env, peaks, s1, s2, T0, t_max=None):
    t = np.arange(len(pcg)) / sr

    if t_max is not None:
        mask = t <= t_max
        t = t[mask]
        pcg = pcg[mask]
        env = env[:len(t)]
        peaks = peaks[peaks < len(t)]
        s1 = s1[s1 < len(t)]
        s2 = s2[s2 < len(t)]

    plt.figure(figsize=(14, 4))
    plt.plot(t, pcg, alpha=0.35, label="PCG")
    plt.plot(t, env, lw=2, label="Envelope")

    plt.scatter(peaks / sr, env[peaks],
                s=30, c="gray", label="Envelope Peaks")
    plt.scatter(s1 / sr, env[s1],
                s=120, c="red", label="S1")
    plt.scatter(s2 / sr, env[s2],
                s=120, c="orange", label="S2")

    for i, ct in enumerate(np.arange(0, t[-1], T0)):
        plt.axvline(
            ct,
            color="blue",
            linestyle="--",
            alpha=0.4,
            label="T0" if i == 0 else None
        )

    plt.title(f"S1/S2 via pure phase-locking (T0 ≈ {T0:.3f}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, stft

def bandpass_filter(x, sr, low, high, order=4):
    nyq = 0.5 * sr
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, x)


def high_freq_energy_envelope(x, sr, low=150, high=600, smooth_ms=50):
    x_hf = bandpass_filter(x, sr, low, high)
    env = np.abs(hilbert(x_hf))**2

    win = int(smooth_ms * sr / 1000)
    if win > 1:
        kernel = np.ones(win) / win
        env = np.convolve(env, kernel, mode='same')

    return env


def murmur_saliency_curve(x, sr, n_fft=1024, hop=256,
                          f_low=150, f_high=600):
    f, t, Zxx = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    P = np.abs(Zxx)**2

    idx = np.where((f >= f_low) & (f <= f_high))[0]
    w_t = P[idx, :].mean(axis=0)

    return t, w_t


