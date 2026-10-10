# ============================================================
# nist_sp800_22.py
#
# Self-contained implementation of the NIST SP 800-22 Rev.1a
# statistical test suite for random number generators.
#
# Written because the `nistrng` PyPI package packs sequences as
# int8 and overflows its internal accumulators on sequences of
# 10^6 bits, which makes every test report p = 0.000000
# regardless of the input. All arithmetic here is float64 or
# int64.
#
# Reference: A. Rukhin et al., "A Statistical Test Suite for
# Random and Pseudorandom Number Generators for Cryptographic
# Applications", NIST Special Publication 800-22 Rev. 1a, 2010.
# ============================================================

import math
import numpy as np
from scipy.special import gammaincc, erfc
from scipy.stats import norm

ALPHA = 0.01


# ---------- 1. Frequency (Monobit) ----------
def frequency_monobit(b):
    n = len(b)
    s = abs(int(np.sum(2 * b.astype(np.int64) - 1))) / math.sqrt(n)
    return erfc(s / math.sqrt(2))


# ---------- 2. Frequency within a Block ----------
def block_frequency(b, M=128):
    n = len(b)
    N = n // M
    if N == 0:
        return None
    blocks = b[:N * M].reshape(N, M).astype(np.float64)
    pi = blocks.mean(axis=1)
    chi2 = 4.0 * M * np.sum((pi - 0.5) ** 2)
    return gammaincc(N / 2.0, chi2 / 2.0)


# ---------- 3. Runs ----------
def runs(b):
    n = len(b)
    pi = float(np.mean(b))
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0
    v = 1 + int(np.sum(b[:-1] != b[1:]))
    num = abs(v - 2.0 * n * pi * (1 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1 - pi)
    return erfc(num / den)


# ---------- 4. Longest Run of Ones in a Block ----------
def longest_run_ones(b):
    n = len(b)
    if n < 128:
        return None
    if n < 6272:
        M, K, N = 8, 3, 16
        v_cls, pi = [1, 2, 3, 4], [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        M, K, N = 128, 5, 49
        v_cls = [4, 5, 6, 7, 8, 9]
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M, K, N = 10000, 6, 75
        v_cls = [10, 11, 12, 13, 14, 15, 16]
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    counts = np.zeros(K + 1, dtype=np.int64)
    for i in range(N):
        blk = b[i * M:(i + 1) * M]
        best = cur = 0
        for x in blk:
            cur = cur + 1 if x else 0
            if cur > best:
                best = cur
        idx = min(max(best, v_cls[0]), v_cls[-1]) - v_cls[0]
        counts[idx] += 1

    chi2 = sum((counts[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(K + 1))
    return gammaincc(K / 2.0, chi2 / 2.0)


# ---------- 5. Binary Matrix Rank ----------
def _gf2_rank(mat):
    m = mat.copy()
    rows, cols = m.shape
    rank = 0
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if m[r, c]:
                piv = r
                break
        if piv is None:
            continue
        if piv != rank:
            m[[rank, piv]] = m[[piv, rank]]
        for r in range(rows):
            if r != rank and m[r, c]:
                m[r] ^= m[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def binary_matrix_rank(b, M=32, Q=32):
    n = len(b)
    N = n // (M * Q)
    if N < 38:
        return None
    F = [0, 0, 0]          # full, full-1, lower
    for i in range(N):
        blk = b[i * M * Q:(i + 1) * M * Q].reshape(M, Q).astype(np.uint8)
        r = _gf2_rank(blk)
        if r == M:
            F[0] += 1
        elif r == M - 1:
            F[1] += 1
        else:
            F[2] += 1
    p_full, p_m1 = 0.2888, 0.5776
    p_low = 1.0 - p_full - p_m1
    chi2 = ((F[0] - p_full * N) ** 2 / (p_full * N) +
            (F[1] - p_m1 * N) ** 2 / (p_m1 * N) +
            (F[2] - p_low * N) ** 2 / (p_low * N))
    return math.exp(-chi2 / 2.0)


# ---------- 6. Discrete Fourier Transform (Spectral) ----------
def dft_spectral(b):
    n = len(b)
    x = 2.0 * b.astype(np.float64) - 1.0
    S = np.abs(np.fft.rfft(x))[:n // 2]
    T = math.sqrt(math.log(1.0 / 0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = float(np.sum(S < T))
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return erfc(abs(d) / math.sqrt(2))


# ---------- 7. Non-overlapping Template Matching ----------
def non_overlapping_template(b, m=9, N=8):
    """Template 0^(m-1)1, non-overlapping scan (vectorised)."""
    n = len(b)
    M = n // N
    if M < m:
        return None
    W = np.zeros(N)
    for i in range(N):
        blk = b[i * M:(i + 1) * M].astype(np.int64)
        # match at j iff blk[j:j+m-1] all zero and blk[j+m-1] == 1
        c = np.concatenate([[0], np.cumsum(blk)])
        L = M - m + 1
        zeros_ok = (c[np.arange(L) + m - 1] - c[np.arange(L)]) == 0
        ones_ok = blk[np.arange(L) + m - 1] == 1
        hits = np.flatnonzero(zeros_ok & ones_ok)
        cnt = 0
        last = -m
        for j in hits:                 # enforce non-overlap
            if j >= last + m:
                cnt += 1
                last = j
        W[i] = cnt
    mu = (M - m + 1) / 2.0 ** m
    var = M * (1.0 / 2.0 ** m - (2.0 * m - 1.0) / 2.0 ** (2 * m))
    chi2 = float(np.sum((W - mu) ** 2) / var)
    return gammaincc(N / 2.0, chi2 / 2.0)


# ---------- 8. Overlapping Template Matching ----------
def overlapping_template(b, m=9, M=1032):
    """Template 1^m, overlapping scan (vectorised)."""
    n = len(b)
    N = n // M
    if N < 1:
        return None
    K = 5
    pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
    nu = np.zeros(K + 1, dtype=np.int64)
    for i in range(N):
        blk = b[i * M:(i + 1) * M].astype(np.int64)
        c = np.concatenate([[0], np.cumsum(blk)])
        L = M - m + 1
        cnt = int(np.sum((c[np.arange(L) + m] - c[np.arange(L)]) == m))
        nu[min(cnt, K)] += 1
    chi2 = sum((nu[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(K + 1))
    return gammaincc(K / 2.0, chi2 / 2.0)


# ---------- 9. Maurer's Universal Statistical ----------
_MU_VAR = {
    6: (5.2177052, 2.954), 7: (6.1962507, 3.125), 8: (7.1836656, 3.238),
    9: (8.1764248, 3.311), 10: (9.1723243, 3.356), 11: (10.170032, 3.384),
    12: (11.168765, 3.401), 13: (12.168070, 3.410), 14: (13.167693, 3.416),
    15: (14.167488, 3.419), 16: (15.167379, 3.421),
}


def maurers_universal(b):
    n = len(b)
    L = 6
    for cand in range(6, 17):
        if n >= (10 * 2 ** cand + 1000 * 2 ** cand) * cand:
            L = cand
    Q = 10 * 2 ** L
    K = n // L - Q
    if K <= 0:
        return None
    tab = np.zeros(2 ** L, dtype=np.int64)
    seq = b[:(Q + K) * L].reshape(-1, L)
    powers = (1 << np.arange(L - 1, -1, -1)).astype(np.int64)
    idx = seq.astype(np.int64) @ powers
    for i in range(Q):
        tab[idx[i]] = i + 1
    total = 0.0
    for i in range(Q, Q + K):
        total += math.log2(i + 1 - tab[idx[i]])
        tab[idx[i]] = i + 1
    fn = total / K
    expected, variance = _MU_VAR[L]
    c = 0.7 - 0.8 / L + (4 + 32.0 / L) * (K ** (-3.0 / L)) / 15
    sigma = c * math.sqrt(variance / K)
    return erfc(abs((fn - expected) / (math.sqrt(2) * sigma)))


# ---------- 10. Linear Complexity ----------
def _berlekamp_massey(s):
    n = len(s)
    c = np.zeros(n, dtype=np.int8); bb = np.zeros(n, dtype=np.int8)
    c[0] = bb[0] = 1
    l = m = 0; N = 0
    while N < n:
        d = s[N]
        for i in range(1, l + 1):
            d ^= c[i] & s[N - i]
        if d == 1:
            t = c.copy()
            for i in range(n - (N - m)):
                c[N - m + i] ^= bb[i]
            if l <= N // 2:
                l = N + 1 - l
                m = N
                bb = t
        N += 1
    return l


def linear_complexity(b, M=500):
    n = len(b)
    N = n // M
    if N < 1:
        return None
    K = 6
    pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    mu = M / 2.0 + (9.0 + (-1) ** (M + 1)) / 36.0 - (M / 3.0 + 2.0 / 9.0) / 2 ** M
    nu = np.zeros(K + 1, dtype=np.int64)
    for i in range(N):
        l = _berlekamp_massey(b[i * M:(i + 1) * M].astype(np.int8))
        t = (-1) ** M * (l - mu) + 2.0 / 9.0
        if t <= -2.5: nu[0] += 1
        elif t <= -1.5: nu[1] += 1
        elif t <= -0.5: nu[2] += 1
        elif t <= 0.5: nu[3] += 1
        elif t <= 1.5: nu[4] += 1
        elif t <= 2.5: nu[5] += 1
        else: nu[6] += 1
    chi2 = sum((nu[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(K + 1))
    return gammaincc(K / 2.0, chi2 / 2.0)


# ---------- 11. Serial ----------
def _psi2(b, m):
    n = len(b)
    if m <= 0:
        return 0.0
    ext = np.concatenate([b, b[:m - 1]]) if m > 1 else b
    idx = np.zeros(n, dtype=np.int64)
    for j in range(m):
        idx = (idx << 1) | ext[j:j + n].astype(np.int64)
    counts = np.bincount(idx, minlength=2 ** m).astype(np.float64)
    return (2 ** m / n) * np.sum(counts ** 2) - n


def serial(b, m=16):
    p0 = _psi2(b, m)
    p1 = _psi2(b, m - 1)
    p2 = _psi2(b, m - 2)
    d1 = p0 - p1
    d2 = p0 - 2 * p1 + p2
    return (gammaincc(2 ** (m - 2), d1 / 2.0),
            gammaincc(2 ** (m - 3), d2 / 2.0))


# ---------- 12. Approximate Entropy ----------
def approximate_entropy(b, m=10):
    n = len(b)
    def phi(mm):
        if mm == 0:
            return 0.0
        ext = np.concatenate([b, b[:mm - 1]]) if mm > 1 else b
        idx = np.zeros(n, dtype=np.int64)
        for j in range(mm):
            idx = (idx << 1) | ext[j:j + n].astype(np.int64)
        c = np.bincount(idx, minlength=2 ** mm).astype(np.float64) / n
        nz = c[c > 0]
        return float(np.sum(nz * np.log(nz)))
    apen = phi(m) - phi(m + 1)
    chi2 = 2.0 * n * (math.log(2) - apen)
    return gammaincc(2 ** (m - 1), chi2 / 2.0)


# ---------- 13. Cumulative Sums ----------
def cumulative_sums(b, mode=0):
    n = len(b)
    x = (2 * b.astype(np.int64) - 1)
    if mode == 1:
        x = x[::-1]
    z = int(np.max(np.abs(np.cumsum(x))))
    if z == 0:
        return 1.0
    total = 0.0
    k0 = int((-n / z + 1) // 4)
    k1 = int((n / z - 1) // 4)
    for k in range(k0, k1 + 1):
        total += norm.cdf((4 * k + 1) * z / math.sqrt(n)) - \
                 norm.cdf((4 * k - 1) * z / math.sqrt(n))
    k0 = int((-n / z - 3) // 4)
    for k in range(k0, k1 + 1):
        total -= norm.cdf((4 * k + 3) * z / math.sqrt(n)) - \
                 norm.cdf((4 * k + 1) * z / math.sqrt(n))
    return max(0.0, min(1.0, 1.0 - total))


# ---------- 14/15. Random Excursions (+ Variant) ----------
_PI_EXC = {
    1: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0312],
    2: [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
    3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804],
    4: [0.8750, 0.0156, 0.0137, 0.0120, 0.0105, 0.0733],
}


def _cycles(b):
    x = 2 * b.astype(np.int64) - 1
    s = np.cumsum(x)
    s = np.concatenate([[0], s, [0]])
    zero_idx = np.where(s == 0)[0]
    return s, zero_idx


def random_excursions(b):
    s, zero_idx = _cycles(b)
    J = len(zero_idx) - 1
    if J < 500:
        return None
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    out = {}
    for st in states:
        nu = np.zeros(6, dtype=np.int64)
        for c in range(J):
            seg = s[zero_idx[c]:zero_idx[c + 1]]
            k = int(np.sum(seg == st))
            nu[min(k, 5)] += 1
        pi = _PI_EXC[abs(st)]
        chi2 = sum((nu[k] - J * pi[k]) ** 2 / (J * pi[k]) for k in range(6))
        out[st] = gammaincc(2.5, chi2 / 2.0)
    return out


def random_excursions_variant(b):
    s, zero_idx = _cycles(b)
    J = len(zero_idx) - 1
    if J < 500:
        return None
    out = {}
    for st in list(range(-9, 0)) + list(range(1, 10)):
        xi = int(np.sum(s == st))
        out[st] = erfc(abs(xi - J) / math.sqrt(2.0 * J * (4 * abs(st) - 2)))
    return out


# ---------- battery ----------
def run_battery(bits, alpha=ALPHA, verbose=True):
    b = np.asarray(bits, dtype=np.uint8)
    results = []

    def add(name, p):
        if p is None:
            results.append((name, None, 'N/A'))
        else:
            results.append((name, float(p), 'PASS' if p >= alpha else 'FAIL'))

    add('Frequency (Monobit)',            frequency_monobit(b))
    add('Block Frequency',                block_frequency(b))
    add('Runs',                           runs(b))
    add('Longest Run of Ones in a Block', longest_run_ones(b))
    add('Binary Matrix Rank',             binary_matrix_rank(b))
    add('Discrete Fourier Transform',     dft_spectral(b))
    add('Non-overlapping Template',       non_overlapping_template(b))
    add('Overlapping Template',           overlapping_template(b))
    add("Maurer's Universal Statistical", maurers_universal(b))
    add('Linear Complexity',              linear_complexity(b))
    sp = serial(b)
    add('Serial (test 1)', sp[0]); add('Serial (test 2)', sp[1])
    add('Approximate Entropy',            approximate_entropy(b))
    add('Cumulative Sums (forward)',      cumulative_sums(b, 0))
    add('Cumulative Sums (reverse)',      cumulative_sums(b, 1))

    re_ = random_excursions(b)
    if re_ is None:
        add('Random Excursions', None)
    else:
        worst = min(re_.values())
        add('Random Excursions (worst of 8 states)', worst)
    rev = random_excursions_variant(b)
    if rev is None:
        add('Random Excursions Variant', None)
    else:
        add('Random Excursions Variant (worst of 18)', min(rev.values()))

    if verbose:
        npass = sum(1 for _, _, v in results if v == 'PASS')
        ntot = sum(1 for _, _, v in results if v != 'N/A')
        for name, p, v in results:
            ps = 'n/a       ' if p is None else '%-10.6f' % p
            print('  %-42s p = %s %s' % (name, ps, v))
        print('  %d of %d applicable tests passed (alpha = %.2f)'
              % (npass, ntot, alpha))
    return results


if __name__ == '__main__':
    print('Self-test on 1,000,000 bits from numpy PCG64 (should pass):')
    rng = np.random.default_rng(20260815)
    run_battery(rng.integers(0, 2, 1_000_000).astype(np.uint8))
