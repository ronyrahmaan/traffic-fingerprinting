"""Feature extraction for all attack algorithms.

Paper algorithms (reproduce Kennedy et al., IEEE CNS 2019):
1. LL-Jaccard: a SET of signed packet sizes (for Jaccard similarity)
2. LL-NB: a HISTOGRAM of signed packet sizes (for Naive Bayes)
3. VNG++: burst histogram + traffic stats (for Naive Bayes)
4. P-SVM/AdaBoost: burst histogram + richer stats (for AdaBoost)

My additions (beyond the paper):
5. CUMUL: cumulative sum features (Panchenko et al., NDSS 2016)
6. Timing: inter-packet timing features (Rahman et al., PETS 2020)
7. Combined: CUMUL + timing + statistical features (my own design)

All start from the same building block: size * direction.
"""

import numpy as np

from traffic_fingerprinting.data_loader import TrafficTrace

# --- Constants ---
# These match the paper's settings. Changing them changes results.

# LL-NB: packet size ranges from -1500 to +1500 (max packet = 1500 bytes)
LLNB_MIN = -1500
LLNB_MAX = 1500
LLNB_ROUNDING = 60  # Sweep tested 10-100; bin=60 best matches paper's 33.8%

# VNG++: burst sizes can be much larger (sum of many packets)
VNG_MIN = -400000
VNG_MAX = 400000
VNG_ROUNDING = 2000  # Paper's repo default is 2000, tested 1000-10000

# P-SVM: uses SVM with RBF kernel (paper calls it "P-SVM" in Table I)
SVM_MIN = -200000
SVM_MAX = 200000
SVM_ROUNDING = 2000  # Same binning approach as VNG++


# --- Helpers ---

def compute_bursts(trace: TrafficTrace) -> list[int]:
    """Group consecutive same-direction packets into bursts.

    A "burst" is a run of packets all going the same direction.
    The burst value = sum of (size * direction) for those packets.

    Example:
        Packets: (+1, 500), (+1, 1500), (-1, 234), (-1, 1500), (+1, 350)
        Directions:  +1, +1, -1, -1, +1
        Bursts: [+2000, -1734, +350]

    Why bursts? They capture the rhythm of communication.
    When Echo sends your voice, that's one upward burst.
    When Amazon replies, that's one downward burst.
    """
    if not trace.packets:
        return []

    bursts = []
    current_burst = trace.packets[0].size * trace.packets[0].direction
    current_dir = trace.packets[0].direction

    for pkt in trace.packets[1:]:
        if pkt.direction == current_dir:
            # Same direction: add to current burst
            current_burst += pkt.size * pkt.direction
        else:
            # Direction changed: save burst, start new one
            bursts.append(current_burst)
            current_burst = pkt.size * pkt.direction
            current_dir = pkt.direction

    # Don't forget the last burst
    bursts.append(current_burst)
    return bursts


def make_histogram(values: list[int], min_val: int, max_val: int, bin_width: int) -> np.ndarray:
    """Bin a list of values into a fixed-width histogram.

    Creates bins from min_val to max_val with the given width.
    Values outside the range go into the first or last bin.

    Example with min=-1500, max=1500, width=100:
        Bins: [-1500,-1400), [-1400,-1300), ..., [1400,1500]
        Total bins: 30

    Returns a numpy array of counts per bin.
    """
    n_bins = (max_val - min_val) // bin_width
    histogram = np.zeros(n_bins, dtype=np.float64)

    for v in values:
        # Find which bin this value falls into
        idx = (v - min_val) // bin_width
        # Clamp to valid range
        idx = max(0, min(idx, n_bins - 1))
        histogram[idx] += 1

    return histogram


# --- Algorithm 1: LL-Jaccard ---

def extract_jaccard_set(trace: TrafficTrace) -> set[int]:
    """Extract feature set for LL-Jaccard.

    Simply: the set of all (size * direction) values in the trace.
    Duplicates are removed because it's a set.

    This is the simplest representation. It only knows WHICH
    packet sizes appeared, not how many times.
    """
    return set(trace.signed_sizes())


# --- Algorithm 2: LL-NB (Naive Bayes) ---

def extract_llnb_features(
    trace: TrafficTrace,
    rounding: int = LLNB_ROUNDING,
) -> np.ndarray:
    """Extract histogram features for LL-NB.

    Takes each (size * direction) value, rounds it, then bins into
    a histogram. The histogram becomes the feature vector.

    Rounding reduces the number of unique values, which helps
    Naive Bayes work better (fewer dimensions = less overfitting).

    With rounding=60 (default) and range [-1500, 1500]:
        A 583-byte outgoing packet → +583 → rounded to +600 → bin 35
    """
    signed = trace.signed_sizes()

    # Round to nearest multiple
    rounded = [round(s / rounding) * rounding for s in signed]

    return make_histogram(rounded, LLNB_MIN, LLNB_MAX + 1, rounding)


# --- Algorithm 3: VNG++ ---

def extract_vngpp_features(
    trace: TrafficTrace,
    rounding: int = VNG_ROUNDING,
) -> np.ndarray:
    """Extract features for VNG++ (Variable N-Gram).

    Uses three types of information:
    1. Burst histogram (the main feature)
    2. Total trace duration
    3. Upstream and downstream byte totals

    These are concatenated into one feature vector.
    """
    bursts = compute_bursts(trace)

    # Round burst values
    rounded_bursts = [round(b / rounding) * rounding for b in bursts]
    burst_hist = make_histogram(rounded_bursts, VNG_MIN, VNG_MAX + 1, rounding)

    # Traffic statistics
    signed = trace.signed_sizes()
    upstream_bytes = sum(s for s in signed if s > 0)
    downstream_bytes = abs(sum(s for s in signed if s < 0))
    total_time = trace.duration

    # Combine: [time, upstream, downstream, ...burst_histogram...]
    stats = np.array([total_time, upstream_bytes, downstream_bytes], dtype=np.float64)
    return np.concatenate([stats, burst_hist])


# --- Algorithm 4: P-SVM (Panchenko SVM) ---

def extract_svm_features(
    trace: TrafficTrace,
    rounding: int = SVM_ROUNDING,
) -> np.ndarray:
    """Extract features for P-SVM (Panchenko et al.).

    The richest feature set. Includes burst histogram plus:
    - Upstream/downstream total bytes
    - Incoming packet ratio
    - Total packet count
    - Number of bursts

    Paper uses SVM with RBF kernel (gamma=2^-19, C=2^17).
    Table I reports 33.4% accuracy with this method.
    """
    bursts = compute_bursts(trace)

    # Round burst values
    rounded_bursts = [round(b / rounding) * rounding for b in bursts]
    burst_hist = make_histogram(rounded_bursts, SVM_MIN, SVM_MAX + 1, rounding)

    # Traffic statistics
    signed = trace.signed_sizes()
    upstream_bytes = sum(s for s in signed if s > 0)
    downstream_bytes = abs(sum(s for s in signed if s < 0))

    num_bursts = len(bursts)
    total_packets = trace.num_packets

    # Incoming packet ratio
    incoming_count = sum(1 for p in trace.packets if p.direction == -1)
    in_ratio = incoming_count / total_packets if total_packets > 0 else 0.0

    # Combine
    stats = np.array(
        [upstream_bytes, downstream_bytes, in_ratio, total_packets, num_bursts],
        dtype=np.float64,
    )
    return np.concatenate([stats, burst_hist])


# --- Feature matrix builders (for sklearn) ---

def build_llnb_matrix(traces: list[TrafficTrace]) -> np.ndarray:
    """Build feature matrix for LL-NB. Each row = one trace."""
    return np.array([extract_llnb_features(t) for t in traces])


def build_llnb_matrix_with_rounding(
    traces: list[TrafficTrace], rounding: int,
) -> np.ndarray:
    """Build LL-NB feature matrix with a custom bin width.

    Used for bin-width sweep experiments (testing rounding=10,20,...,150).
    """
    return np.array([extract_llnb_features(t, rounding=rounding) for t in traces])


def build_vngpp_matrix(traces: list[TrafficTrace]) -> np.ndarray:
    """Build feature matrix for VNG++. Each row = one trace."""
    return np.array([extract_vngpp_features(t) for t in traces])


def build_svm_matrix(traces: list[TrafficTrace]) -> np.ndarray:
    """Build feature matrix for P-SVM. Each row = one trace."""
    return np.array([extract_svm_features(t) for t in traces])


# =====================================================================
# IMPROVEMENTS BEYOND THE PAPER
# =====================================================================

# CUMUL: cumulative sum features (Panchenko et al., NDSS 2016)
# Instead of histograms, track how total bytes accumulate over time.
# This captures the "shape" of the traffic flow, not just counts.
CUMUL_N_FEATURES = 100  # Sample the cumulative curve at 100 evenly-spaced points


def extract_cumul_features(trace: TrafficTrace) -> np.ndarray:
    """Extract CUMUL features: cumulative sum of packet sizes.

    The idea (Panchenko et al., NDSS 2016):
    - Compute cumulative sum of signed packet sizes over time
    - Sample this curve at N evenly-spaced points
    - The curve shape is a fingerprint of the traffic pattern

    Why this works better than histograms:
    - Histograms lose ORDER information (packet 1 vs packet 100)
    - CUMUL preserves the temporal flow (early vs late traffic)
    - A command that sends big packets early looks different from
      one that sends big packets late, even if totals are the same

    Features (104 total):
    - [0]: total incoming bytes
    - [1]: total outgoing bytes
    - [2]: total incoming packets
    - [3]: total outgoing packets
    - [4:104]: cumulative sum sampled at 100 points
    """
    signed = trace.signed_sizes()

    if not signed:
        return np.zeros(4 + CUMUL_N_FEATURES, dtype=np.float64)

    # Basic statistics
    incoming_bytes = abs(sum(s for s in signed if s < 0))
    outgoing_bytes = sum(s for s in signed if s > 0)
    incoming_pkts = sum(1 for s in signed if s < 0)
    outgoing_pkts = sum(1 for s in signed if s > 0)

    # Cumulative sum of signed sizes
    cumsum = np.cumsum(signed).astype(np.float64)

    # Sample at N evenly-spaced points
    if len(cumsum) >= CUMUL_N_FEATURES:
        indices = np.linspace(0, len(cumsum) - 1, CUMUL_N_FEATURES, dtype=int)
        sampled = cumsum[indices]
    else:
        # Pad with the last value if trace is shorter than N
        sampled = np.zeros(CUMUL_N_FEATURES, dtype=np.float64)
        sampled[:len(cumsum)] = cumsum
        sampled[len(cumsum):] = cumsum[-1]

    stats = np.array(
        [incoming_bytes, outgoing_bytes, incoming_pkts, outgoing_pkts],
        dtype=np.float64,
    )
    return np.concatenate([stats, sampled])


# Timing features (inspired by Rahman et al., PETS 2020 "Tik-Tok")
# Inter-packet timing carries information that pure sizes miss.
TIMING_N_BINS = 50  # Number of bins for timing histogram


def extract_timing_features(trace: TrafficTrace) -> np.ndarray:
    """Extract timing-based features from inter-packet delays.

    The idea (Rahman et al., PETS 2020):
    - Packet sizes alone miss temporal patterns
    - A fast burst of 10 packets in 0.01s is very different from
      10 packets spread over 2 seconds
    - Inter-packet timing is a separate information channel

    Features (70 total):
    - [0:5]: timing statistics (mean, std, min, max, median of inter-packet gaps)
    - [5:10]: burst timing (mean, std, min, max, median of inter-burst gaps)
    - [10:20]: percentiles of inter-packet gaps (10th, 20th, ..., 100th)
    - [20:70]: histogram of inter-packet gaps (50 bins, log-scaled)
    """
    packets = trace.packets

    if len(packets) < 2:
        return np.zeros(70, dtype=np.float64)

    # Inter-packet timing gaps
    gaps = [packets[i + 1].time - packets[i].time for i in range(len(packets) - 1)]
    gaps = [max(g, 1e-6) for g in gaps]  # Floor at 1 microsecond

    # Timing statistics
    gap_arr = np.array(gaps)
    timing_stats = np.array([
        np.mean(gap_arr),
        np.std(gap_arr),
        np.min(gap_arr),
        np.max(gap_arr),
        np.median(gap_arr),
    ], dtype=np.float64)

    # Burst timing: gaps between direction changes
    burst_gaps = []
    for i in range(len(packets) - 1):
        if packets[i].direction != packets[i + 1].direction:
            burst_gaps.append(packets[i + 1].time - packets[i].time)

    if burst_gaps:
        bg_arr = np.array(burst_gaps)
        burst_stats = np.array([
            np.mean(bg_arr),
            np.std(bg_arr),
            np.min(bg_arr),
            np.max(bg_arr),
            np.median(bg_arr),
        ], dtype=np.float64)
    else:
        burst_stats = np.zeros(5, dtype=np.float64)

    # Percentiles of inter-packet gaps
    percentiles = np.percentile(gap_arr, np.arange(10, 101, 10)).astype(np.float64)

    # Histogram of log-scaled inter-packet gaps
    log_gaps = np.log10(gap_arr + 1e-6)
    hist_vals, _ = np.histogram(log_gaps, bins=TIMING_N_BINS, range=(-6, 1))
    timing_hist = hist_vals.astype(np.float64)

    return np.concatenate([timing_stats, burst_stats, percentiles, timing_hist])


def extract_combined_features(trace: TrafficTrace) -> np.ndarray:
    """Extract the combined feature set: CUMUL + timing + stats.

    My own design — combining the best from each approach:
    - CUMUL captures the cumulative traffic shape (104 features)
    - Timing captures inter-packet delay patterns (70 features)
    - Burst stats from VNG++ (5 features: count, mean, std, max, min burst size)
    - Total: 179 features

    Why combine? Each feature type captures different information:
    - CUMUL = what bytes were sent
    - Timing = when they were sent
    - Bursts = how they were grouped
    """
    cumul = extract_cumul_features(trace)
    timing = extract_timing_features(trace)

    # Add compact burst statistics
    bursts = compute_bursts(trace)
    if bursts:
        abs_bursts = [abs(b) for b in bursts]
        burst_stats = np.array([
            len(bursts),
            np.mean(abs_bursts),
            np.std(abs_bursts),
            max(abs_bursts),
            min(abs_bursts),
        ], dtype=np.float64)
    else:
        burst_stats = np.zeros(5, dtype=np.float64)

    return np.concatenate([cumul, timing, burst_stats])


# --- Feature matrix builders (improvements) ---

def build_cumul_matrix(traces: list[TrafficTrace]) -> np.ndarray:
    """Build feature matrix for CUMUL. Each row = one trace."""
    return np.array([extract_cumul_features(t) for t in traces])


def build_timing_matrix(traces: list[TrafficTrace]) -> np.ndarray:
    """Build feature matrix for timing features. Each row = one trace."""
    return np.array([extract_timing_features(t) for t in traces])


def build_combined_matrix(traces: list[TrafficTrace]) -> np.ndarray:
    """Build feature matrix for combined features. Each row = one trace."""
    return np.array([extract_combined_features(t) for t in traces])
