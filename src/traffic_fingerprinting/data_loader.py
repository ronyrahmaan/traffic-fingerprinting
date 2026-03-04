"""Load traffic trace CSV files and extract labels from filenames.

Each CSV file represents one traffic trace (one voice command instance).
The label (which command it was) is encoded in the filename.

CSV format:
    ,time,size,direction
    0,0.0,176.0,1.0
    1,0.0005,1500.0,1.0
    ...

Each file gets converted into a TrafficTrace: a list of Packet tuples
with (timestamp, size, direction) plus the command label.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Packet:
    """A single network packet.

    time: seconds since first packet in this trace
    size: packet size in bytes (1-1500)
    direction: +1 (device to cloud) or -1 (cloud to device)
    """

    time: float
    size: int
    direction: int


@dataclass
class TrafficTrace:
    """One complete traffic trace for a single voice command.

    packets: ordered list of packets in this trace
    label: the voice command (e.g., "what_is_the_weather")
    filename: original CSV filename for reference
    """

    packets: list[Packet] = field(default_factory=list)
    label: str = ""
    filename: str = ""

    @property
    def num_packets(self) -> int:
        """Total number of packets in this trace."""
        return len(self.packets)

    @property
    def duration(self) -> float:
        """Total duration of the trace in seconds."""
        if not self.packets:
            return 0.0
        return self.packets[-1].time - self.packets[0].time

    def signed_sizes(self) -> list[int]:
        """Get size * direction for each packet.

        This is the core feature used by every algorithm in the paper.
        Positive = outgoing, negative = incoming.
        Example: a 1500-byte incoming packet becomes -1500.
        """
        return [p.size * p.direction for p in self.packets]


# Regex to extract command name from filename.
# Filenames look like: what_is_the_weather_5_60_80_100_120_140_capture_1.csv
# or: alexa_5_30s_L_1.csv
# Just the command name part before the numeric suffixes.
LABEL_PATTERN = re.compile(r"^([a-zA-Z][a-zA-Z_' ]*?)_\d")


def extract_label(filename: str) -> str:
    """Pull the voice command name out of a CSV filename.

    Examples:
        "alexa_5_30s_L_1.csv" -> "alexa"
        "what_is_the_weather_5_60_80_100_120_140_capture_1.csv" -> "what_is_the_weather"
        "do_dogs_dream_5_1.csv" -> "do_dogs_dream"
    """
    stem = Path(filename).stem
    match = LABEL_PATTERN.match(stem)
    if match:
        return match.group(1)
    # Fallback: return the whole stem if regex fails
    return stem


def load_trace(filepath: Path) -> TrafficTrace:
    """Load a single CSV file into a TrafficTrace.

    The CSV has columns: (index), time, size, direction.
    Reads the CSV and converts to Packet format.
    """
    df = pd.read_csv(filepath)

    packets = []
    for _, row in df.iterrows():
        packets.append(
            Packet(
                time=float(row["time"]),
                size=int(row["size"]),
                direction=int(row["direction"]),
            )
        )

    return TrafficTrace(
        packets=packets,
        label=extract_label(filepath.name),
        filename=filepath.name,
    )


def load_dataset(data_dir: str | Path) -> list[TrafficTrace]:
    """Load all CSV trace files from a directory.

    Returns a list of TrafficTrace objects sorted by filename.
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    traces = []
    for f in csv_files:
        traces.append(load_trace(f))

    return traces


def get_labels_and_ids(traces: list[TrafficTrace]) -> tuple[list[str], dict[str, int]]:
    """Get sorted unique labels and a label-to-integer mapping.

    Returns:
        labels: sorted list of unique command names
        label_to_id: dict mapping each command name to an integer
    """
    labels = sorted(set(t.label for t in traces))
    label_to_id = {label: i for i, label in enumerate(labels)}
    return labels, label_to_id


def traces_to_arrays(
    traces: list[TrafficTrace],
    label_to_id: dict[str, int],
) -> tuple[list[TrafficTrace], np.ndarray]:
    """Convert traces into a list + integer label array for sklearn.

    Returns:
        traces: the same traces (for feature extraction later)
        y: numpy array of integer labels
    """
    y = np.array([label_to_id[t.label] for t in traces])
    return traces, y
