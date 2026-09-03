"""MIT-BIH download, loading, filtering, and annotated beat segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import wfdb
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ALL_RECORDS, LABEL_MAP, SignalConfig
from .preprocess import bandpass_filter, normalize_beat


@dataclass(frozen=True)
class BeatDataset:
    beats: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    sample_indices: np.ndarray
    symbols: np.ndarray

    def select_records(self, records: tuple[str, ...]) -> "BeatDataset":
        mask = np.isin(self.record_ids, np.asarray(records))
        return BeatDataset(
            beats=self.beats[mask],
            labels=self.labels[mask],
            record_ids=self.record_ids[mask],
            sample_indices=self.sample_indices[mask],
            symbols=self.symbols[mask],
        )


def download_mitdb(data_dir: Path) -> None:
    """Download the required records with retry and existing-file reuse."""
    data_dir.mkdir(parents=True, exist_ok=True)
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET",)),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))

    base_url = "https://physionet.org/files/mitdb/1.0.0"
    for record_id in ALL_RECORDS:
        for extension in ("hea", "dat", "atr"):
            destination = data_dir / f"{record_id}.{extension}"
            if destination.exists() and destination.stat().st_size > 0:
                continue

            url = f"{base_url}/{record_id}.{extension}"
            temporary = destination.with_suffix(destination.suffix + ".part")
            print(f"Downloading {destination.name}")
            with session.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            temporary.replace(destination)


def _select_lead(record: wfdb.Record, preferred: str) -> int:
    if preferred in record.sig_name:
        return record.sig_name.index(preferred)
    return 0


def extract_record(
    data_dir: Path,
    record_id: str,
    config: SignalConfig,
) -> BeatDataset:
    """Extract fixed-length N and V beats using reference annotations."""
    record_path = str(data_dir / record_id)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    if int(record.fs) != config.sampling_rate:
        raise ValueError(
            f"Record {record_id} has fs={record.fs}; expected {config.sampling_rate}"
        )

    lead_index = _select_lead(record, config.preferred_lead)
    signal = np.asarray(record.p_signal[:, lead_index], dtype=np.float64)
    signal = bandpass_filter(signal, record.fs, config)

    beats: list[np.ndarray] = []
    labels: list[int] = []
    sample_indices: list[int] = []
    symbols: list[str] = []

    for peak, symbol in zip(annotation.sample, annotation.symbol, strict=True):
        if symbol not in LABEL_MAP:
            continue

        start = int(peak) - config.pre_r_samples
        end = int(peak) + config.post_r_samples
        if start < 0 or end > signal.shape[0]:
            continue

        beat = signal[start:end]
        if beat.shape[0] != config.beat_length:
            continue
        if config.normalize_each_beat:
            beat = normalize_beat(beat)

        beats.append(beat)
        labels.append(LABEL_MAP[symbol])
        sample_indices.append(int(peak))
        symbols.append(symbol)

    return BeatDataset(
        beats=np.asarray(beats, dtype=np.float64).reshape(-1, config.beat_length),
        labels=np.asarray(labels, dtype=np.int64),
        record_ids=np.full(len(beats), record_id, dtype="U3"),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        symbols=np.asarray(symbols, dtype="U2"),
    )


def build_dataset(data_dir: Path, config: SignalConfig) -> BeatDataset:
    missing = [
        data_dir / f"{record_id}.{extension}"
        for record_id in ALL_RECORDS
        for extension in ("hea", "dat", "atr")
        if not (data_dir / f"{record_id}.{extension}").exists()
    ]
    if missing:
        preview = ", ".join(path.name for path in missing[:5])
        raise FileNotFoundError(
            f"MIT-BIH is incomplete ({len(missing)} files missing, e.g. {preview}). "
            "Run 'pcghe-ecg download' first."
        )

    parts = []
    for record_id in ALL_RECORDS:
        print(f"Extracting record {record_id}")
        parts.append(extract_record(data_dir, record_id, config))
    return BeatDataset(
        beats=np.concatenate([p.beats for p in parts]),
        labels=np.concatenate([p.labels for p in parts]),
        record_ids=np.concatenate([p.record_ids for p in parts]),
        sample_indices=np.concatenate([p.sample_indices for p in parts]),
        symbols=np.concatenate([p.symbols for p in parts]),
    )


def save_dataset(dataset: BeatDataset, cache_file: Path) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        beats=dataset.beats,
        labels=dataset.labels,
        record_ids=dataset.record_ids,
        sample_indices=dataset.sample_indices,
        symbols=dataset.symbols,
    )


def load_dataset(cache_file: Path) -> BeatDataset:
    with np.load(cache_file, allow_pickle=False) as data:
        return BeatDataset(
            beats=data["beats"],
            labels=data["labels"],
            record_ids=data["record_ids"],
            sample_indices=data["sample_indices"],
            symbols=data["symbols"],
        )


def load_or_build_dataset(
    data_dir: Path,
    cache_file: Path,
    config: SignalConfig,
    force: bool = False,
) -> BeatDataset:
    if cache_file.exists() and not force:
        return load_dataset(cache_file)
    dataset = build_dataset(data_dir, config)
    save_dataset(dataset, cache_file)
    return dataset
