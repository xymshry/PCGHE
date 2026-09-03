"""Fixed experiment protocol for the MIT-BIH N-versus-V study."""

from __future__ import annotations

from dataclasses import asdict, dataclass


# De Chazal-style inter-patient split. Records 102, 104, 107, and 217 are
# excluded because they contain paced beats. Record 112 is part of DS1.
DS1_RECORDS = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118",
    "119", "122", "124", "201", "203", "205", "207", "208", "209",
    "215", "220", "223", "230",
)

DS2_RECORDS = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200",
    "202", "210", "212", "213", "214", "219", "221", "222", "228",
    "231", "232", "233", "234",
)

# Primary hyperparameters are fixed in advance, so all DS1 records train the
# final model. Any future tuning must use record-grouped cross-validation
# within DS1 rather than reserving an arbitrary set of four records.
VALIDATION_RECORDS: tuple[str, ...] = ()
TRAIN_RECORDS = DS1_RECORDS
ALL_RECORDS = DS1_RECORDS + DS2_RECORDS

LABEL_MAP = {"N": 0, "V": 1}


@dataclass(frozen=True)
class SignalConfig:
    sampling_rate: int = 360
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    filter_order: int = 4
    pre_r_samples: int = 90
    post_r_samples: int = 166
    beat_length: int = 256
    preferred_lead: str = "MLII"
    normalize_each_beat: bool = True

    def __post_init__(self) -> None:
        if self.pre_r_samples + self.post_r_samples != self.beat_length:
            raise ValueError("Pre- and post-R samples must equal beat_length")


@dataclass(frozen=True)
class FeatureConfig:
    kind: str = "dwt"
    wavelet: str = "db4"
    wavelet_level: int = 4
    wavelet_mode: str = "periodization"


@dataclass(frozen=True)
class ModelConfig:
    pca_components: int | None = 32
    class_weight: str = "balanced"
    max_iter: int = 2000
    c: float = 1.0
    random_state: int = 2026
    threshold: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    signal: SignalConfig = SignalConfig()
    feature: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["protocol"] = {
            "train_records": list(TRAIN_RECORDS),
            "test_records": list(DS2_RECORDS),
            "label_map": LABEL_MAP,
        }
        return result
