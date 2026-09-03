from pcghe_ecg.config import DS2_RECORDS, TRAIN_RECORDS, VALIDATION_RECORDS, SignalConfig


def test_record_splits_are_disjoint():
    assert set(TRAIN_RECORDS).isdisjoint(VALIDATION_RECORDS)
    assert set(TRAIN_RECORDS).isdisjoint(DS2_RECORDS)
    assert set(VALIDATION_RECORDS).isdisjoint(DS2_RECORDS)


def test_beat_window_is_exactly_256_samples():
    config = SignalConfig()
    assert config.pre_r_samples + config.post_r_samples == config.beat_length == 256
