from scripts.prepare_dataset import split_indices


def test_split_indices_determinism():
    a = split_indices(10, 0.2, seed=42)
    b = split_indices(10, 0.2, seed=42)
    assert a == b

