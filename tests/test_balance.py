import numpy as np
import pytest

from cadgnn.training.balance import balance_indices, smote

Y = np.array([0] * 8 + [1] * 3)


def test_no_resampling_keeps_everything():
    for method in ("none", "class_weight"):
        assert balance_indices(Y, method, seed=0).tolist() == list(range(len(Y)))


def test_oversampling_copies_the_smaller_class():
    picked = balance_indices(Y, "oversample", seed=0)
    assert np.bincount(Y[picked]).tolist() == [8, 8]
    assert set(range(len(Y))) <= set(picked.tolist())


def test_undersampling_drops_part_of_the_larger_class():
    picked = balance_indices(Y, "undersample", seed=0)
    assert np.bincount(Y[picked]).tolist() == [3, 3]
    assert len(set(picked.tolist())) == 6
    assert {8, 9, 10} <= set(picked.tolist())


def test_resampling_is_reproducible():
    a = balance_indices(Y, "oversample", seed=4)
    assert a.tolist() == balance_indices(Y, "oversample", seed=4).tolist()


def test_bad_resampling_requests():
    with pytest.raises(ValueError, match="unknown resampling"):
        balance_indices(Y, "smote", seed=0)
    with pytest.raises(ValueError, match="both classes"):
        balance_indices(np.zeros(4), "oversample", seed=0)


def test_smote_adds_points_between_minority_samples():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (8, 2)), rng.normal(10, 1, (3, 2))])
    new_x, new_y = smote(x, Y, seed=0, k=2)
    assert np.bincount(new_y).tolist() == [8, 8]
    assert np.array_equal(new_x[: len(x)], x)
    added = new_x[len(x):]
    low, high = x[Y == 1].min(axis=0), x[Y == 1].max(axis=0)
    assert ((added >= low - 1e-9) & (added <= high + 1e-9)).all()


def test_smote_needs_two_minority_samples():
    with pytest.raises(ValueError, match="at least two"):
        smote(np.zeros((4, 2)), np.array([0, 0, 0, 1]), seed=0)
