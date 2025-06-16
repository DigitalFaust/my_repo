import math
import numpy as np
import pytest

from cosine_similarity import cosine_similarity


def test_identical_vectors():
    assert math.isclose(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)


def test_opposite_vectors():
    assert math.isclose(cosine_similarity([1, 0], [-1, 0]), -1.0)


def test_orthogonal_vectors():
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)


def test_list_and_numpy_array():
    vec = np.array([1, 2, 3, 4])
    assert math.isclose(cosine_similarity([1, 2, 3, 4], vec), 1.0)


def test_three_dimension():
    a = [1, 2, 3]
    b = [3, 2, 1]
    expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert math.isclose(cosine_similarity(a, b), expected)


def test_shape_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0, 0], [1, 2, 3])


def test_two_dimensional_arrays():
    a = np.array([[1, 0], [0, 1]])
    b = np.array([[1, 0], [0, 1]])
    assert math.isclose(cosine_similarity(a, b), 1.0)


def test_random_vectors_seed():
    rng = np.random.default_rng(0)
    a = rng.random(10)
    b = rng.random(10)
    expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert math.isclose(cosine_similarity(a, b), expected)


def test_negative_values():
    a = [-1, -2, -3]
    b = [1, 0, -1]
    expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert math.isclose(cosine_similarity(a, b), expected)
