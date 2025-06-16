import numpy as np

def cosine_similarity(a, b):
    """Compute the cosine similarity between two vectors.

    Parameters
    ----------
    a : array_like
        First input vector.
    b : array_like
        Second input vector.

    Returns
    -------
    float
        The cosine similarity between ``a`` and ``b``.

    Raises
    ------
    ValueError
        If the input vectors do not have the same size or if either is
        a zero vector.
    """
    a = np.ravel(np.asarray(a, dtype=float))
    b = np.ravel(np.asarray(b, dtype=float))

    if a.size != b.size:
        raise ValueError("Input vectors must have the same number of elements")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Input vectors must not be zero vectors")

    dot_prod = float(np.dot(a, b))
    return dot_prod / (norm_a * norm_b)
