import numpy as np
from scipy.spatial.distance import hamming as skl_hamming, euclidean as skl_euclidean, cosine as skl_cosine
from scipy.stats import pearsonr

from starter import pearson_dist, cos_dist, euclidean, hamming

# Test vectors
a = [1, 2, 3, 4, 5]
b = [2, 2, 3, 5, 6]
c = [1, 0, 1, 0, 1]
d = [1, 1, 0, 0, 1]

# Test cases
def test_pearson_dist():
    custom_result = pearson_dist(a, b)
    sklearn_result = 1 - pearsonr(a, b)[0]
    print(f"Pearson Distance Test: Custom Result = {custom_result}, sklearn Result = {sklearn_result}")
    assert np.isclose(custom_result, sklearn_result), "Pearson distance test failed."

def test_hamming():
    custom_result = hamming(c, d)
    sklearn_result = skl_hamming(c, d) * len(c)
    print(f"Hamming Distance Test: Custom Result = {custom_result}, sklearn Result = {sklearn_result}")
    assert custom_result == sklearn_result, "Hamming distance test failed."

def test_euclidean():
    custom_result = euclidean(a, b)
    sklearn_result = skl_euclidean(a, b)
    print(f"Euclidean Distance Test: Custom Result = {custom_result}, sklearn Result = {sklearn_result}")
    assert np.isclose(custom_result, sklearn_result), "Euclidean distance test failed."

def test_cosine_distance():
    custom_result = cos_dist(a, b)
    sklearn_result = skl_cosine(a, b)
    print(f"Cosine Distance Test: Custom Result = {custom_result}, sklearn Result = {sklearn_result}")
    assert np.isclose(custom_result, sklearn_result), "Cosine distance test failed."

# Run tests
test_pearson_dist()
test_hamming()
test_euclidean()
test_cosine_distance()