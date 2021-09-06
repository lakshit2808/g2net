from g2net.eda import merge_iters, concat_series, get_indexed_items
import pandas as pd
import pytest

# Given
@pytest.mark.parametrize(
    'iters, expected_merged_iters', 
    (
        pytest.param(
            [[1], [3], [9, 8], [], [0, 2]], 
            [1, 3, 9, 8, 0, 2],
            id='list-of-lists'), 
        pytest.param(
            ([], [], []), 
            [],
            id='tuple-of-empty-lists'), 
    )
)
def test_merge_iters(iters, expected_merged_iters):
    # When
    merged_iters = merge_iters(iters) 

    # Then
    assert expected_merged_iters == merged_iters


# Given
TEST_SERIES_1 = pd.Series([1, 2, 3])
TEST_SERIES_2 = pd.Series([])
TEST_SERIES_3 = pd.Series([10])
@pytest.mark.parametrize(
    'serieses, expected_concated_series', 
    (
        pytest.param(
            [TEST_SERIES_1, TEST_SERIES_2, TEST_SERIES_3], 
            pd.Series([1, 2, 3, 10]),
            id='list-of-serieses'), 
        pytest.param(
            pd.Series([TEST_SERIES_2, TEST_SERIES_3, TEST_SERIES_1]), 
            pd.Series([10, 1, 2, 3]),
            id='series-of-serieses'), 
    )
)
def test_concat_series(serieses, expected_concated_series):
    # When
    concated_series = concat_series(serieses)

    # Then
    assert (expected_concated_series == concated_series).all()


# Given
@pytest.mark.parametrize(
    'source_list, indices, expected_extracted_list', 
    (
        pytest.param(
            [1, 2, 3, 4, 5], 
            [0, 3, 2],
            [1, 4, 3],
            id='list-of-indices-from-list'), 
        pytest.param(
            [1, 2, 3, 4, 5],
            [],
            [],
            id='empty-index-list'), 
    )
)
def test_get_indexed_items(source_list, indices, expected_extracted_list):
    # When
    extracted_list = get_indexed_items(source_list, indices)

    # Then 
    assert expected_extracted_list == extracted_list
