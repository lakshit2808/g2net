from g2net.input import extract_dict_from_df
import pandas as pd
import pytest


@pytest.mark.parametrize(
    'data_dict, key_col, val_col, expected_dict', 
    (
        pytest.param(
            {
                'col1': [1, 2, 5], 
                'col2': [3, 4, 6]
            }, 
            'col1', 
            'col2', 
            {
                1: 3, 
                2: 4, 
                5: 6
            }, 
            id='2-columns-only'), 
        pytest.param(
            {
                'col1': [1, 2, 5], 
                'col2': [3, 4, 6], 
                'col3': [-1, -2, -3]
            }, 
            'col3', 
            'col1', 
            {
                -1: 1, 
                -2: 2, 
                -3: 5
            }, 
            id='3-columns'), 
    )
)
def test_extract_dict_from_df(data_dict, key_col, val_col, expected_dict):
    # Given
    source_df = pd.DataFrame(data=data_dict)

    # When
    result_dict = extract_dict_from_df(source_df, key_col, val_col)
    
    # Then
    assert expected_dict == result_dict
