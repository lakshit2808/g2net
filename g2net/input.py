from typing import Dict, Any, Iterable, List, Optional, Tuple
import os
import pandas as pd
import numpy as np


def find_files_with_suffix_rooted_at_path(suffix: str, root_path: str, max_file_count: int) -> Dict[str, str]:
    """
    Find all the files with the given suffix, inside the directory and subdirectories of root_path.
    """
    all_files = {}

    for (dir_path, _, file_names) in os.walk(root_path):
        for file_name in file_names:
            if file_name.endswith(suffix):
                all_files[file_name.replace(suffix, '')] = os.path.join(dir_path, file_name)
                
                if max_file_count and len(all_files) >= max_file_count:
                    return all_files
    
    return all_files


def load_n_samples_with_label(
    all_file_names: List[str], 
    all_labels: Dict[str, Any], 
    batch_begin: int,
    n_sample: int,
    expected_shape: Tuple[int, int]
) -> Tuple[List, List]:
    raw_features = []
    raw_labels = []

    for sample_id, file_path in list(all_file_names.items())[batch_begin:]:
        # Stop when we have processed enough samples.
        if n_sample and len(raw_features) >= n_sample:
            break

        # Make sure the sample has a label in the training label file.
        if sample_id not in all_labels:
            continue
        
        # Load the training signal, note we know this will be a 3 channel time series.
        try:
            sample_x_raw = np.load(file_path)
        except FileNotFoundError as f:
            print(f)
            continue
        if not sample_x_raw.shape == expected_shape:
            continue

        # Convert the ND-Array into a dictionary.
        sample_x = [pd.Series(sample_x_raw[c_id, :]) for c_id in range(expected_shape[0])]

        # Save the samples into a list.
        raw_features.append(sample_x)
        raw_labels.append(all_labels[sample_id])

    return raw_features, raw_labels


def extract_dict_from_df(source_df: pd.DataFrame, key_col: str, val_col: str) -> Dict[str, Any]:
    if not key_col in source_df.columns or not val_col in source_df.columns:
        raise ValueError('Invalid Key or Value column name')
    
    return {k: v for (k, v) in zip(source_df.loc[:, key_col], source_df.loc[:, val_col])}
