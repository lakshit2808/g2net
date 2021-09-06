from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple
from .input import load_n_samples_with_label
import joblib
import logging
import os


class CkptClassifier:
    def __init__(self, classifier_class, ckpt_path: Optional[str]=None, **model_params) -> None:
        self._classifier_class = classifier_class
        self._ckpt_path = ckpt_path
        self.classifier = self._classifier_class(**model_params)

    def __enter__(self):
        # Create a classifier from saved data        
        if self._ckpt_path and os.path.isfile(self._ckpt_path):
            try:
                with open(self._ckpt_path, 'rb') as ckpt_file:
                    # Load the checkpoint file
                    self.classifier = joblib.load(ckpt_file)
                
            except EOFError as e:
                logging.warning(f'Failed to load checkpoint: {e}')

        return self.classifier
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._ckpt_path: 
            logging.warning('Checkpoint path not given, model parameters will not be saved')
            return
        
        with open(self._ckpt_path, 'wb') as ckpt_file:
            joblib.dump(self.classifier, ckpt_file)


_BATCH_ID = 'batch_id'


@dataclass
class CkptDataLoader:
    all_file_names: List[str]
    all_labels: Dict[str, Any]
    n_batch: int
    batch_size: int
    expected_shape: Tuple[int, int]
    ckpt_path: Optional[str] = None
    _batch_id: Optional[int] = 0
    
    def __enter__(self):
        # Create a DateLoader from saved data        
        if self.ckpt_path and os.path.isfile(self.ckpt_path):
            try:
                with open(self.ckpt_path, 'rb') as ckpt_file:
                    # Load the checkpoint file
                    ckpt_data = joblib.load(ckpt_file)
                    self._batch_id = ckpt_data.get(_BATCH_ID, 0)
            except EOFError as e:
                logging.warning(f'Failed to load checkpoint: {e}')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.ckpt_path: 
            logging.warning('Checkpoint path not given, batch ID will not be saved')
            return
        
        with open(self.ckpt_path, 'wb') as ckpt_file:
            joblib.dump({_BATCH_ID: self._batch_id}, ckpt_file)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._batch_id < self.n_batch:
            samples = load_n_samples_with_label(self.all_file_names, 
                                                self.all_labels, 
                                                self._batch_id * self.batch_size, 
                                                self.batch_size, 
                                                self.expected_shape)
            self._batch_id += 1
            return samples
        else:
            raise StopIteration
