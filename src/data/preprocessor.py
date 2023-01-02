from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from src.model.config import SEQ_MAX_LENGTH


class IPreprocessor(metaclass=ABCMeta):

    @abstractmethod
    def transform(sefl, x_batch, y_batch=None):
        return NotImplemented("not implemented")

    
class GPT2Preprocessor(IPreprocessor):

    def __init__(self, tokenizer, eos_token, bos_token) -> None:
        self.tokenizer = tokenizer

        self.bos_token = bos_token
        self.eos_token = eos_token


    def transform(self, x: str) -> Tuple:
        
        if self.bos_token:
            x_inp = x
        else:
            x_inp = x[1: ]


        x_batch = self.tokenizer(self.bos_token + x, 
                                padding='max_length', 
                                truncation=True, 
                                max_length= SEQ_MAX_LENGTH, 
                                return_tensors='pt')


        y_batch = self.tokenizer(x_inp + self.eos_token, 
                                padding='max_length', 
                                truncation=True, 
                                max_length= SEQ_MAX_LENGTH, 
                                return_tensors='pt')

        return (x_batch, y_batch, )
        


    
