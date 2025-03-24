# Abstract encoder
from abc import ABC, abstractmethod

class BaseEncoder(ABC):
    '''Abstract encoder for all encoder class'''

    @abstractmethod
    def load_model(self, model_name: str):
        '''Load a pretrain model with a path'''
        pass

    @abstractmethod
    def encode(self, data):
        '''encode data to vector'''
        pass

    @abstractmethod
    def save_model(self,save_path: str):
        '''save the current model to given path'''
        pass

    @abstractmethod
    def model_info(self) -> dict:
        '''return model metadata'''