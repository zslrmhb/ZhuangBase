# Abstract preprocessor
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    '''Abstract preprocessor for all preprocessor class'''

    @abstractmethod
    def preprocess(self, data):
        '''Preprocess the input data'''
        pass 