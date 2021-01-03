"""
Trading-Technical-Indicators (tti) python library

File name: _machine_learning_api.py
    Defines the API for the Machine Learning features of the tti library.
"""

from abc import ABC, abstractmethod


class MachineLearningAPI(ABC):
    """
    Machine Learning API class implementation.
    """
    
    def __init__(self):

        raise NotImplementedError

    def mlSaveModel(self):
        """
        Saves a trained model to a file.
        """

        raise NotImplementedError

    def mlLoadModel(self):
        """
        Loads a trained model from a file.
        """

        raise NotImplementedError

    def mlModelDetails(self):
        """
        Returns details about the trained model.
        """

        raise NotImplementedError

    @abstractmethod
    def mlTrainModel(self):
        """
        Trains a machine learning model for prices direction predictions.
        """

        raise NotImplementedError

    @abstractmethod
    def mlPredict(self):
        """
        Returns a prediction.
        """

        raise NotImplementedError
