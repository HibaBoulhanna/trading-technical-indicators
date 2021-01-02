"""
Trading-Technical-Indicators (tti) python library

File name: _machine_learning_data.py
    Implements Data related functions for the Machine Learning features of
    the tti library.
"""


class MachineLearningData:
    """
    Machine Learning Data class implementation.
    """

    def __init__(self):

        raise NotImplementedError

    def createMLData(self):
        """
        Constructs data to be used for the ML model creation.
        """

        raise NotImplementedError

    def _createMLDataFeatures(self):
        """
        Creates the features part of the ML data.
        """

        raise NotImplementedError

    def _createMLDataLabels(self, ):
        """
        Creates the labels part of the ML data.
        """

        raise NotImplementedError

    def loadMLData(self):
        """
        Loads ML data (features and labels) from a csv file.
        """

        raise NotImplementedError

    def saveMLData(self):
        """
        Saves ML data (features and labels) to a csv file.
        """

        raise NotImplementedError
