"""
Trading-Technical-Indicators (tti) python library

File name: _machine_learning_rnn.py
    Implements a Recurrent Neural Network for the Machine Learning features of
    the tti library.
"""

from ._machine_learning_api import MachineLearningAPI
from ._machine_learning_data import MachineLearningData
from ..utils.constants import DEFAULT_TI_FEATURES


class MachineLearningRNN(MachineLearningAPI):
    """
    Machine Learning RNN class implementation.
    """

    def __init__(self):

        super().__init__(model_type='RNN')

    def mlTrainModel(self, input_data, ti_features=DEFAULT_TI_FEATURES,
                     price_diff_periods=1, pool_size=None, verbose=False):
        """
        Trains a RNN machine learning model for prices direction predictions.

        Args:
        input_data (pandas.DataFrame): The input data. Required input columns
            are dependent from the technical indicators which will be used as
            features (see ti_features argument). The largest column set is
            ``high``, ``low``, ``open``, ``close`` and ``volume``. The index is
            of type ``pandas.DatetimeIndex``.

        ti_features (list, default=DEFAULT_TI_FEATURES): List of indicators to
            be used as features for the ML data. The format of the list is
            ``[{'ti': 'indicator_class_name', 'kwargs': {...}, ...]``. Use
            ``DEFAULT_TI_FEATURES`` as usage example. The default value is to
            use all the 62 indicators.

        price_diff_periods (int, default=1): The price periods distance for
            which the price direction is evaluated. The default value is for
            ``1`` period (i.e. 1 day).

        pool_size (int, default=None): Pool size of the processes used when
            concurrency is applied. When None, then the created processes are
            equal to the number of the available cpu cores.

        verbose (bool, default=False): If set to True, processing information
            is sent to the console.
        """

        raise NotImplementedError

    def mlPredict(self, features_values):
        """
        Returns a prediction.

        Args:
        features_values (list): The features values for which a classification
            prediction should be returned, by the trained model.

        Returns:
            int: The class predicted by the trained model.
        """

        raise NotImplementedError
