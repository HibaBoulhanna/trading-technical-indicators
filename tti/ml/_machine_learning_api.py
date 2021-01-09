"""
Trading-Technical-Indicators (tti) python library

File name: _machine_learning_api.py
    Defines the API for the Machine Learning features of the tti library.
"""

import pickle

from abc import ABC, abstractmethod
from ..utils.constants import ALL_TI_FEATURES, ML_CLASSES


class MachineLearningAPI(ABC):
    """
    Machine Learning API class implementation.

    Args:
        model_type (str): The model type of the trained model (i.e. RNN, LSTM
            etc). It is set by the child class.

    Attributes:
        _model (object): The trained model object.

        _scaler (sklearn.preprocessing.StandardScaler): Scaler used with the
            training data.

        _model_details (dict): Dictionary with model details. It contains the
            ``model_type`` see model_type argument, ``training_score`` the
            score of the trained model on the training data, ``test_score`` the
            score of the trained model on the test data,
            ``number_of_training_instances`` the number of instances used to
            train the model, ``classes`` a definition (dict) of the classes,
            ``scaler_used`` whether a scaler used with the training data, and
            ``dump_file`` the file where the model is saved.
    """
    
    def __init__(self, model_type):

        # Trained model, when None there is not any trained model yet
        self._model = None

        # Data scaler, when None there is not any scaler used
        self._scaler = None

        # Training score, test score, number of training instances are set by
        # the child class
        self._model_details = {'model_type': model_type,
                               'training_score': None,
                               'test_score': None,
                               'number_of_training_instances': None,
                               'classes': ML_CLASSES,
                               'scaler_used': False,
                               'dump_file': None}

    def mlSaveModel(self, file_name):
        """
        Saves the trained model to a file.

        Args:
        file_name (str): The file where the serialized trained model should be
            saved.
        """

        self._model_details['dump_file'] = file_name

        self._model_details['scaler_used'] = False if self._scaler is None \
            else True

        with open(file_name, 'wb') as out_file:
            pickle.dump((self._scaler, self._model, self._model_details),
                        out_file)

    def mlLoadModel(self, file_name):
        """
        Loads a trained model from a file.

        Args:
        file_name (str): The file from where the trained model should be
            deserialized.
        """

        with open(file_name, 'rb') as in_file:
            self._scaler, self._model, self._model_details = \
                pickle.load(in_file)

    def mlModelDetails(self):
        """
        Returns details about the trained model.

        Returns:
            dict: The model details dictionary. See class attributes for
            details.
        """

        self._model_details['scaler_used'] = False if self._scaler is None \
            else True

        return self._model_details

    @abstractmethod
    def mlTrainModel(self, input_data, ti_features=ALL_TI_FEATURES,
                 include_close_feature=False, include_volume_feature=False,
                 price_diff_periods=1, pool_size=None, verbose=False):
        """
        Trains a machine learning model for prices direction predictions.

        Args:
            input_data (pandas.DataFrame): The input data. Required input
                columns are dependent from the technical indicators which will
                be used as features (see ti_features argument). The largest
                column set is ``high``, ``low``, ``open``, ``close`` and
                ``volume``. The index is of type ``pandas.DatetimeIndex``. The
                ``close`` and ``volume`` columns are required in case they are
                included also as features (see ``include_close_feature`` and
                ``include_volume_feature`` arguments).

            ti_features (list or None, default=ALL_TI_FEATURES): List of
                indicators to be used as features for the ML data. The format
                of the list is ``[{'ti': 'indicator_class_name', 'kwargs':
                {...}, ...]``. See ``ALL_TI_FEATURES`` as usage example. The
                default value is to use all the 62 indicators. If None, or
                empty list is given, then none ti signal is included as feature
                in the created ML data.

            include_close_feature (bool, default=False): Indicates whether the
                ``close`` value from the input data should be included as a
                feature in the produced ML data.

            include_volume_feature (bool, default=False): Indicates whether the
                ``volume`` value from the input data should be included as a
                feature in the produced ML data.

            price_diff_periods (int, default=1): The number of periods ahead
                for which the price direction is evaluated. The default value
                is for ``1`` period (i.e. 1 day).

            pool_size (int, default=None): Pool size parallel computing when
                concurrency applies.  When None, then the created processes are
                equal to the number of the available cpu cores.

            verbose (bool, default=False): If set to True, processing
                information is sent to the console.

        Raises:
            WrongTypeForInputParameter: Input argument has wrong type
            WrongValueForInputParameter: Unsupported value for input argument
            NotEnoughInputData: Not enough data for calculating the indicator
            TypeError: Type error occurred when validating the ``input_data``
            ValueError: Value error occurred when validating the ``input_data``
            NoFeaturesSelectedForMLData: No features selected for ML data
        """

        raise NotImplementedError

    @abstractmethod
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
