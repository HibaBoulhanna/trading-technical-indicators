"""
Trading-Technical-Indicators (tti) python library

File name: _machine_learning_data.py
    Implements Data related functions for the Machine Learning features of
    the tti library.
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

from ..indicators import *
from ..utils.constants import ALL_TI_FEATURES, ML_CLASSES
from ..utils.exceptions import WrongValueForInputParameter, \
    WrongTypeForInputParameter, NotEnoughInputData, NoFeaturesSelectedForMLData
from ..utils import fillMissingValues


class MachineLearningData:
    """
    Machine Learning Data class implementation. Creates ML data to be used for
    the ML model training. The columns can be the trading signals for chosen
    technical indicators (``ti_features``), the ``close`` price, and the
    ``volume`` for a equity. The exact columns are defined from the received
    input arguments. The labels are the price direction between two data points
    (n_periods_future_price - current_price). The number of periods in the
    future is defined by the ``price_diff_periods`` input argument. The
    possible values of the labels are ``0`` for ``downward`` price direction or
    when price does not change, and ``1`` for ``upward`` price direction.

    Args:
        input_data (pandas.DataFrame): The input data. Required input columns
            are dependent from the technical indicators which will be used as
            features (see ti_features argument). The largest column set is
            ``high``, ``low``, ``open``, ``close`` and ``volume``. The index is
            of type ``pandas.DatetimeIndex``. The ``close`` and ``volume``
            columns are required in case they are included also as features
            (see ``include_close_feature`` and ``include_volume_feature``
            arguments).

        ti_features (list or None, default=ALL_TI_FEATURES): List of
            indicators to be used as features for the ML data. The format of
            the list is ``[{'ti': 'indicator_class_name', 'kwargs': {...},
            ...]``. See ``ALL_TI_FEATURES`` as usage example. The default
            value is to use all the 62 indicators. If None, or empty list is
            given, then none ti signal is included as feature in the created ML
            data.

        include_close_feature (bool, default=False): Indicates whether the
            ``close`` value from the input data should be included as a feature
            in the produced ML data.

        include_volume_feature (bool, default=False): Indicates whether the
            ``volume`` value from the input data should be included as a 
            feature in the produced ML data.

        price_diff_periods (int, default=1): The number of periods ahead for
            which the price direction is evaluated. The default value is for
            ``1`` period (i.e. 1 day).

        pool_size (int, default=None): Pool size of the processes used when
            calculating the features values for the ML Data. When None, then
            the created processes are equal to the number of the available cpu
            cores.

        verbose (bool, default=False): If set to True, processing information
            is sent to the console.

    Attributes:
        _input_data (pandas.DataFrame): See input_data argument for
            description.

        _ti_features (list): See ti_features argument for description.
        
        _include_close_feature (bool, default=False): See include_close_feature
            argument for description.

        _include_volume_feature (bool, default=False): See 
            include_volume_feature argument for description.

        _price_diff_periods (int): See price_diff_periods argument for
            description.

        _pool_size (int): See pool_size argument for description.

        _verbose (bool): See verbose argument for description.

        _indicators_set (list): List of the indicators instances created based
            on the ti_features input argument.

        _ml_data (numpy.ndarray): The created ML data. Last column is the data
            labels, and the rest of the columns are the data features.

    Raises:
        WrongTypeForInputParameter: Input argument has wrong type.
        WrongValueForInputParameter: Unsupported value for input argument.
        NotEnoughInputData: Not enough data for calculating the indicator.
        TypeError: Type error occurred when validating the ``input_data``.
        ValueError: Value error occurred when validating the ``input_data``.
        NoFeaturesSelectedForMLData: No features selected for ML data.
    """

    def __init__(self, input_data, ti_features=ALL_TI_FEATURES, 
                 include_close_feature=False, include_volume_feature=False,
                 price_diff_periods=1, pool_size=None, verbose=False):

        self._input_data = fillMissingValues(input_data=input_data)
        self._ti_features = ti_features
        self._include_close_feature = include_close_feature
        self._include_volume_feature = include_volume_feature
        self._price_diff_periods = price_diff_periods
        self._pool_size = pool_size
        self._verbose = verbose

        # Context added in the _validateInputArguments
        self._indicators_set = []

        # If fails, exception is raised
        self._validateInputArguments()

        # Context is updated in createMLData
        # Columns are the signals for the selected ti_features, optionally the
        # close price and volume, and the labels. Rows are same as the number
        # of the input data, reduced by the price ahead lookup period used.
        self._ml_data = np.zeros(
            shape=(
                len(self._input_data.index) - price_diff_periods,
                len(self._ti_features) +
                (1 if self._include_close_feature else 0) +
                (1 if self._include_volume_feature else 0) + 1),
            dtype=np.float64)

    def _validateInputArguments(self):
        """
        Validates the input arguments and sets the _indicators_set by
        instantiating the indicators which will be used as features.

        Raises:
            WrongTypeForInputParameter: Input argument has wrong type
            WrongValueForInputParameter: Unsupported value for input argument
            NotEnoughInputData: Not enough data for calculating the indicator
            TypeError: Type error occurred when validating the ``input_data``
            ValueError: Value error occurred when validating the ``input_data``
            NoFeaturesSelectedForMLData: No features selected for ML data.
        """

        if not isinstance(self._verbose, bool):

            raise WrongTypeForInputParameter(type(self._verbose),
                                             'verbose', 'bool')

        if not isinstance(self._include_close_feature, bool):

            raise WrongTypeForInputParameter(type(self._include_close_feature),
                                             'include_close_feature', 'bool')

        if not isinstance(self._include_close_feature, bool):

            raise WrongTypeForInputParameter(
                type(self._include_volume_feature), 'include_volume_feature',
                'bool')

        if self._ti_features is None:
            self._ti_features = []

        if len(self._ti_features) == 0 and not self._include_close_feature \
                and not self._include_volume_feature:

            raise NoFeaturesSelectedForMLData(
                ti_features=self._ti_features,
                include_close_feature=self._include_close_feature,
                include_volume_feature=self._include_volume_feature)

        if not isinstance(self._verbose, bool):

            raise WrongTypeForInputParameter(type(self._verbose),
                                             'verbose', 'bool')

        if isinstance(self._price_diff_periods, int):
            if ((self._price_diff_periods < 1) or
                    (self._price_diff_periods >= len(self._input_data.index))):

                raise WrongValueForInputParameter(
                    self._price_diff_periods, 'price_diff_periods',
                    '>0 and <' + str(len(self._input_data.index)))

        else:
            raise WrongTypeForInputParameter(
                type(self._price_diff_periods), 'price_diff_periods', 'int')

        if isinstance(self._pool_size, int):
            if self._pool_size < 1:
                raise WrongValueForInputParameter(
                    self._pool_size, 'pool_size', '>0')
        elif self._pool_size is not None:
            raise WrongTypeForInputParameter(
                type(self._pool_size), 'pool_size', 'int or None')

        # Validate ti_features
        supported_indicators = []
        for item in ALL_TI_FEATURES:
            supported_indicators.append(item['ti'])

        if isinstance(self._ti_features, list) and len(self._ti_features) > 0:
            for item in self._ti_features:

                for key in item.keys():
                    if key not in ['ti', 'kwargs']:
                        raise WrongTypeForInputParameter(
                            self._ti_features, 'ti_features',
                            '[{\'ti\': \'indicator_class_name\', \'kwargs\':' +
                            ' {...}, ...]')

                if 'ti' not in item.keys() or 'kwargs' not in item.keys():
                    raise WrongTypeForInputParameter(
                        self._ti_features, 'ti_features',
                        '[{\'ti\': \'indicator_class_name\', \'kwargs\':' +
                        ' {...}, ...]')

                if item['ti'] not in supported_indicators:

                    raise WrongValueForInputParameter(
                        item['ti'], 'ti_features[\'ti\']',
                        str(supported_indicators))

                if not isinstance(item['kwargs'], dict):

                    raise WrongTypeForInputParameter(
                        self._ti_features, 'ti_features',
                        '[{\'ti\': \'indicator_class_name\', \'kwargs\': ' +
                        '{...}, ...]')

        elif isinstance(self._ti_features, list) and \
                len(self._ti_features) == 0:

            pass

        else:
            raise WrongTypeForInputParameter(
                self._ti_features, 'ti_features',
                '[{\'ti\': \'indicator_class_name\', \'kwargs\': ' +
                '{...}, ...]')

        if len(self._input_data.index) - self._price_diff_periods <= 0:
            raise NotEnoughInputData(
                '', '> ' + str(self._price_diff_periods),
                len(self._input_data.index),
                'Not enough input data. Minimum required data are '
                '(<req_data_num>), but (<data_num>) found.')

        # Construct and validate the indicators set
        if self._verbose:
            print('\nIndicators set (ML data features)')

        for indicator in self._ti_features:

            if self._verbose:
                print('- create indicator instance:', indicator)

            self._indicators_set.append(eval(indicator['ti'])(
                input_data=self._input_data, **indicator['kwargs']))

        # Make columns case insensitive
        self._input_data.columns = \
            [c.lower() for c in self._input_data.columns]

    def createMLData(self):
        """
        Constructs data to be used for the ML model creation.

        Returns:
            numpy.ndarray: The created ML data.
        """

        # Add ti_features
        self._ml_data[:, :len(self._ti_features)] = \
            self._createMLDataFeatures()[:-self._price_diff_periods, :]

        # Add close and volume features if requested
        features_index = len(self._ti_features)

        if self._include_close_feature:
            self._ml_data[:, features_index] = \
                self._input_data['close'].iloc[:-self._price_diff_periods]
            features_index += 1

        if self._include_volume_feature:
            self._ml_data[:, features_index] = \
                self._input_data['volume'].iloc[:-self._price_diff_periods]

        # Add labels, Don't include first labels row
        self._ml_data[:, -1] = \
            self._createMLDataLabels()[:-self._price_diff_periods]

        return self._ml_data

    def _getSignalsSequence(self, ti):
        """
        Constructs data to be used for the ML model creation.

        Args:
            ti (tti.indicators): Instance of a Trading Technical Indicator.

        Returns:
            numpy.ndarray: The signals for the whole period defined in
            input_data.
        """

        signals = np.zeros(shape=(len(self._input_data.index), 1),
                           dtype=np.int)

        # Full input and indicator data
        full_ti_data = ti._ti_data
        full_input_data = ti._input_data

        # Calculate signals for the whole input period
        for i in range(len(ti._ti_data.index)):
            ti._input_data = full_input_data[
                full_input_data.index <= full_input_data.index[i]]

            ti._ti_data = full_ti_data[
                full_ti_data.index <= full_ti_data.index[i]]

            signals[i] = ti.getTiSignal()[1]

        return signals

    def _createMLDataFeatures(self):
        """
        Creates the features part of the ML data.

        Returns:
            numpy.ndarray: The features values for the whole period defined in
            input_data.
        """

        features = np.zeros(
            shape=(len(self._input_data.index), len(self._indicators_set)),
            dtype=np.int)

        processes_pool = Pool(self._pool_size)

        results = [
            processes_pool.apply_async(
                self._getSignalsSequence, (self._indicators_set[i], ))
            for i in range(len(self._indicators_set))
        ]

        processes_pool.close()
        processes_pool.join()

        features_values = [r.get() for r in results]
        for i in range(len(features_values)):
            features[:, i] = features_values[i].transpose()

        return features

    def _createMLDataLabels(self):
        """
        Creates the labels part of the ML data.

        Returns:
            numpy.ndarray: The labels values for the whole period defined in
            input_data.
        """

        labels = np.roll(
            a=self._input_data['close'].values,
            shift=-self._price_diff_periods, axis=0
        ) - self._input_data['close'].values

        labels[labels > 0] = ML_CLASSES['UP']
        labels[labels <= 0] = ML_CLASSES['DOWN']

        return labels

    def saveMLData(self, destination_file):
        """
        Saves ML data (features and labels) to a csv file.

        Args:
            destination_file (str): The destination file where the ML data will
                be redirected.
        """

        # Columns for the csv file
        columns = []
        for indicator in self._ti_features:

            arguments = ''
            for key, value in indicator['kwargs'].items():
                arguments += '_' + key + '_' + str(value)

            columns.append(indicator['ti'] + arguments)

        if self._include_close_feature:
            columns += ['close']

        if self._include_volume_feature:
            columns += ['volume']

        columns.append('labels')

        pd.DataFrame(index=self._input_data.index[:-self._price_diff_periods],
                     columns=columns, data=self._ml_data, dtype=np.float64
                     ).to_csv(destination_file)
