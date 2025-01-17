"""
Trading-Technical-Indicators (tti) python library

File name: _triple_exponential_moving_average.py
    Implements the Triple Exponential Moving Average technical indicator.
"""

import pandas as pd
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _technical_indicator_MM3 import TechnicalIndicator
from utils.constants import TRADE_SIGNALS
from utils.exceptions import NotEnoughInputData, WrongTypeForInputParameter,\
    WrongValueForInputParameter


class TripleExponentialMovingAverage(TechnicalIndicator):
    """
    Triple Exponential Moving Average Technical Indicator class implementation.

    Args:
        input_data (pandas.DataFrame): The input data. Required input column
            is ``close``. The index is of type ``pandas.DatetimeIndex``.

        period (int, default=5): The past periods to be used for the
            calculation of the indicator.

        fill_missing_values (bool, default=True): If set to True, missing
            values in the input data are being filled.

    RAttributes:
        _input_data (pandas.DataFrame): The ``input_data`` after preprocessing.

        _ti_data (pandas.DataFrame): The calculated indicator. Index is of type
            ``pandas.DatetimeIndex``. It contains one column, the ``tema``.

        _properties (dict): Indicator properties.

        _calling_instance (str): The name of the class.

    Raises:
        WrongTypeForInputParameter: Input argument has wrong type.
        WrongValueForInputParameter: Unsupported value for input argument.
        NotEnoughInputData: Not enough data for calculating the indicator.
        TypeError: Type error occurred when validating the ``input_data``.
        ValueError: Value error occurred when validating the ``input_data``.
    """
    def __init__(self, input_data,fill_missing_values=True):
   
        # Control is passing to the parent class
        super().__init__(calling_instance=self.__class__.__name__,
                         input_data=input_data,
                         fill_missing_values=fill_missing_values)

    def _calculateTi(self,period):
        """
        Calculates the technical indicator for the given input data. The input
        data are taken from an attribute of the parent class.

        Returns:
            pandas.DataFrame: The calculated indicator. Index is of type
            ``pandas.DatetimeIndex``. It contains one column, the ``tema``.

        Raises:
            NotEnoughInputData: Not enough data for calculating the indicator.
        """

        # Not enough data for the requested period
        if len(self._input_data.index) < self._period:
            raise NotEnoughInputData('Triple Exponential Moving Average',
                                     self._period, len(self._input_data.index))

        tema = pd.DataFrame(index=self._input_data.index, columns=['tema'],
                            data=0, dtype='float64')

        # Exponential moving average of prices
        ema = self._input_data.ewm(
            span=period, min_periods=period, adjust=False, axis=0
        ).mean()

        # Double Exponential moving average of prices
        double_ema = ema.ewm(
            span=period, min_periods=period, adjust=False,
            axis=0).mean()

        # Triple Exponential moving average of prices
        triple_ema = double_ema.ewm(
            span=period, min_periods=period, adjust=False,
            axis=0).mean()

        tema['tema'] = (3 * ema) - (3 * double_ema) + triple_ema

        return tema

    def getTiSignal(self):
        """
        Calculates and returns the trading signal for the calculated technical
        indicator.

        Returns:
            {('hold', 0), ('buy', -1), ('sell', 1)}: The calculated trading
            signal.
        """

        # Not enough data for calculating trading signal
        if len(self._ti_data.index) < 1:
            return TRADE_SIGNALS['hold']

        # Close price is below Moving Average
        if self._input_data['close'].iat[-1] < self._ti_data['tema'].iat[-1]:
            return TRADE_SIGNALS['buy']

        # Close price is above Moving Average
        if self._input_data['close'].iat[-1] > self._ti_data['tema'].iat[-1]:
            return TRADE_SIGNALS['sell']

        return TRADE_SIGNALS['hold']
