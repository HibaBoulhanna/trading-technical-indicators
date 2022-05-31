"""
Trading-Technical-Indicators (tti) python library

File name: _chaikin_oscillator.py
    Implements the Chaikin Oscillator technical indicator.
"""

import pandas as pd
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _technical_indicator_cho import TechnicalIndicator
#from _accumulation_distribution_line import AccumulationDistributionLine
from utils.constants import TRADE_SIGNALS
from utils.exceptions import NotEnoughInputData


class ChaikinOscillator(TechnicalIndicator):
    """
    Chaikin Oscillator Technical Indicator class implementation.

    Args:
        input_data (pandas.DataFrame): The input data. Required input columns
            are ``high``, ``low``, ``close``, ``volume``. The index is of type
            ``pandas.DatetimeIndex``.

        fill_missing_values (bool, default=True): If set to True, missing
            values in the input data are being filled.

    Attributes:
        _input_data (pandas.DataFrame): The ``input_data`` after preprocessing.

        _ti_data (pandas.DataFrame): The calculated indicator. Index is of type
            ``pandas.DatetimeIndex``. It contains one column, the ``co``.

        _properties (dict): Indicator properties.

        _calling_instance (str): The name of the class.

    Raises:
        WrongTypeForInputParameter: Input argument has wrong type.
        WrongValueForInputParameter: Unsupported value for input argument.
        NotEnoughInputData: Not enough data for calculating the indicator.
        TypeError: Type error occurred when validating the ``input_data``.
        ValueError: Value error occurred when validating the ``input_data``.
    """
    def __init__(self, input_data, fill_missing_values=True):

        # Control is passing to the parent class
        super().__init__(calling_instance=self.__class__.__name__,
                         input_data=input_data,
                         fill_missing_values=fill_missing_values)

    def _calculateTi(self,ws,wl):
        """
        Calculates the technical indicator for the given input data. The input
        data are taken from an attribute of the parent class.

        Returns:
            pandas.DataFrame: The calculated indicator. Index is of type
            ``pandas.DatetimeIndex``. It contains one column, the ``co``.

        Raises:
            NotEnoughInputData: Not enough data for calculating the indicator.
        """
        adl = pd.DataFrame(index=self._input_data.index, columns=['adl'],
                           data=0, dtype='int64')

        adl['adl'] = self._input_data['Volume'] * (
                (self._input_data['Close'] - self._input_data['Low']) -
                (self._input_data['High'] - self._input_data['Close'])
        ) / (self._input_data['High'] - self._input_data['Low'])

        adl = adl.cumsum(axis=0)
        

        # Not enough data for the requested period
        if len(self._input_data.index) < 10:
            raise NotEnoughInputData('Chaikin Oscillator', 10,
                                     len(self._input_data.index))

        co = pd.DataFrame(index=self._input_data.index, columns=['co'],
                          data=0, dtype='float64')

        #adl = AccumulationDistributionLine(self._input_data).getTiData()

        co['co'] = \
            adl.ewm(span=ws, min_periods=ws, adjust=False, axis=0).mean() - \
            adl.ewm(span=wl, min_periods=wl, adjust=False, axis=0).mean()

        return co

    def getTiSignal(self):
        """
        Calculates and returns the trading signal for the calculated technical
        indicator.

        Returns:
            {('hold', 0), ('buy', -1), ('sell', 1)}: The calculated trading
            signal.
        """

        # Not enough data for calculating trading signal
        if len(self._ti_data.index) < 90:
            return TRADE_SIGNALS['hold']

        # 90-periods moving average
        ma_90 = self._input_data['close'].iloc[-90:].mean()

        # Buy signal when price above 90-MA and indicator upturns in the
        # negative area
        if self._input_data['close'].iat[-1] > ma_90 and \
           self._ti_data['co'].iat[-2] < self._ti_data['co'].iat[-1] < 0.0:
            return TRADE_SIGNALS['buy']

        # Sell signal when price below 90-MA and indicator downturns in the
        # positive area
        elif self._input_data['close'].iat[-1] < ma_90 and \
                self._ti_data['co'].iat[-2] > \
                self._ti_data['co'].iat[-1] > 0.0:
            return TRADE_SIGNALS['sell']

        else:
            return TRADE_SIGNALS['hold']
