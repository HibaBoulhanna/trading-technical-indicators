"""
Trading-Technical-Indicators (tti) python library

File name: _momentum.py
    Implements the Momentum technical indicator.
"""

import pandas as pd
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _technical_indicator_mom import TechnicalIndicator
from utils.constants import TRADE_SIGNALS
from utils.exceptions import NotEnoughInputData, WrongTypeForInputParameter,\
    WrongValueForInputParameter


class Momentum(TechnicalIndicator):
    """
    Momentum Technical Indicator class implementation.

    Args:
        input_data (pandas.DataFrame): The input data. Required input column
            is ``close``. The index is of type ``pandas.DatetimeIndex``.

        period (int, default=12): The past periods to be used for the
            calculation of the indicator.

        fill_missing_values (bool, default=True): If set to True, missing
            values in the input data are being filled.

    Attributes:
        _input_data (pandas.DataFrame): The ``input_data`` after preprocessing.

        _ti_data (pandas.DataFrame): The calculated indicator. Index is of type
            ``pandas.DatetimeIndex``. It contains one column, the ``mom``.

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
        super().__init__(calling_instance=self.__class__.__name__,input_data=input_data,fill_missing_values=fill_missing_values)

    def _calculateTi(self,period,wsig=9):
        """
        Calculates the technical indicator for the given input data. The input
        data are taken from an attribute of the parent class.

        Returns:
            pandas.DataFrame: The calculated indicator. Index is of type
            ``pandas.DatetimeIndex``. It contains one column, the ``mom``.

        Raises:
            NotEnoughInputData: Not enough data for calculating the indicator.
            
        """
        """
        Momentum
          Paramètre: df: le  vecteur des prix
                      w: ordre
                      wsig ; ordre de signal ligne
          Retour:   Momentume (pndas.DataFrame)
        """
        MOM=pd.Series(self._input_data.close.diff(period),name="MOM")
        MOMsignal=pd.Series(MOM.rolling(wsig, min_periods=wsig).mean(), name= "MOMsignal")
        df=pd.DataFrame(self._input_data.close)
        df=df.join(MOM)
        df=df.join(MOMsignal)
        return df

        """
        # Not enough data for the requested period
        if len(self._input_data.index) < period:
            raise NotEnoughInputData('Momentum', period,
                                     len(self._input_data.index))

        mom = pd.DataFrame(index=self._input_data.index, columns=['mom'],
                           data=None, dtype='float64')

        mom['mom'].iloc[period:] = \
            100. * self._input_data['close'].iloc[period:] / \
            self._input_data['close'].shift(period).iloc[period:]

        return mom
        """

    def getTiSignal(self):
    
        """
        Calculates and returns the trading signal for the calculated technical
        indicator.

        Returns:
            {('hold', 0), ('buy', -1), ('sell', 1)}: The calculated trading
            signal.
        """

        # Not enough data for calculating trading signal
        if len(self._ti_data.index) < 9:
            return TRADE_SIGNALS['hold']

        # Short term moving average for determining the bottoming and peaking
        ema = self._ti_data['mom'].ewm(span=9, min_periods=9, adjust=False,
                                       axis=0).mean()

        # Indicator value goes above Moving Average
        if self._ti_data['mom'].iat[-2] < ema[-2] and \
                self._ti_data['mom'].iat[-1] > ema[-1]:
            return TRADE_SIGNALS['sell']

        # Indicator value goes below Moving Average
        if self._ti_data['mom'].iat[-2] > ema[-2] and \
                self._ti_data['mom'].iat[-1] < ema[-1]:
            return TRADE_SIGNALS['buy']

        return TRADE_SIGNALS['hold']
