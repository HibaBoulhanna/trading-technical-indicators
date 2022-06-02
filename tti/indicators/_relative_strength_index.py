"""
Trading-Technical-Indicators (tti) python library

File name: _relative_strength_index.py
    Implements the Relative Strength Index technical indicator.
"""

import pandas as pd
import numpy as np 
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _technical_indicator_rsi import TechnicalIndicator
from utils.constants import TRADE_SIGNALS
from utils.exceptions import NotEnoughInputData, WrongTypeForInputParameter,\
    WrongValueForInputParameter


class RelativeStrengthIndex(TechnicalIndicator):
    """
    Relative Strength Index Technical Indicator class implementation.

    Args:
        input_data (pandas.DataFrame): The input data. Required input column
            is ``close``. The index is of type ``pandas.DatetimeIndex``.

        period (int, default=14): The past periods to be used for the
            calculation of the indicator. Popular values are 14, 9 and 25.

        fill_missing_values (bool, default=True): If set to True, missing
            values in the input data are being filled.

    Attributes:
        _input_data (pandas.DataFrame): The ``input_data`` after preprocessing.

        _ti_data (pandas.DataFrame): The calculated indicator. Index is of type
            ``pandas.DatetimeIndex``. It contains one column ``rsi``.

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
        """
        # Validate and store if needed, the input parameters
        if isinstance(period, int):
            if period > 0:
                self._period = period
            else:
                raise WrongValueForInputParameter(
                    period, 'period', '>0')
        else:
            raise WrongTypeForInputParameter(
                type(period), 'period', 'int')
       """
        # Control is passing to the parent class
        super().__init__(calling_instance=self.__class__.__name__,
                         input_data=input_data,
                         fill_missing_values=fill_missing_values)

    def _calculateTi(self,period):
        """
         Relative Strength index
          :Paramètre:
           df: pandas.DataFrame
           n : ordre
          :return:
           pandas.DataFrame

        """
        diff=self._input_data.diff(1)
        t=[]
        for i in diff.values :
            if i > 0:
                t.append(i)
            else :
                t.append(0)
        pos=pd.DataFrame(t,index=self._input_data.index)
        diff=np.abs(pd.DataFrame(diff))
        RSI=pos.rolling(period,min_periods=period).sum()/np.array((diff.rolling(period,min_periods=period).sum()))
        df=pd.DataFrame(self._input_data)
        df=df.join(RSI)
        df.columns=["COURS_CLOTURE","rsi"] 
        return df
    
    def getTiSignal(self):
        """
        Calculates and returns the trading signal for the calculated technical
        indicator.

        Returns:
            {('hold', 0), ('buy', -1), ('sell', 1)}: The calculated trading
            signal.
        """

        # Not enough data for calculating trading signal
        if len(self._ti_data.index) < 2:
            return TRADE_SIGNALS['hold']

        # Overbought region
        if self._ti_data['rsi'].iat[-2] < 70. < self._ti_data['rsi'].iat[-1]:
            return TRADE_SIGNALS['sell']

        # Oversold region
        if self._ti_data['rsi'].iat[-2] > 30. > self._ti_data['rsi'].iat[-1]:
            return TRADE_SIGNALS['buy']

        return TRADE_SIGNALS['hold']
