"""
Trading-Technical-Indicators (tti) python library

File name: _parabolic_sar.py
    Implements the Parabolic SAR technical indicator.
"""

import pandas as pd

from ._technical_indicator import TechnicalIndicator
from ..utils.constants import TRADE_SIGNALS
from ..utils.exceptions import NotEnoughInputData


class ParabolicSAR(TechnicalIndicator):
    """
    Parabolic SAR Technical Indicator class implementation.

    Parameters:
        input_data (pandas.DataFrame): The input data.

        fill_missing_values (boolean, default is True): If set to True,
            missing values in the input data are being filled.

    Attributes:
        -

    Raises:
        -
    """

    def __init__(self, input_data, fill_missing_values=True):

        # Indicator parameters, currently constants but expose to user should
        # be considered.
        self._af_increase = 0.02
        self._af_max = 0.2

        # Control is passing to the parent class
        super().__init__(calling_instance=self.__class__.__name__,
                         input_data=input_data,
                         fill_missing_values=fill_missing_values)

    def _calculateTi(self):
        """
        Calculates the technical indicator for the given input data. The input
        data are taken from an attribute of the parent class.

        Parameters:
            -

        Raises:
            -

        Returns:
            pandas.DataFrame: The calculated indicator. Index is of type date.
                It contains one column, the 'sar'.
        """

        # Not enough data for calculating SAR
        if len(self._input_data.index) < 2:
            raise NotEnoughInputData('Parabolic SAR', 2,
                                     len(self._input_data.index))

        sar = pd.DataFrame(index=self._input_data.index,
                           columns=['af', 'ep', 'sar', 'position'],
                           data=None)

        position_start_index = 0

        for i in range(len(self._input_data.index)):

            sar.iloc[i, :], position_start_index = self._calculateSarRow(
                current_index=i, position_start_index=position_start_index,
                previous_sar=sar.iloc[i-1, :], position_changed=False)

        return sar[['sar']].astype(dtype='float64').round(4)

    def _calculateSarRow(self, current_index, position_start_index,
                         previous_sar=None, position_changed=False):
        """
        Calculate SAR in case we are in a `LONG` position.

        Parameters:
            current_index (int): The current dataframe index for which the SAR
                calculation is requested.

            position_start_index (int): The dataframe index (starting from 0)
                in which the current position was started.

            previous_sar (pd.DataFrame, default is None): The SAR row for the
                previous period. Is not required when this is the first period
                of the input data (current_index = 0).

            position_changed (boolean, default is False): Indicates if this is
                the first calculation for a new position. Is not required when
                this is the first period of the input data (current_index = 0).

        Raises:
            -

        Returns:
            list (af, ep, sar, position):
                - af (float): Acceleration factor for the current index.
                - ep (float): Extreme price for the current index.
                - sar (float): SAR indicator value for the current index.
                - position (str, one of `LONG`, `SHORT`): The position for
                    the current index.

            int: Current position start index.
        """

        # In case this is for the first period of the input data, calculation
        # is based on an initial position assumption (guess the initial
        #  position by checking the high values direction for the first two
        #  days)
        if current_index == 0:

            af = self._af_increase

            if self._input_data['high'].iat[1] > \
                    self._input_data['high'].iat[0]:
                position = 'LONG'
                ep = self._input_data['high'].iat[0]
                sar = self._input_data['low'].iat[0]
            else:
                position = 'SHORT'
                ep = self._input_data['low'].iat[0]
                sar = self._input_data['high'].iat[0]

            return [af, ep, sar, position], current_index

        # Check if position changed, so to re-initialize the values
        if position_changed:
            af = self._af_increase

            if previous_sar['position'] == 'LONG':
                position = 'SHORT'
                ep = self._input_data['low'].iat[current_index]
                sar = self._input_data['high'].iloc[
                      position_start_index:current_index].max()

            else:
                position = 'LONG'
                ep = self._input_data['high'].iat[current_index]
                sar = self._input_data['low'].iloc[
                      position_start_index:current_index].min()

            return [af, ep, sar, position], current_index

        # Calculate Extreme Price, Highest price reached when in current
        # `LONG` position, or Lowest price reached when in current `SHORT`
        # position
        if previous_sar['position'] == 'LONG':

            ep = self._input_data['high'].iloc[
                 position_start_index:current_index+1].max()

        else:

            ep = self._input_data['low'].iloc[
                 position_start_index:current_index+1].min()

        # Calculate Acceleration Factor, new high is reached when in `LONG` or
        # new low is reached when in `SHORT`
        if previous_sar['position'] == 'LONG' and ep > previous_sar['ep']:

            af = min(self._af_max,
                     previous_sar['af'] + self._af_increase)

        elif previous_sar['position'] == 'SHORT' and ep < previous_sar['ep']:

            af = min(self._af_max,
                     previous_sar['af'] + self._af_increase)

        else:
            af = previous_sar['af']

        # Calculate SAR, when `LONG` not above the two prior lows, when `SHORT`
        # not below the two prior highs
        sar = previous_sar['sar'] + previous_sar['af'] * (
                previous_sar['ep'] - previous_sar['sar'])

        if previous_sar['position'] == 'LONG':
            sar = min(sar, self._input_data['low'].iloc[
                           max(0, current_index-2):current_index].min())

        else:
            sar = max(sar, self._input_data['high'].iloc[
                           max(0, current_index - 2):current_index].max())

        # Check if position changes
        if (previous_sar['position'] == 'SHORT' and
            self._input_data['high'].iat[current_index] > sar) or \
                (previous_sar['position'] == 'LONG' and
                 self._input_data['low'].iat[current_index] < sar):

            return self._calculateSarRow(
                current_index=current_index,
                position_start_index=position_start_index,
                previous_sar=previous_sar, position_changed=True)

        else:
            position = previous_sar['position']

        return [af, ep, sar, position], position_start_index

    def getTiSignal(self):
        """
        Calculates and returns the signal of the technical indicator. The
        Technical Indicator data are taken from an attribute of the parent
        class.

        Parameters:
            -

        Raises:
            -

        Returns:
            tuple (string, integer): The Trading signal. Possible values are
                ('hold', 0), ('buy', -1), ('sell', 1). See TRADE_SIGNALS
                constant in the tti.utils package, constants.py module.
        """

        # Not enough data for calculating trading signal
        if len(self._ti_data.index) < 2:
            return TRADE_SIGNALS['hold']

        if ((self._input_data['close'].iat[-2] >
                self._ti_data['sar'].iat[-2]) and
                (self._input_data['close'].iat[-1] <
                 self._ti_data['sar'].iat[-1])):
            return TRADE_SIGNALS['buy']

        elif ((self._input_data['close'].iat[-2] <
               self._ti_data['sar'].iat[-2]) and
              (self._input_data['close'].iat[-1] >
               self._ti_data['sar'].iat[-1])):
            return TRADE_SIGNALS['sell']

        else:
            return TRADE_SIGNALS['hold']