from backtester.trading_system_parameters import TradingSystemParameters
from backtester.features.feature import Feature
from datetime import timedelta
from backtester.dataSource.csv_data_source import CsvDataSource
from backtester.timeRule.nse_time_rule import NSETimeRule
from backtester.executionSystem.simple_execution_system import SimpleExecutionSystem
from backtester.orderPlacer.backtesting_order_placer import BacktestingOrderPlacer
from backtester.trading_system import TradingSystem
from backtester.version import updateCheck
from backtester.constants import *
from backtester.features.feature import Feature
import pandas as pd
import numpy as np
from problem2_trading_params import  MyTradingParams

## Make your changes to the functions below.
## You need to specify features you want to use in getInstrumentFeatureConfigDicts() and getMarketFeatureConfigDicts()
## and create your predictions using these features in getPrediction()
## SPECIFY number of stocks in each basket below
## Don't change any other function
## The toolbox does the rest for you, from downloading and loading data to running backtest


class MyLeagueFunctions():

    def __init__(self):  #Put any global variables here
        self.count = 0
        self.params = {}
        self.lookback = 720  ## max number of historical datapoints you want at any given time

    def getPlayersInTeam(self):
        return 12


    ####################################
    ## FILL THESE FOUR FUNCTIONS BELOW ##
    ####################################

    '''
    Specify all Features you want to use by  by creating config dictionaries.
    Create one dictionary per feature and return them in an array.

    Feature config Dictionary have the following keys:

        featureId: a str for the type of feature you want to use
        featureKey: {optional} a str for the key you will use to call this feature
                    If not present, will just use featureId
        params: {optional} A dictionary with which contains other optional params if needed by the feature

    msDict = {'featureKey': 'ms_5',
              'featureId': 'moving_sum',
              'params': {'period': 5,
                         'featureName': 'basis'}}

    return [msDict]

    You can now use this feature by in getPRediction() calling it's featureKey, 'ms_5'
    '''

    def getPlayerFeatureConfigDicts(self):

        #############################################################################
        ### TODO: FILL THIS FUNCTION TO CREATE DESIRED FEATURES for each stock.   ###
        ### USE TEMPLATE BELOW AS EXAMPLE                                         ###
        #############################################################################
        mom1Dict = {'featureKey': 'mom_5',
                   'featureId': 'momentum',
                   'params': {'period': 5,
                              'featureName': 'F5'}}
        mom2Dict = {'featureKey': 'mom_10',
                   'featureId': 'momentum',
                   'params': {'period': 10,
                              'featureName': 'F5'}}
        return [mom1Dict, mom2Dict]



    def getTeamFeatureConfigDicts(self):
        #############################################################################
        ### TODO: FILL THIS FUNCTION TO CREATE a ratio feature                    ###
        ### AND DESIRED FEATURES for each ratio.                                  ###
        ### USE TEMPLATE BELOW AS EXAMPLE                                         ###
        #############################################################################
        
        # customFeatureDict = {'featureKey': 'custom_mrkt_feature',
        #                      'featureId': 'my_custom_mrkt_feature',
        #                      'params': {'param1': 'value1'}}
        return []

    '''
    Combine all the features to create the desired predictions for each stock.
    'predictions' is Pandas Series with stock as index and predictions as values
    We first call the holder for all the instrument features for all stocks as
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()
    Then call the dataframe for a feature using its feature_key as
        ms5Data = lookbackInstrumentFeatures.getFeatureDf('ms_5')
    This returns a dataFrame for that feature for ALL stocks for all times upto lookback time
    Now you can call just the last data point for ALL stocks as
        ms5 = ms5Data.iloc[-1]
    You can call last datapoint for one stock 'ABC' as
        value_for_abs = ms5['ABC']

    Output of the prediction function is used by the toolbox to make further trading decisions and evaluate your score.
    '''


    def getPlayersAndPosition(self, time, updateNum, playerManager):

        # self.updateCount() - uncomment if you want a counter
        # holder for all the instrument features for all instruments
        #############################################################################################
        ###  TODO : FILL THIS FUNCTION TO RETURN A BUY (1) or SELL (0) prediction for each pair   ###
        ###  USE TEMPLATE BELOW AS EXAMPLE                                                        ###
        #############################################################################################

        lookbackPlayerFeatures = playerManager.getLookbackInstrumentFeatures()

        lookbackTeamFeatures = playerManager.getDataDf()
        mom1 = lookbackPlayerFeatures.getFeatureDf('mom_5')
        mom2 = lookbackPlayerFeatures.getFeatureDf('mom_10')

        #Create some factor to rank their skill on
        #You can create one or multiple factors
        #In this example we assume that this factor ranks players in order of forward to defense ability
        #Top ranked are best forwards, bottom ranked are best defenders
        factorValues = (mom1/mom2).iloc[-1]
        factorValues.sort_values(0, ascending=False, inplace=True)

        #Derive player ranks from factor values
        ranks = factorValues.rank(0)

        # Put players with no rank in the middle
        ranks.fillna(len(factorValues.index)/2, inplace=True)

        ## Choose players and their position

        players = pd.Series(index=ranks.index[(ranks<=self.getPlayersInTeam()/2)|(ranks>(len(ranks.index)-self.getPlayersInTeam()/2))])
        players[ranks<=self.getPlayersInTeam()/2] = 1
        players[ranks>=(len(ranks.index)-self.getPlayersInTeam()/2)] = 0 

        return players

# def getPrediction(self, time, updateNum, instrumentManager, predictions):

#         # self.updateCount() - uncomment if you want a counter
#         # holder for all the instrument features for all instruments
#         lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()

#         #############################################################################################
#         ###  TODO : FILL THIS FUNCTION TO RETURN A BUY (1) or SELL (0) prediction for each pair   ###
#         ###  USE TEMPLATE BELOW AS EXAMPLE                                                        ###
#         #############################################################################################

#         lookbackMarketFeatures = instrumentManager.getDataDf()
#         mom1 = lookbackInstrumentFeatures.getFeatureDf('mom_5')
#         mom2 = lookbackInstrumentFeatures.getFeatureDf('mom_10')
#         #Get latest factor values
#         factorValues = (mom1/mom2).iloc[-1]
#         factorValues.sort_values(0, inplace=True)

#         #Derive rank from factor values
#         ranks = factorValues.rank(0)

#         # Put stocks with no rank in the middle
#         ranks.fillna(len(predictions.index)/2, inplace=True)

#         if len(mom1.index) > 1:
#             oldFactorValues = (mom1/mom2).iloc[-2]
#             oldranks = oldFactorValues.rank(0)
#             oldranks.fillna(len(predictions.index)/2, inplace=True)
#         else:
#             oldranks = pd.Series(len(predictions.index)/2, index = ranks.index)

#         # Short position in the lowest ranked stocks
#         predictions[ranks<=self.getSymbolsInBasket()] = 0
#         # we are already short stocks that ranked lowest previous time, don't sell again them
#         predictions[(oldranks<=self.getSymbolsInBasket()) & (ranks<=self.getSymbolsInBasket())] = 0.25
#         predictions[ranks>self.getSymbolsInBasket()] = 0.5

#         # Long position in the highest ranked stocks
#         predictions[ranks>(len(predictions.index)-self.getSymbolsInBasket())] = 1
#         # we are already long stocks that ranked highest previous time, don't buy again them
#         predictions[(oldranks>(len(predictions.index)-self.getSymbolsInBasket())) &\
#                  (ranks>(len(predictions.index)-self.getSymbolsInBasket()))] = 0.75    

#         return predictions

#     def updateCount(self):
#         self.count = self.count + 1

    ###########################################
    ##         DONOT CHANGE THESE            ##
    ###########################################

    def getLookbackSize(self):
        return self.lookback

    ###############################################
    ##  CHANGE ONLY IF YOU HAVE CUSTOM FEATURES  ##
    ###############################################

    def getCustomFeatures(self):
        return {'my_custom_feature_identifier': MyCustomFeatureClassName}

class MyCustomFeature(Feature):
    ''''
    Custom Feature to implement for instrument. This function would return the value of the feature you want to implement.
    1. create a new class MyCustomFeatureClassName for the feature and implement your logic in the function computeForInstrument() -

    2. modify function getCustomFeatures() to return a dictionary with Id for this class
        (follow formats like {'my_custom_feature_identifier': MyCustomFeatureClassName}.
        Make sure 'my_custom_feature_identifier' doesnt conflict with any of the pre defined feature Ids

        def getCustomFeatures(self):
            return {'my_custom_feature_identifier': MyCustomFeatureClassName}

    3. create a dict for this feature in getInstrumentFeatureConfigDicts() above. Dict format is:
            customFeatureDict = {'featureKey': 'my_custom_feature_key',
                                'featureId': 'my_custom_feature_identifier',
                                'params': {'param1': 'value1'}}
    You can now use this feature by calling it's featureKey, 'my_custom_feature_key' in getPrediction()
    '''
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        # Custom parameter which can be used as input to computation of this feature
        param1Value = featureParams['param1']

        # A holder for the all the instrument features
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()

        # dataframe for a historical instrument feature (basis in this case). The index is the timestamps
        # atmost upto lookback data points. The columns of this dataframe are the stocks/instrumentIds.
        lookbackInstrumentValue = lookbackInstrumentFeatures.getFeatureDf('stockVWAP')

        # The last row of the previous dataframe gives the last calculated value for that feature (basis in this case)
        # This returns a series with stocks/instrumentIds as the index.
        currentValue = lookbackInstrumentValue.iloc[-1]

        if param1Value == 'value1':
            return currentValue * 0.1
        else:
            return currentValue * 0.5

if __name__ == "__main__":
    if updateCheck():
        print('Your version of the auquan toolbox package is old. Please update by running the following command:')
        print('pip install -U auquan_toolbox')
    else:
        tf = MyLeagueFunctions()
        tsParams = MyTradingParams(tf)
        tradingSystem = TradingSystem(tsParams)
    # Set onlyAnalyze to True to quickly generate csv files with all the features
    # Set onlyAnalyze to False to run a full backtest
    # Set makeInstrumentCsvs to False to not make instrument specific csvs in runLogs. This improves the performance BY A LOT
        tradingSystem.startTrading(onlyAnalyze=False, shouldPlot=True, makeInstrumentCsvs=True)