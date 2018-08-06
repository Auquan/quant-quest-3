from backtester.trading_system_parameters import TradingSystemParameters
from backtester.features.feature import Feature
from datetime import timedelta
from backtester.dataSource.csv_data_source import CsvDataSource
from backtester.timeRule.nse_time_rule import NSETimeRule
from backtester.orderPlacer.backtesting_order_placer import BacktestingOrderPlacer
from backtester.trading_system import TradingSystem
from backtester.version import updateCheck
from backtester.constants import *
from backtester.features.feature import Feature
from backtester.logger import *
import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
from sklearn import metrics as sm
from problem2_execution_system import Problem2ExecutionSystem
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen


## Make your changes to the functions below.
## SPECIFY the symbols you are modeling for in getSymbolsToTrade() below
## You need to specify features you want to use in getInstrumentFeatureConfigDicts() and getMarketFeatureConfigDicts()
## and create your predictions using these features in getPrediction()

## Don't change any other function
## The toolbox does the rest for you, from downloading and loading data to running backtest




class MyTradingParams(TradingSystemParameters):
    '''
    initialize class
    place any global variables here
    '''
    def __init__(self, leagueFunctions):
        self.__leagueFunctions = leagueFunctions
        url = "https://raw.githubusercontent.com/Auquan/data_set_id/master/dsiqq3p2"
        response = urlopen(url)
        self.__dataSetId = response.read().decode('utf8').rstrip()
        self.__instrumentIds = []
        self.__priceKey = 'F5'
        self.__additionalInstrumentFeatureConfigDicts = []
        self.__additionalMarketFeatureConfigDicts = []
        self.__fees = {'brokerage': 0.0001,'spread': 0.05}
        self.__startDate = '2010/06/02'
        self.__endDate = '2013/02/07'
        super(MyTradingParams, self).__init__()


    '''
    Returns an instance of class DataParser. Source of data for instruments
    '''

    def getDataParser(self):
        instrumentIds = self.__instrumentIds
        return CsvDataSource(cachedFolderName='historicalData/',
                             dataSetId=self.__dataSetId,
                             instrumentIds=self.__instrumentIds,
                             downloadUrl = 'https://raw.githubusercontent.com/Auquan/qq3Data/master',
                             timeKey = 'datetime',
                             timeStringFormat = '%Y-%m-%d %H:%M:%S',
                             startDateStr=self.__startDate,
                             endDateStr=self.__endDate,
                             liveUpdates=True,
                             pad=True)

    '''
    Returns an instance of class TimeRule, which describes the times at which
    we should update all the features and try to execute any trades based on
    execution logic.
    For eg, for intra day data, you might have a system, where you get data
    from exchange at a very fast rate (ie multiple times every second). However,
    you might want to run your logic of computing features or running your execution
    system, only at some fixed intervals (like once every 5 seconds). This depends on your
    strategy whether its a high, medium, low frequency trading strategy. Also, performance
    is another concern. if your execution system and features computation are taking
    a lot of time, you realistically wont be able to keep upto pace.
    '''
    def getTimeRuleForUpdates(self):
        return NSETimeRule(startDate=self.__startDate, endDate=self.__endDate, startTime='15:30', frequency='M', sample='1440')

    '''
    Returns a timedetla object to indicate frequency of updates to features
    Any updates within this frequncy to instruments do not trigger feature updates.
    Consequently any trading decisions that need to take place happen with the same
    frequency
    '''

    def getFrequencyOfFeatureUpdates(self):
        return timedelta(3,0, 0)  # minutes, seconds

    def getStartingCapital(self):
        return 3*10000*self.__leagueFunctions.getPlayersInTeam()

    '''
    This is a way to use any custom features you might have made.
    Returns a dictionary where
    key: featureId to access this feature (Make sure this doesnt conflict with any of the pre defined feature Ids)
    value: Your custom Class which computes this feature. The class should be an instance of Feature
    Eg. if your custom class is MyCustomFeature, and you want to access this via featureId='my_custom_feature',
    you will import that class, and return this function as {'my_custom_feature': MyCustomFeature}
    '''

    def getCustomFeatures(self):
        customFeatures = {'prediction': TrainingPredictionFeature,
                'fees_and_spread': FeesCalculator,
                'RankPnL':RankPnL,
                'Separation_PnL': AverageSeparation,
                'DollarExposure':DollarExposure}
        customFeatures.update(self.__leagueFunctions.getCustomFeatures())


        return customFeatures


    def getInstrumentFeatureConfigDicts(self):
        # ADD RELEVANT FEATURES HERE

        predictionDict = {'featureKey': 'prediction',
                                'featureId': 'prediction',
                                'params': {'leagueFunctions':self.__leagueFunctions}}
        feesConfigDict = {'featureKey': 'fees',
                          'featureId': 'fees_and_spread',
                          'params': {'feeDict': self.__fees,
                                    'price': self.getPriceFeatureKey(),
                                    'position' : 'position'}}
        profitlossConfigDict = {'featureKey': 'pnl',
                                'featureId': 'pnl',
                                'params': {'price': self.getPriceFeatureKey(),
                                           'fees': 'fees'}}
        capitalConfigDict = {'featureKey': 'capital',
                             'featureId': 'capital',
                             'params': {'price': self.getPriceFeatureKey(),
                                        'fees': 'fees',
                                        'capitalReqPercent': 0.95}}
        scoreDict = {'featureKey': 'score',
                     'featureId': 'RankPnL',
                     'params': {'price': self.getPriceFeatureKey()}}


        stockFeatureConfigs = self.__leagueFunctions.getPlayerFeatureConfigDicts()


        return {INSTRUMENT_TYPE_STOCK: stockFeatureConfigs + [predictionDict,
                feesConfigDict,profitlossConfigDict,capitalConfigDict,scoreDict]
                + self.__additionalInstrumentFeatureConfigDicts}

    '''
    Returns an array of market feature config dictionaries
        market feature config Dictionary has the following keys:
        featureId: a string representing the type of feature you want to use
        featureKey: a string representing the key you will use to access the value of this feature.this
        params: A dictionary with which contains other optional params if needed by the feature
    '''

    def getMarketFeatureConfigDicts(self):
    # ADD RELEVANT FEATURES HERE
        scoreDict = {'featureKey': 'score',
                     'featureId': 'Separation_PnL',
                     'params': {'price': self.getPriceFeatureKey()}}
        betaDict = {'featureKey': 'Beta',
                     'featureId': 'DollarExposure',
                     'params': {'price': self.getPriceFeatureKey()}}

        marketFeatureConfigs = self.__leagueFunctions.getTeamFeatureConfigDicts()
        return marketFeatureConfigs + [scoreDict, betaDict]


    def getPrediction(self, time, updateNum, instrumentManager):

        predictions = pd.Series(index = instrumentManager.getAllInstrumentsByInstrumentId())

       

        return predictions

    '''
    Returns the type of execution system we want to use. Its an implementation of the class ExecutionSystem
    It converts prediction to intended positions for different instruments.
    '''

    def getExecutionSystem(self):
        return Problem2ExecutionSystem(enter_threshold=0.99, 
                                    exit_threshold=0.55, 
                                    longLimit=10000,
                                    shortLimit=10000, 
                                    capitalUsageLimit=0.10 * self.getStartingCapital(), 
                                    enterlotSize = 10000, 
                                    limitType='D', price=self.getPriceFeatureKey())

    '''
    Returns the type of order placer we want to use. its an implementation of the class OrderPlacer.
    It helps place an order, and also read confirmations of orders being placed.
    For Backtesting, you can just use the BacktestingOrderPlacer, which places the order which you want, and automatically confirms it too.
    '''

    def getOrderPlacer(self):
        return BacktestingOrderPlacer()

    '''
    Returns the amount of lookback data you want for your calculations. The historical market features and instrument features are only
    stored upto this amount.
    This number is the number of times we have updated our features.
    '''


    def getSymbolsInBasket(self):
        return self.__leagueFunctions.getPlayersInTeam()

    def getLookbackSize(self):
        return max(720, self.__leagueFunctions.getLookbackSize())

    def getPriceFeatureKey(self):
        return self.__priceKey

    def setPriceFeatureKey(self, priceKey='Adj_Close'):
        self.__priceKey = priceKey

    def getDataSetId(self):
        return self.__dataSetId

    def setDataSetId(self, dataSetId):
        self.__dataSetId = dataSetId

    def getInstrumentsIds(self):
        return self.__instrumentIds

    def setInstrumentsIds(self, instrumentIds):
        self.__instrumentIds = instrumentIds

    def getDates(self):
        return {'startDate':self.__startDate,
                'endDate':self.__endDate}

    def setDates(self, dateDict):
        self.__startDate = dateDict['startDate']
        self.__endDate = dateDict['endDate']

    def setFees(self, feeDict={'brokerage': 0.0001,'spread': 0.05}):
        self.__fees = feeDict

    def setAdditionalInstrumentFeatureConfigDicts(self, dicts = []):
        self.__additionalInstrumentFeatureConfigDicts = dicts

    def setAdditionalMarketFeatureConfigDicts(self, dicts = []):
        self.__additionalMarketFeatureConfigDicts = dicts


class TrainingPredictionFeature(Feature):

    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        predictions = pd.Series(0.5,index = instrumentManager.getAllInstrumentsByInstrumentId())
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        lf = featureParams['leagueFunctions']
        players = lf.getPlayersAndPosition(time, updateNum, instrumentManager)
        
        if len(players)>0:
            
            # print(pd.Series([instrumentManager.getInstrument(x).getCurrentPosition() for x in predictions.index], index=predictions.index))
            # print(players)
            predictions.update(players)

        return predictions

class FeesCalculator(Feature):

    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()

        priceData = instrumentLookbackData.getFeatureDf(featureParams['price'])
        positionData = instrumentLookbackData.getFeatureDf(featureParams['position'])
        currentPosition = currentPosition = pd.Series([instrumentManager.getInstrument(x).getCurrentPosition() for x in priceData.columns], index=priceData.columns)
        previousPosition = 0 if updateNum < 2 else positionData.iloc[-1]
        changeInPosition = currentPosition - previousPosition
        fees = pd.Series(np.abs(changeInPosition)*featureParams['feeDict']['brokerage'],index = instrumentManager.getAllInstrumentsByInstrumentId())
        if len(priceData)>1:
            currentPrice = priceData.iloc[-1]
        else:
            currentPrice = 0

        fees = fees*currentPrice + np.abs(changeInPosition)*featureParams['feeDict']['spread']

        return fees


class RankPnL(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()

        priceData = instrumentLookbackData.getFeatureDf(featureParams['price'])
        score = pd.Series(0,index = instrumentManager.getAllInstrumentsByInstrumentId())
        if len(priceData)>2:
            changeInPrice = priceData.iloc[-1] - priceData.iloc[-2]
            ranks = changeInPrice.rank(0)

            # Put stocks with no rank in the middle
            ranks.fillna(len(changeInPrice.index)/2, inplace=True)
            scoreDict = instrumentLookbackData.getFeatureDf(featureKey)
            score = scoreDict.iloc[-1]
            score[ranks<=6] = scoreDict.iloc[-1] + 1
            score[ranks>(len(ranks)-6)] = scoreDict.iloc[-1] - 1
        return score

class AverageSeparation(Feature):
    @classmethod
    def computeForMarket(cls, updateNum, time, featureParams, featureKey, currentMarketFeatures, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        scoreDict = instrumentManager.getDataDf()[featureKey]
        score = 0
        priceData = instrumentLookbackData.getFeatureDf(featureParams['price'])
        if updateNum>1:
            currentPosition = pd.Series([instrumentManager.getInstrument(x).getCurrentPosition() for x in priceData.columns], index=priceData.columns)
            changeInPrice = priceData.iloc[-1] /priceData.iloc[-2] - 1 
            Pnl = changeInPrice * np.sign(currentPosition)
            
            separation = np.nan_to_num(Pnl[currentPosition>0].mean()) + np.nan_to_num(Pnl[currentPosition<0].mean())
            cumulativeScore = scoreDict.values[-2]
            score = (1000*separation + cumulativeScore*(updateNum-1)) / float(updateNum)
        return score

class DollarExposure(Feature):
    @classmethod
    def computeForMarket(cls, updateNum, time, featureParams, featureKey, currentMarketFeatures, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        scoreDict = instrumentManager.getDataDf()[featureKey]
        score = 0
        priceData = instrumentLookbackData.getFeatureDf(featureParams['price'])
        if updateNum>1:
            
            currentPosition = pd.Series([instrumentManager.getInstrument(x).getCurrentPosition() for x in priceData.columns], index=priceData.columns)
            exposure = priceData.iloc[-1] * currentPosition
            capital = priceData.iloc[-1] * currentPosition.abs()
            ratio = 0
            if capital.sum()!=0:
                ratio = np.abs(exposure.sum()/float(capital.sum()))
            cumulativeScore = scoreDict.values[-2]
            score = (10*ratio + cumulativeScore*(updateNum-1)) / float(updateNum)
        return score
