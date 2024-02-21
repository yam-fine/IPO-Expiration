import pandas as pd
# Import the main functionality from the SimFin Python API.
import simfin as sf
# Import names used for easy access to SimFin's data-columns.
from simfin.names import *
import numpy as np
import datetime as dt
# # Import python libraries required in this example:
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from sklearn.model_selection import train_test_split

DATES_BACK = 3
COLUMNS_LEAVE_ME_ALONE_DECREASE_ME_THERE = [0,1,2,3,9,10,11]

class Simfin_data():
    def __init__(self, stockpath):
        sf.set_data_dir('~/simfin_data/')
        # # Replace YOUR_API_KEY with your actual API-key.
        sf.set_api_key(api_key='NmFN8XBW6Se0bGezCbgvREZzWfKbVUWl')
        # self.df_income = sf.load_income(variant='annual', market='us')

        # We are interested in the US stock-market.
        market = 'us'

        # Add this date-offset to the fundamental data such as
        # Income Statements etc., because the REPORT_DATE is not
        # when it was actually made available to the public,
        # which can be 1, 2 or even 3 months after the Report Date.
        offset = pd.DateOffset(days=60)

        # Refresh the fundamental datasets (Income Statements etc.)
        # every 30 days.
        refresh_days = 30

        # Refresh the dataset with shareprices every 10 days.
        refresh_days_shareprices = 10

        # hub = sf.StockHub(market=market, offset=offset,
        #                   refresh_days=refresh_days,
        #                   refresh_days_shareprices=refresh_days_shareprices)

        #open excel file
        listOfStocks = pd.ExcelFile(stockpath).parse('Sheet1')
        listOfStocks["Date"] = listOfStocks["Date"].dt.strftime('%d/%m/%Y')

        df_share_signals = sf.load_shareprices(variant='daily', market=market, index=['Ticker'])

        # remove tickers that don't intersect in both df's
        df_share_signals = df_share_signals[df_share_signals.index.isin(listOfStocks["Stock"])]
        listOfStocks = listOfStocks[listOfStocks["Stock"].isin(df_share_signals.index.unique())]
        listOfStocks = listOfStocks.sort_values(by="Date", ascending=False).drop_duplicates(subset="Stock", keep="first", inplace=False)

        # removes lines without exp dates
        stockNames = self.getNonIpoExpData(df_share_signals, listOfStocks)
        for name in stockNames:
            listOfStocks.drop(listOfStocks[listOfStocks['Stock'] == name].index, inplace=True)
            df_share_signals.drop(df_share_signals[df_share_signals.index == name].index, inplace=True)

        # process
        df_share_dates_back_signals = self.getLastConstDays(df_share_signals, listOfStocks)
        nnXData = self.normalizedTimeSeries(df_share_dates_back_signals)
        nnYData = self.YVal(df_share_signals,listOfStocks.sort_values("Stock"))

        nnXData = self.columnNormalizer(nnXData)
        # nnYData = nnYData / nnYData.abs().max()

        self.X = nnXData
        self.y = nnYData



    #might not be the most eff
    def getNonIpoExpData(self, df_share_signals, listOfStocks):
        stockList = []
        lcl_df = df_share_signals.reset_index()
        lcl_df['Date'] = lcl_df['Date'].apply(lambda x: x.strftime("%d/%m/%Y"))
        for name, ipodate in zip(listOfStocks['Stock'], listOfStocks['Date']):
            setA = set(lcl_df.index[lcl_df['Ticker'] == name].tolist())
            setB = set(lcl_df.index[lcl_df['Date'] == ipodate].tolist())
            if not(setA & setB):
                stockList = np.append(stockList,name)
        return stockList

    def normalizedTimeSeries(self, df: pd.DataFrame):
        ans_df = pd.DataFrame(columns = range(DATES_BACK * (df.shape[1]+1)),index = range(df["Ticker"].unique().size))
        lcl_df = df.reset_index()
        counter = 0
        for name in sorted(lcl_df["Ticker"].unique()):
            numArr = lcl_df[lcl_df["Ticker"] == name].sort_values("Date")
            high = numArr["High"]
            low = numArr["Low"]
            open = numArr["Open"]
            close  = numArr["Close"]
            adjClose = numArr["Adj. Close"]
            numArr["High"] = ((high - open)/open)*100
            numArr["Low"] = ((low - open)/open)*100
            numArr["Open"] = ((high - low)/open)*100
            numArr["Close"] = ((close - open) / open) * 100
            numArr["Adj. Close"] = adjClose - close
            ans_df.iloc[counter] = numArr.to_numpy().reshape(1,numArr.shape[0]*numArr.shape[1])
            counter += 1
        colsToRemove = []
        numberOfFeatures = lcl_df.shape[1]
        for row in COLUMNS_LEAVE_ME_ALONE_DECREASE_ME_THERE:
            for val in range(0,DATES_BACK):
                colsToRemove = np.append(colsToRemove, int(row + numberOfFeatures*val))

        ans_df.drop(columns=colsToRemove, axis=1, inplace=True)
        return ans_df

    def timeSeries(self, df: pd.DataFrame):
        ans_df = pd.DataFrame(columns = range(DATES_BACK * (df.shape[1]+1)),index = range(df["Ticker"].unique().size))
        lcl_df = df.reset_index()
        counter = 0
        for name in sorted(lcl_df["Ticker"].unique()):
            numArr = lcl_df.iloc[lcl_df.index[lcl_df["Ticker"] == name].to_list()].sort_values("Date").to_numpy()
            ans_df.iloc[counter] = numArr.reshape(1,numArr.shape[0]*numArr.shape[1])
            counter += 1

        colsToRemove = []
        numberOfFeatures = lcl_df.shape[1]
        for row in COLUMNS_LEAVE_ME_ALONE_DECREASE_ME_THERE:
            for val in range(0,DATES_BACK):
                colsToRemove = np.append(colsToRemove, int(row + numberOfFeatures*val))

        ans_df.drop(columns=colsToRemove, axis=1, inplace=True)
        return ans_df

    def getLastConstDays(self, df: pd.DataFrame, list_of_stocks: pd.DataFrame):
        list_of_stocks_loc = list_of_stocks
        list_of_stocks_loc = list_of_stocks_loc.sort_values("Stock")
        p_los = 0
        ans_df = df.copy()
        # remove all dates that proceed the one in list of stocks
        ans_df.reset_index(inplace=True)
        for name, df_group in ans_df.sort_values("Date").groupby('Ticker'):
            # if keyerror add if is in
            ipo_date = list_of_stocks_loc["Date"][list_of_stocks_loc["Stock"].index[p_los]]
            ipo_date = dt.datetime.strptime(ipo_date, "%d/%m/%Y")
            p_los += 1
            for i, row in df_group.iterrows():
                if row[2] >= ipo_date:
                    ans_df.drop(i, inplace=True, axis=0)
        return ans_df.sort_values("Date").groupby('Ticker').tail(DATES_BACK)

    def get_train_test(self, train_percentage=0.8):
        # List of all unique stock-tickers in the dataset.
        tickers = self.df_sig_rets.reset_index()[TICKER].unique()

        # Split the tickers into training- and test-sets.
        tickers_train, tickers_test = \
            train_test_split(tickers, train_size=train_percentage, random_state=1234)

        return self.df_sig_rets.loc[tickers_train], self.df_sig_rets.loc[tickers_test]

    def YVal(self, df: pd.DataFrame, stock_list: pd.DataFrame):
        yDf = pd.DataFrame(columns=df.columns.insert(0, "Ticker"),index=range(df.index.unique().size))
        index = 0
        df = df.reset_index()
        for name, ipodate in zip(stock_list['Stock'], stock_list['Date']):
            stockDf = df.loc[df["Ticker"] == name]
            stockDf['Date'] = stockDf['Date'].apply(lambda x: x.strftime("%d/%m/%Y"))
            yDf.iloc[index] = stockDf[(stockDf["Ticker"] == name) & (stockDf["Date"] == ipodate)]
            index += 1
        return ((yDf["High"] - yDf["Open"])/yDf["Open"])*100


    def columnNormalizer(self, df: pd.DataFrame):
        for (columnName, columnData) in df.iteritems():
            df[columnName] = df[columnName] / df[columnName].abs().max()
        return df




