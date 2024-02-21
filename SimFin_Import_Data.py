import pandas as pd
# Import the main functionality from the SimFin Python API.
import simfin as sf
# Import names used for easy access to SimFin's data-columns.
from simfin.names import *

class Simfin_data():
    def __init__(self):
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
        hub = sf.StockHub(market=market, offset=offset,
                          refresh_days=refresh_days,
                          refresh_days_shareprices=refresh_days_shareprices)

        # signals
        df_fin_signals = hub.fin_signals(variant='daily')
        df_growth_signals = hub.growth_signals(variant='daily')
        df_val_signals = hub.val_signals(variant='daily')
        df_share_signals = hub.price_signals()

        dfs = [df_fin_signals, df_growth_signals, df_val_signals, df_share_signals]
        df_signals = pd.concat(dfs, axis=1)




        # PREPROCESS
        # Remove all rows with only NaN values.
        df = df_signals.dropna(how='all').reset_index(drop=True)

        # For each column, show the fraction of the rows that are NaN.
        (df.isnull().sum() / len(df)).sort_values(ascending=False)

        # List of the columns before removing any.
        columns_before = df_signals.columns

        # Threshold for the number of rows that must be NaN for each column.
        thresh = 0.75 * len(df_signals.dropna(how='all'))

        # Remove all columns which don't have sufficient data.
        df_signals = df_signals.dropna(axis='columns', thresh=thresh)

        # List of the columns after the removal.
        columns_after = df_signals.columns

        # Show the columns that were removed.
        columns_before.difference(columns_after)
        
        # add column for results
        TOTAL_RETURN_1_3Y = 'Total Return 1-3 Years'
        df_returns_1_3y = \
            hub.mean_log_returns(name=TOTAL_RETURN_1_3Y,
                                 future=True, annualized=True,
                                 min_years=1, max_years=3)

        # combine data into one df
        dfs = [df_signals, df_returns_1_3y]
        self.df_sig_rets = pd.concat(dfs, axis=1)

        # clean data (winsorize it)
        # Clip both the signals and returns at their 5% and 95% quantiles.
        # We do not set them to NaN because it would remove too much data.
        self.df_sig_rets = sf.winsorize(self.df_sig_rets)

        # Remove all rows with missing values (NaN)
        # because TensorFlow cannot handle that.
        self.df_sig_rets = self.df_sig_rets.dropna(how='any')

        # Remove all tickers which have less than 200 data-rows.
        self.df_sig_rets = self.df_sig_rets.groupby(TICKER) \
            .filter(lambda df: len(df) > 200)

        self.df_sig_rets.head()
        print("lol")

    def get_train_test(self, train_percentage=0.8):
        # List of all unique stock-tickers in the dataset.
        tickers = self.df_sig_rets.reset_index()[TICKER].unique()

        # Split the tickers into training- and test-sets.
        tickers_train, tickers_test = \
            train_test_split(tickers, train_size=train_percentage, random_state=1234)

        return self.df_sig_rets.loc[tickers_train], self.df_sig_rets.loc[tickers_test]

    # def get_revenue(self):
    #     df_income.loc['MSFT'][REVENUE]
