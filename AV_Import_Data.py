from alpha_vantage.timeseries import TimeSeries


class AV_Import_Data():
    def __init__(self):
        key = '9UVZTQZ64K3XQHQ4'
        ts = TimeSeries(key=key, output_format='pandas')
        features = []
        self.data, meta_data = ts.get_intraday('GOOGL', interval='15min')
        data2, meta2 = ts
    def get_data(self):
        return self.data
