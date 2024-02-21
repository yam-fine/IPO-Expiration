class DailyData():
    def __init__(self):
        self.dayDict = {}

    def set_dailyData(self,date, data):
        self.dayDict[date] = data

    def get_dailyData(self,date):
        return self.dayDict[date]