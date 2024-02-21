class Company():
    def __init__(self,name,sector,dailyData,quarterData):
        self.name = name
        self.sector = sector
        self.dailyData = dailyData
        self.quarterData = quarterData
        
    def get_name(self):
        return self.name
    
    def get_sector(self):
        return self.sector
    
    def get_dailyData(self):
        return self.dailyData
    
    def get_quarterData(self):
        return self.quarterData
    