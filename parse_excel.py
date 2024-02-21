import pandas as pd
import datetime as dt


class ParseExcel():
    def __init__(self, stockpath):
        self.file = pd.ExcelFile(stockpath).parse('Sheet1')
        # self.file["Date"] = self.file["Date"].dt.strftime('%d/%m/%Y')
        tmp1 = self.file["Stock"].iloc[::2]
        tmp1 = tmp1.reset_index()["Stock"]
        tmp2 = self.file["Date"][self.file["Date"].notna()]
        self.file = pd.DataFrame({"Stock": tmp1, "Date": tmp2})

    def get_file(self):
        return self.file

    def save_excel(self, savepath):
        self.file.to_excel(savepath, index=False)
