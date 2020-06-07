from pandas_datareader import data



def generateData(stocks, start_date, end_date):
    for stock in stocks:
        panel_data = data.DataReader(stock, start=start_date, end=end_date, data_source='yahoo')
        panel_data.to_csv("data/" + stock + "_" + start_date + "_" + end_date + ".csv")
    print("done")







if __name__ == "__main__":
    start_date = "2020-03-01"
    end_date = "2020-05-31"
    generateData(["INVE-B.ST", "EVO.ST"], start_date, end_date)