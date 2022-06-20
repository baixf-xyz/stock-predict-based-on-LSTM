from jqdatasdk import *
import pandas as pd
from datetime import datetime, timedelta

auth('17860271144','qwer1234Q')  #聚宽换成自己的token

startDate = '2006-05-01'
endDate = '2022-05-01'
stock_code = ["600519.XSHG","000002.XSHE","000538.XSHE","002049.XSHE"]

for stock in stock_code:
    df = get_price(stock,start_date=startDate, end_date=endDate, frequency='daily', skip_paused=True, fq='pre',panel=False)
    df.to_csv('./data/'+stock+'.csv')
    print('finishied')