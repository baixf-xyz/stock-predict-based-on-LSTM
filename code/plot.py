from jqdatasdk import *
import pandas as pd
from datetime import datetime, timedelta

auth('17860271144','qwer1234Q')

startDate = '2006-05-01'
endDate = '2022-05-01'
stock_mt = "600519.XSHG"
dfmt = get_price(stock_mt,start_date=startDate, end_date=endDate, frequency='daily', skip_paused=True, fq='pre',panel=False)
stock_wk="000002.XSHE"
dfwk = get_price(stock_wk,start_date=startDate, end_date=endDate, frequency='daily', skip_paused=True, fq='pre',panel=False)
stock_ynby="000538.XSHE"
dfynby = get_price(stock_ynby,start_date=startDate, end_date=endDate, frequency='daily', skip_paused=True, fq='pre',panel=False)
stock_zggw="002049.XSHE"
dfzggw = get_price(stock_zggw,start_date=startDate, end_date=endDate, frequency='daily', skip_paused=True, fq='pre',panel=False)
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')
plt.plot(dfmt['open'],color="blue",label="open")
plt.plot(dfmt['close'],color="red",label="close")
plt.plot(dfmt['high'],color="green",label="high")
plt.plot(dfmt['low'],color="yellow",label="low")
plt.legend()
plt.show()
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')
# plt.plot(dfmt['open'],color="blue",label="MTOpen")
plt.plot(dfwk['open'],color="red",label="WKOpen")
plt.plot(dfynby['open'],color="green",label="YNBYOpen")
plt.plot(dfzggw['open'],color="yellow",label="ZGGWOpen")
plt.legend()
plt.show()