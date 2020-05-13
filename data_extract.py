import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from utils import Option

data_path = RAW_DATA_PATH
mf = pd.read_csv(os.path.join(data_path, "RAW_CSV"))
options = mf[mf.INSTRUMENT == "OPTSTK"]
colgate = options[options.SYMBOL == 'ACTIVE NAME']
callinds = colgate[colgate.OPTION_TYP == "CE"]

duples = colgate[["EXPIRY_DT", "STRIKE_PR",  "EXPIRY_DT", "TIMESTAMP"]]
treasury = pd.read_csv(os.path.join(data_path, "RISK_FREE_RATE"))
prices = pd.read_csv(os.path.join(data_path, "STOCK_PRICES"))
treasury.Date = pd.to_datetime(treasury.Date)
prices.Date = pd.to_datetime(prices.Date)
treasury = treasury.merge(prices[['Date', "CloseCOL"]], on="Date", how='left')


treasury = treasury.fillna(method='pad')
colgate.TIMESTAMP = pd.to_datetime(colgate.TIMESTAMP)
colgate.EXPIRY_DT = pd.to_datetime(colgate.EXPIRY_DT)


dubl_inds = dict()
for i in tqdm(callinds.index):
    notdup = duples.drop(i, axis=0)
    values = duples.loc[i]
    put_ind = (values == notdup).mean(axis=1).index[(
        values == notdup).mean(axis=1) == 1]
    dubl_inds[i] = put_ind[0]


datas = []
for k in tqdm(dubl_inds.items()):
    c1 = colgate.loc[k[0]]
    p1 = colgate.loc[k[1]]

    T = c1.EXPIRY_DT - c1.TIMESTAMP
    T = T.days
    T /= 356
    tres = treasury[treasury['Date'] == c1.TIMESTAMP]
    # print(tres)
    price = tres['CloseCOL'].values[0]
    # print(pricel)
    rf = tres['Close']
    opt = Option(strike_price=c1.STRIKE_PR,
                 call_price=c1.SETTLE_PR,
                 put_price=p1.SETTLE_PR,
                 asset_price=price,
                 rf_rate=rf.values[0] / 100,
                 T=T)
    try:
        moneyness = opt.logforward_moneyness()
        vol = opt.implied_volatility()

        datas.append([moneyness, T, vol])
    except ZeroDivisionError:
        continue
datas = np.array(datas)
datas = pd.DataFrame(datas, columns=['m', 't', 'v'])

datas.to_csv("OUTPUT_PATH", index=False)
