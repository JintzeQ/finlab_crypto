#!/usr/bin/env python3
import pandas as pd
import alpha_research_v3 as v3

def zero_funding(sym):
    idx=pd.DatetimeIndex([pd.Timestamp(v3.START,tz='UTC'),v3.T1-pd.Timedelta(hours=1)])
    return pd.Series([0.0,0.0],index=idx)

v3.read_funding=zero_funding
v3.main()
