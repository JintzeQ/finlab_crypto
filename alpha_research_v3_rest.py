#!/usr/bin/env python3
import json,time,urllib.request,urllib.error
import pandas as pd
import alpha_research_v3 as v3

API='https://fapi.binance.com/fapi/v1/fundingRate'

def read_funding_rest(sym):
    start=int(pd.Timestamp(v3.START,tz='UTC').timestamp()*1000)
    end=int(v3.T1.timestamp()*1000)-1
    cur=start
    rows=[]
    while cur<=end:
        url=f'{API}?symbol={sym}&startTime={cur}&endTime={end}&limit=1000'
        batch=None
        for attempt in range(6):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(req,timeout=30) as r:
                    batch=json.loads(r.read().decode('utf-8'))
                break
            except urllib.error.HTTPError as e:
                if e.code in (418,429):
                    delay=int(e.headers.get('Retry-After','2'))
                    time.sleep(max(delay,2))
                    continue
                raise
            except Exception:
                if attempt==5: raise
                time.sleep(1+attempt)
        if not batch: break
        rows.extend(batch)
        nxt=max(int(x['fundingTime']) for x in batch)+1
        if nxt<=cur: break
        cur=nxt
        if len(batch)<1000: break
        time.sleep(0.12)
    if not rows:
        return pd.Series(dtype=float)
    d=pd.DataFrame(rows)
    idx=pd.to_datetime(pd.to_numeric(d['fundingTime'],errors='coerce'),unit='ms',utc=True,errors='coerce')
    s=pd.Series(pd.to_numeric(d['fundingRate'],errors='coerce').values,index=idx).dropna().sort_index()
    return s[~s.index.duplicated(keep='last')]

v3.read_funding=read_funding_rest
v3.main()
