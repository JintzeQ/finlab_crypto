#!/usr/bin/env python3
import io, json, math, zipfile, urllib.request, urllib.error
from pathlib import Path
import numpy as np
import pandas as pd

UNIVERSE=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','AVAXUSDT','SUIUSDT']
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
BASE='https://data.binance.vision/data/futures/um/monthly/klines'
START='2022-01-01'; END='2026-07-31'; WIN=168; MINP=72
HOURS_YR=365.25*24

def months(a,b):
    return [str(x) for x in pd.period_range(pd.Period(a,'M'),pd.Period(b,'M'),freq='M')]

def load_symbol(sym, cache='cache'):
    p=Path(cache); p.mkdir(exist_ok=True)
    fs=[]
    for ym in months(START,END):
        fn=p/f'{sym}-1h-{ym}.zip'; url=f'{BASE}/{sym}/1h/{sym}-1h-{ym}.zip'
        if not fn.exists():
            try:
                with urllib.request.urlopen(url,timeout=90) as r: fn.write_bytes(r.read())
            except urllib.error.HTTPError as e:
                if e.code==404: continue
                raise
        with zipfile.ZipFile(fn) as z:
            raw=z.read(z.namelist()[0]); fs.append(pd.read_csv(io.BytesIO(raw),header=None,names=COLS))
    if not fs: return pd.DataFrame()
    d=pd.concat(fs,ignore_index=True)
    # Binance archive transitioned timestamps from ms to us for some recent files; infer unit row-wise by magnitude.
    ot=pd.to_numeric(d.open_time,errors='coerce')
    ms=ot.where(ot<1e15, ot/1000.0)
    d['date']=pd.to_datetime(ms,unit='ms',utc=True,errors='coerce')
    for c in ['open','high','low','close','volume','quote_volume','taker_buy_quote']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['date','open','high','low','close','quote_volume','taker_buy_quote']).drop_duplicates('date').sort_values('date').set_index('date')
    lo=pd.Timestamp(START,tz='UTC'); hi=pd.Timestamp(END,tz='UTC')+pd.Timedelta(days=1)
    return d[(d.index>=lo)&(d.index<hi)]

def rz(s):
    med=s.rolling(WIN,min_periods=MINP).median().shift(1)
    mad=(s-med).abs().rolling(WIN,min_periods=MINP).median().shift(1)
    scale=1.4826*mad
    std=s.rolling(WIN,min_periods=MINP).std().shift(1)
    scale=scale.where(scale>1e-12,std)
    return (s-med)/scale.replace(0,np.nan)

def feats(d):
    x=d.copy()
    x['ret1']=x.close.pct_change(); x['range']=(x.high-x.low)/x.open.replace(0,np.nan)
    x['log_qv']=np.log1p(x.quote_volume.clip(lower=0))
    x['imb']=2*x.taker_buy_quote/x.quote_volume.replace(0,np.nan)-1
    x['eff']=((x.close-x.open).abs()/(x.high-x.low).replace(0,np.nan)).clip(0,1)
    x['zr']=rz(x.ret1); x['zrange']=rz(x['range']); x['zvol']=rz(x.log_qv)
    directional=np.sign(x.ret1)*x.imb
    ex=-x.zr*x.zrange.clip(lower=0)*x.zvol.clip(lower=0)*directional.clip(lower=0)
    ab=-np.sign(x.imb)*x.imb.abs()*(1-x.eff)*x.zvol.clip(lower=0)
    x['alpha']=0.6*ex+0.4*ab
    x['exec1']=x.open.shift(-2)/x.open.shift(-1)-1
    for h in [1,2,3,6]: x[f'fwd{h}']=x.close.shift(-h)/x.close-1
    return x

def spear(a,b):
    m=a.notna()&b.notna()
    return a[m].rank().corr(b[m].rank()) if m.sum()>=5 else np.nan

def nw_t(x,L=24):
    x=pd.Series(x).dropna().values; n=len(x)
    if n<50:return np.nan
    u=x-x.mean(); lrv=np.dot(u,u)/n
    for lag in range(1,min(L,n-1)+1):
        g=np.dot(u[lag:],u[:-lag])/n; lrv+=2*(1-lag/(L+1))*g
    se=math.sqrt(max(lrv,0)/n); return x.mean()/se if se>0 else np.nan

def perf(r,turn=None):
    r=pd.Series(r).dropna(); n=len(r)
    if not n:return {}
    eq=(1+r).cumprod(); total=float(eq.iloc[-1]-1); yrs=n/HOURS_YR
    cagr=float(eq.iloc[-1]**(1/yrs)-1) if yrs>0 and eq.iloc[-1]>0 else np.nan
    sd=float(r.std(ddof=1)); vol=sd*math.sqrt(HOURS_YR); sh=float(r.mean()/sd*math.sqrt(HOURS_YR)) if sd>0 else np.nan
    dn=float(r[r<0].std(ddof=1)); so=float(r.mean()/dn*math.sqrt(HOURS_YR)) if np.isfinite(dn) and dn>0 else np.nan
    peak=eq.cummax(); dd=eq/peak-1; mdd=float(dd.min()); cal=cagr/abs(mdd) if np.isfinite(cagr) and mdd<0 else np.nan
    pos=float(r[r>0].sum()); neg=float(-r[r<0].sum()); pf=pos/neg if neg>0 else np.nan
    out={'Hours':n,'TotalReturn':total,'CAGR':cagr,'AnnVol':vol,'Sharpe':sh,'Sortino':so,'MaxDD':mdd,'Calmar':cal,'HitRate':float((r>0).mean()),'ProfitFactor':pf,'AvgHourlyRet_bps':float(r.mean()*1e4)}
    if turn is not None:
        tt=turn.reindex(r.index); out['AvgHourlyTurnover']=float(tt.mean()); out['AnnualTurnover']=float(tt.mean()*HOURS_YR)
    return out

def main():
    data={}
    for s in UNIVERSE:
        d=load_symbol(s); data[s]=feats(d)
        print('DATA',s,len(d),str(d.index.min()),str(d.index.max()),flush=True)
    A=pd.concat({s:x.alpha for s,x in data.items()},axis=1)
    R=pd.concat({s:x.exec1 for s,x in data.items()},axis=1)
    A=A.loc[A.notna().sum(axis=1)>=8]; R=R.reindex(A.index)
    W=pd.DataFrame(0.,index=A.index,columns=A.columns)
    for t in A.index:
        a=A.loc[t].dropna().sort_values()
        if len(a)>=8:
            W.loc[t,a.index[:2]]=-0.25; W.loc[t,a.index[-2:]]=0.25
    gross=(W*R).sum(axis=1); turn=W.diff().abs().sum(axis=1)
    if len(turn):turn.iloc[0]=W.iloc[0].abs().sum()
    net=gross-turn*5e-4
    idx=net.dropna().index; cut=int(len(idx)*.70); iidx=idx[:cut]; oidx=idx[cut:]
    summary={'period':{'start':START,'end':END,'signal_hours':len(idx),'is_hours':len(iidx),'oos_hours':len(oidx)},'performance':{}}
    for name,rr in [('FULL_GROSS',gross),('FULL_NET_5BPS',net),('IS_NET_5BPS',net.reindex(iidx)),('OOS_NET_5BPS',net.reindex(oidx))]:
        summary['performance'][name]=perf(rr,turn)
    summary['cost_stress']={}
    for cb in [0,3,5,8,12]: summary['cost_stress'][str(cb)]=perf(gross-turn*(cb/1e4),turn)
    summary['ic']={}
    for h in [1,2,3,6]:
        Y=pd.concat({s:x[f'fwd{h}'] for s,x in data.items()},axis=1).reindex(A.index)
        ic=pd.Series([spear(A.loc[t],Y.loc[t]) for t in A.index],index=A.index).dropna()
        io=ic.reindex(oidx).dropna()
        def icstat(z):
            sd=z.std(ddof=1)
            return {'N':len(z),'MeanIC':float(z.mean()),'MedianIC':float(z.median()),'ICStd':float(sd),'ICIR':float(z.mean()/sd) if sd>0 else np.nan,'PositiveIC':float((z>0).mean()),'NW_tstat_24lags':float(nw_t(z))}
        summary['ic'][f'{h}H']={'FULL':icstat(ic),'OOS':icstat(io)}
    bucket={}
    for q in range(1,11):
        vals=[]
        for t in A.index:
            a=A.loc[t].dropna().sort_values(); y=R.loc[t]
            if len(a)==10 and pd.notna(y.get(a.index[q-1],np.nan)): vals.append(y[a.index[q-1]])
        bucket[str(q)]=float(np.mean(vals)*1e4) if vals else np.nan
    summary['rank_bucket_bps']=bucket
    summary['top_minus_bottom_bps']=bucket['10']-bucket['1']
    Path('backtest_output').mkdir(exist_ok=True)
    with open('backtest_output/summary.json','w') as f: json.dump(summary,f,indent=2,allow_nan=True)
    pd.DataFrame({'gross':gross,'net5':net,'turnover':turn,'equity':(1+net.fillna(0)).cumprod()}).to_csv('backtest_output/equity.csv')
    pd.DataFrame({'alpha_count':A.notna().sum(axis=1)}).to_csv('backtest_output/coverage.csv')
    print('\n===SUMMARY_JSON===')
    print(json.dumps(summary,indent=2,allow_nan=True),flush=True)

if __name__=='__main__': main()
