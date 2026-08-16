#!/usr/bin/env python3
import io,json,math,zipfile,urllib.request,urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
U=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','AVAXUSDT','SUIUSDT']
C=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
BASE='https://data.binance.vision/data/futures/um/monthly/klines'; START='2022-01-01'; END='2026-07-31'; WIN=168; MINP=72; HY=365.25*24

def months(): return [str(x) for x in pd.period_range(pd.Period(START,'M'),pd.Period(END,'M'),freq='M')]
def dl(sym):
 p=Path('cache');p.mkdir(exist_ok=True);fs=[]
 for ym in months():
  fn=p/f'{sym}-1h-{ym}.zip';url=f'{BASE}/{sym}/1h/{sym}-1h-{ym}.zip'
  if not fn.exists():
   try:
    with urllib.request.urlopen(url,timeout=90) as r: fn.write_bytes(r.read())
   except urllib.error.HTTPError as e:
    if e.code==404: continue
    raise
  try:
   with zipfile.ZipFile(fn) as z: fs.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None,names=C))
  except zipfile.BadZipFile:
   fn.unlink(missing_ok=True);raise
 if not fs:return pd.DataFrame()
 d=pd.concat(fs,ignore_index=True);ot=pd.to_numeric(d.open_time,errors='coerce');ms=ot.where(ot<1e15,ot/1000.)
 d['date']=pd.to_datetime(ms,unit='ms',utc=True,errors='coerce')
 for c in ['open','high','low','close','volume','quote_volume','taker_buy_quote']: d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna(subset=['date','open','high','low','close','quote_volume','taker_buy_quote']).drop_duplicates('date').sort_values('date').set_index('date')
 lo=pd.Timestamp(START,tz='UTC');hi=pd.Timestamp(END,tz='UTC')+pd.Timedelta(days=1);return d[(d.index>=lo)&(d.index<hi)]
def rz(s):
 med=s.rolling(WIN,min_periods=MINP).median().shift(1);mad=(s-med).abs().rolling(WIN,min_periods=MINP).median().shift(1);sc=1.4826*mad;sd=s.rolling(WIN,min_periods=MINP).std().shift(1);sc=sc.where(sc>1e-12,sd);return (s-med)/sc.replace(0,np.nan)
def feat(d):
 x=d.copy();x['r']=x.close.pct_change();x['rg']=(x.high-x.low)/x.open.replace(0,np.nan);x['lq']=np.log1p(x.quote_volume.clip(lower=0));x['im']=2*x.taker_buy_quote/x.quote_volume.replace(0,np.nan)-1;x['ef']=((x.close-x.open).abs()/(x.high-x.low).replace(0,np.nan)).clip(0,1);zr=rz(x.r);zg=rz(x.rg);zv=rz(x.lq);flow=np.sign(x.r)*x.im;ex=-zr*zg.clip(lower=0)*zv.clip(lower=0)*flow.clip(lower=0);ab=-np.sign(x.im)*x.im.abs()*(1-x.ef)*zv.clip(lower=0);x['a']=.6*ex+.4*ab;x['e1']=x.open.shift(-2)/x.open.shift(-1)-1
 for h in [1,2,3,6]:x[f'f{h}']=x.close.shift(-h)/x.close-1
 return x
def sp(a,b):
 m=a.notna()&b.notna();return a[m].rank().corr(b[m].rank()) if m.sum()>=5 else np.nan
def nwt(x,L=24):
 x=pd.Series(x).dropna().values;n=len(x)
 if n<50:return np.nan
 u=x-x.mean();v=np.dot(u,u)/n
 for k in range(1,min(L,n-1)+1):v+=2*(1-k/(L+1))*np.dot(u[k:],u[:-k])/n
 se=math.sqrt(max(v,0)/n);return x.mean()/se if se>0 else np.nan
def perf(r,t=None):
 r=pd.Series(r).dropna();n=len(r)
 if not n:return {}
 eq=(1+r).cumprod();yrs=n/HY;total=float(eq.iloc[-1]-1);cagr=float(eq.iloc[-1]**(1/yrs)-1) if eq.iloc[-1]>0 else np.nan;sd=float(r.std(ddof=1));sh=float(r.mean()/sd*math.sqrt(HY)) if sd>0 else np.nan;dn=float(r[r<0].std(ddof=1));so=float(r.mean()/dn*math.sqrt(HY)) if np.isfinite(dn) and dn>0 else np.nan;dd=eq/eq.cummax()-1;mdd=float(dd.min());pos=float(r[r>0].sum());neg=float(-r[r<0].sum());o={'Hours':n,'TotalReturn':total,'CAGR':cagr,'AnnVol':sd*math.sqrt(HY),'Sharpe':sh,'Sortino':so,'MaxDD':mdd,'Calmar':cagr/abs(mdd) if mdd<0 else np.nan,'HitRate':float((r>0).mean()),'ProfitFactor':pos/neg if neg>0 else np.nan,'AvgHourlyRet_bps':float(r.mean()*1e4)}
 if t is not None:o['AvgHourlyTurnover']=float(t.reindex(r.index).mean());o['AnnualTurnover']=float(t.reindex(r.index).mean()*HY)
 return o
def main():
 raw={}
 with ThreadPoolExecutor(max_workers=10) as ex:
  fs={ex.submit(dl,s):s for s in U}
  for f in as_completed(fs):
   s=fs[f];d=f.result();raw[s]=d;print('DATA',s,len(d),d.index.min(),d.index.max(),flush=True)
 D={s:feat(raw[s]) for s in U};A=pd.concat({s:D[s].a for s in U},axis=1);R=pd.concat({s:D[s].e1 for s in U},axis=1);A=A.loc[A.notna().sum(axis=1)>=8];R=R.reindex(A.index);W=pd.DataFrame(0.,index=A.index,columns=A.columns)
 for t in A.index:
  a=A.loc[t].dropna().sort_values()
  if len(a)>=8:W.loc[t,a.index[:2]]=-.25;W.loc[t,a.index[-2:]]=.25
 g=(W*R).sum(axis=1);tr=W.diff().abs().sum(axis=1)
 if len(tr):tr.iloc[0]=W.iloc[0].abs().sum()
 n5=g-tr*.0005;idx=n5.dropna().index;cut=int(len(idx)*.7);ii=idx[:cut];oo=idx[cut:];S={'period':{'start':START,'end':END,'signal_hours':len(idx),'is_hours':len(ii),'oos_hours':len(oo)},'performance':{}}
 for k,r in [('FULL_GROSS',g),('FULL_NET_5BPS',n5),('IS_NET_5BPS',n5.reindex(ii)),('OOS_NET_5BPS',n5.reindex(oo))]:S['performance'][k]=perf(r,tr)
 S['cost_stress']={str(c):perf(g-tr*c/1e4,tr) for c in [0,3,5,8,12]};S['ic']={}
 for h in [1,2,3,6]:
  Y=pd.concat({s:D[s][f'f{h}'] for s in U},axis=1).reindex(A.index);ic=pd.Series([sp(A.loc[t],Y.loc[t]) for t in A.index],index=A.index).dropna();io=ic.reindex(oo).dropna()
  def st(z):
   sd=z.std(ddof=1);return {'N':len(z),'MeanIC':float(z.mean()),'MedianIC':float(z.median()),'ICStd':float(sd),'ICIR':float(z.mean()/sd) if sd>0 else np.nan,'PositiveIC':float((z>0).mean()),'NW_tstat_24lags':float(nwt(z))}
  S['ic'][f'{h}H']={'FULL':st(ic),'OOS':st(io)}
 B={}
 for q in range(10):
  v=[]
  for t in A.index:
   a=A.loc[t].dropna().sort_values();y=R.loc[t]
   if len(a)==10 and pd.notna(y.get(a.index[q],np.nan)):v.append(y[a.index[q]])
  B[str(q+1)]=float(np.mean(v)*1e4) if v else np.nan
 S['rank_bucket_bps']=B;S['top_minus_bottom_bps']=B['10']-B['1'];Path('backtest_output').mkdir(exist_ok=True);Path('backtest_output/summary.json').write_text(json.dumps(S,indent=2,allow_nan=True));pd.DataFrame({'gross':g,'net5':n5,'turnover':tr,'equity':(1+n5.fillna(0)).cumprod()}).to_csv('backtest_output/equity.csv');print('===SUMMARY_JSON===');print(json.dumps(S,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
