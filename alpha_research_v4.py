#!/usr/bin/env python3
import io,json,math,zipfile,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','AVAXUSDT','SUIUSDT']
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
ROOT='https://data.binance.vision/data/futures/um/monthly/klines'; HRY=365.25*24
START=pd.Timestamp('2022-01-01',tz='UTC'); END=pd.Timestamp('2026-08-01',tz='UTC')
FOLDS=[('f1','2022-07-01','2023-01-01'),('f2','2023-01-01','2023-07-01'),('f3','2023-07-01','2024-01-01'),('f4','2024-01-01','2024-07-01'),('f5','2024-07-01','2025-01-01'),('f6','2025-01-01','2025-07-01')]
C0=pd.Timestamp('2025-07-01',tz='UTC');C1=pd.Timestamp('2026-01-01',tz='UTC');T0=C1;T1=END

def months():return [str(x) for x in pd.period_range('2022-01','2026-07',freq='M')]
def load(s):
 fs=[]
 for ym in months():
  u=f'{ROOT}/{s}/1h/{s}-1h-{ym}.zip'
  try:
   with urllib.request.urlopen(u,timeout=45) as r:b=r.read()
  except urllib.error.HTTPError as e:
   if e.code==404:continue
   raise
  with zipfile.ZipFile(io.BytesIO(b)) as z:fs.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None,names=COLS))
 d=pd.concat(fs,ignore_index=True);t=pd.to_numeric(d.open_time,errors='coerce');t=t.where(t<1e15,t/1000);d['date']=pd.to_datetime(t,unit='ms',utc=True,errors='coerce')
 for c in ['open','close','quote_volume']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['date','open','close']).drop_duplicates('date').sort_values('date').set_index('date')
def weights(sig,reb):
 a=sig.to_numpy(float);o=np.zeros_like(a);prev=np.zeros(a.shape[1])
 for i,row in enumerate(a):
  if i%reb:o[i]=prev;continue
  ok=np.isfinite(row);w=np.zeros(a.shape[1])
  if ok.sum()>=8:
   ids=np.where(ok)[0];v=row[ids];order=ids[np.argsort(v)];lo=order[:2];hi=order[-2:];w[hi]=.25;w[lo]=-.25
  prev=w;o[i]=w
 return pd.DataFrame(o,index=sig.index,columns=sig.columns)
def perf(r,turn):
 r=r.fillna(0);eq=(1+r).cumprod();sd=r.std(ddof=1);sh=r.mean()/sd*math.sqrt(HRY) if sd>0 else np.nan;yrs=len(r)/HRY;cagr=float(eq.iloc[-1]**(1/yrs)-1) if len(r) and eq.iloc[-1]>0 else -1.;mdd=float((eq/eq.cummax()-1).min());return {'Sharpe':float(sh),'CAGR':cagr,'Return':float(eq.iloc[-1]-1),'MaxDD':mdd,'AvgRet_bps':float(r.mean()*1e4),'Turnover':float(turn.mean())}
def calc(W,R,sl,cost=5):
 w=W.loc[sl];rr=R.reindex(w.index);g=(w*rr).sum(1);t=w.diff().abs().sum(1);t.iloc[0]=w.iloc[0].abs().sum();return perf(g-t*cost/1e4,t),perf(g,t)
def main():
 raw={}
 with ThreadPoolExecutor(max_workers=10) as ex:
  fs={ex.submit(load,s):s for s in SYMS}
  for f in as_completed(fs):s=fs[f];raw[s]=f.result();print('DATA',s,len(raw[s]),flush=True)
 idx=pd.date_range(START,END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in SYMS});R=op.shift(-2)/op.shift(-1)-1;ret1=cl.pct_change();rv24=ret1.rolling(24,min_periods=12).std().shift(1)
 cand={}
 for h in [12,24,48,72,168,336]:
  rr=cl/cl.shift(h)-1; vol=ret1.rolling(h,min_periods=max(6,h//2)).std()*np.sqrt(h); cons=ret1.rolling(h,min_periods=max(6,h//2)).apply(lambda x: np.nanmean(np.sign(x)),raw=True)
  sigs={'raw':rr,'risk':rr/vol.replace(0,np.nan),'cons':cons*np.sqrt(h)}
  for typ,sig0 in sigs.items():
   # cross-sectional demeaning only, no future info
   sig0=sig0.sub(sig0.mean(1),axis=0)
   for side in [1,-1]:
    for reb in [6,12,24,48]:cand[f'{"mom" if side==1 else "rev"}_{typ}_h{h}_r{reb}']=weights(side*sig0,reb)
 rows=[]
 for n,W in cand.items():
  vals=[]
  for fn,a,b in FOLDS:
   p,_=calc(W,R,slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(b,tz='UTC')-pd.Timedelta(hours=1)),5);vals.append(p['Sharpe'])
  rows.append({'name':n,'median_sh':float(np.nanmedian(vals)),'mean_sh':float(np.nanmean(vals)),'worst_sh':float(np.nanmin(vals)),'positive_folds':int(np.sum(np.array(vals)>0)),'folds':vals})
 tab=pd.DataFrame(rows).sort_values(['median_sh','positive_folds','worst_sh'],ascending=False).reset_index(drop=True)
 # Predeclared robust selector: at least 4/6 positive folds; maximize median Sharpe, tie by worst fold.
 elig=tab[tab.positive_folds>=4];win=(elig.iloc[0] if len(elig) else tab.iloc[0]);W=cand[win['name']]
 out={'winner':win.to_dict(),'top20':tab.head(20).to_dict('records'),'confirmation':{},'test':{}}
 cs=slice(C0,C1-pd.Timedelta(hours=1));ts=slice(T0,T1-pd.Timedelta(hours=1))
 for c in [0,2,3,5,8,12]:out['confirmation'][str(c)]=calc(W,R,cs,c)[0];out['test'][str(c)]=calc(W,R,ts,c)[0]
 out['accepted']=bool(win['median_sh']>.5 and win['positive_folds']>=5 and out['confirmation']['5']['Sharpe']>.5 and out['test']['5']['Sharpe']>1 and out['test']['8']['Sharpe']>0 and out['test']['5']['CAGR']>0)
 Path('alpha_v4_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v4_output/summary.json','w'),indent=2,allow_nan=True);tab.drop(columns=['folds']).to_csv('alpha_v4_output/grid.csv',index=False);print('===V4===');print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
