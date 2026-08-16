#!/usr/bin/env python3
import io,json,math,zipfile,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','AVAXUSDT','SUIUSDT']
ALTS=[s for s in SYMS if s!='BTCUSDT']
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
ROOT='https://data.binance.vision/data/futures/um/monthly/klines'; HRY=365.25*24
START=pd.Timestamp('2022-01-01',tz='UTC'); END=pd.Timestamp('2026-08-01',tz='UTC')
FOLDS=[('f1','2022-07-01','2023-01-01'),('f2','2023-01-01','2023-07-01'),('f3','2023-07-01','2024-01-01'),('f4','2024-01-01','2024-07-01'),('f5','2024-07-01','2025-01-01'),('f6','2025-01-01','2025-07-01')]
DEV=FOLDS[:4]; HOLD=FOLDS[4:]
C0=pd.Timestamp('2025-07-01',tz='UTC'); C1=pd.Timestamp('2026-01-01',tz='UTC'); T0=C1; T1=END

def months(): return [str(x) for x in pd.period_range('2022-01','2026-07',freq='M')]
def load(s):
 fs=[]
 for ym in months():
  try:
   with urllib.request.urlopen(f'{ROOT}/{s}/1h/{s}-1h-{ym}.zip',timeout=45) as r:b=r.read()
  except urllib.error.HTTPError as e:
   if e.code==404:continue
   raise
  with zipfile.ZipFile(io.BytesIO(b)) as z:fs.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None,names=COLS))
 d=pd.concat(fs,ignore_index=True);t=pd.to_numeric(d.open_time,errors='coerce');t=t.where(t<1e15,t/1000);d['date']=pd.to_datetime(t,unit='ms',utc=True,errors='coerce')
 for c in ['open','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['date','open','close']).drop_duplicates('date').sort_values('date').set_index('date')
def cs_z(df):
 m=df.mean(axis=1);sd=df.std(axis=1,ddof=1).replace(0,np.nan);return df.sub(m,axis=0).div(sd,axis=0).clip(-4,4)
def perf(r,t):
 r=r.fillna(0); eq=(1+r).cumprod(); sd=r.std(ddof=1); sh=r.mean()/sd*math.sqrt(HRY) if sd>0 else np.nan; yrs=len(r)/HRY
 cagr=float(eq.iloc[-1]**(1/yrs)-1) if len(r) and eq.iloc[-1]>0 else -1.; mdd=float((eq/eq.cummax()-1).min()); pos=r[r>0].sum();neg=-r[r<0].sum()
 return {'Sharpe':float(sh),'CAGR':cagr,'Return':float(eq.iloc[-1]-1),'MaxDD':mdd,'PF':float(pos/neg) if neg>0 else np.nan,'AvgRet_bps':float(r.mean()*1e4),'Turnover':float(t.mean()),'AnnualTurnover':float(t.mean()*HRY)}
def make_weights(sig,beta,reb,hedge):
 A=sig.to_numpy(float); B=beta.reindex(sig.index).to_numpy(float); n,m=A.shape; out=np.zeros((n,m+1)); prev=np.zeros(m+1)
 for i in range(n):
  if i%reb: out[i]=prev;continue
  row=A[i]; ok=np.isfinite(row)
  w=np.zeros(m+1)
  if ok.sum()>=7:
   ids=np.where(ok)[0];order=ids[np.argsort(row[ids])];wa=np.zeros(m);wa[order[-2:]]=.25;wa[order[:2]]=-.25
   if hedge=='btc':
    bb=B[i]; valid=np.isfinite(bb); hb=-float(np.nansum(wa[valid]*bb[valid])); hb=float(np.clip(hb,-.5,.5)); w[1:]=wa;w[0]=hb
    gross=np.abs(w).sum()
    if gross>0:w/=gross
   else:w[1:]=wa
  prev=w;out[i]=w
 return pd.DataFrame(out,index=sig.index,columns=['BTCUSDT']+ALTS)
def pnl_arrays(W,R):
 rr=R.reindex(W.index).to_numpy(float);w=W.to_numpy(float); gross=np.nansum(w*rr,axis=1);turn=np.abs(np.diff(w,axis=0,prepend=np.zeros((1,w.shape[1])))).sum(axis=1);return pd.Series(gross,index=W.index),pd.Series(turn,index=W.index)
def calc_from(gt,sl,cost=5):
 g,t=gt;g=g.loc[sl];t=t.loc[sl];return perf(g-t*cost/1e4,t),perf(g,t)
def main():
 raw={}
 with ThreadPoolExecutor(max_workers=10) as ex:
  fs={ex.submit(load,s):s for s in SYMS}
  for f in as_completed(fs):s=fs[f];raw[s]=f.result();print('DATA',s,len(raw[s]),flush=True)
 idx=pd.date_range(START,END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in SYMS})
 r=cl.pct_change(); R=op.shift(-2)/op.shift(-1)-1; btc=r.BTCUSDT
 betas={};resids={}
 for bw in [168,336]:
  var=btc.rolling(bw,min_periods=bw//2).var().shift(1)
  b=pd.DataFrame(index=idx,columns=ALTS,dtype=float)
  e=pd.DataFrame(index=idx,columns=ALTS,dtype=float)
  for s in ALTS:
   bs=r[s].rolling(bw,min_periods=bw//2).cov(btc).shift(1)/var
   bs=bs.clip(-1,3);b[s]=bs;e[s]=r[s]-bs*btc
  betas[bw]=b;resids[bw]=e
 # Closed family: 2 beta windows x 6 horizons x 2 skips x 2 scales x 4 rebalance x 2 hedge = 384.
 candidates=[]
 for bw in [168,336]:
  e=resids[bw];b=betas[bw]
  for h in [72,120,168,240,336,504]:
   for skip in [0,24]:
    mom=e.shift(skip).rolling(h,min_periods=max(48,h//2)).sum()
    vol=e.shift(skip).rolling(h,min_periods=max(48,h//2)).std()*np.sqrt(h)
    for scale in ['raw','risk']:
     sig=cs_z(mom if scale=='raw' else mom/vol.replace(0,np.nan))
     for reb in [24,48,72,96]:
      for hedge in ['none','btc']:
       n=f'b{bw}_h{h}_s{skip}_{scale}_r{reb}_{hedge}'
       W=make_weights(sig,b,reb,hedge);gt=pnl_arrays(W,R[['BTCUSDT']+ALTS]); candidates.append((n,gt))
 print('CANDIDATES',len(candidates),flush=True)
 rows=[]
 for n,gt in candidates:
  vals=[];turn=[]
  for fn,a,z in DEV:
   p,_=calc_from(gt,slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1)),5);vals.append(p['Sharpe']);turn.append(p['Turnover'])
  rows.append({'name':n,'median_sh':float(np.nanmedian(vals)),'mean_sh':float(np.nanmean(vals)),'worst_sh':float(np.nanmin(vals)),'positive_folds':int(np.sum(np.array(vals)>0)),'median_turn':float(np.nanmedian(turn)),'folds':vals})
 tab=pd.DataFrame(rows).sort_values(['positive_folds','worst_sh','median_sh'],ascending=False).reset_index(drop=True)
 # Selector locked before holdout: 4/4 positive, then highest worst fold, then median.
 elig=tab[(tab.positive_folds==4)&(tab.worst_sh>0)];win=(elig.iloc[0] if len(elig) else tab.iloc[0]);gt=dict(candidates)[win['name']]
 out={'winner':win.to_dict(),'top20':tab.head(20).to_dict('records'),'family_holdouts':{},'confirmation':{},'secondary_2026':{}}
 for fn,a,z in HOLD:
  sl=slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1));out['family_holdouts'][fn]={str(c):calc_from(gt,sl,c)[0] for c in [0,3,5,8]}
 cs=slice(C0,C1-pd.Timedelta(hours=1));ts=slice(T0,T1-pd.Timedelta(hours=1))
 for c in [0,2,3,5,8,12]:out['confirmation'][str(c)]=calc_from(gt,cs,c)[0];out['secondary_2026'][str(c)]=calc_from(gt,ts,c)[0]
 # Strong gate: both family holdouts positive at 5bps; 2025H2 >.5 at 5bps and >0 at 8bps; 2026 secondary >0 at 5bps.
 out['accepted']=bool(win['positive_folds']==4 and win['worst_sh']>0 and all(out['family_holdouts'][f]['5']['Sharpe']>0 for f in ['f5','f6']) and out['confirmation']['5']['Sharpe']>.5 and out['confirmation']['8']['Sharpe']>0 and out['confirmation']['5']['CAGR']>0 and out['secondary_2026']['5']['Sharpe']>0)
 Path('alpha_v6_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v6_output/summary.json','w'),indent=2,allow_nan=True);tab.drop(columns=['folds']).to_csv('alpha_v6_output/grid.csv',index=False);print('===V6===');print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
