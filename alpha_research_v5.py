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
def cs_z(df): return df.sub(df.mean(axis=1),axis=0).div(df.std(axis=1,ddof=1).replace(0,np.nan),axis=0).clip(-4,4)
def rolling_pct(s,w=720,minp=240):
 return s.rolling(w,min_periods=minp).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],raw=False).shift(1)
def weights(sig,reb,gate=None):
 a=sig.to_numpy(float);g=np.ones(len(sig),dtype=bool) if gate is None else gate.reindex(sig.index).fillna(False).to_numpy(bool);o=np.zeros_like(a);prev=np.zeros(a.shape[1])
 for i,row in enumerate(a):
  if not g[i]: prev=np.zeros(a.shape[1]);o[i]=prev;continue
  if i%reb:o[i]=prev;continue
  ok=np.isfinite(row);w=np.zeros(a.shape[1])
  if ok.sum()>=8:
   ids=np.where(ok)[0];order=ids[np.argsort(row[ids])];w[order[-2:]]=.25;w[order[:2]]=-.25
  prev=w;o[i]=w
 return pd.DataFrame(o,index=sig.index,columns=sig.columns)
def perf(r,t):
 r=r.fillna(0);eq=(1+r).cumprod();sd=r.std(ddof=1);sh=r.mean()/sd*math.sqrt(HRY) if sd>0 else np.nan;yrs=len(r)/HRY;cagr=float(eq.iloc[-1]**(1/yrs)-1) if len(r) and eq.iloc[-1]>0 else -1.;mdd=float((eq/eq.cummax()-1).min());pos=r[r>0].sum();neg=-r[r<0].sum();return {'Sharpe':float(sh),'CAGR':cagr,'Return':float(eq.iloc[-1]-1),'MaxDD':mdd,'PF':float(pos/neg) if neg>0 else np.nan,'AvgRet_bps':float(r.mean()*1e4),'Turnover':float(t.mean()),'ActiveFrac':float((t>0).mean())}
def calc(W,R,sl,cost=5):
 w=W.loc[sl];rr=R.reindex(w.index);gross=(w*rr).sum(1);t=w.diff().abs().sum(1);t.iloc[0]=w.iloc[0].abs().sum();return perf(gross-t*cost/1e4,t),perf(gross,t)
def main():
 raw={}
 with ThreadPoolExecutor(max_workers=10) as ex:
  fs={ex.submit(load,s):s for s in SYMS}
  for f in as_completed(fs):s=fs[f];raw[s]=f.result();print('DATA',s,len(raw[s]),flush=True)
 idx=pd.date_range(START,END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in SYMS});R=op.shift(-2)/op.shift(-1)-1;r1=cl.pct_change()
 # Base slow momentum components; all use data available at close(t).
 bases={}
 for h in [72,168,336]:
  rr=cl/cl.shift(h)-1; vol=r1.rolling(h,min_periods=max(24,h//2)).std()*np.sqrt(h); bases[f'risk{h}']=cs_z(rr/vol.replace(0,np.nan)); bases[f'raw{h}']=cs_z(rr)
 bases['blend72_168']=(bases['risk72']+bases['risk168'])/2
 bases['blend72_168_336']=(bases['risk72']+bases['risk168']+bases['risk336'])/3
 # Point-in-time regimes. Thresholds are trailing distribution based, not fixed using future info.
 alt24=(cl.drop(columns=['BTCUSDT'])/cl.drop(columns=['BTCUSDT']).shift(24)-1)
 dispersion=alt24.std(axis=1); disp_pct=rolling_pct(dispersion)
 btc24=r1.BTCUSDT.rolling(24,min_periods=12).std(); btcvol_pct=rolling_pct(btc24)
 btctrend=(cl.BTCUSDT/cl.BTCUSDT.shift(168)-1).abs()/(r1.BTCUSDT.rolling(168,min_periods=84).std()*np.sqrt(168)).replace(0,np.nan); trend_pct=rolling_pct(btctrend)
 breadth=alt24.apply(lambda x: abs(np.nanmean(np.sign(x))) if x.notna().sum()>=5 else np.nan,axis=1); breadth_pct=rolling_pct(breadth)
 regimes={'all':pd.Series(True,index=idx),'disp_hi':disp_pct>=.50,'disp_q60':disp_pct>=.60,'btcvol_lo':btcvol_pct<=.60,'btcvol_hi':btcvol_pct>=.40,'trend_hi':trend_pct>=.50,'breadth_hi':breadth_pct>=.50,'disp_trend':(disp_pct>=.50)&(trend_pct>=.50),'disp_breadth':(disp_pct>=.50)&(breadth_pct>=.50),'trend_vollo':(trend_pct>=.50)&(btcvol_pct<=.60)}
 # Fixed compact v5 space: 8 bases x 10 regimes x 3 rebalance = 240 configs.
 cand={}
 for bn,b in bases.items():
  for gn,g in regimes.items():
   for reb in [24,48,72]:cand[f'{bn}_{gn}_r{reb}']=weights(b,reb,g)
 rows=[]
 for n,W in cand.items():
  vals=[];turn=[]
  for fn,a,b in FOLDS:
   p,_=calc(W,R,slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(b,tz='UTC')-pd.Timedelta(hours=1)),5);vals.append(p['Sharpe']);turn.append(p['Turnover'])
  rows.append({'name':n,'median_sh':float(np.nanmedian(vals)),'mean_sh':float(np.nanmean(vals)),'worst_sh':float(np.nanmin(vals)),'positive_folds':int(np.sum(np.array(vals)>0)),'median_turn':float(np.nanmedian(turn)),'folds':vals})
 tab=pd.DataFrame(rows).sort_values(['median_sh','positive_folds','worst_sh'],ascending=False).reset_index(drop=True)
 # Robust selector fixed before confirmation: >=5/6 positive, positive worst fold preferred, median Sharpe highest.
 elig=tab[(tab.positive_folds>=5)&(tab.worst_sh>0)];win=(elig.iloc[0] if len(elig) else tab[tab.positive_folds>=5].iloc[0] if len(tab[tab.positive_folds>=5]) else tab.iloc[0]);W=cand[win['name']]
 cs=slice(C0,C1-pd.Timedelta(hours=1));ts=slice(T0,T1-pd.Timedelta(hours=1));out={'winner':win.to_dict(),'top20':tab.head(20).to_dict('records'),'confirmation':{},'secondary_2026':{}}
 for c in [0,2,3,5,8,12]:out['confirmation'][str(c)]=calc(W,R,cs,c)[0];out['secondary_2026'][str(c)]=calc(W,R,ts,c)[0]
 # Because 2026 was viewed in prior family research, it is secondary, not pristine OOS.
 out['accepted']=bool(win['median_sh']>.75 and win['positive_folds']>=5 and win['worst_sh']>0 and out['confirmation']['5']['Sharpe']>.5 and out['confirmation']['8']['Sharpe']>0 and out['confirmation']['5']['CAGR']>0 and out['secondary_2026']['5']['Sharpe']>0)
 Path('alpha_v5_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v5_output/summary.json','w'),indent=2,allow_nan=True);tab.drop(columns=['folds']).to_csv('alpha_v5_output/grid.csv',index=False);print('===V5===');print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
