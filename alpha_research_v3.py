#!/usr/bin/env python3
import io,json,math,zipfile,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','AVAXUSDT','SUIUSDT']; ALTS=[s for s in SYMS if s!='BTCUSDT']
KCOL=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
ROOT='https://data.binance.vision/data/futures/um/monthly'; START='2022-01-01'; END='2026-07-31'; HRY=365.25*24; WIN=168; MINP=72
V0=pd.Timestamp('2024-01-01',tz='UTC'); V1=pd.Timestamp('2025-07-01',tz='UTC'); C0=V1; C1=pd.Timestamp('2026-01-01',tz='UTC'); T0=C1; T1=pd.Timestamp('2026-08-01',tz='UTC')
def months(): return [str(x) for x in pd.period_range('2022-01','2026-07',freq='M')]
def getzip(url):
 try:
  with urllib.request.urlopen(url,timeout=45) as r:return r.read()
 except urllib.error.HTTPError as e:
  if e.code==404:return None
  raise
def read_kline(sym,dtype):
 fs=[]
 for ym in months():
  b=getzip(f'{ROOT}/{dtype}/{sym}/1h/{sym}-1h-{ym}.zip')
  if b is None:continue
  with zipfile.ZipFile(io.BytesIO(b)) as z:fs.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None,names=KCOL))
 if not fs:return pd.DataFrame()
 d=pd.concat(fs,ignore_index=True); ot=pd.to_numeric(d.open_time,errors='coerce'); ms=ot.where(ot<1e15,ot/1000); d['date']=pd.to_datetime(ms,unit='ms',utc=True,errors='coerce')
 for c in ['open','high','low','close','quote_volume','taker_buy_quote']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['date','open','close']).drop_duplicates('date').sort_values('date').set_index('date')
def read_funding(sym):
 fs=[]
 for ym in months():
  b=getzip(f'{ROOT}/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip')
  if b is None:continue
  with zipfile.ZipFile(io.BytesIO(b)) as z:
   q=pd.read_csv(io.BytesIO(z.read(z.namelist()[0]))); q.columns=[str(c).strip().lower() for c in q.columns]; fs.append(q)
 if not fs:return pd.Series(dtype=float)
 d=pd.concat(fs,ignore_index=True); tc=next((c for c in d.columns if 'time' in c),d.columns[0]); rc=next((c for c in d.columns if 'funding' in c and 'rate' in c),d.columns[-1]); t=pd.to_numeric(d[tc],errors='coerce'); ms=t.where(t<1e15,t/1000); idx=pd.to_datetime(ms,unit='ms',utc=True,errors='coerce'); s=pd.Series(pd.to_numeric(d[rc],errors='coerce').values,index=idx).dropna().sort_index(); return s[~s.index.duplicated(keep='last')]
def rz(s):
 med=s.rolling(WIN,min_periods=MINP).median().shift(1); mad=(s-med).abs().rolling(WIN,min_periods=MINP).median().shift(1); sc=(1.4826*mad).where(mad>1e-12,s.rolling(WIN,min_periods=MINP).std().shift(1)); return ((s-med)/sc.replace(0,np.nan)).clip(-8,8)
def cs_z(x):return x.sub(x.mean(1),axis=0).div(x.std(1,ddof=1).replace(0,np.nan),axis=0).clip(-4,4)
def perf(r,turn):
 r=pd.Series(r).fillna(0); n=len(r); eq=(1+r).cumprod(); sd=r.std(ddof=1); sh=r.mean()/sd*math.sqrt(HRY) if sd>0 else np.nan; yrs=n/HRY; cagr=float(eq.iloc[-1]**(1/yrs)-1) if n and eq.iloc[-1]>0 else -1.; mdd=float((eq/eq.cummax()-1).min()); pos=r[r>0].sum();neg=-r[r<0].sum();return {'Hours':n,'Return':float(eq.iloc[-1]-1),'CAGR':cagr,'Sharpe':float(sh),'MaxDD':mdd,'PF':float(pos/neg) if neg>0 else np.nan,'AvgRet_bps':float(r.mean()*1e4),'Turnover':float(turn.reindex(r.index).fillna(0).mean())}
def pnl(W,R,F,cost,sl):
 W=W.loc[sl]; R=R.reindex(W.index); F=F.reindex(W.index).fillna(0); gross=(W*R).sum(1); funding=(-W*F).sum(1); turn=W.diff().abs().sum(1); turn.iloc[0]=W.iloc[0].abs().sum(); net=gross+funding-turn*cost/1e4; return perf(net,turn),{'gross_price':float(gross.mean()*1e4),'funding_bps_h':float(funding.mean()*1e4)}
def event_weights(event,hold):
 # event entries are -1/0/+1 on alts. Hold last H events, then BTC hedge net alt exposure; total gross normalized to 1.
 h=event.rolling(hold,min_periods=1).sum(); h=np.sign(h); arr=h.to_numpy(float); out=np.zeros((len(h),len(SYMS))); amap={s:i+1 for i,s in enumerate(ALTS)}
 for i,row in enumerate(arr):
  active=np.where(np.isfinite(row)&(row!=0))[0]
  if len(active)==0:continue
  for j in active:out[i,amap[ALTS[j]]]=row[j]/len(active)
  out[i,0]=-out[i,1:].sum(); g=np.abs(out[i]).sum(); out[i]/=g if g>0 else 1
 return pd.DataFrame(out,index=h.index,columns=SYMS)
def rank_weights(sig,reb=8,thr=.5):
 a=sig.to_numpy(float); out=np.zeros_like(a); prev=np.zeros(a.shape[1])
 for i,row in enumerate(a):
  if i%reb:out[i]=prev;continue
  ok=np.isfinite(row); w=np.zeros(a.shape[1])
  if ok.sum()>=8:
   z=(row-np.nanmean(row))/np.nanstd(row,ddof=1); pos=np.where(z>=thr)[0];neg=np.where(z<=-thr)[0]
   if len(pos) and len(neg):
    p=pos[np.argsort(z[pos])[-2:]];n=neg[np.argsort(z[neg])[:2]];w[p]=.5/len(p);w[n]=-.5/len(n)
  prev=w;out[i]=w
 return pd.DataFrame(out,index=sig.index,columns=sig.columns)
def main():
 raw={}; prem={}; fund={}
 def one(s):return s,read_kline(s,'klines'),read_kline(s,'premiumIndexKlines'),read_funding(s)
 with ThreadPoolExecutor(max_workers=10) as ex:
  fut=[ex.submit(one,s) for s in SYMS]
  for f in as_completed(fut):
   s,k,p,fu=f.result();raw[s]=k;prem[s]=p;fund[s]=fu;print('DATA',s,len(k),len(p),len(fu),list(fu.head(1).items()),flush=True)
 idx=pd.date_range(pd.Timestamp(START,tz='UTC'),T1-pd.Timedelta(hours=1),freq='h'); close=pd.DataFrame({s:raw[s].close.reindex(idx) for s in SYMS}); op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in SYMS}); qv=pd.DataFrame({s:raw[s].quote_volume.reindex(idx) for s in SYMS}); imb=pd.DataFrame({s:(2*raw[s].taker_buy_quote/raw[s].quote_volume-1).reindex(idx) for s in SYMS}); R=op.shift(-2)/op.shift(-1)-1
 pm=pd.DataFrame({s:prem[s].close.reindex(idx) if len(prem[s]) else np.nan for s in SYMS}); pz=pd.DataFrame({s:rz(pm[s]) for s in SYMS}); ret=close.pct_change(); btc=ret.BTCUSDT; resid=ret[ALTS].sub(btc,axis=0); rzres=pd.DataFrame({s:rz(resid[s]) for s in ALTS}); zvol=pd.DataFrame({s:rz(np.log1p(qv[s])) for s in ALTS}); same=(np.sign(resid)*imb[ALTS]).clip(lower=0)
 # last known funding rate is point-in-time; funding PnL uses settlement at t+2 boundary, conservatively after entry.
 flast=pd.DataFrame({s:fund[s].reindex(idx,method='ffill') if len(fund[s]) else np.nan for s in SYMS}); fevent=pd.DataFrame(0.,index=idx,columns=SYMS)
 for s in SYMS:
  if len(fund[s]):
   hh=fund[s].copy();hh.index=hh.index.floor('h'); tmp=hh.groupby(level=0).last().reindex(idx);fevent[s]=tmp.shift(-2).fillna(0)
 candidates={}
 # Fixed sparse event family: residual shock + volume + same-direction taker flow; optional premium crowding confirmation.
 for zthr in [2.0,2.5,3.0,3.5]:
  for vthr in [.5,1.0]:
   for ithr in [.10,.20]:
    gate=(rzres.abs()>=zthr)&(zvol>=vthr)&(same>=ithr); ev=(-np.sign(rzres)).where(gate,0)
    for h in [1,2,3,6]:candidates[f'event_z{zthr}_v{vthr}_i{ithr}_h{h}']=event_weights(ev,h)
    pc=(np.sign(resid)*pz[ALTS]>=.5); ev2=(-np.sign(rzres)).where(gate&pc,0)
    for h in [1,2,3,6]:candidates[f'eventP_z{zthr}_v{vthr}_i{ithr}_h{h}']=event_weights(ev2,h)
 # Fixed carry/basis family. All signals known at candle close / from last settled funding.
 fz=cs_z(flast); ppz=cs_z(pm)
 for reb in [8,16,24]:
  for thr in [.5,1.0]:
   candidates[f'carryF_r{reb}_t{thr}']=rank_weights(-fz,reb,thr);candidates[f'carryP_r{reb}_t{thr}']=rank_weights(-ppz,reb,thr);candidates[f'carryFP_r{reb}_t{thr}']=rank_weights(-(fz+ppz),reb,thr)
 # Select only on validation 2024H1-2025H1; confirmation and 2026 are not used for ranking.
 vs=slice(V0,V1-pd.Timedelta(hours=1)); cs=slice(C0,C1-pd.Timedelta(hours=1)); ts=slice(T0,T1-pd.Timedelta(hours=1)); rows=[]
 for name,W in candidates.items():
  pv,_=pnl(W,R,fevent,5,vs);rows.append({'name':name,'val_sh':pv['Sharpe'],'val_cagr':pv['CAGR'],'val_mdd':pv['MaxDD'],'val_turn':pv['Turnover']})
 tab=pd.DataFrame(rows).sort_values(['val_sh','val_cagr'],ascending=False).reset_index(drop=True); win=tab.iloc[0].to_dict(); W=candidates[win['name']]; out={'winner':win,'top20':tab.head(20).to_dict('records'),'validation':{},'confirmation':{},'test':{}}
 for c in [0,2,3,5,8]:
  out['validation'][str(c)]=pnl(W,R,fevent,c,vs)[0];out['confirmation'][str(c)]=pnl(W,R,fevent,c,cs)[0];out['test'][str(c)]=pnl(W,R,fevent,c,ts)[0]
 out['accepted']=bool(out['validation']['5']['Sharpe']>.75 and out['confirmation']['5']['Sharpe']>.5 and out['test']['5']['Sharpe']>1 and out['test']['5']['CAGR']>0 and out['test']['8']['Sharpe']>0)
 out['test_monthly']={}
 for p,g in W.loc[ts].groupby(W.loc[ts].index.to_period('M')):
  sl=slice(g.index.min(),g.index.max());out['test_monthly'][str(p)]=pnl(W,R,fevent,5,sl)[0]
 Path('alpha_v3_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v3_output/summary.json','w'),indent=2,allow_nan=True);tab.to_csv('alpha_v3_output/validation_grid.csv',index=False);print('===V3===');print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
