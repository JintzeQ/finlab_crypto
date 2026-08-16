#!/usr/bin/env python3
import io,json,math,zipfile,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','AVAXUSDT','SUIUSDT']; ALTS=SYMS[1:]
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
ROOT='https://data.binance.vision/data/futures/um/monthly/klines'; START=pd.Timestamp('2022-01-01',tz='UTC'); END=pd.Timestamp('2026-08-01',tz='UTC'); HRY=365.25*24

def months(): return [str(x) for x in pd.period_range('2022-01','2026-07',freq='M')]
def load(s):
 fs=[]
 for ym in months():
  try:
   with urllib.request.urlopen(f'{ROOT}/{s}/1h/{s}-1h-{ym}.zip',timeout=45) as r:b=r.read()
  except urllib.error.HTTPError as e:
   if e.code==404: continue
   raise
  with zipfile.ZipFile(io.BytesIO(b)) as z: fs.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None,names=COLS))
 d=pd.concat(fs,ignore_index=True); t=pd.to_numeric(d.open_time,errors='coerce'); t=t.where(t<1e15,t/1000); d['date']=pd.to_datetime(t,unit='ms',utc=True,errors='coerce')
 for c in ['open','close']: d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['date','open','close']).drop_duplicates('date').sort_values('date').set_index('date')
def cs_z(df): return df.sub(df.mean(axis=1),axis=0).div(df.std(axis=1,ddof=1).replace(0,np.nan),axis=0).clip(-4,4)
def signal_for(cl,bw=168,h=336,skip=24):
 r=cl.pct_change(); btc=r.BTCUSDT; var=btc.rolling(bw,min_periods=bw//2).var().shift(1); e=pd.DataFrame(index=cl.index,columns=ALTS,dtype=float)
 for s in ALTS:
  b=(r[s].rolling(bw,min_periods=bw//2).cov(btc).shift(1)/var).clip(-1,3); e[s]=r[s]-b*btc
 mom=e.shift(skip).rolling(h,min_periods=max(48,h//2)).sum(); vol=e.shift(skip).rolling(h,min_periods=max(48,h//2)).std()*np.sqrt(h)
 return cs_z(mom/vol.replace(0,np.nan))
def cohort_weights(sig,reb=48,phase=0):
 A=sig.to_numpy(float); out=np.zeros_like(A); prev=np.zeros(A.shape[1])
 for i,row in enumerate(A):
  if (i-phase)%reb: out[i]=prev; continue
  ok=np.isfinite(row); w=np.zeros(A.shape[1])
  if ok.sum()>=7:
   ids=np.where(ok)[0]; o=ids[np.argsort(row[ids])]; w[o[-2:]]=.25; w[o[:2]]=-.25
  prev=w; out[i]=w
 return pd.DataFrame(out,index=sig.index,columns=sig.columns)
def staggered(sig): return sum((cohort_weights(sig,48,p) for p in range(48)))/48

def simulate(Wa,p1a,p2a,hours,initial=70.0,fee_bps=5.0,slip_bps=0.0,funding_bps_8h=0.0,min_notional=0.0,leverage=1.0):
 qty=np.zeros(len(ALTS)); eq=initial; peak=initial; min_eq=initial; maxdd=0.; fees=slips=funds=turn=0.; trades=skipped=0; equities=[]; grosses=[]
 for i in range(len(Wa)):
  p1=p1a[i]; p2=p2a[i]; trad=np.isfinite(p1)&np.isfinite(p2)
  if trad.sum()<6:
   equities.append(eq); grosses.append(0.); continue
  target=np.nan_to_num(Wa[i],nan=0.0)*eq*leverage; cur=np.zeros(len(ALTS)); cur[trad]=qty[trad]*p1[trad]; delta=target-cur
  nz=trad & (np.abs(delta)>1e-12); eligible=nz & (np.abs(delta)>=min_notional); skipped+=int((nz & ~eligible).sum())
  if eligible.any():
   traded=np.abs(delta[eligible]); tv=float(traded.sum()); turn+=tv; trades+=int(eligible.sum()); fees+=tv*fee_bps/1e4; slips+=tv*slip_bps/1e4; eq-=tv*(fee_bps+slip_bps)/1e4; qty[eligible]+=delta[eligible]/p1[eligible]
  gross=float(np.abs(qty[trad]*p1[trad]).sum()); fund=gross*funding_bps_8h/1e4 if hours[i]%8==0 else 0.; funds+=fund; eq-=fund
  eq+=float((qty[trad]*(p2[trad]-p1[trad])).sum()); peak=max(peak,eq); min_eq=min(min_eq,eq); maxdd=min(maxdd,eq/peak-1 if peak>0 else -1.); equities.append(eq); grosses.append(gross)
  if eq<=0: break
 e=np.asarray(equities); r=np.zeros(len(e)); r[1:]=e[1:]/e[:-1]-1; sd=r.std(ddof=1) if len(r)>1 else np.nan; sh=r.mean()/sd*math.sqrt(HRY) if np.isfinite(sd) and sd>0 else np.nan; yrs=len(e)/HRY
 return {'final_equity':float(eq),'return_pct':float((eq/initial-1)*100),'CAGR':float((eq/initial)**(1/yrs)-1) if eq>0 and yrs>0 else -1,'Sharpe':float(sh),'MaxDD':float(maxdd),'min_equity':float(min_eq),'fees_usdt':float(fees),'slippage_usdt':float(slips),'funding_usdt':float(funds),'traded_notional_usdt':float(turn),'trade_count':int(trades),'skipped_small_deltas':int(skipped),'avg_gross_notional':float(np.mean(grosses)) if grosses else 0.0,'max_gross_notional':float(np.max(grosses)) if grosses else 0.0}

def main():
 raw={}
 with ThreadPoolExecutor(max_workers=10) as ex:
  fs={ex.submit(load,s):s for s in SYMS}
  for f in as_completed(fs): raw[fs[f]]=f.result(); print('DATA',fs[f],len(raw[fs[f]]),flush=True)
 idx=pd.date_range(START,END-pd.Timedelta(hours=1),freq='h'); cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in SYMS}); op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in SYMS}); sig=signal_for(cl); W=staggered(sig)
 W=W.loc[W.index>=pd.Timestamp('2022-07-01',tz='UTC')]; p1=op[ALTS].reindex(W.index+pd.Timedelta(hours=1)).to_numpy(float); p2=op[ALTS].reindex(W.index+pd.Timedelta(hours=2)).to_numpy(float); Wa=W.to_numpy(float); hours=W.index.hour.to_numpy()
 gross=W.abs().sum(axis=1); net=W.sum(axis=1); out={'audit':{'mean_target_gross':float(gross.mean()),'max_target_gross':float(gross.max()),'mean_abs_net':float(net.abs().mean()),'max_abs_net':float(net.abs().max()),'cohort_count':48,'aggregation':'sum of 48 cohort weights divided by 48; only aggregate net delta is executed'},'scenarios':{}}
 scenarios=[('ideal_fee5',5,0,0,0,1),('min5_fee5',5,0,0,5,1),('min5_fee5_slip2',5,2,0,5,1),('min5_fee5_slip2_fund1',5,2,1,5,1),('min5_fee5_slip5_fund1',5,5,1,5,1),('min10_fee5_slip2_fund1',5,2,1,10,1)]
 for name,fee,slip,fund,mn,lev in scenarios:
  print('RUN',name,flush=True); out['scenarios'][name]=simulate(Wa,p1,p2,hours,70,fee,slip,fund,mn,lev)
 Path('alpha_v8_account70_output').mkdir(exist_ok=True); json.dump(out,open('alpha_v8_account70_output/summary.json','w'),indent=2,allow_nan=True); print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__': main()
