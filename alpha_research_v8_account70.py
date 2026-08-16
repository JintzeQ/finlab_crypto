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

def simulate(W,op,initial=70.0,fee_bps=5.0,slip_bps=0.0,funding_bps_8h=0.0,min_notional=0.0,leverage=1.0):
 idx=W.index; qty=pd.Series(0.0,index=ALTS); eq=initial; peak=initial; min_eq=initial; maxdd=0.; fees=slips=funds=turn=0.; trades=skipped=0; rows=[]
 for t in idx:
  p1=op.loc[t+pd.Timedelta(hours=1),ALTS] if t+pd.Timedelta(hours=1) in op.index else pd.Series(np.nan,index=ALTS)
  p2=op.loc[t+pd.Timedelta(hours=2),ALTS] if t+pd.Timedelta(hours=2) in op.index else pd.Series(np.nan,index=ALTS)
  tradable=np.isfinite(p1)&np.isfinite(p2)
  if tradable.sum()<6: rows.append((t,eq,0,0,maxdd)); continue
  target_notional=W.loc[t].fillna(0)*eq*leverage; cur_notional=pd.Series(0.0,index=ALTS); cur_notional.loc[tradable]=qty.loc[tradable]*p1.loc[tradable]; delta=target_notional-cur_notional
  for s in np.array(ALTS)[tradable.values]:
   dn=float(delta[s])
   if abs(dn)<1e-12: continue
   if abs(dn)<min_notional: skipped+=1; continue
   dq=dn/p1[s]; qty[s]+=dq; traded=abs(dn); turn+=traded; trades+=1; fees+=traded*fee_bps/1e4; slips+=traded*slip_bps/1e4; eq-=traded*(fee_bps+slip_bps)/1e4
  gross=float((qty.loc[tradable].abs()*p1.loc[tradable]).sum())
  fund=gross*funding_bps_8h/1e4 if t.hour%8==0 else 0.; funds+=fund; eq-=fund
  pnl=float((qty.loc[tradable]*(p2.loc[tradable]-p1.loc[tradable])).sum()); eq+=pnl
  peak=max(peak,eq); min_eq=min(min_eq,eq); dd=eq/peak-1 if peak>0 else -1.; maxdd=min(maxdd,dd)
  rows.append((t,eq,pnl,gross,dd))
  if eq<=0: break
 d=pd.DataFrame(rows,columns=['time','equity','pnl','gross_notional','dd']).set_index('time'); n=len(d); yrs=n/HRY; r=d.equity.pct_change().fillna(0); sd=r.std(ddof=1); sh=r.mean()/sd*math.sqrt(HRY) if sd>0 else np.nan
 return {'final_equity':float(eq),'return_pct':float((eq/initial-1)*100),'CAGR':float((eq/initial)**(1/yrs)-1) if eq>0 and yrs>0 else -1,'Sharpe':float(sh),'MaxDD':float(maxdd),'min_equity':float(min_eq),'fees_usdt':float(fees),'slippage_usdt':float(slips),'funding_usdt':float(funds),'traded_notional_usdt':float(turn),'trade_count':int(trades),'skipped_small_deltas':int(skipped),'avg_gross_notional':float(d.gross_notional.mean()) if n else 0.0,'max_gross_notional':float(d.gross_notional.max()) if n else 0.0}

def main():
 raw={}
 with ThreadPoolExecutor(max_workers=10) as ex:
  fs={ex.submit(load,s):s for s in SYMS}
  for f in as_completed(fs): raw[fs[f]]=f.result(); print('DATA',fs[f],len(raw[fs[f]]),flush=True)
 idx=pd.date_range(START,END-pd.Timedelta(hours=1),freq='h'); cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in SYMS}); op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in SYMS}); sig=signal_for(cl); W=staggered(sig)
 sl=W.index>=pd.Timestamp('2022-07-01',tz='UTC'); W=W.loc[sl]; op=op.reindex(pd.date_range(W.index.min(),END+pd.Timedelta(hours=2),freq='h'))
 gross=W.abs().sum(axis=1); net=W.sum(axis=1)
 out={'audit':{'mean_target_gross':float(gross.mean()),'max_target_gross':float(gross.max()),'mean_abs_net':float(net.abs().mean()),'max_abs_net':float(net.abs().max()),'cohort_count':48,'aggregation':'sum of 48 cohort weights divided by 48; only aggregate net delta is executed'},'scenarios':{}}
 scenarios=[('ideal_fee5',5,0,0,0,1),('min5_fee5',5,0,0,5,1),('min5_fee5_slip2',5,2,0,5,1),('min5_fee5_slip2_fund1',5,2,1,5,1),('min5_fee5_slip5_fund1',5,5,1,5,1),('min10_fee5_slip2_fund1',5,2,1,10,1)]
 for name,fee,slip,fund,mn,lev in scenarios:
  print('RUN',name,flush=True); out['scenarios'][name]=simulate(W,op,70,fee,slip,fund,mn,lev)
 Path('alpha_v8_account70_output').mkdir(exist_ok=True); json.dump(out,open('alpha_v8_account70_output/summary.json','w'),indent=2,allow_nan=True); print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__': main()
