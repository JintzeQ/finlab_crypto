#!/usr/bin/env python3
import io,json,math,zipfile,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','AVAXUSDT','SUIUSDT']; ALTS=SYMS[1:]
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
ROOT='https://data.binance.vision/data/futures/um/monthly/klines'; HRY=365.25*24
START=pd.Timestamp('2022-01-01',tz='UTC'); END=pd.Timestamp('2026-08-01',tz='UTC')
PERIODS={'2022H2':('2022-07-01','2023-01-01'),'2023H1':('2023-01-01','2023-07-01'),'2023H2':('2023-07-01','2024-01-01'),'2024H1':('2024-01-01','2024-07-01'),'2024H2':('2024-07-01','2025-01-01'),'2025H1':('2025-01-01','2025-07-01'),'2025H2':('2025-07-01','2026-01-01'),'2026':('2026-01-01','2026-08-01')}
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
def cs_z(df):return df.sub(df.mean(axis=1),axis=0).div(df.std(axis=1,ddof=1).replace(0,np.nan),axis=0).clip(-4,4)
def signal_for(cl,bw=168,h=336,skip=24):
 r=cl.pct_change();btc=r.BTCUSDT;var=btc.rolling(bw,min_periods=bw//2).var().shift(1);e=pd.DataFrame(index=cl.index,columns=ALTS,dtype=float)
 for s in ALTS:
  b=(r[s].rolling(bw,min_periods=bw//2).cov(btc).shift(1)/var).clip(-1,3);e[s]=r[s]-b*btc
 mom=e.shift(skip).rolling(h,min_periods=max(48,h//2)).sum();vol=e.shift(skip).rolling(h,min_periods=max(48,h//2)).std()*np.sqrt(h);return cs_z(mom/vol.replace(0,np.nan))
def cohort_weights(sig,reb=48,phase=0,cols=None):
 cols=cols or list(sig.columns);A=sig[cols].to_numpy(float);out=np.zeros_like(A);prev=np.zeros(A.shape[1])
 for i,row in enumerate(A):
  if (i-phase)%reb:out[i]=prev;continue
  ok=np.isfinite(row);w=np.zeros(A.shape[1])
  if ok.sum()>=max(6,len(cols)-2):
   ids=np.where(ok)[0];o=ids[np.argsort(row[ids])];w[o[-2:]]=.25;w[o[:2]]=-.25
  prev=w;out[i]=w
 return pd.DataFrame(out,index=sig.index,columns=cols)
def staggered_weights(sig,reb=48,cols=None):
 cols=cols or list(sig.columns);acc=None
 for p in range(reb):
  w=cohort_weights(sig,reb,p,cols);acc=w if acc is None else acc+w
 return acc/reb
def perf(r,t,mkt=None):
 r=r.fillna(0);eq=(1+r).cumprod();sd=r.std(ddof=1);sh=r.mean()/sd*math.sqrt(HRY) if sd>0 else np.nan;yrs=len(r)/HRY;cagr=float(eq.iloc[-1]**(1/yrs)-1) if len(r) and eq.iloc[-1]>0 else -1.;mdd=float((eq/eq.cummax()-1).min());pos=r[r>0].sum();neg=-r[r<0].sum();o={'Sharpe':float(sh),'CAGR':cagr,'Return':float(eq.iloc[-1]-1),'MaxDD':mdd,'Calmar':float(cagr/-mdd) if mdd<0 else np.nan,'PF':float(pos/neg) if neg>0 else np.nan,'AvgRet_bps':float(r.mean()*1e4),'Turnover':float(t.mean()),'AnnualTurnover':float(t.mean()*HRY)}
 if mkt is not None:
  m=mkt.reindex(r.index).fillna(0);v=m.var();o['BTC_beta']=float(r.cov(m)/v) if v>0 else np.nan;o['BTC_corr']=float(r.corr(m))
 return o
def path(W,op,cols,cost=5,funding_bps_8h=0):
 R=op[cols].shift(-2)/op[cols].shift(-1)-1;g=(W*R).sum(axis=1);t=W.diff().abs().sum(axis=1);t.iloc[0]=W.iloc[0].abs().sum();gross_exposure=W.abs().sum(axis=1)
 # Conservative synthetic funding stress: charge absolute gross exposure every 8 hours, regardless of funding sign.
 fund=pd.Series(0.0,index=W.index);mask=(W.index.hour%8)==0;fund.loc[mask]=gross_exposure.loc[mask]*funding_bps_8h/1e4
 net=g-t*cost/1e4-fund;mkt=op.BTCUSDT.shift(-2)/op.BTCUSDT.shift(-1)-1
 return g,t,net,mkt,gross_exposure
def main():
 raw={}
 with ThreadPoolExecutor(max_workers=10) as ex:
  fs={ex.submit(load,s):s for s in SYMS}
  for f in as_completed(fs):s=fs[f];raw[s]=f.result();print('DATA',s,len(raw[s]),flush=True)
 idx=pd.date_range(START,END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in SYMS})
 sig=signal_for(cl);W=staggered_weights(sig,48,ALTS);g,t,net,mkt,expo=path(W,op,ALTS,5,0)
 out={'fixed':'v6_signal_b168_h336_s24_risk_staggered48','periods':{},'cost_stress':{},'funding_stress':{},'phase_diagnostic':{},'horizon_neighbors':{},'leave_one_out':{},'monthly':{},'concentration':{}}
 for n,(a,z) in PERIODS.items():
  sl=slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1));out['periods'][n]=perf(net.loc[sl],t.loc[sl],mkt.loc[sl])
 allsl=slice(pd.Timestamp('2022-07-01',tz='UTC'),END-pd.Timedelta(hours=1))
 for c in [0,2,3,5,8,12,16,20]:
  _,tt,nn,mm,_=path(W,op,ALTS,c,0);out['cost_stress'][str(c)]=perf(nn.loc[allsl],tt.loc[allsl],mm.loc[allsl])
 gross_bps=float(g.loc[allsl].mean()*1e4);turn=float(t.loc[allsl].mean());out['break_even_cost_bps']=gross_bps/turn if turn>0 else np.nan
 # Funding stress locked ex ante: 0.5/1/2/3 bps per 8h charged on gross exposure.
 for fb in [0.5,1.0,2.0,3.0]:
  _,tt,nn,mm,_=path(W,op,ALTS,5,fb);out['funding_stress'][str(fb)]={}
  for n,(a,z) in PERIODS.items():
   sl=slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1));out['funding_stress'][str(fb)][n]=perf(nn.loc[sl],tt.loc[sl],mm.loc[sl])
 # All 48 raw phases diagnostic; no phase selection.
 for p in range(48):
  wp=cohort_weights(sig,48,p,ALTS);_,tp,np_,mp,_=path(wp,op,ALTS,5,0);out['phase_diagnostic'][str(p)]={}
  for n in ['2025H2','2026']:
   a,z=PERIODS[n];sl=slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1));out['phase_diagnostic'][str(p)][n]=perf(np_.loc[sl],tp.loc[sl])['Sharpe']
 # Only economic neighbor horizons, no selection.
 for h in [240,336,504]:
  ss=signal_for(cl,168,h,24);ww=staggered_weights(ss,48,ALTS);_,tt,nn,mm,_=path(ww,op,ALTS,5,0);out['horizon_neighbors'][str(h)]={}
  for n,(a,z) in PERIODS.items():
   sl=slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1));out['horizon_neighbors'][str(h)][n]=perf(nn.loc[sl],tt.loc[sl])['Sharpe']
 for exs in ALTS:
  cols=[s for s in ALTS if s!=exs];ww=staggered_weights(sig,48,cols);_,tt,nn,mm,_=path(ww,op,cols,5,0);out['leave_one_out'][exs]={}
  for n in ['2025H2','2026']:
   a,z=PERIODS[n];sl=slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1));out['leave_one_out'][exs][n]=perf(nn.loc[sl],tt.loc[sl])['Sharpe']
 mon=net.loc[allsl].groupby(net.loc[allsl].index.to_period('M')).sum();out['monthly']={'positive_frac':float((mon>0).mean()),'median_bps':float(mon.median()*1e4),'worst_bps':float(mon.min()*1e4),'best_bps':float(mon.max()*1e4),'n':int(len(mon))}
 R=op[ALTS].shift(-2)/op[ALTS].shift(-1)-1;coin=(W*R).loc[allsl].sum();ta=float(coin.abs().sum());out['concentration']={'coin_pnl':{k:float(v) for k,v in coin.items()},'top_abs_share':float(coin.abs().max()/ta) if ta>0 else np.nan}
 per=out['periods'];pos=sum(per[n]['Sharpe']>0 for n in PERIODS);ph=list(out['phase_diagnostic'].values());phase_both=sum(v['2025H2']>0 and v['2026']>0 for v in ph)/48;hn=list(out['horizon_neighbors'].values());hboth=sum(v['2025H2']>0 and v['2026']>0 for v in hn);loo=list(out['leave_one_out'].values());lboth=sum(v['2025H2']>0 and v['2026']>0 for v in loo)
 out['robust_stats']={'positive_halfyears':pos,'phase_both_positive_frac':phase_both,'horizon_both_positive':hboth,'loo_both_positive':lboth}
 # Gate locked before this run.
 out['accepted']=bool(pos>=7 and per['2025H2']['Sharpe']>.5 and per['2026']['Sharpe']>.5 and abs(per['2025H2']['BTC_beta'])<.15 and abs(per['2026']['BTC_beta'])<.15 and out['cost_stress']['8']['Sharpe']>1.0 and out['break_even_cost_bps']>8 and out['monthly']['positive_frac']>=.60 and phase_both>=.60 and hboth>=2 and lboth>=7 and out['funding_stress']['1.0']['2025H2']['Sharpe']>0 and out['funding_stress']['1.0']['2026']['Sharpe']>0)
 Path('alpha_v8_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v8_output/summary.json','w'),indent=2,allow_nan=True);print('===V8===');print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
