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
def months():return [str(x) for x in pd.period_range('2022-01','2026-07',freq='M')]
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
def cs_z(df):return df.sub(df.mean(1),axis=0).div(df.std(1,ddof=1).replace(0,np.nan),axis=0).clip(-4,4)
def make_weights(sig,reb,cols):
 A=sig[cols].to_numpy(float);out=np.zeros_like(A);prev=np.zeros(A.shape[1])
 for i,row in enumerate(A):
  if i%reb:out[i]=prev;continue
  ok=np.isfinite(row);w=np.zeros(A.shape[1])
  if ok.sum()>=max(6,len(cols)-2):
   ids=np.where(ok)[0];o=ids[np.argsort(row[ids])];w[o[-2:]]=.25;w[o[:2]]=-.25
  prev=w;out[i]=w
 return pd.DataFrame(out,index=sig.index,columns=cols)
def perf(r,t,mkt=None):
 r=r.fillna(0);eq=(1+r).cumprod();sd=r.std(ddof=1);sh=r.mean()/sd*math.sqrt(HRY) if sd>0 else np.nan;yrs=len(r)/HRY;cagr=float(eq.iloc[-1]**(1/yrs)-1) if len(r) and eq.iloc[-1]>0 else -1.;mdd=float((eq/eq.cummax()-1).min());pos=r[r>0].sum();neg=-r[r<0].sum();o={'Sharpe':float(sh),'CAGR':cagr,'Return':float(eq.iloc[-1]-1),'MaxDD':mdd,'Calmar':float(cagr/-mdd) if mdd<0 else np.nan,'PF':float(pos/neg) if neg>0 else np.nan,'AvgRet_bps':float(r.mean()*1e4),'Turnover':float(t.mean()),'AnnualTurnover':float(t.mean()*HRY)}
 if mkt is not None:
  m=mkt.reindex(r.index).fillna(0);v=m.var();o['BTC_beta']=float(r.cov(m)/v) if v>0 else np.nan;o['BTC_corr']=float(r.corr(m))
 return o
def signal_for(cl,bw,h,skip):
 r=cl.pct_change();btc=r.BTCUSDT;var=btc.rolling(bw,min_periods=bw//2).var().shift(1);e=pd.DataFrame(index=cl.index,columns=ALTS,dtype=float)
 for s in ALTS:
  b=(r[s].rolling(bw,min_periods=bw//2).cov(btc).shift(1)/var).clip(-1,3);e[s]=r[s]-b*btc
 mom=e.shift(skip).rolling(h,min_periods=max(48,h//2)).sum();vol=e.shift(skip).rolling(h,min_periods=max(48,h//2)).std()*np.sqrt(h);return cs_z(mom/vol.replace(0,np.nan))
def ensemble_signal(cl, betaws=(168,336), horizons=(240,336,504), skip=24):
 parts=[signal_for(cl,bw,h,skip) for bw in betaws for h in horizons]
 return sum(parts)/len(parts)
def path_metrics(sig,op,reb,cols,cost=5):
 W=make_weights(sig,reb,cols);R=op[cols].shift(-2)/op[cols].shift(-1)-1;g=(W*R).sum(1);t=W.diff().abs().sum(1);t.iloc[0]=W.iloc[0].abs().sum();net=g-t*cost/1e4;mkt=op.BTCUSDT.shift(-2)/op.BTCUSDT.shift(-1)-1;return W,g,t,net,mkt
def ic_series(sig,op):
 f=op[ALTS].shift(-2)/op[ALTS].shift(-1)-1
 vals=[]
 for i in range(len(sig)):
  a=sig.iloc[i];b=f.iloc[i];ok=a.notna()&b.notna();vals.append(a[ok].corr(b[ok],method='spearman') if ok.sum()>=6 else np.nan)
 return pd.Series(vals,index=sig.index)
def main():
 raw={}
 with ThreadPoolExecutor(max_workers=10) as ex:
  fs={ex.submit(load,s):s for s in SYMS}
  for f in as_completed(fs):s=fs[f];raw[s]=f.result();print('DATA',s,len(raw[s]),flush=True)
 idx=pd.date_range(START,END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in SYMS})
 sig=ensemble_signal(cl);W,g,t,net,mkt=path_metrics(sig,op,48,ALTS,5)
 out={'fixed':'ensemble_b168_336_h240_336_504_s24_r48_none','periods':{},'cost_stress':{},'neighbors':{},'leave_one_out':{},'IC':{},'monthly':{},'concentration':{}}
 for n,(a,z) in PERIODS.items():
  sl=slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1));out['periods'][n]=perf(net.loc[sl],t.loc[sl],mkt.loc[sl])
 allsl=slice(pd.Timestamp('2022-07-01',tz='UTC'),END-pd.Timedelta(hours=1))
 for c in [0,2,3,5,8,12,16,20]:out['cost_stress'][str(c)]=perf((g-t*c/1e4).loc[allsl],t.loc[allsl],mkt.loc[allsl])
 gross_bps=float(g.loc[allsl].mean()*1e4);turn=float(t.loc[allsl].mean());out['break_even_cost_bps']=gross_bps/turn if turn>0 else np.nan
 variants={'core':((168,336),(240,336,504),24),'b168':((168,),(240,336,504),24),'b336':((336,),(240,336,504),24),'broad':((168,336),(168,240,336,504),24),'noskip':((168,336),(240,336,504),0),'shorter':((168,336),(168,240,336),24),'longer':((168,336),(336,504),24)}
 for vn,(bws,hs,sk) in variants.items():
  ss=ensemble_signal(cl,bws,hs,sk)
  for reb in [24,48,72,96]:
   _,gg,tt,nn,mm=path_metrics(ss,op,reb,ALTS,5);key=f'{vn}_r{reb}';out['neighbors'][key]={}
   for n,(a,z) in PERIODS.items():
    sl=slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1));out['neighbors'][key][n]=perf(nn.loc[sl],tt.loc[sl])['Sharpe']
 for exs in ALTS:
  cols=[s for s in ALTS if s!=exs];_,gg,tt,nn,mm=path_metrics(sig,op,48,cols,5);out['leave_one_out'][exs]={}
  for n,(a,z) in PERIODS.items():
   sl=slice(pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1));out['leave_one_out'][exs][n]=perf(nn.loc[sl],tt.loc[sl])['Sharpe']
 ics=ic_series(sig,op)
 for n,(a,z) in PERIODS.items():
  x=ics.loc[pd.Timestamp(a,tz='UTC'):pd.Timestamp(z,tz='UTC')-pd.Timedelta(hours=1)].dropna();out['IC'][n]={'mean':float(x.mean()),'std':float(x.std(ddof=1)),'ICIR_hourly':float(x.mean()/x.std(ddof=1)*np.sqrt(HRY)) if len(x)>2 and x.std(ddof=1)>0 else np.nan,'n':int(len(x))}
 mon=net.loc[allsl].groupby(net.loc[allsl].index.to_period('M')).sum();out['monthly']={'positive_frac':float((mon>0).mean()),'median_bps':float(mon.median()*1e4),'worst_bps':float(mon.min()*1e4),'best_bps':float(mon.max()*1e4),'n':int(len(mon))}
 R=op[ALTS].shift(-2)/op[ALTS].shift(-1)-1;coin=(W*R).loc[allsl].sum();total_abs=float(coin.abs().sum());out['concentration']={'coin_pnl':{k:float(v) for k,v in coin.items()},'top_abs_share':float(coin.abs().max()/total_abs) if total_abs>0 else np.nan}
 per=out['periods']; pos=sum(per[n]['Sharpe']>0 for n in PERIODS); neigh=list(out['neighbors'].values()); ng=sum(v['2025H2']>0 and v['2026']>0 for v in neigh)/len(neigh); loo=list(out['leave_one_out'].values()); lg=sum(v['2025H2']>0 and v['2026']>0 for v in loo)
 out['robust_stats']={'positive_halfyears':pos,'neighbor_both_positive_frac':ng,'loo_both_positive':lg}
 out['robust_accepted']=bool(pos>=6 and per['2025H2']['Sharpe']>.5 and per['2026']['Sharpe']>.5 and abs(per['2025H2']['BTC_beta'])<.15 and abs(per['2026']['BTC_beta'])<.15 and out['break_even_cost_bps']>8 and out['monthly']['positive_frac']>=.60 and ng>=.60 and lg>=7)
 Path('alpha_v7_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v7_output/summary.json','w'),indent=2,allow_nan=True);print('===V7===');print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
