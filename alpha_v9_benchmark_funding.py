#!/usr/bin/env python3
import io,json,math,zipfile,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','AVAXUSDT','SUIUSDT']; ALTS=SYMS[1:]
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
ROOT='https://data.binance.vision/data/futures/um/monthly'; HRY=365.25*24
START=pd.Timestamp('2022-01-01',tz='UTC'); TRADE_START=pd.Timestamp('2022-07-01',tz='UTC'); END=pd.Timestamp('2026-08-01',tz='UTC')
INITIAL=70.0

def months(): return [str(x) for x in pd.period_range('2022-01','2026-07',freq='M')]
def getzip(url):
 try:
  with urllib.request.urlopen(url,timeout=60) as r:return r.read()
 except urllib.error.HTTPError as e:
  if e.code==404:return None
  raise

def load_kline(s):
 fs=[]
 for ym in months():
  b=getzip(f'{ROOT}/klines/{s}/1h/{s}-1h-{ym}.zip')
  if b is None: continue
  with zipfile.ZipFile(io.BytesIO(b)) as z: fs.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None,names=COLS))
 d=pd.concat(fs,ignore_index=True);t=pd.to_numeric(d.open_time,errors='coerce');t=t.where(t<1e15,t/1000);d['date']=pd.to_datetime(t,unit='ms',utc=True,errors='coerce')
 for c in ['open','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['date','open','close']).drop_duplicates('date').sort_values('date').set_index('date')

def load_funding(s):
 fs=[]
 for ym in months():
  b=getzip(f'{ROOT}/fundingRate/{s}/{s}-fundingRate-{ym}.zip')
  if b is None: continue
  with zipfile.ZipFile(io.BytesIO(b)) as z:
   x=pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None)
   # Binance Vision USD-M fundingRate: calc_time, funding_rate, symbol
   if x.shape[1] < 2: continue
   x=x.iloc[:,:3].copy(); x.columns=['calc_time','funding_rate','symbol'][:x.shape[1]]
   fs.append(x)
 if not fs:return pd.Series(dtype=float)
 d=pd.concat(fs,ignore_index=True)
 t=pd.to_numeric(d['calc_time'],errors='coerce');t=t.where(t<1e15,t/1000)
 idx=pd.to_datetime(t,unit='ms',utc=True,errors='coerce')
 r=pd.to_numeric(d['funding_rate'],errors='coerce')
 out=pd.Series(r.values,index=idx).dropna().groupby(level=0).last().sort_index()
 return out

def cs_z(df):return df.sub(df.mean(axis=1),axis=0).div(df.std(axis=1,ddof=1).replace(0,np.nan),axis=0).clip(-4,4)
def signal_for(cl,bw=168,h=336,skip=24):
 r=cl.pct_change();btc=r.BTCUSDT;var=btc.rolling(bw,min_periods=bw//2).var().shift(1);e=pd.DataFrame(index=cl.index,columns=ALTS,dtype=float)
 for s in ALTS:
  b=(r[s].rolling(bw,min_periods=bw//2).cov(btc).shift(1)/var).clip(-1,3);e[s]=r[s]-b*btc
 mom=e.shift(skip).rolling(h,min_periods=max(48,h//2)).sum();vol=e.shift(skip).rolling(h,min_periods=max(48,h//2)).std()*np.sqrt(h)
 return cs_z(mom/vol.replace(0,np.nan))
def cohort_weights(sig,reb=48,phase=0):
 A=sig[ALTS].to_numpy(float);out=np.zeros_like(A);prev=np.zeros(A.shape[1])
 for i,row in enumerate(A):
  if (i-phase)%reb:out[i]=prev;continue
  ok=np.isfinite(row);w=np.zeros(A.shape[1])
  if ok.sum()>=7:
   ids=np.where(ok)[0];o=ids[np.argsort(row[ids])];w[o[-2:]]=.25;w[o[:2]]=-.25
  prev=w;out[i]=w
 return pd.DataFrame(out,index=sig.index,columns=ALTS)
def staggered(sig):
 acc=None
 for p in range(48):
  w=cohort_weights(sig,48,p);acc=w if acc is None else acc+w
 return acc/48

def metrics(eq,btc_ret):
 r=eq.pct_change().fillna(0); yrs=(eq.index[-1]-eq.index[0]).total_seconds()/(365.25*86400); cagr=(eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1
 sd=r.std(ddof=1); sh=r.mean()/sd*np.sqrt(HRY) if sd>0 else np.nan
 downside=r[r<0].std(ddof=1); sortino=r.mean()/downside*np.sqrt(HRY) if downside>0 else np.nan
 dd=eq/eq.cummax()-1;mdd=dd.min();annvol=sd*np.sqrt(HRY); b=btc_ret.reindex(r.index).fillna(0); bv=b.var();beta=r.cov(b)/bv if bv>0 else np.nan;corr=r.corr(b)
 # hourly regression intercept annualized linearly; descriptive only
 alpha=(r.mean()-beta*b.mean())*HRY if np.isfinite(beta) else np.nan
 return {'final_equity':float(eq.iloc[-1]),'Return':float(eq.iloc[-1]/eq.iloc[0]-1),'CAGR':float(cagr),'Sharpe':float(sh),'Sortino':float(sortino),'AnnVol':float(annvol),'MaxDD':float(mdd),'Calmar':float(cagr/-mdd) if mdd<0 else np.nan,'BTC_beta':float(beta),'BTC_corr':float(corr),'ann_regression_alpha':float(alpha)}

def simulate(W,op,fund,fee_bps=5,slip_bps=0,min_delta=0,use_funding=True):
 idx=op.index; cols=ALTS; P=op[cols].to_numpy(float); T=W[cols].shift(1).fillna(0).to_numpy(float) # close(t) -> open(t+1)
 F=fund.reindex(idx).fillna(0).to_numpy(float)
 n=len(idx);q=np.zeros(len(cols));eq=np.full(n,np.nan);equity=INITIAL;fees=slips=funds=turnover=0.;trades=skips=0
 prevp=np.full(len(cols),np.nan)
 for k in range(n):
  p=P[k]
  valid=np.isfinite(p)
  if k>0:
   both=valid & np.isfinite(prevp); equity += float(np.nansum(q[both]*(p[both]-prevp[both])))
  # actual funding event at this timestamp; positive funding => longs pay, shorts receive
  if use_funding:
   fr=F[k]; fm=valid & np.isfinite(fr) & (fr!=0)
   if fm.any():
    pay=float(np.nansum(q[fm]*p[fm]*fr[fm])); equity -= pay; funds += pay
  if idx[k] >= TRADE_START and idx[k] < END:
   tw=T[k].copy(); tw[~valid]=0
   desired=np.zeros(len(cols));desired[valid]=tw[valid]*equity/p[valid]
   delta=desired-q; dn=np.abs(delta)*np.where(valid,p,0); execute=valid & (dn>=min_delta-1e-12)
   skips += int((valid & (dn>1e-10) & ~execute).sum())
   traded=float(dn[execute].sum()); cost=traded*(fee_bps+slip_bps)/1e4
   fees += traded*fee_bps/1e4; slips += traded*slip_bps/1e4; turnover += traded; trades += int(execute.sum()); equity -= cost; q[execute]=desired[execute]
  eq[k]=equity;prevp=p.copy()
 ser=pd.Series(eq,index=idx)
 ser=ser.loc[(ser.index>=TRADE_START)&(ser.index<END)]
 return ser,{'fees_usdt':fees,'slippage_usdt':slips,'net_funding_paid_usdt':funds,'traded_notional_usdt':turnover,'trade_count':trades,'skipped_small_deltas':skips}

def yearly(eq):
 out={}
 for y,g in eq.groupby(eq.index.year):
  if len(g)>1: out[str(y)]=float(g.iloc[-1]/g.iloc[0]-1)
 return out

def main():
 raw={};fund={}
 with ThreadPoolExecutor(max_workers=10) as ex:
  fs={ex.submit(load_kline,s):s for s in SYMS}
  for f in as_completed(fs):raw[fs[f]]=f.result();print('KLINE',fs[f],len(raw[fs[f]]),flush=True)
 with ThreadPoolExecutor(max_workers=9) as ex:
  fs={ex.submit(load_funding,s):s for s in ALTS}
  for f in as_completed(fs):fund[fs[f]]=f.result();print('FUND',fs[f],len(fund[fs[f]]),flush=True)
 idx=pd.date_range(START,END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in SYMS});F=pd.DataFrame({s:fund[s].reindex(idx) for s in ALTS})
 sig=signal_for(cl);W=staggered(sig)
 btc=op.BTCUSDT.loc[(op.index>=TRADE_START)&(op.index<END)].dropna();btc_eq=INITIAL*btc/btc.iloc[0];btc_ret=btc.pct_change().fillna(0)
 cases={}
 for name,kw in {
  'strategy_fee5_no_funding':dict(fee_bps=5,slip_bps=0,min_delta=5,use_funding=False),
  'strategy_fee5_slip2_no_funding':dict(fee_bps=5,slip_bps=2,min_delta=5,use_funding=False),
  'strategy_fee5_actual_funding':dict(fee_bps=5,slip_bps=0,min_delta=5,use_funding=True),
  'strategy_fee5_slip2_actual_funding':dict(fee_bps=5,slip_bps=2,min_delta=5,use_funding=True),
 }.items():
  eq,cost=simulate(W,op,F,**kw);m=metrics(eq,btc_ret);m.update(cost);m['yearly']=yearly(eq);cases[name]=m
 bm=metrics(btc_eq,btc_ret);bm['yearly']=yearly(btc_eq);bm['start_price']=float(btc.iloc[0]);bm['end_price']=float(btc.iloc[-1]);
 # Fair execution sensitivity: one 5bp entry fee only, no exit liquidation assumed.
 btc_fee_eq=btc_eq*(1-5/1e4);bmfee=metrics(btc_fee_eq,btc_ret);bmfee['yearly']=yearly(btc_fee_eq)
 funding_stats={}
 for s in ALTS:
  x=fund[s].loc[(fund[s].index>=TRADE_START)&(fund[s].index<END)]
  funding_stats[s]={'events':int(len(x)),'mean_bps':float(x.mean()*1e4) if len(x) else np.nan,'median_bps':float(x.median()*1e4) if len(x) else np.nan,'positive_frac':float((x>0).mean()) if len(x) else np.nan,'sum_bps':float(x.sum()*1e4) if len(x) else np.nan}
 out={'period':[str(TRADE_START),str(END)],'initial_usdt':INITIAL,'strategy':'v6 residual momentum b168 h336 skip24 risk-adjusted; 48 staggered cohorts aggregated to one net target','execution':'signal at close(t), target executed open(t+1); futures quantity carried between hours; fee/slippage charged on actual delta notional; min delta 5 USDT','funding':'Binance Vision monthly fundingRate; positive rate: long pays / short receives; funding applied to signed notional at event timestamp using 1h open as mark-price proxy','cases':cases,'btc_buy_hold':bm,'btc_buy_hold_entry_fee5':bmfee,'funding_stats':funding_stats}
 Path('alpha_v9_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v9_output/summary.json','w'),indent=2,allow_nan=True);pd.DataFrame({'btc':btc_eq}).to_csv('alpha_v9_output/btc_equity.csv');print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
