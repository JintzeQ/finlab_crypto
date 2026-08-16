#!/usr/bin/env python3
import io,json,math,zipfile,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','AVAXUSDT','SUIUSDT'];ALTS=SYMS[1:]
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
ROOT='https://data.binance.vision/data/futures/um/monthly';HRY=365.25*24
START=pd.Timestamp('2022-01-01',tz='UTC');TRADE_START=pd.Timestamp('2022-07-01',tz='UTC');END=pd.Timestamp('2026-08-01',tz='UTC');INITIAL=70.
DEV_END=pd.Timestamp('2024-07-01',tz='UTC');HOLD1_END=pd.Timestamp('2025-07-01',tz='UTC');CONF_END=pd.Timestamp('2026-01-01',tz='UTC')
def months():return [str(x) for x in pd.period_range('2022-01','2026-07',freq='M')]
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
  if b is None:continue
  with zipfile.ZipFile(io.BytesIO(b)) as z:fs.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None,names=COLS))
 d=pd.concat(fs,ignore_index=True);t=pd.to_numeric(d.open_time,errors='coerce');t=t.where(t<1e15,t/1000);d['date']=pd.to_datetime(t,unit='ms',utc=True,errors='coerce')
 for c in ['open','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['date','open','close']).drop_duplicates('date').sort_values('date').set_index('date')
def load_funding(s):
 fs=[]
 for ym in months():
  b=getzip(f'{ROOT}/fundingRate/{s}/{s}-fundingRate-{ym}.zip')
  if b is None:continue
  with zipfile.ZipFile(io.BytesIO(b)) as z:
   x=pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None)
   if x.shape[1]<2:continue
   x=x.iloc[:,:3].copy();x.columns=['calc_time','funding_rate','symbol'][:x.shape[1]];fs.append(x)
 if not fs:return pd.Series(dtype=float)
 d=pd.concat(fs,ignore_index=True);t=pd.to_numeric(d.calc_time,errors='coerce');t=t.where(t<1e15,t/1000);idx=pd.to_datetime(t,unit='ms',utc=True,errors='coerce');r=pd.to_numeric(d.funding_rate,errors='coerce')
 return pd.Series(r.values,index=idx).dropna().groupby(level=0).last().sort_index()
def cs_z(x):return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1,ddof=1).replace(0,np.nan),axis=0).clip(-4,4)
def residuals(cl,bw=168):
 r=cl.pct_change();btc=r.BTCUSDT;var=btc.rolling(bw,min_periods=bw//2).var().shift(1);e=pd.DataFrame(index=cl.index,columns=ALTS,dtype=float)
 for s in ALTS:
  b=(r[s].rolling(bw,min_periods=bw//2).cov(btc).shift(1)/var).clip(-1,3);e[s]=r[s]-b*btc
 return e
def rms_signal(cl):
 e=residuals(cl);mom=e.shift(24).rolling(336,min_periods=168).sum();vol=e.shift(24).rolling(336,min_periods=168).std()*np.sqrt(336);return cs_z(mom/vol.replace(0,np.nan))
def weights_from_signal(sig,reb,phases=None):
 phases=reb if phases is None else phases;A=sig[ALTS].to_numpy(float);acc=np.zeros_like(A)
 for phase in range(phases):
  out=np.zeros_like(A);prev=np.zeros(A.shape[1])
  for i,row in enumerate(A):
   if (i-phase)%reb:out[i]=prev;continue
   ok=np.isfinite(row);w=np.zeros(A.shape[1])
   if ok.sum()>=7:
    ids=np.where(ok)[0];o=ids[np.argsort(row[ids])];w[o[-2:]]=.25;w[o[:2]]=-.25
   prev=w;out[i]=w
  acc+=out
 return pd.DataFrame(acc/phases,index=sig.index,columns=ALTS)
def build_candidates(cl,F):
 e=residuals(cl);out={}
 # Funding carry: long persistently low/negative funding; short persistently high funding.
 # Sparse event series is zero between actual funding timestamps; rolling SUM preserves cumulative carry pressure.
 for days in [3,7,14]:
  fs=F.fillna(0).rolling(days*24,min_periods=max(24,days*12)).sum();sig=cs_z(-fs)
  for reb in [8,24]:out[f'carry_d{days}_r{reb}']=weights_from_signal(sig,reb)
 # Short idiosyncratic reversal, economically opposite to RMS medium-term momentum.
 for h in [6,12,24]:
  rr=e.shift(1).rolling(h,min_periods=max(3,h//2)).sum();vv=e.shift(1).rolling(max(24,h*2),min_periods=12).std()*np.sqrt(h);sig=cs_z(-rr/vv.replace(0,np.nan))
  for reb in [6,12]:out[f'rev_h{h}_r{reb}']=weights_from_signal(sig,reb)
 # Hybrid: carry filters squeeze-prone reversal; no tuned coefficient, equal z blend.
 f7=cs_z(-F.fillna(0).rolling(7*24,min_periods=84).sum())
 for h in [12,24]:
  rr=e.shift(1).rolling(h,min_periods=h//2).sum();vv=e.shift(1).rolling(max(24,h*2),min_periods=12).std()*np.sqrt(h);rz=cs_z(-rr/vv.replace(0,np.nan));sig=cs_z((f7+rz)/2)
  for reb in [12,24]:out[f'hybrid_f7_rev{h}_r{reb}']=weights_from_signal(sig,reb)
 return out
def simulate(W,op,F,fee=5,slip=2,min_delta=5):
 idx=op.index;P=op[ALTS].to_numpy(float);T=W[ALTS].shift(1).fillna(0).to_numpy(float);FR=F.reindex(idx).fillna(0).to_numpy(float);q=np.zeros(len(ALTS));eq=np.full(len(idx),np.nan);equity=INITIAL;prev=np.full(len(ALTS),np.nan);fees=slips=funds=0.
 for k in range(len(idx)):
  p=P[k];valid=np.isfinite(p)
  if k>0:
   both=valid&np.isfinite(prev);equity+=float(np.nansum(q[both]*(p[both]-prev[both])))
  fr=FR[k];fm=valid&np.isfinite(fr)&(fr!=0)
  if fm.any():
   pay=float(np.nansum(q[fm]*p[fm]*fr[fm]));equity-=pay;funds+=pay
  if TRADE_START<=idx[k]<END and equity>0:
   tw=T[k].copy();tw[~valid]=0;desired=np.zeros(len(ALTS));desired[valid]=tw[valid]*equity/p[valid];delta=desired-q;dn=np.abs(delta)*np.where(valid,p,0);ex=valid&(dn>=min_delta-1e-12);tr=float(dn[ex].sum());equity-=tr*(fee+slip)/1e4;fees+=tr*fee/1e4;slips+=tr*slip/1e4;q[ex]=desired[ex]
  eq[k]=equity;prev=p.copy()
 return pd.Series(eq,index=idx).loc[(idx>=TRADE_START)&(idx<END)],{'fees':fees,'slippage':slips,'funding_paid':funds}
def metrics(eq,btc_ret=None):
 r=eq.pct_change().fillna(0);yrs=(eq.index[-1]-eq.index[0]).total_seconds()/(365.25*86400);cagr=(eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1 if eq.iloc[-1]>0 else -1;sd=r.std(ddof=1);sh=r.mean()/sd*np.sqrt(HRY) if sd>0 else np.nan;dd=eq/eq.cummax()-1;mdd=dd.min();o={'final':float(eq.iloc[-1]),'CAGR':float(cagr),'Sharpe':float(sh),'MaxDD':float(mdd),'Calmar':float(cagr/-mdd) if mdd<0 else np.nan,'AnnVol':float(sd*np.sqrt(HRY))}
 if btc_ret is not None:
  b=btc_ret.reindex(r.index).fillna(0);o['BTC_corr']=float(r.corr(b));o['BTC_beta']=float(r.cov(b)/b.var()) if b.var()>0 else np.nan
 return o
def submetrics(eq,a,z):return metrics(eq.loc[(eq.index>=a)&(eq.index<z)])
def corr_eq(a,b,start,end):
 ra=a.pct_change();rb=b.pct_change();x=pd.concat([ra,rb],axis=1).loc[(ra.index>=start)&(ra.index<end)].dropna();return float(x.iloc[:,0].corr(x.iloc[:,1])) if len(x)>10 else np.nan
def main():
 raw={};fund={}
 with ThreadPoolExecutor(max_workers=10) as ex:
  fs={ex.submit(load_kline,s):s for s in SYMS}
  for f in as_completed(fs):raw[fs[f]]=f.result();print('K',fs[f],len(raw[fs[f]]),flush=True)
 with ThreadPoolExecutor(max_workers=9) as ex:
  fs={ex.submit(load_funding,s):s for s in ALTS}
  for f in as_completed(fs):fund[fs[f]]=f.result();print('F',fs[f],len(fund[fs[f]]),flush=True)
 idx=pd.date_range(START,END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in SYMS});F=pd.DataFrame({s:fund[s].reindex(idx) for s in ALTS})
 rmsW=weights_from_signal(rms_signal(cl),48);rmsEq,rmsCost=simulate(rmsW,op,F);btc=op.BTCUSDT.loc[(op.index>=TRADE_START)&(op.index<END)].pct_change().fillna(0)
 cands=build_candidates(cl,F);res={};equities={}
 for n,W in cands.items():
  eq,cost=simulate(W,op,F);equities[n]=eq;res[n]={'full':metrics(eq,btc),'dev':submetrics(eq,TRADE_START,DEV_END),'holdout1':submetrics(eq,DEV_END,HOLD1_END),'confirm':submetrics(eq,HOLD1_END,CONF_END),'secondary2026':submetrics(eq,CONF_END,END),'corr_rms_full':corr_eq(eq,rmsEq,TRADE_START,END),'corr_rms_dev':corr_eq(eq,rmsEq,TRADE_START,DEV_END),'cost':cost}
 # Locked selection: on DEV only, require positive Sharpe and |corr RMS|<=0.35. Rank by Sharpe*(1-|corr|), tie lower MDD.
 eligible=[n for n,v in res.items() if v['dev']['Sharpe']>0 and abs(v['corr_rms_dev'])<=.35]
 eligible.sort(key=lambda n:(res[n]['dev']['Sharpe']*(1-abs(res[n]['corr_rms_dev'])),-abs(res[n]['dev']['MaxDD'])),reverse=True)
 winner=eligible[0] if eligible else max(res,key=lambda n:res[n]['dev']['Sharpe']*(1-abs(res[n]['corr_rms_dev'])))
 wW=cands[winner]
 # Allocation chosen DEV only. Actual account target is aggregated before simulation, so overlapping trades net.
 alloc={};rms_dev=submetrics(rmsEq,TRADE_START,DEV_END)
 for wrms in [1.0,.9,.8,.7,.6,.5]:
  W=wrms*rmsW+(1-wrms)*wW;eq,cost=simulate(W,op,F);k=f'{wrms:.1f}/{1-wrms:.1f}';alloc[k]={'dev':submetrics(eq,TRADE_START,DEV_END),'holdout1':submetrics(eq,DEV_END,HOLD1_END),'confirm':submetrics(eq,HOLD1_END,CONF_END),'secondary2026':submetrics(eq,CONF_END,END),'full':metrics(eq,btc),'corr_components_full':corr_eq(rmsEq,equities[winner],TRADE_START,END),'cost':cost}
 # Choose lowest DEV MDD subject to >=95% RMS DEV CAGR; tie higher CAGR.
 ok=[k for k,v in alloc.items() if v['dev']['CAGR']>=.95*rms_dev['CAGR']]
 chosen=min(ok,key=lambda k:(abs(alloc[k]['dev']['MaxDD']),-alloc[k]['dev']['CAGR'])) if ok else max(alloc,key=lambda k:alloc[k]['dev']['Calmar'])
 out={'baseline_rms48':{'full':metrics(rmsEq,btc),'dev':rms_dev,'holdout1':submetrics(rmsEq,DEV_END,HOLD1_END),'confirm':submetrics(rmsEq,HOLD1_END,CONF_END),'secondary2026':submetrics(rmsEq,CONF_END,END),'cost':rmsCost},'candidate_results':res,'selection_rule':'DEV 2022H2-2024H1 only: Sharpe>0, abs(corr RMS)<=0.35, maximize Sharpe*(1-abs(corr)); no later-period selection','winner':winner,'allocation_results':alloc,'allocation_rule':'DEV only: among RMS weights 1.0..0.5, minimize abs MaxDD subject to CAGR >=95% of RMS DEV CAGR','chosen_allocation':chosen,'periods':{'dev':'2022-07..2024-06','holdout1':'2024-07..2025-06','confirm':'2025-07..2025-12','secondary':'2026-01..2026-07'},'cost_model':'70USDT account; 5bps fee + 2bps slippage + actual funding; 5USDT min delta; aggregate portfolio targets before trading'}
 Path('alpha_v10_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v10_output/summary.json','w'),indent=2,allow_nan=True);rmsEq.to_csv('alpha_v10_output/rms_equity.csv');equities[winner].to_csv('alpha_v10_output/diversifier_equity.csv');print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
