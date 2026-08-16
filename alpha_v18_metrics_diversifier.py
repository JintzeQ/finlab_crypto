#!/usr/bin/env python3
import io,json,zipfile,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
import alpha_v12_portfolio as v12
m=v12.m; v11=v12.v11; m.load_funding=v11.robust_load_funding
BASE='https://data.binance.vision/data/futures/um/daily/metrics'
MCOLS=['create_time','symbol','sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio','sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio']
MSTART=pd.Timestamp('2022-06-01',tz='UTC'); MEND=m.END-pd.Timedelta(days=1)

def fetch_metric_day(sym,day):
    ds=day.strftime('%Y-%m-%d');url=f'{BASE}/{sym}/{sym}-metrics-{ds}.zip'
    try:
      r=requests.get(url,timeout=15,headers={'User-Agent':'Mozilla/5.0'});
      if r.status_code!=200:return sym,None
      z=zipfile.ZipFile(io.BytesIO(r.content));df=pd.read_csv(z.open(z.namelist()[0]))
      if set(MCOLS).issubset(df.columns): df=df[MCOLS]
      else: df=pd.read_csv(z.open(z.namelist()[0]),header=None,names=MCOLS)
      x=pd.to_numeric(df['create_time'],errors='coerce');unit='ms' if x.dropna().median()>1e11 else 's';df['time']=pd.to_datetime(x,unit=unit,utc=True)
      for c in MCOLS[2:]:df[c]=pd.to_numeric(df[c],errors='coerce')
      return sym,df[['time']+MCOLS[2:]].dropna(subset=['time'])
    except Exception:return sym,None

def load_metrics():
    dates=list(pd.date_range(MSTART,MEND,freq='D'));acc={s:[] for s in m.ALTS};jobs=[(s,d) for s in m.ALTS for d in dates]
    with ThreadPoolExecutor(max_workers=64) as ex:
      fs={ex.submit(fetch_metric_day,s,d):(s,d) for s,d in jobs}
      for j,f in enumerate(as_completed(fs),1):
        s,df=f.result()
        if df is not None and len(df):acc[s].append(df)
        if j%1000==0:print('metrics files',j,'/',len(jobs),flush=True)
    fields={c:pd.DataFrame() for c in MCOLS[2:]};idx=pd.date_range(m.START,m.END-pd.Timedelta(hours=1),freq='h')
    for s in m.ALTS:
      if not acc[s]:continue
      d=pd.concat(acc[s],ignore_index=True).sort_values('time');d['hour']=d.time.dt.floor('h');g=d.groupby('hour')
      h=pd.DataFrame(index=d.hour.drop_duplicates().sort_values());h['sum_open_interest_value']=g.sum_open_interest_value.last();h['sum_open_interest']=g.sum_open_interest.last()
      for c in ['count_toptrader_long_short_ratio','sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio']:h[c]=g[c].mean()
      for c in fields:fields[c][s]=h[c].reindex(idx)
      print('metric',s,len(d),flush=True)
    return {c:df.reindex(idx) for c,df in fields.items()}

def csz(df):
    mu=df.mean(axis=1);sd=df.std(axis=1).replace(0,np.nan);return df.sub(mu,axis=0).div(sd,axis=0)
def signal_weights(sig,reb):return m.weights_from_signal(csz(sig),reb)
def candidates(M):
    oi=M['sum_open_interest_value'].replace(0,np.nan);taker=M['sum_taker_long_short_vol_ratio'].replace(0,np.nan);top=M['sum_toptrader_long_short_ratio'].replace(0,np.nan);crowd=M['count_long_short_ratio'].replace(0,np.nan)
    logtak=np.log(taker.clip(lower=1e-6));smart=np.log((top/crowd).clip(lower=1e-6));out={}
    for h in [24,72]:
      sd=smart.shift(1).rolling(h,min_periods=max(12,h//2)).mean();oig=np.log(oi).diff(h).shift(1);oz=csz(oig);tz=csz(logtak.shift(1).rolling(h,min_periods=max(12,h//2)).mean());build=oz.clip(lower=0);contra=-(tz*build);flow=tz*build
      for reb in [8,24]:
        out[f'smart_h{h}_r{reb}']=signal_weights(sd,reb);out[f'oi_contra_h{h}_r{reb}']=signal_weights(contra,reb);out[f'oi_flow_h{h}_r{reb}']=signal_weights(flow,reb)
    return out
def corr_eq(a,b,a0,b0):
    x=pd.concat([a.pct_change(),b.pct_change()],axis=1).loc[(a.index>=a0)&(a.index<b0)].dropna();return float(x.iloc[:,0].corr(x.iloc[:,1])) if len(x)>10 else np.nan

def main():
  raw={};fund={}
  with ThreadPoolExecutor(max_workers=10) as ex:
    fs={ex.submit(v11.load_kline_full,s):s for s in m.SYMS}
    for f in as_completed(fs):raw[fs[f]]=f.result()
  with ThreadPoolExecutor(max_workers=9) as ex:
    fs={ex.submit(m.load_funding,s):s for s in m.ALTS}
    for f in as_completed(fs):fund[fs[f]]=f.result()
  idx=pd.date_range(m.START,m.END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in m.SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in m.SYMS});F=pd.DataFrame({s:fund[s].reindex(idx) for s in m.ALTS})
  M=load_metrics();rms=m.weights_from_signal(m.rms_signal(cl),48);fcd=v11.fcd14_weights(F);rmsEq,_=m.simulate(rms,op,F);fcdEq,_=m.simulate(fcd,op,F);rdev=m.submetrics(rmsEq,m.TRADE_START,m.DEV_END);rfull=m.metrics(rmsEq)
  C=candidates(M);cres={}
  for n,W in C.items():
    eq,cost=m.simulate(W,op,F);cres[n]={'dev':m.submetrics(eq,m.TRADE_START,m.DEV_END),'full':m.metrics(eq),'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'corr_rms_dev':corr_eq(eq,rmsEq,m.TRADE_START,m.DEV_END),'corr_fcd_dev':corr_eq(eq,fcdEq,m.TRADE_START,m.DEV_END),'corr_rms_full':corr_eq(eq,rmsEq,m.TRADE_START,m.END),'corr_fcd_full':corr_eq(eq,fcdEq,m.TRADE_START,m.END),'cost':cost}
  eligible=[n for n,v in cres.items() if v['dev']['Sharpe']>0 and abs(v['corr_rms_dev'])<=.40 and abs(v['corr_fcd_dev'])<=.40];winner=max(eligible,key=lambda n:cres[n]['dev']['Sharpe']*(1-abs(cres[n]['corr_rms_dev']))*(1-abs(cres[n]['corr_fcd_dev']))) if eligible else max(cres,key=lambda n:cres[n]['dev']['Sharpe']);nw=C[winner]
  alloc={};target=.95*rdev['CAGR']
  for wf in np.arange(0,.31,.05):
   for wn in np.arange(.05,.31,.05):
    wr=1-wf-wn
    if wr<.50:continue
    base=wr*rms+wf*fcd+wn*nw
    for lev in [1.0,1.1,1.2,1.3]:
      eq,cost=m.simulate(base*lev,op,F);name=f'R{wr:.2f}_F{wf:.2f}_N{wn:.2f}_L{lev:.1f}';alloc[name]={'weights':{'RMS':wr,'FCD':wf,'NEW':wn,'gross_scalar':lev},'dev':m.submetrics(eq,m.TRADE_START,m.DEV_END),'full':m.metrics(eq),'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'cost':cost}
  ok=[n for n,v in alloc.items() if v['dev']['CAGR']>=target];chosen=min(ok,key=lambda n:(abs(alloc[n]['dev']['MaxDD']),-alloc[n]['dev']['CAGR'],-alloc[n]['dev']['Sharpe'])) if ok else max(alloc,key=lambda n:alloc[n]['dev']['Calmar']);devgoal=[n for n,v in alloc.items() if v['dev']['CAGR']>=target and v['dev']['MaxDD']>-.15];fullgoal=[n for n,v in alloc.items() if v['full']['CAGR']>=.95*rfull['CAGR'] and v['full']['MaxDD']>-.15]
  out={'goal':'CAGR>=95% RMS and MDD<15%; DEV-only strategy/allocation selection','data':'Binance Vision daily futures metrics 5m: OI, top/global L/S ratios, taker ratio','baseline_rms':{'dev':rdev,'full':rfull},'candidate_results':cres,'candidate_selection':'DEV only: Sharpe>0, |corr RMS|<=.40, |corr FCD|<=.40; maximize Sharpe*(1-|corrR|)*(1-|corrF|)','winner':winner,'winner_result':cres[winner],'chosen_allocation':chosen,'chosen_result':alloc[chosen],'dev_goal_count':len(devgoal),'dev_goal_names':devgoal,'full_goal_count_descriptive_only':len(fullgoal),'full_goal_names_descriptive_only':fullgoal,'allocation_selection':'DEV only: minimize abs MDD subject CAGR>=95% RMS DEV','all_allocations':alloc}
  Path('alpha_v18_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v18_output/summary.json','w'),indent=2,allow_nan=True);print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
# trigger-v18-utc
