#!/usr/bin/env python3
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
import alpha_v12_portfolio as v12
m=v12.m; v11=v12.v11; m.load_funding=v11.robust_load_funding

def simulate_governor(B,op,F,calm=1.15,trigger=.075,def_scale=.50,recover=.025,fee=5,slip=2,min_delta=5):
    idx=op.index;P=op[m.ALTS].to_numpy(float);T=B[m.ALTS].shift(1).fillna(0).to_numpy(float);FR=F.reindex(idx).fillna(0).to_numpy(float)
    q=np.zeros(len(m.ALTS));eq=np.full(len(idx),np.nan);equity=m.INITIAL;prev=np.full(len(m.ALTS),np.nan);peak=m.INITIAL;defensive=False;fees=slips=funds=0.;state=np.zeros(len(idx),dtype=bool);scale_hist=np.ones(len(idx))
    for k in range(len(idx)):
      p=P[k];valid=np.isfinite(p)
      if k>0:
        both=valid&np.isfinite(prev);equity+=float(np.nansum(q[both]*(p[both]-prev[both])))
      fr=FR[k];fm=valid&np.isfinite(fr)&(fr!=0)
      if fm.any():
        pay=float(np.nansum(q[fm]*p[fm]*fr[fm]));equity-=pay;funds+=pay
      if equity>peak: peak=equity
      dd=equity/peak-1 if peak>0 else -1
      # All state decisions use currently observed account equity only.
      if defensive:
        if dd>=-recover: defensive=False
      else:
        if dd<=-trigger: defensive=True
      scale=def_scale if defensive else calm;state[k]=defensive;scale_hist[k]=scale
      if m.TRADE_START<=idx[k]<m.END and equity>0:
        tw=T[k].copy()*scale;tw[~valid]=0;desired=np.zeros(len(m.ALTS));desired[valid]=tw[valid]*equity/p[valid];delta=desired-q;dn=np.abs(delta)*np.where(valid,p,0);ex=valid&(dn>=min_delta-1e-12);tr=float(dn[ex].sum());equity-=tr*(fee+slip)/1e4;fees+=tr*fee/1e4;slips+=tr*slip/1e4;q[ex]=desired[ex]
      eq[k]=equity;prev=p.copy()
    ser=pd.Series(eq,index=idx).loc[(idx>=m.TRADE_START)&(idx<m.END)];st=pd.Series(state,index=idx).loc[ser.index];sc=pd.Series(scale_hist,index=idx).loc[ser.index]
    return ser,{'fees':fees,'slippage':slips,'funding_paid':funds,'defensive_frac':float(st.mean()),'avg_scale':float(sc.mean()),'max_scale':float(sc.max()),'min_scale':float(sc.min())}

def main():
  raw={};fund={}
  with ThreadPoolExecutor(max_workers=10) as ex:
    fs={ex.submit(v11.load_kline_full,s):s for s in m.SYMS}
    for f in as_completed(fs):raw[fs[f]]=f.result()
  with ThreadPoolExecutor(max_workers=9) as ex:
    fs={ex.submit(m.load_funding,s):s for s in m.ALTS}
    for f in as_completed(fs):fund[fs[f]]=f.result()
  idx=pd.date_range(m.START,m.END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in m.SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in m.SYMS});F=pd.DataFrame({s:fund[s].reindex(idx) for s in m.ALTS})
  rms=m.weights_from_signal(m.rms_signal(cl),48);fcd=v11.fcd14_weights(F);rmsEq,_=m.simulate(rms,op,F);rmsdev=m.submetrics(rmsEq,m.TRADE_START,m.DEV_END);rmsfull=m.metrics(rmsEq);target=.95*rmsdev['CAGR']
  bases={'R90F10':.9*rms+.1*fcd,'R80F20':.8*rms+.2*fcd}
  res={}
  # Small locked grid: no more than 1.35x calm gross; defensive gross 0.40-0.70x of base.
  for bn,B in bases.items():
   for calm in [1.10,1.20,1.30,1.35]:
    for trig in [.05,.075,.10]:
     for ds in [.40,.55,.70]:
      rec=min(.025,trig/2);n=f'{bn}_L{int(calm*100)}_T{int(trig*1000):03d}_D{int(ds*100)}';eq,cost=simulate_governor(B,op,F,calm,trig,ds,rec);res[n]={'params':{'base':bn,'calm_scale':calm,'trigger':trig,'def_scale':ds,'recover':rec},'dev':m.submetrics(eq,m.TRADE_START,m.DEV_END),'full':m.metrics(eq),'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'cost':cost}
  eligible=[n for n,v in res.items() if v['dev']['CAGR']>=target];winner=min(eligible,key=lambda n:(abs(res[n]['dev']['MaxDD']),-res[n]['dev']['CAGR'],-res[n]['dev']['Sharpe'])) if eligible else max(res,key=lambda n:res[n]['dev']['Calmar'])
  full_goal=[(n,v) for n,v in res.items() if v['full']['CAGR']>=.95*rmsfull['CAGR'] and v['full']['MaxDD']>-.15];full_goal=sorted(full_goal,key=lambda x:(abs(x[1]['full']['MaxDD']),-x[1]['full']['CAGR']))
  out={'goal':'CAGR >=95% RMS and MDD<15%; DEV selection only','baseline_rms':{'dev':rmsdev,'full':rmsfull},'winner':winner,'winner_result':res[winner],'dev_goal_met':bool(res[winner]['dev']['CAGR']>=target and res[winner]['dev']['MaxDD']>-.15),'full_goal_count_descriptive_only':len(full_goal),'full_goal_best_descriptive_only':[{'name':n,**v} for n,v in full_goal[:15]],'all_results':res,'selection':'DEV min abs MDD subject CAGR>=95% RMS DEV; no later-period selection','governor':'stateful, based only on observed account drawdown from running peak at current hour; hysteresis recovery; calm leverage capped 1.35x'}
  Path('alpha_v16_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v16_output/summary.json','w'),indent=2,allow_nan=True);print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
