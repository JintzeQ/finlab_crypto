#!/usr/bin/env python3
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
import alpha_v12_portfolio as v12
m=v12.m; v11=v12.v11; m.load_funding=v11.robust_load_funding

def vol_scale(W,op,target_ann,lookback,cap=1.25,floor=.35):
    # Ex-ante proxy realized return: previous target weights times current open-to-open returns.
    r=op[m.ALTS].pct_change(); pr=(W.shift(1)*r).sum(axis=1);rv=pr.rolling(lookback,min_periods=lookback//2).std().shift(1)*np.sqrt(m.HRY)
    s=(target_ann/rv.replace(0,np.nan)).clip(floor,cap).fillna(1.0)
    return W.mul(s,axis=0),s

def main():
    raw={};fund={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        fs={ex.submit(v11.load_kline_full,s):s for s in m.SYMS}
        for f in as_completed(fs):raw[fs[f]]=f.result()
    with ThreadPoolExecutor(max_workers=9) as ex:
        fs={ex.submit(m.load_funding,s):s for s in m.ALTS}
        for f in as_completed(fs):fund[fs[f]]=f.result()
    idx=pd.date_range(m.START,m.END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in m.SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in m.SYMS});F=pd.DataFrame({s:fund[s].reindex(idx) for s in m.ALTS})
    rms=m.weights_from_signal(m.rms_signal(cl),48);fcd=v11.fcd14_weights(F);season=m.weights_from_signal(v12.seasonal_signal(m.residuals(cl),28),24)
    rmsEq,_=m.simulate(rms,op,F);rmsfull=m.metrics(rmsEq);rmsdev=m.submetrics(rmsEq,m.TRADE_START,m.DEV_END);target_dev=.95*rmsdev['CAGR']
    bases={
      '80R_20F':.8*rms+.2*fcd,
      '80R_10F_10S':.8*rms+.1*fcd+.1*season,
      '70R_20F_10S':.7*rms+.2*fcd+.1*season,
      '70R_10F_20S':.7*rms+.1*fcd+.2*season,
    }
    tests=[]
    for bn,B in bases.items():
      # Include unscaled and predeclared vol-target grid; cap gross scaling at 1.25x.
      configs=[('plain',B,pd.Series(1.0,index=B.index))]
      for lb in [168,336,720]:
       for tv in [.16,.18,.20,.22]:
        W,s=vol_scale(B,op,tv,lb,1.25,.35);configs.append((f'vt{int(tv*100)}_lb{lb}',W,s))
      for cn,W,s in configs:
        eq,cost=m.simulate(W,op,F);dev=m.submetrics(eq,m.TRADE_START,m.DEV_END);full=m.metrics(eq)
        tests.append({'base':bn,'config':cn,'dev':dev,'full':full,'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'avg_scale_dev':float(s.loc[(s.index>=m.TRADE_START)&(s.index<m.DEV_END)].mean()),'max_scale':float(s.max()),'min_scale':float(s.min()),'cost':cost})
    feasible=[x for x in tests if x['dev']['CAGR']>=target_dev]
    best=min(feasible,key=lambda x:(abs(x['dev']['MaxDD']),-x['dev']['CAGR'])) if feasible else max(tests,key=lambda x:x['dev']['Calmar'])
    full_goal=[x for x in tests if x['full']['CAGR']>=.95*rmsfull['CAGR'] and x['full']['MaxDD']>-.15]
    out={'goal':'full economic target CAGR >=95% RMS and MDD <15%; all selection DEV only','baseline_rms':{'dev':rmsdev,'full':rmsfull},'best_dev_selected':best,'dev_goal_met':bool(best['dev']['CAGR']>=target_dev and best['dev']['MaxDD']>-.15),'full_goal_count_descriptive_only':len(full_goal),'full_goal_descriptive_only':sorted(full_goal,key=lambda x:(abs(x['full']['MaxDD']),-x['full']['CAGR'])),'tests':tests,'risk_rule':'ex-ante rolling annualized proxy vol, lagged 1h; target 16/18/20/22%; lookback 168/336/720h; scale floor .35 cap 1.25; 70U actual account costs'}
    Path('alpha_v13_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v13_output/summary.json','w'),indent=2,allow_nan=True);print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
