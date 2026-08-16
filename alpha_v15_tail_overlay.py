#!/usr/bin/env python3
import io,json,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
import alpha_v12_portfolio as v12
m=v12.m; v11=v12.v11; m.load_funding=v11.robust_load_funding

def proxy_returns(W,op):
    # purely lagged/ex-ante strategy proxy, no execution costs; used only for state classification
    rr=op[m.ALTS].pct_change();return (W.shift(1)*rr).sum(axis=1).fillna(0)

def rank_flip(sig,h):
    # Cross-sectional rank instability: mean absolute percentile rank change vs h hours ago.
    rk=sig.rank(axis=1,pct=True);return (rk-rk.shift(h)).abs().mean(axis=1)

def overlays(rmsW,fcdW,sig,op):
    pr=proxy_returns(rmsW,op);out={}
    # Locked small grid. State metrics all lagged one hour.
    for trend_h in [72,168,336]:
      trend=pr.rolling(trend_h,min_periods=trend_h//2).sum().shift(1)
      for vol_h in [168,336]:
        vol=pr.rolling(vol_h,min_periods=vol_h//2).std().shift(1)*np.sqrt(m.HRY)
        vthr=vol.rolling(720,min_periods=360).quantile(.70).shift(1)
        for flip_h in [24,48]:
          flip=rank_flip(sig,flip_h).shift(1);fthr=flip.rolling(720,min_periods=360).quantile(.70).shift(1)
          # Three transparent regimes: any 2 of 3 stress flags.
          flags=(trend<0).astype(int)+(vol>vthr).astype(int)+(flip>fthr).astype(int);stress=flags>=2
          for defensive in [.25,.50,.75]:
            # In stress: shift defensive fraction from RMS to FCD, keeping gross ~1x.
            a=pd.Series(1.0,index=rmsW.index);a.loc[stress]=1-defensive
            W=rmsW.mul(a,axis=0)+fcdW.mul(1-a,axis=0)
            out[f'switch_t{trend_h}_v{vol_h}_f{flip_h}_d{int(defensive*100)}']=(W,stress)
          for cut in [.25,.50]:
            # Alternative: retain FCD10 base and cut total gross in stress; no compensating leverage in calm regime.
            base=.9*rmsW+.1*fcdW;s=pd.Series(1.0,index=rmsW.index);s.loc[stress]=1-cut;W=base.mul(s,axis=0)
            out[f'cut_t{trend_h}_v{vol_h}_f{flip_h}_c{int(cut*100)}']=(W,stress)
    return out

def main():
    raw={};fund={}
    with ThreadPoolExecutor(max_workers=10) as ex:
      fs={ex.submit(v11.load_kline_full,s):s for s in m.SYMS}
      for f in as_completed(fs):raw[fs[f]]=f.result()
    with ThreadPoolExecutor(max_workers=9) as ex:
      fs={ex.submit(m.load_funding,s):s for s in m.ALTS}
      for f in as_completed(fs):fund[fs[f]]=f.result()
    idx=pd.date_range(m.START,m.END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in m.SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in m.SYMS});F=pd.DataFrame({s:fund[s].reindex(idx) for s in m.ALTS})
    sig=m.rms_signal(cl);rmsW=m.weights_from_signal(sig,48);fcdW=v11.fcd14_weights(F);rmsEq,_=m.simulate(rmsW,op,F);rmsfull=m.metrics(rmsEq);rmsdev=m.submetrics(rmsEq,m.TRADE_START,m.DEV_END)
    cands=overlays(rmsW,fcdW,sig,op);res={}
    for n,(W,stress) in cands.items():
      eq,cost=m.simulate(W,op,F);dev=m.submetrics(eq,m.TRADE_START,m.DEV_END);full=m.metrics(eq)
      res[n]={'dev':dev,'full':full,'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'stress_frac_dev':float(stress.loc[(stress.index>=m.TRADE_START)&(stress.index<m.DEV_END)].mean()),'stress_frac_full':float(stress.loc[(stress.index>=m.TRADE_START)&(stress.index<m.END)].mean()),'cost':cost}
    # DEV selection objective: minimize MDD subject to CAGR >=95% RMS DEV; tie higher CAGR then Sharpe.
    target=.95*rmsdev['CAGR'];eligible=[n for n,v in res.items() if v['dev']['CAGR']>=target]
    winner=min(eligible,key=lambda n:(abs(res[n]['dev']['MaxDD']),-res[n]['dev']['CAGR'],-res[n]['dev']['Sharpe'])) if eligible else max(res,key=lambda n:res[n]['dev']['Calmar'])
    # Report full target only descriptively, never used for selection.
    full_goal=[(n,v) for n,v in res.items() if v['full']['CAGR']>=.95*rmsfull['CAGR'] and v['full']['MaxDD']>-.15]
    full_goal=sorted(full_goal,key=lambda x:(abs(x[1]['full']['MaxDD']),-x[1]['full']['CAGR']))
    out={'goal':'preserve >=95% RMS CAGR and MDD<15%; rules selected DEV only','baseline_rms':{'dev':rmsdev,'full':rmsfull},'selection_rule':'DEV 2022-07..2024-06: min |MDD| subject to CAGR>=95% RMS DEV, no later-period selection','winner':winner,'winner_result':res[winner],'dev_goal_met':bool(res[winner]['dev']['CAGR']>=target and res[winner]['dev']['MaxDD']>-.15),'full_goal_count_descriptive_only':len(full_goal),'full_goal_best_descriptive_only':[{'name':n,**v} for n,v in full_goal[:10]],'all_results':res,'cost_model':'70U actual account, fee5+slip2+actual funding, min delta5, aggregate target before trade'}
    Path('alpha_v15_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v15_output/summary.json','w'),indent=2,allow_nan=True);print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
