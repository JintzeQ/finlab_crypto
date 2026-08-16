#!/usr/bin/env python3
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
import alpha_v11_portfolio as v11
m=v11.m; m.load_funding=v11.robust_load_funding

def seasonal_signal(e,days):
    # Forecast next-hour residual from the same UTC hour in prior days; only past observations.
    acc=e*0;cnt=e*0
    for k in range(1,days+1):
        x=e.shift(24*k-1);acc=acc+x.fillna(0);cnt=cnt+x.notna().astype(float)
    mu=acc/cnt.replace(0,np.nan)
    sd=e.shift(1).rolling(days*24,min_periods=max(168,days*12)).std()
    return m.cs_z(mu/sd.replace(0,np.nan))

def candidates(cl,F):
    e=m.residuals(cl);out={}
    # Intraday cross-sectional seasonality: independent of current trend direction.
    for d in [28,56,84]:
        sig=seasonal_signal(e,d)
        for reb in [12,24]:out[f'SEASON_d{d}_r{reb}']=m.weights_from_signal(sig,reb)
    # Residual acceleration: recent trend minus preceding trend, distinct from level momentum.
    for h in [48,72,120]:
        recent=e.shift(6).rolling(h,min_periods=h//2).sum();prior=e.shift(6+h).rolling(h,min_periods=h//2).sum();vol=e.shift(6).rolling(2*h,min_periods=h).std()*np.sqrt(h);sig=m.cs_z((recent-prior)/vol.replace(0,np.nan))
        for reb in [24,48]:out[f'ACCEL_h{h}_r{reb}']=m.weights_from_signal(sig,reb)
    # Short/medium residual trend, substantially faster than RMS48's 336h/skip24.
    for h in [72,120,168]:
        mom=e.shift(6).rolling(h,min_periods=h//2).sum();vol=e.shift(6).rolling(h,min_periods=h//2).std()*np.sqrt(h);sig=m.cs_z(mom/vol.replace(0,np.nan))
        for reb in [24,48]:out[f'FASTTREND_h{h}_r{reb}']=m.weights_from_signal(sig,reb)
    # Funding-conditioned trend: require price strength with cheap funding / weakness with rich funding.
    fc=m.cs_z(-F.fillna(0).rolling(14*24,min_periods=7*24).sum())
    for h in [72,168]:
        mom=e.shift(6).rolling(h,min_periods=h//2).sum();vol=e.shift(6).rolling(h,min_periods=h//2).std()*np.sqrt(h);mz=m.cs_z(mom/vol.replace(0,np.nan));sig=m.cs_z((mz+fc)/2)
        for reb in [24,48]:out[f'FUND_TREND_h{h}_r{reb}']=m.weights_from_signal(sig,reb)
    return out

def corr(a,b,a0,z0):return v11.corr(a,b,a0,z0)
def main():
    raw={};fund={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        fs={ex.submit(v11.load_kline_full,s):s for s in m.SYMS}
        for f in as_completed(fs):raw[fs[f]]=f.result();print('K',fs[f],len(raw[fs[f]]),flush=True)
    with ThreadPoolExecutor(max_workers=9) as ex:
        fs={ex.submit(m.load_funding,s):s for s in m.ALTS}
        for f in as_completed(fs):fund[fs[f]]=f.result()
    idx=pd.date_range(m.START,m.END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in m.SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in m.SYMS});F=pd.DataFrame({s:fund[s].reindex(idx) for s in m.ALTS})
    rmsW=m.weights_from_signal(m.rms_signal(cl),48);fcdW=v11.fcd14_weights(F);rmsEq,_=m.simulate(rmsW,op,F);fcdEq,_=m.simulate(fcdW,op,F);rmsfull=m.metrics(rmsEq)
    if not (220<rmsfull['final']<255):raise RuntimeError('RMS guard failed')
    cands=candidates(cl,F);cres={};ceq={}
    for n,W in cands.items():
        eq,_=m.simulate(W,op,F);ceq[n]=eq;cres[n]={'family':n.split('_')[0],'dev':m.submetrics(eq,m.TRADE_START,m.DEV_END),'full':m.metrics(eq),'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'corr_rms_dev':corr(eq,rmsEq,m.TRADE_START,m.DEV_END),'corr_fcd_dev':corr(eq,fcdEq,m.TRADE_START,m.DEV_END),'corr_rms_full':corr(eq,rmsEq,m.TRADE_START,m.END),'corr_fcd_full':corr(eq,fcdEq,m.TRADE_START,m.END)}
    # DEV-only select one per family: positive Sharpe; <=0.45 corr to RMS, <=0.35 to FCD.
    selected=[]
    for fam in ['SEASON','ACCEL','FASTTREND','FUND']:
        ns=[n for n,v in cres.items() if v['family']==fam and v['dev']['Sharpe']>0 and abs(v['corr_rms_dev'])<=.45 and abs(v['corr_fcd_dev'])<=.35]
        if ns:
            ns.sort(key=lambda n:cres[n]['dev']['Sharpe']*(1-abs(cres[n]['corr_rms_dev']))*(1-abs(cres[n]['corr_fcd_dev'])),reverse=True);selected.append(ns[0])
    # Keep max two extras, strongest dev adjusted score, to constrain search dimension.
    selected=sorted(selected,key=lambda n:cres[n]['dev']['Sharpe']*(1-abs(cres[n]['corr_rms_dev']))*(1-abs(cres[n]['corr_fcd_dev'])),reverse=True)[:2]
    engines={'RMS48':rmsW,'FCD14':fcdW};eqs={'RMS48':rmsEq,'FCD14':fcdEq}
    for n in selected:engines[n]=cands[n];eqs[n]=ceq[n]
    names=list(engines);rmsdev=m.submetrics(rmsEq,m.TRADE_START,m.DEV_END);target=.95*rmsdev['CAGR']
    # 10% simplex grid, RMS>=40%, each satellite<=30%.
    tuples=[];G=[i/10 for i in range(11)]
    if len(names)==2:tuples=[(a,1-a) for a in G if a>=.4 and 1-a<=.6]
    elif len(names)==3:
      for a in G:
       for b in G:
        c=round(1-a-b,10)
        if c>=0 and a>=.4 and b<=.4 and c<=.3:tuples.append((a,b,c))
    else:
      for a in G:
       for b in G:
        for c in G:
         d=round(1-a-b-c,10)
         if d>=0 and a>=.4 and max(b,c,d)<=.3:tuples.append((a,b,c,d))
    combos=[]
    for ws in tuples:
        W=sum(w*engines[n] for w,n in zip(ws,names));eq,cost=m.simulate(W,op,F);dev=m.submetrics(eq,m.TRADE_START,m.DEV_END);full=m.metrics(eq)
        combos.append({'weights':dict(zip(names,ws)),'dev':dev,'full':full,'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'cost':cost})
    feas=[x for x in combos if x['dev']['CAGR']>=target];best=min(feas,key=lambda x:(abs(x['dev']['MaxDD']),-x['dev']['CAGR'])) if feas else max(combos,key=lambda x:x['dev']['Calmar'])
    full_goal=[x for x in combos if x['full']['CAGR']>=.95*rmsfull['CAGR'] and x['full']['MaxDD']>-.15]
    out={'goal':'CAGR >=95% RMS and MDD <15%, DEV selection only','baseline_rms':{'full':rmsfull,'dev':rmsdev},'selected_extra_engines':selected,'candidate_results':cres,'engine_names':names,'best_dev_selected':best,'dev_goal_met':bool(best['dev']['CAGR']>=target and best['dev']['MaxDD']>-.15),'full_goal_count_descriptive_only':len(full_goal),'full_goal_best_descriptive_only':sorted(full_goal,key=lambda x:abs(x['full']['MaxDD']))[:5],'n_combos':len(combos)}
    Path('alpha_v12_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v12_output/summary.json','w'),indent=2,allow_nan=True);print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
