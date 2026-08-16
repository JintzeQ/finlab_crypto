#!/usr/bin/env python3
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import alpha_v19_btc_trend as v19

m=v19.m; v11=v19.v11
ALL=v19.ALL

# PREDECLARED V20 HYPOTHESIS / PARAMETERS.
# BTC trend is useful primarily when its recent realized sleeve return is
# diversifying versus both RMS and FCD. No 2025/2026 outcome is used here.
CORR_WINDOW=30*24
CORR_MIN=14*24
CORR_MAX=0.25
HARD_CAGR=0.333
HARD_MDD=-0.15


def conditional_gate(bt_eq,rms_eq,fcd_eq,idx):
    r=pd.DataFrame({
        'btc':bt_eq.pct_change(),
        'rms':rms_eq.pct_change(),
        'fcd':fcd_eq.pct_change(),
    }).reindex(idx)
    # shift(1): today's allocation can use only correlations known before this hour.
    cr=r['btc'].rolling(CORR_WINDOW,min_periods=CORR_MIN).corr(r['rms']).shift(1)
    cf=r['btc'].rolling(CORR_WINDOW,min_periods=CORR_MIN).corr(r['fcd']).shift(1)
    gate=((cr<=CORR_MAX)&(cf<=CORR_MAX)).fillna(False).astype(float)
    return gate,cr,cf


def pack(eq,cost=None):
    out={
        'dev':m.submetrics(eq,m.TRADE_START,m.DEV_END),
        'full':m.metrics(eq),
        'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),
        'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),
        'secondary2026':m.submetrics(eq,m.CONF_END,m.END),
    }
    if cost is not None: out['cost']=cost
    return out


def main():
    raw={};fund={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        fs={ex.submit(v11.load_kline_full,s):s for s in ALL}
        for f in as_completed(fs): raw[fs[f]]=f.result()
    with ThreadPoolExecutor(max_workers=10) as ex:
        fs={ex.submit(m.load_funding,s):s for s in ALL}
        for f in as_completed(fs): fund[fs[f]]=f.result()

    idx=pd.date_range(m.START,m.END-pd.Timedelta(hours=1),freq='h')
    cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in ALL})
    op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in ALL})
    F=pd.DataFrame({s:fund[s].reindex(idx) for s in ALL})

    rms=m.weights_from_signal(m.rms_signal(cl),48)
    fcd=v11.fcd14_weights(F[m.ALTS])
    zero=pd.DataFrame(0.,index=idx,columns=m.ALTS)
    bt=v19.btc_trend_weight(cl)

    rmsEq,_=v19.joint_sim(rms,pd.Series(0.,index=idx),op,F)
    fcdEq,_=v19.joint_sim(fcd,pd.Series(0.,index=idx),op,F)
    btEq,_=v19.joint_sim(zero,bt,op,F)
    gate,cr,cf=conditional_gate(btEq,rmsEq,fcdEq,idx)
    cbt=bt*gate
    cbtEq,cbtCost=v19.joint_sim(zero,cbt,op,F)

    # Fixed ablation allocations. These are diagnostic, not parameter selection.
    ablation_specs={
        'RMS_only':(1.0,0.0,0.0),
        'FCD_only':(0.0,1.0,0.0),
        'BTC_cond_only':(0.0,0.0,1.0),
        'RMS_FCD_70_30':(0.7,0.3,0.0),
        'RMS_BTC_80_20':(0.8,0.0,0.2),
        'FCD_BTC_80_20':(0.0,0.8,0.2),
        'TRIPLE_60_20_20':(0.6,0.2,0.2),
    }
    ablations={}
    for name,(wr,wf,wb) in ablation_specs.items():
        eq,cost=v19.joint_sim(wr*rms+wf*fcd,wb*cbt,op,F)
        ablations[name]=pack(eq,cost)

    # Same allocation/leverage search domain as v19; no newly tuned grid values.
    alloc={};params={}
    for wf in np.arange(0,.31,.05):
        for wb in np.arange(.05,.31,.05):
            wr=round(float(1-wf-wb),10)
            if wr<.50: continue
            alt=wr*rms+wf*fcd; btc=wb*cbt
            for lev in [1.0,1.1,1.2,1.3]:
                eq,cost=v19.joint_sim(alt,btc,op,F,lev)
                n=f'R{wr:.2f}_F{wf:.2f}_B{wb:.2f}_L{lev:.1f}'
                alloc[n]={'weights':{'RMS':wr,'FCD':float(wf),'BTCconditional':float(wb),'scalar':lev},**pack(eq,cost)}
                params[n]=(round(wr,2),round(float(wf),2),round(float(wb),2),round(lev,1))

    # DEV-only constrained selection: first enforce MDD<15%, then maximize CAGR.
    feasible=[n for n,v in alloc.items() if v['dev']['MaxDD']>HARD_MDD]
    if feasible:
        chosen=max(feasible,key=lambda n:(alloc[n]['dev']['CAGR'],alloc[n]['dev']['Sharpe'],alloc[n]['dev']['Calmar']))
    else:
        chosen=max(alloc,key=lambda n:alloc[n]['dev']['Calmar'])

    cw=params[chosen]
    neighbors=[]
    for n,p in params.items():
        if n==chosen: continue
        diffs=[abs(p[i]-cw[i]) for i in range(4)]
        alloc_step=(sum(d>1e-9 for d in diffs[:3])==2 and max(diffs[:3])<=.051 and diffs[3]<1e-9)
        lev_step=(sum(d>1e-9 for d in diffs[:3])==0 and 0.099<=diffs[3]<=.101)
        if alloc_step or lev_step: neighbors.append(n)

    chosen_dev=alloc[chosen]['dev']
    robust_good=[n for n in neighbors if alloc[n]['dev']['MaxDD']>HARD_MDD and alloc[n]['dev']['CAGR']>=.90*chosen_dev['CAGR']]
    robust_ratio=(len(robust_good)/len(neighbors)) if neighbors else 0.0
    hard_gate=bool(chosen_dev['dev']['CAGR']>=HARD_CAGR and chosen_dev['dev']['MaxDD']>HARD_MDD) if 'dev' in chosen_dev else False
    hard_gate=bool(chosen_dev['CAGR']>=HARD_CAGR and chosen_dev['MaxDD']>HARD_MDD)
    robustness_gate=bool(neighbors and robust_ratio>=.50)

    dev_mask=(idx>=m.TRADE_START)&(idx<m.DEV_END)
    gate_dev=gate.loc[dev_mask]
    out={
        'hypothesis':'BTC trend contributes mainly through conditional diversification; enable only when lagged 30d rolling correlation to both RMS and FCD is <=0.25.',
        'anti_overfit':'All selection and robustness gates use DEV only. 2025/2026 metrics are descriptive only and are never referenced by selection logic.',
        'fixed_parameters':{'corr_window_hours':CORR_WINDOW,'corr_min_hours':CORR_MIN,'corr_max':CORR_MAX,'btc_trend_horizons_hours':[168,336,720],'hard_cagr':HARD_CAGR,'hard_mdd':HARD_MDD},
        'conditional_btc':{**pack(cbtEq,cbtCost),'dev_gate_on_fraction':float(gate_dev.mean()),'dev_corr_rms_mean':float(cr.loc[dev_mask].mean()),'dev_corr_fcd_mean':float(cf.loc[dev_mask].mean())},
        'ablations':ablations,
        'selection':'DEV-only: maximize CAGR among allocations with MaxDD > -15%; tie break Sharpe then Calmar.',
        'chosen':chosen,
        'chosen_result':alloc[chosen],
        'hard_gate_pass':hard_gate,
        'robustness_definition':'Immediate grid neighbors; >=50% must retain MDD<15% and >=90% of chosen DEV CAGR.',
        'neighbor_count':len(neighbors),
        'neighbor_names':neighbors,
        'robust_neighbor_count':len(robust_good),
        'robust_neighbor_names':robust_good,
        'robust_neighbor_ratio':robust_ratio,
        'robustness_gate_pass':robustness_gate,
        'overall_pass':bool(hard_gate and robustness_gate),
        'all_allocations':alloc,
    }
    Path('alpha_v20_output').mkdir(exist_ok=True)
    with open('alpha_v20_output/summary.json','w') as f: json.dump(out,f,indent=2,allow_nan=True)
    print(json.dumps(out,indent=2,allow_nan=True),flush=True)

if __name__=='__main__': main()
# v20 execution trigger; parameters above remain frozen.
