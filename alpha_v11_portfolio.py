#!/usr/bin/env python3
import io,json,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
import alpha_v10_diversifier as m

# Harden funding parser exactly as audited in v10-fixed.
def robust_load_funding(s):
    fs=[]
    for ym in m.months():
        b=m.getzip(f'{m.ROOT}/fundingRate/{s}/{s}-fundingRate-{ym}.zip')
        if b is None: continue
        with zipfile.ZipFile(io.BytesIO(b)) as z: fs.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None))
    if not fs:return pd.Series(dtype=float)
    d=pd.concat(fs,ignore_index=True); nums={c:pd.to_numeric(d[c],errors='coerce') for c in d.columns}
    tcol=max(nums,key=lambda c: float(nums[c].dropna().median()) if len(nums[c].dropna()) else -np.inf); t=nums[tcol]
    cand=[]
    for c,v in nums.items():
        if c==tcol:continue
        a=v.dropna().abs()
        if len(a)<10:continue
        if float(a.quantile(.99))<=.10 and float((a>0).mean())>.01:
            med=float(a.median());cand.append((abs(np.log10(max(med,1e-12))-np.log10(1e-4)),c))
    if not cand:raise RuntimeError(f'No plausible funding column {s}')
    _,rcol=min(cand);r=nums[rcol]
    if float(r.dropna().abs().max())>.10:raise RuntimeError(f'Implausible funding {s}')
    tt=t.where(t<1e15,t/1000);idx=pd.to_datetime(tt,unit='ms',utc=True,errors='coerce')
    out=pd.Series(r.values,index=idx).dropna().groupby(level=0).last().sort_index()
    print('FUND',s,'rcol',rcol,'median_bps',float(out.median()*1e4),'max_bps',float(out.abs().max()*1e4),flush=True)
    return out
m.load_funding=robust_load_funding

# Need volume in addition to prices.
def load_kline_full(s):
    fs=[]
    for ym in m.months():
        b=m.getzip(f'{m.ROOT}/klines/{s}/1h/{s}-1h-{ym}.zip')
        if b is None:continue
        with zipfile.ZipFile(io.BytesIO(b)) as z:fs.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None,names=m.COLS))
    d=pd.concat(fs,ignore_index=True);t=pd.to_numeric(d.open_time,errors='coerce');t=t.where(t<1e15,t/1000);d['date']=pd.to_datetime(t,unit='ms',utc=True,errors='coerce')
    for c in ['open','close','quote_volume']:d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.dropna(subset=['date','open','close']).drop_duplicates('date').sort_values('date').set_index('date')

def fcd14_weights(F):
    fs=F.fillna(0).rolling(14*24,min_periods=7*24).sum();return m.weights_from_signal(m.cs_z(-fs),24)

def extra_candidates(cl,qv):
    e=m.residuals(cl);out={}
    # Family A: fast idiosyncratic mean reversion, opposite horizon to RMS48.
    for h in [3,6,12,24]:
        rr=e.shift(1).rolling(h,min_periods=max(2,h//2)).sum();vv=e.shift(1).rolling(max(24,2*h),min_periods=12).std()*np.sqrt(h)
        sig=m.cs_z(-rr/vv.replace(0,np.nan))
        for reb in [4,8,12]:out[f'REV_h{h}_r{reb}']=m.weights_from_signal(sig,reb)
    # Family B: low idiosyncratic volatility anomaly; different from direction/momentum.
    for vh in [72,168,336]:
        vol=e.shift(1).rolling(vh,min_periods=vh//2).std();sig=m.cs_z(-np.log(vol.replace(0,np.nan)))
        for reb in [24,48]:out[f'LOWVOL_h{vh}_r{reb}']=m.weights_from_signal(sig,reb)
    # Family C: volume-shock reversal. Fade residual moves that arrive with unusually high quote volume.
    lq=np.log1p(qv[m.ALTS]);vmu=lq.shift(1).rolling(168,min_periods=84).mean();vsd=lq.shift(1).rolling(168,min_periods=84).std().replace(0,np.nan);vz=((lq-vmu)/vsd).clip(-3,3)
    for h in [3,6,12]:
        rr=e.shift(1).rolling(h,min_periods=max(2,h//2)).sum();rv=e.shift(1).rolling(72,min_periods=36).std()*np.sqrt(h)
        shock=(rr/rv.replace(0,np.nan))*vz.shift(1).rolling(h,min_periods=max(2,h//2)).mean();sig=m.cs_z(-shock)
        for reb in [4,8,12]:out[f'VSHOCK_h{h}_r{reb}']=m.weights_from_signal(sig,reb)
    return out

def eqret(eq):return eq.pct_change().fillna(0)
def corr(a,b,a0,z0):
    x=pd.concat([eqret(a),eqret(b)],axis=1).loc[(a.index>=a0)&(a.index<z0)].dropna();return float(x.iloc[:,0].corr(x.iloc[:,1]))

def main():
    raw={};fund={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        fs={ex.submit(load_kline_full,s):s for s in m.SYMS}
        for f in as_completed(fs):raw[fs[f]]=f.result();print('K',fs[f],len(raw[fs[f]]),flush=True)
    with ThreadPoolExecutor(max_workers=9) as ex:
        fs={ex.submit(m.load_funding,s):s for s in m.ALTS}
        for f in as_completed(fs):fund[fs[f]]=f.result()
    idx=pd.date_range(m.START,m.END-pd.Timedelta(hours=1),freq='h')
    cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in m.SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in m.SYMS});qv=pd.DataFrame({s:raw[s].quote_volume.reindex(idx) for s in m.SYMS});F=pd.DataFrame({s:fund[s].reindex(idx) for s in m.ALTS})
    rmsW=m.weights_from_signal(m.rms_signal(cl),48);fcdW=fcd14_weights(F)
    rmsEq,_=m.simulate(rmsW,op,F);fcdEq,_=m.simulate(fcdW,op,F)
    # Baseline reproduction guard from v10/v9.
    rmsfull=m.metrics(rmsEq)
    if not (220 < rmsfull['final'] < 255 and .30 < rmsfull['CAGR'] < .40): raise RuntimeError(f'RMS baseline failed audit {rmsfull}')
    cands=extra_candidates(cl,qv); cres={};ceq={}
    for n,W in cands.items():
        eq,cost=m.simulate(W,op,F);ceq[n]=eq
        cres[n]={'family':n.split('_')[0],'dev':m.submetrics(eq,m.TRADE_START,m.DEV_END),'full':m.metrics(eq),'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'corr_rms_dev':corr(eq,rmsEq,m.TRADE_START,m.DEV_END),'corr_fcd_dev':corr(eq,fcdEq,m.TRADE_START,m.DEV_END),'corr_rms_full':corr(eq,rmsEq,m.TRADE_START,m.END),'corr_fcd_full':corr(eq,fcdEq,m.TRADE_START,m.END)}
    # Pick at most one per economic family, using DEV only. Require positive DEV Sharpe and low correlation to both existing engines.
    selected=[]
    for fam in ['REV','LOWVOL','VSHOCK']:
        names=[n for n,v in cres.items() if v['family']==fam and v['dev']['Sharpe']>0 and abs(v['corr_rms_dev'])<=.30 and abs(v['corr_fcd_dev'])<=.30]
        if names:
            names.sort(key=lambda n:cres[n]['dev']['Sharpe']*(1-abs(cres[n]['corr_rms_dev']))*(1-abs(cres[n]['corr_fcd_dev'])),reverse=True);selected.append(names[0])
    engines={'RMS48':rmsW,'FCD14':fcdW};eng_eq={'RMS48':rmsEq,'FCD14':fcdEq}
    for n in selected:engines[n]=cands[n];eng_eq[n]=ceq[n]
    # Pairwise correlation matrix full and dev.
    names=list(engines);corrdev={};corrfull={}
    for a in names:
        corrdev[a]={};corrfull[a]={}
        for b in names:
            corrdev[a][b]=1.0 if a==b else corr(eng_eq[a],eng_eq[b],m.TRADE_START,m.DEV_END)
            corrfull[a][b]=1.0 if a==b else corr(eng_eq[a],eng_eq[b],m.TRADE_START,m.END)
    # Enumerate weights in 10% steps, no leverage, sum=1. RMS must remain >=40%; each diversifier <=30% to avoid a weak engine dominating.
    combos=[];grid=[i/10 for i in range(11)];rmsdev=m.submetrics(rmsEq,m.TRADE_START,m.DEV_END);target_dev=.95*rmsdev['CAGR']
    if len(names)==2:
        tuples=[(wr,1-wr) for wr in grid if wr>=.4]
    elif len(names)==3:
        tuples=[]
        for a in grid:
          for b in grid:
            c=round(1-a-b,10)
            if c>=0 and a>=.4 and b<=.3 and c<=.3:tuples.append((a,b,c))
    else:
        tuples=[]
        for a in grid:
          for b in grid:
           for c in grid:
            d=round(1-a-b-c,10)
            if d>=0 and a>=.4 and max(b,c,d)<=.3:tuples.append((a,b,c,d))
    for ws in tuples:
        W=sum(w*engines[n] for w,n in zip(ws,names));eq,cost=m.simulate(W,op,F);dev=m.submetrics(eq,m.TRADE_START,m.DEV_END);full=m.metrics(eq)
        combos.append({'weights':dict(zip(names,ws)),'dev':dev,'full':full,'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'goal_dev':bool(dev['CAGR']>=target_dev and dev['MaxDD']>-.15),'cost':cost})
    feasible=[x for x in combos if x['dev']['CAGR']>=target_dev]
    best=min(feasible,key=lambda x:(abs(x['dev']['MaxDD']),-x['dev']['CAGR'])) if feasible else max(combos,key=lambda x:x['dev']['Calmar'])
    # Also report any full-period combos that meet user's economic target, but NEVER use them for selection.
    full_goal=[x for x in combos if x['full']['CAGR']>=.95*rmsfull['CAGR'] and x['full']['MaxDD']>-.15]
    full_goal_sorted=sorted(full_goal,key=lambda x:(abs(x['full']['MaxDD']),-x['full']['CAGR']))[:10]
    out={'goal':'CAGR >=95% RMS48 and MaxDD <15%; selection DEV only','baseline_rms':{'full':rmsfull,'dev':rmsdev},'baseline_fcd':{'full':m.metrics(fcdEq),'dev':m.submetrics(fcdEq,m.TRADE_START,m.DEV_END),'corr_rms_full':corr(fcdEq,rmsEq,m.TRADE_START,m.END)},'candidate_results':cres,'selected_extra_engines':selected,'engine_names':names,'corr_dev':corrdev,'corr_full':corrfull,'best_dev_selected':best,'full_goal_count_descriptive_only':len(full_goal),'full_goal_examples_descriptive_only':full_goal_sorted,'n_combos':len(combos),'cost_model':'70U, 5bps fee +2bps slippage + actual funding, 5U min delta, aggregate targets before trading'}
    Path('alpha_v11_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v11_output/summary.json','w'),indent=2,allow_nan=True);print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
