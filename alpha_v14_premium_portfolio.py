#!/usr/bin/env python3
import io,json,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
import alpha_v12_portfolio as v12
m=v12.m; v11=v12.v11; m.load_funding=v11.robust_load_funding
PREM_COLS=['open_time','open','high','low','close','ignore1','close_time','ignore2','count','ignore3','ignore4','ignore5']

def load_premium(s):
    fs=[]
    for ym in m.months():
        b=m.getzip(f'{m.ROOT}/premiumIndexKlines/{s}/1h/{s}-1h-{ym}.zip')
        if b is None:continue
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            x=pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None,names=PREM_COLS)
            fs.append(x)
    if not fs:return pd.DataFrame(columns=['close'])
    d=pd.concat(fs,ignore_index=True);t=pd.to_numeric(d.open_time,errors='coerce');t=t.where(t<1e15,t/1000);d['date']=pd.to_datetime(t,unit='ms',utc=True,errors='coerce');d['close']=pd.to_numeric(d['close'],errors='coerce')
    return d.dropna(subset=['date','close']).drop_duplicates('date').sort_values('date').set_index('date')

def premium_candidates(P):
    out={}
    # Cross-sectional premium carry: long cheap/discounted perps, short rich perps.
    for h in [24,72,168]:
        lev=P.shift(1).rolling(h,min_periods=h//2).mean();sig=m.cs_z(-lev)
        for reb in [12,24,48]:out[f'PREM_CARRY_h{h}_r{reb}']=m.weights_from_signal(sig,reb)
    # Own-history premium dislocation mean reversion, then rank cross-sectionally.
    for h in [168,336,720]:
        mu=P.shift(1).rolling(h,min_periods=h//2).mean();sd=P.shift(1).rolling(h,min_periods=h//2).std().replace(0,np.nan);dev=(P.shift(1)-mu)/sd;sig=m.cs_z(-dev)
        for reb in [8,24]:out[f'PREM_DEV_h{h}_r{reb}']=m.weights_from_signal(sig,reb)
    # Premium change reversal: fade rapid expansion/compression of basis.
    for h in [12,24,72]:
        ch=P.shift(1)-P.shift(1+h);sd=P.shift(1).rolling(168,min_periods=84).std().replace(0,np.nan);sig=m.cs_z(-ch/sd)
        for reb in [8,24]:out[f'PREM_CHG_h{h}_r{reb}']=m.weights_from_signal(sig,reb)
    return out

def corr(a,b,a0,z0):return v11.corr(a,b,a0,z0)
def main():
    raw={};fund={};prem={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        fs={ex.submit(v11.load_kline_full,s):s for s in m.SYMS}
        for f in as_completed(fs):raw[fs[f]]=f.result()
    with ThreadPoolExecutor(max_workers=9) as ex:
        fs={ex.submit(m.load_funding,s):s for s in m.ALTS}
        for f in as_completed(fs):fund[fs[f]]=f.result()
    with ThreadPoolExecutor(max_workers=9) as ex:
        fs={ex.submit(load_premium,s):s for s in m.ALTS}
        for f in as_completed(fs):prem[fs[f]]=f.result();print('PREM',fs[f],len(prem[fs[f]]),flush=True)
    idx=pd.date_range(m.START,m.END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in m.SYMS});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in m.SYMS});F=pd.DataFrame({s:fund[s].reindex(idx) for s in m.ALTS});P=pd.DataFrame({s:prem[s]['close'].reindex(idx) if len(prem[s]) else np.nan for s in m.ALTS})
    if P.notna().sum().sum()<10000:raise RuntimeError('premium data unavailable')
    rmsW=m.weights_from_signal(m.rms_signal(cl),48);fcdW=v11.fcd14_weights(F);seasonW=m.weights_from_signal(v12.seasonal_signal(m.residuals(cl),28),24)
    rmsEq,_=m.simulate(rmsW,op,F);fcdEq,_=m.simulate(fcdW,op,F);seasonEq,_=m.simulate(seasonW,op,F);rmsfull=m.metrics(rmsEq);rmsdev=m.submetrics(rmsEq,m.TRADE_START,m.DEV_END)
    cands=premium_candidates(P);cres={};ceq={}
    for n,W in cands.items():
        eq,_=m.simulate(W,op,F);ceq[n]=eq;cres[n]={'dev':m.submetrics(eq,m.TRADE_START,m.DEV_END),'full':m.metrics(eq),'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'corr_rms_dev':corr(eq,rmsEq,m.TRADE_START,m.DEV_END),'corr_fcd_dev':corr(eq,fcdEq,m.TRADE_START,m.DEV_END),'corr_rms_full':corr(eq,rmsEq,m.TRADE_START,m.END),'corr_fcd_full':corr(eq,fcdEq,m.TRADE_START,m.END)}
    elig=[n for n,v in cres.items() if v['dev']['Sharpe']>0 and abs(v['corr_rms_dev'])<=.35 and abs(v['corr_fcd_dev'])<=.35]
    elig.sort(key=lambda n:cres[n]['dev']['Sharpe']*(1-abs(cres[n]['corr_rms_dev']))*(1-abs(cres[n]['corr_fcd_dev'])),reverse=True);winner=elig[0] if elig else max(cres,key=lambda n:cres[n]['dev']['Sharpe'])
    premW=cands[winner];premEq=ceq[winner]
    engines={'RMS48':rmsW,'FCD14':fcdW,'PREM':premW,'SEASON28':seasonW};names=list(engines);target=.95*rmsdev['CAGR'];G=[i/20 for i in range(21)] # 5% grid
    combos=[]
    for a in G:
      for b in G:
       for c in G:
        d=round(1-a-b-c,10)
        if d<0:continue
        # RMS core >=45%; satellites each <=25%.
        if a<.45 or max(b,c,d)>.25:continue
        W=a*rmsW+b*fcdW+c*premW+d*seasonW;eq,cost=m.simulate(W,op,F);dev=m.submetrics(eq,m.TRADE_START,m.DEV_END);full=m.metrics(eq)
        combos.append({'weights':{'RMS48':a,'FCD14':b,'PREM':c,'SEASON28':d},'dev':dev,'full':full,'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'cost':cost})
    feas=[x for x in combos if x['dev']['CAGR']>=target];best=min(feas,key=lambda x:(abs(x['dev']['MaxDD']),-x['dev']['CAGR'])) if feas else max(combos,key=lambda x:x['dev']['Calmar'])
    full_goal=[x for x in combos if x['full']['CAGR']>=.95*rmsfull['CAGR'] and x['full']['MaxDD']>-.15]
    out={'goal':'CAGR>=95% RMS and MDD<15%; selection DEV only','premium_winner':winner,'premium_results':cres,'baseline_rms':{'dev':rmsdev,'full':rmsfull},'premium_winner_corr':{'rms_dev':corr(premEq,rmsEq,m.TRADE_START,m.DEV_END),'fcd_dev':corr(premEq,fcdEq,m.TRADE_START,m.DEV_END),'rms_full':corr(premEq,rmsEq,m.TRADE_START,m.END),'fcd_full':corr(premEq,fcdEq,m.TRADE_START,m.END)},'best_dev_selected':best,'dev_goal_met':bool(best['dev']['CAGR']>=target and best['dev']['MaxDD']>-.15),'full_goal_count_descriptive_only':len(full_goal),'full_goal_best_descriptive_only':sorted(full_goal,key=lambda x:(abs(x['full']['MaxDD']),-x['full']['CAGR']))[:10],'n_combos':len(combos)}
    Path('alpha_v14_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v14_output/summary.json','w'),indent=2,allow_nan=True);print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
