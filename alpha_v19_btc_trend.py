#!/usr/bin/env python3
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
import alpha_v12_portfolio as v12
m=v12.m; v11=v12.v11; m.load_funding=v11.robust_load_funding
ALL=['BTCUSDT']+m.ALTS

def btc_trend_weight(cl):
    p=cl['BTCUSDT']; lr=np.log(p).diff()
    # Fixed diversified horizons: 7d, 14d, 30d. No horizon selection.
    votes=[]
    for h in [168,336,720]: votes.append(np.sign(np.log(p/p.shift(h))))
    vote=pd.concat(votes,axis=1).mean(axis=1)
    annvol=lr.shift(1).rolling(336,min_periods=168).std()*np.sqrt(8760)
    # Fixed 30% annual vol target, max 1x notional for the BTC sleeve.
    mag=(0.30/annvol.replace(0,np.nan)).clip(upper=1.0)
    raw=(vote*mag).fillna(0)
    # Phase invariant daily rebalance: average all 24 daily phases.
    vals=raw.to_numpy(float);acc=np.zeros(len(vals))
    for ph in range(24):
      o=np.zeros(len(vals));prev=0.
      for i,x in enumerate(vals):
        if (i-ph)%24==0: prev=x
        o[i]=prev
      acc+=o
    return pd.Series(acc/24,index=raw.index,name='BTCUSDT')

def joint_sim(altW,btcW,op,F,scalar=1.0,fee=5,slip=2,min_delta=5):
    idx=op.index; P=op[ALL].to_numpy(float)
    W=pd.DataFrame(0.,index=idx,columns=ALL);W[m.ALTS]=altW[m.ALTS];W['BTCUSDT']=btcW.reindex(idx).fillna(0);T=(W*scalar).shift(1).fillna(0).to_numpy(float)
    FR=F.reindex(idx).fillna(0).to_numpy(float);q=np.zeros(len(ALL));eq=np.full(len(idx),np.nan);equity=m.INITIAL;prev=np.full(len(ALL),np.nan);fees=slips=funds=0.
    for k in range(len(idx)):
      p=P[k];valid=np.isfinite(p)
      if k>0:
        both=valid&np.isfinite(prev);equity+=float(np.nansum(q[both]*(p[both]-prev[both])))
      fr=FR[k];fm=valid&np.isfinite(fr)&(fr!=0)
      if fm.any():
        pay=float(np.nansum(q[fm]*p[fm]*fr[fm]));equity-=pay;funds+=pay
      if m.TRADE_START<=idx[k]<m.END and equity>0:
        tw=T[k].copy();tw[~valid]=0;desired=np.zeros(len(ALL));desired[valid]=tw[valid]*equity/p[valid];delta=desired-q;dn=np.abs(delta)*np.where(valid,p,0);ex=valid&(dn>=min_delta-1e-12);tr=float(dn[ex].sum());equity-=tr*(fee+slip)/1e4;fees+=tr*fee/1e4;slips+=tr*slip/1e4;q[ex]=desired[ex]
      eq[k]=equity;prev=p.copy()
    ser=pd.Series(eq,index=idx).loc[(idx>=m.TRADE_START)&(idx<m.END)];return ser,{'fees':fees,'slippage':slips,'funding_paid':funds}

def corr(a,b,a0,b0):
    x=pd.concat([a.pct_change(),b.pct_change()],axis=1).loc[(a.index>=a0)&(a.index<b0)].dropna();return float(x.iloc[:,0].corr(x.iloc[:,1]))

def main():
  raw={};fund={}
  with ThreadPoolExecutor(max_workers=10) as ex:
    fs={ex.submit(v11.load_kline_full,s):s for s in ALL}
    for f in as_completed(fs):raw[fs[f]]=f.result()
  with ThreadPoolExecutor(max_workers=10) as ex:
    fs={ex.submit(m.load_funding,s):s for s in ALL}
    for f in as_completed(fs):fund[fs[f]]=f.result()
  idx=pd.date_range(m.START,m.END-pd.Timedelta(hours=1),freq='h');cl=pd.DataFrame({s:raw[s].close.reindex(idx) for s in ALL});op=pd.DataFrame({s:raw[s].open.reindex(idx) for s in ALL});F=pd.DataFrame({s:fund[s].reindex(idx) for s in ALL})
  rms=m.weights_from_signal(m.rms_signal(cl),48);fcd=v11.fcd14_weights(F[m.ALTS]);zero=pd.DataFrame(0.,index=idx,columns=m.ALTS);bt=btc_trend_weight(cl)
  rmsEq,_=joint_sim(rms,pd.Series(0.,index=idx),op,F);fcdEq,_=joint_sim(fcd,pd.Series(0.,index=idx),op,F);btEq,btCost=joint_sim(zero,bt,op,F)
  btres={'dev':m.submetrics(btEq,m.TRADE_START,m.DEV_END),'full':m.metrics(btEq),'holdout1':m.submetrics(btEq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(btEq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(btEq,m.CONF_END,m.END),'corr_rms_dev':corr(btEq,rmsEq,m.TRADE_START,m.DEV_END),'corr_fcd_dev':corr(btEq,fcdEq,m.TRADE_START,m.DEV_END),'corr_rms_full':corr(btEq,rmsEq,m.TRADE_START,m.END),'corr_fcd_full':corr(btEq,fcdEq,m.TRADE_START,m.END),'cost':btCost}
  rdev=m.submetrics(rmsEq,m.TRADE_START,m.DEV_END);rfull=m.metrics(rmsEq);alloc={};target=.95*rdev['CAGR']
  # Fixed allocation grid, no strategy tuning. RMS>=50%, FCD/BTC trend each <=30%, scalar <=1.3.
  for wf in np.arange(0,.31,.05):
   for wb in np.arange(.05,.31,.05):
    wr=1-wf-wb
    if wr<.50:continue
    alt=wr*rms+wf*fcd;btc=wb*bt
    for lev in [1.0,1.1,1.2,1.3]:
      eq,cost=joint_sim(alt,btc,op,F,lev);n=f'R{wr:.2f}_F{wf:.2f}_B{wb:.2f}_L{lev:.1f}';alloc[n]={'weights':{'RMS':wr,'FCD':wf,'BTCtrend':wb,'scalar':lev},'dev':m.submetrics(eq,m.TRADE_START,m.DEV_END),'full':m.metrics(eq),'holdout1':m.submetrics(eq,m.DEV_END,m.HOLD1_END),'confirm':m.submetrics(eq,m.HOLD1_END,m.CONF_END),'secondary2026':m.submetrics(eq,m.CONF_END,m.END),'cost':cost}
  ok=[n for n,v in alloc.items() if v['dev']['CAGR']>=target];chosen=min(ok,key=lambda n:(abs(alloc[n]['dev']['MaxDD']),-alloc[n]['dev']['CAGR'],-alloc[n]['dev']['Sharpe'])) if ok else max(alloc,key=lambda n:alloc[n]['dev']['Calmar'])
  dg=[n for n,v in alloc.items() if v['dev']['CAGR']>=target and v['dev']['MaxDD']>-.15];fg=[n for n,v in alloc.items() if v['full']['CAGR']>=.95*rfull['CAGR'] and v['full']['MaxDD']>-.15]
  out={'goal':'CAGR>=95% RMS and MDD<15%; DEV-only allocation','baseline_rms':{'dev':rdev,'full':rfull},'btc_trend':btres,'btc_strategy':'fixed equal-vote sign of 168/336/720h BTC log return; 30% ann-vol target cap1x; 24-phase daily stagger','chosen':chosen,'chosen_result':alloc[chosen],'dev_goal_count':len(dg),'dev_goal_names':dg,'full_goal_count_descriptive_only':len(fg),'full_goal_names_descriptive_only':fg,'all_allocations':alloc,'selection':'DEV min abs MDD subject CAGR>=95% RMS DEV'}
  Path('alpha_v19_output').mkdir(exist_ok=True);json.dump(out,open('alpha_v19_output/summary.json','w'),indent=2,allow_nan=True);print(json.dumps(out,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
