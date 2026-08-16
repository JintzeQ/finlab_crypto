#!/usr/bin/env python3
import io, json, math, zipfile, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

UNIVERSE=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','AVAXUSDT','SUIUSDT']
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
BASE='https://data.binance.vision/data/futures/um/monthly/klines'
START='2022-01-01'; END='2026-07-31'; WIN=168; MINP=72
VAL0=pd.Timestamp('2025-01-01',tz='UTC'); VAL1=pd.Timestamp('2026-01-01',tz='UTC')
TEST0=pd.Timestamp('2026-01-01',tz='UTC'); TEST1=pd.Timestamp('2026-08-01',tz='UTC')
HOURS_YR=365.25*24

def months(a,b): return [str(x) for x in pd.period_range(pd.Period(a,'M'),pd.Period(b,'M'),freq='M')]

def load_symbol(sym):
    fs=[]
    for ym in months(START,END):
        url=f'{BASE}/{sym}/1h/{sym}-1h-{ym}.zip'
        try:
            with urllib.request.urlopen(url,timeout=45) as r: rawzip=r.read()
        except urllib.error.HTTPError as e:
            if e.code==404: continue
            raise
        with zipfile.ZipFile(io.BytesIO(rawzip)) as z:
            raw=z.read(z.namelist()[0]); fs.append(pd.read_csv(io.BytesIO(raw),header=None,names=COLS))
    d=pd.concat(fs,ignore_index=True)
    ot=pd.to_numeric(d.open_time,errors='coerce'); ms=ot.where(ot<1e15,ot/1000.0)
    d['date']=pd.to_datetime(ms,unit='ms',utc=True,errors='coerce')
    for c in ['open','high','low','close','quote_volume','taker_buy_quote']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['date','open','high','low','close','quote_volume','taker_buy_quote']).drop_duplicates('date').sort_values('date').set_index('date')
    return d[(d.index>=pd.Timestamp(START,tz='UTC'))&(d.index<TEST1)]

def robust_z(s):
    med=s.rolling(WIN,min_periods=MINP).median().shift(1)
    dev=(s-med).abs(); mad=dev.rolling(WIN,min_periods=MINP).median().shift(1)
    sc=1.4826*mad; st=s.rolling(WIN,min_periods=MINP).std().shift(1)
    sc=sc.where(sc>1e-12,st)
    return ((s-med)/sc.replace(0,np.nan)).clip(-8,8)

def prep(d):
    x=pd.DataFrame(index=d.index)
    x['open']=d.open; x['close']=d.close
    x['ret1']=d.close.pct_change(); x['ret3']=d.close.pct_change(3); x['ret6']=d.close.pct_change(6); x['ret12']=d.close.pct_change(12)
    x['range']=(d.high-d.low)/d.open.replace(0,np.nan); x['logq']=np.log1p(d.quote_volume.clip(lower=0))
    x['imb']=(2*d.taker_buy_quote/d.quote_volume.replace(0,np.nan)-1).clip(-1,1)
    x['eff']=((d.close-d.open).abs()/(d.high-d.low).replace(0,np.nan)).clip(0,1)
    x['zr1']=robust_z(x.ret1); x['zr3']=robust_z(x.ret3); x['zr6']=robust_z(x.ret6)
    x['zrange']=robust_z(x.range); x['zvol']=robust_z(x.logq)
    x['exec1']=d.open.shift(-2)/d.open.shift(-1)-1
    return x

def cs_z(df):
    mu=df.mean(axis=1); sd=df.std(axis=1,ddof=1).replace(0,np.nan)
    return df.sub(mu,axis=0).div(sd,axis=0).clip(-4,4)

def ewma_matrix(x,span): return x.ewm(span=span,adjust=False,min_periods=1).mean() if span>1 else x

def make_weights(sig, rebalance=1, threshold=0.6, max_side=2):
    idx=sig.index; cols=sig.columns; W=pd.DataFrame(0.0,index=idx,columns=cols); prev=pd.Series(0.0,index=cols)
    for i,t in enumerate(idx):
        if i%rebalance!=0:
            W.loc[t]=prev; continue
        a=sig.loc[t].dropna()
        if len(a)<8:
            prev[:]=0; W.loc[t]=prev; continue
        z=(a-a.mean())/a.std(ddof=1) if a.std(ddof=1)>0 else a*0
        pos=z[z>=threshold].sort_values(ascending=False).head(max_side)
        neg=z[z<=-threshold].sort_values().head(max_side)
        w=pd.Series(0.0,index=cols)
        if len(pos)>0 and len(neg)>0:
            w.loc[pos.index]=0.5/len(pos); w.loc[neg.index]=-0.5/len(neg)
        prev=w; W.loc[t]=w
    return W

def perf(r,turn=None):
    r=pd.Series(r).replace([np.inf,-np.inf],np.nan).dropna(); n=len(r)
    if n<100:return {'Hours':n,'Sharpe':np.nan}
    eq=(1+r).cumprod(); sd=r.std(ddof=1); sh=r.mean()/sd*math.sqrt(HOURS_YR) if sd>0 else np.nan
    peak=eq.cummax(); mdd=float((eq/peak-1).min()); yrs=n/HOURS_YR
    cagr=float(eq.iloc[-1]**(1/yrs)-1) if eq.iloc[-1]>0 else -1.0
    pos=r[r>0].sum(); neg=-r[r<0].sum(); pf=float(pos/neg) if neg>0 else np.nan
    o={'Hours':n,'TotalReturn':float(eq.iloc[-1]-1),'CAGR':cagr,'Sharpe':float(sh),'MaxDD':mdd,'HitRate':float((r>0).mean()),'ProfitFactor':pf,'AvgRet_bps':float(r.mean()*1e4)}
    if turn is not None:o['AvgTurnover']=float(turn.reindex(r.index).mean())
    return o

def eval_cfg(sig,R,reb,thr,cost_bps,sl):
    W=make_weights(sig,reb,thr); gross=(W*R).sum(axis=1); turn=W.diff().abs().sum(axis=1); turn.iloc[0]=W.iloc[0].abs().sum()
    net=gross-turn*(cost_bps/1e4)
    return perf(net.loc[sl],turn.loc[sl]), perf(gross.loc[sl],turn.loc[sl])

def main():
    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        fut={ex.submit(load_symbol,s):s for s in UNIVERSE}
        for f in as_completed(fut):
            s=fut[f]; d=f.result(); data[s]=prep(d); print('DATA',s,len(d),d.index.min(),d.index.max(),flush=True)
    common=sorted(set().union(*[set(x.index) for x in data.values()]))
    idx=pd.DatetimeIndex(common)
    mats={k:pd.concat({s:x[k] for s,x in data.items()},axis=1).reindex(idx) for k in ['ret1','ret3','ret6','zr1','zr3','zr6','zrange','zvol','imb','eff','exec1']}
    R=mats['exec1']
    cs1=cs_z(mats['ret1']); cs3=cs_z(mats['ret3']); cs6=cs_z(mats['ret6'])
    same=(np.sign(mats['ret1'])*mats['imb']).clip(lower=0)
    zv=mats['zvol'].clip(lower=0); zr=mats['zrange'].clip(lower=0)
    signals={
      'exhaustion': -mats['zr1']*zr*zv*same,
      'absorption': -np.sign(mats['imb'])*mats['imb'].abs()*(1-mats['eff'])*zv,
      'cs_rev1': -cs1,
      'cs_rev3': -cs3,
      'cs_rev6': -cs6,
      'cs_rev1_flow': -cs1*same*(1+zv),
      'imb_rev': -mats['imb']*(1-mats['eff'])*(1+zv),
      'imb_cont': mats['imb']*mats['eff']*(1+zv),
      'efficient_cont': np.sign(mats['ret1'])*mats['eff']*(1+zr)*(1+zv),
      'mom3': cs3,
      'mom6': cs6,
    }
    # Fixed candidate space before validation: 11 families x 3 smoothing x 3 rebalance x 2 thresholds = 198 configs.
    val_rows=[]
    val_slice=slice(VAL0,VAL1-pd.Timedelta(hours=1))
    for name,base in signals.items():
      for span in [1,3,6]:
        sig=ewma_matrix(base,span)
        for reb in [1,3,6]:
          for thr in [0.6,1.0]:
            net5,gross=eval_cfg(sig,R,reb,thr,5,val_slice)
            val_rows.append({'family':name,'span':span,'reb':reb,'thr':thr,'val_net5_sharpe':net5.get('Sharpe',np.nan),'val_net5_cagr':net5.get('CAGR',np.nan),'val_net5_mdd':net5.get('MaxDD',np.nan),'val_turn':net5.get('AvgTurnover',np.nan),'val_gross_sharpe':gross.get('Sharpe',np.nan)})
    V=pd.DataFrame(val_rows).sort_values(['val_net5_sharpe','val_net5_cagr'],ascending=False).reset_index(drop=True)
    # Selection rule fixed: highest validation 5bps Sharpe, requiring positive validation CAGR and <=0.45 avg hourly turnover if possible.
    eligible=V[(V.val_net5_cagr>0)&(V.val_turn<=0.45)&np.isfinite(V.val_net5_sharpe)]
    winner=(eligible.iloc[0] if len(eligible) else V.iloc[0]).to_dict()
    fam=winner['family']; span=int(winner['span']); reb=int(winner['reb']); thr=float(winner['thr'])
    sig=ewma_matrix(signals[fam],span)
    test_slice=slice(TEST0,TEST1-pd.Timedelta(hours=1)); val_slice=slice(VAL0,VAL1-pd.Timedelta(hours=1))
    detail={'winner_config':{'family':fam,'span':span,'rebalance_h':reb,'threshold':thr},'validation_selection':winner,'top_validation':V.head(15).to_dict(orient='records'),'validation':{},'test':{}}
    for cb in [0,2,3,5,8]:
        vn,vg=eval_cfg(sig,R,reb,thr,cb,val_slice); tn,tg=eval_cfg(sig,R,reb,thr,cb,test_slice)
        detail['validation'][str(cb)]={'net':vn,'gross':vg}; detail['test'][str(cb)]={'net':tn,'gross':tg}
    # Year/month stability on untouched test at 5bps
    W=make_weights(sig,reb,thr); gross=(W*R).sum(axis=1); turn=W.diff().abs().sum(axis=1); turn.iloc[0]=W.iloc[0].abs().sum(); net5=gross-turn*5e-4
    test_net=net5.loc[test_slice]; test_turn=turn.loc[test_slice]
    detail['test_months_5bps']={}
    for per,g in test_net.groupby(test_net.index.to_period('M')):
        detail['test_months_5bps'][str(per)]=perf(g,test_turn.reindex(g.index))
    # Coin contribution (gross) on test, for concentration diagnostic.
    contrib=(W*R).loc[test_slice].sum(); detail['test_gross_coin_contribution']= {k:float(v) for k,v in contrib.items()}
    detail['acceptance']={
      'test_net5_sharpe_gt_1': bool(detail['test']['5']['net'].get('Sharpe',-999)>1.0),
      'test_net5_cagr_positive': bool(detail['test']['5']['net'].get('CAGR',-999)>0),
      'test_net8_sharpe_positive': bool(detail['test']['8']['net'].get('Sharpe',-999)>0),
      'val_net5_sharpe_positive': bool(detail['validation']['5']['net'].get('Sharpe',-999)>0),
    }
    detail['accepted']=all(detail['acceptance'].values())
    Path('alpha_v2_output').mkdir(exist_ok=True)
    with open('alpha_v2_output/summary.json','w') as f: json.dump(detail,f,indent=2,allow_nan=True)
    V.to_csv('alpha_v2_output/validation_grid.csv',index=False)
    pd.DataFrame({'gross':gross,'net5':net5,'turnover':turn,'equity5':(1+net5.fillna(0)).cumprod()}).to_csv('alpha_v2_output/winner_equity.csv')
    print('===ALPHA_V2_SUMMARY===',flush=True); print(json.dumps(detail,indent=2,allow_nan=True),flush=True)

if __name__=='__main__': main()
