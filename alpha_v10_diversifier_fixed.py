#!/usr/bin/env python3
import io,zipfile,numpy as np,pandas as pd
import alpha_v10_diversifier as m

def robust_load_funding(s):
    fs=[]
    for ym in m.months():
        b=m.getzip(f'{m.ROOT}/fundingRate/{s}/{s}-fundingRate-{ym}.zip')
        if b is None: continue
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            x=pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),header=None)
            fs.append(x)
    if not fs:return pd.Series(dtype=float)
    d=pd.concat(fs,ignore_index=True)
    nums={c:pd.to_numeric(d[c],errors='coerce') for c in d.columns}
    # timestamp: column whose non-null median has Unix-ms/us magnitude
    tcol=max(nums,key=lambda c: float(nums[c].dropna().median()) if len(nums[c].dropna()) else -np.inf)
    t=nums[tcol]
    # funding rate: among other numeric columns, require plausible rate scale; prefer median abs closest to 1e-4
    cand=[]
    for c,v in nums.items():
        if c==tcol: continue
        a=v.dropna().abs()
        if len(a)<10: continue
        q99=float(a.quantile(.99)); med=float(a.median()); nz=float((a>0).mean())
        if q99<=0.10 and nz>0.01:
            cand.append((abs(np.log10(max(med,1e-12))-np.log10(1e-4)),c))
    if not cand: raise RuntimeError(f'No plausible funding-rate column for {s}')
    _,rcol=min(cand)
    r=nums[rcol]
    if float(r.dropna().abs().max())>0.10: raise RuntimeError(f'Implausible funding rate {s}')
    tt=t.where(t<1e15,t/1000)
    idx=pd.to_datetime(tt,unit='ms',utc=True,errors='coerce')
    out=pd.Series(r.values,index=idx).dropna().groupby(level=0).last().sort_index()
    print('FUND_PARSE',s,'tcol',tcol,'rcol',rcol,'median_bps',float(out.median()*1e4),'max_bps',float(out.abs().max()*1e4),flush=True)
    return out

m.load_funding=robust_load_funding
m.main()
