from __future__ import annotations
import argparse, json, math, urllib.request, zipfile
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"
COLS = ["agg_trade_id","price","quantity","first_trade_id","last_trade_id","transact_time","is_buyer_maker"]

def daterange(start: str, end: str):
    d0 = date.fromisoformat(start); d1 = date.fromisoformat(end); d = d0
    while d <= d1:
        yield d.isoformat(); d += timedelta(days=1)

def _normalize_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool: return s
    return s.astype(str).str.lower().isin(["true","1","t"])

def download_and_aggregate(symbol: str, day: str, cache_dir: Path, bin_ms: int=500, chunksize: int=1_000_000) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zpath = cache_dir / f"{symbol}-aggTrades-{day}.zip"
    if not zpath.exists():
        url = BASE_URL.format(symbol=symbol, day=day)
        print(f"download {url}", flush=True)
        urllib.request.urlretrieve(url, zpath)
    parts = []
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1: raise RuntimeError(f"Expected one CSV in {zpath}, got {names}")
        with zf.open(names[0]) as raw:
            first = raw.readline().decode("utf-8", errors="ignore").strip(); raw.seek(0)
            has_header = not first.split(",")[0].strip().lstrip("-").isdigit()
            reader = pd.read_csv(raw, names=None if has_header else COLS, header=0 if has_header else None,
                                 chunksize=chunksize, low_memory=False)
            for chunk in reader:
                if has_header:
                    ren = {}
                    for c in chunk.columns:
                        lc = str(c).strip().lower()
                        if lc in {"agg_trade_id","aggtradeid","a"}: ren[c]="agg_trade_id"
                        elif lc in {"price","p"}: ren[c]="price"
                        elif lc in {"quantity","qty","q"}: ren[c]="quantity"
                        elif lc in {"first_trade_id","firsttradeid","f"}: ren[c]="first_trade_id"
                        elif lc in {"last_trade_id","lasttradeid","l"}: ren[c]="last_trade_id"
                        elif lc in {"transact_time","timestamp","time","t"}: ren[c]="transact_time"
                        elif lc in {"is_buyer_maker","isbuyermaker","m"}: ren[c]="is_buyer_maker"
                    chunk = chunk.rename(columns=ren)
                missing = [c for c in COLS if c not in chunk.columns]
                if missing: raise RuntimeError(f"Missing columns {missing}; columns={list(chunk.columns)}")
                px = pd.to_numeric(chunk["price"], errors="coerce")
                qty = pd.to_numeric(chunk["quantity"], errors="coerce")
                ts = pd.to_numeric(chunk["transact_time"], errors="coerce")
                bm = _normalize_bool(chunk["is_buyer_maker"])
                ok = px.notna() & qty.notna() & ts.notna()
                px = px[ok].astype("float64"); qty = qty[ok].astype("float64")
                ts = ts[ok].astype("int64"); bm = bm[ok]
                if ts.median() > 10**14: ts = ts // 1000
                sign = np.where(bm.to_numpy(), -1.0, 1.0)
                b = (ts.to_numpy() // bin_ms) * bin_ms
                tmp = pd.DataFrame({
                    "bin_ms": b, "price": px.to_numpy(),
                    "signed_notional": px.to_numpy()*qty.to_numpy()*sign,
                    "total_notional": px.to_numpy()*qty.to_numpy(),
                    "signed_qty": qty.to_numpy()*sign, "total_qty": qty.to_numpy(),
                    "signed_count": sign, "trade_count": 1.0,
                })
                g = tmp.groupby("bin_ms", sort=False).agg(
                    last_price=("price","last"), signed_notional=("signed_notional","sum"),
                    total_notional=("total_notional","sum"), signed_qty=("signed_qty","sum"),
                    total_qty=("total_qty","sum"), signed_count=("signed_count","sum"),
                    trade_count=("trade_count","sum"))
                parts.append(g)
    df = pd.concat(parts)
    return df.groupby(level=0).agg(
        last_price=("last_price","last"), signed_notional=("signed_notional","sum"),
        total_notional=("total_notional","sum"), signed_qty=("signed_qty","sum"),
        total_qty=("total_qty","sum"), signed_count=("signed_count","sum"),
        trade_count=("trade_count","sum")).sort_index()

def complete_grid(df: pd.DataFrame, bin_ms: int) -> pd.DataFrame:
    idx = np.arange(int(df.index.min()), int(df.index.max()) + bin_ms, bin_ms, dtype=np.int64)
    out = df.reindex(idx); out["last_price"] = out["last_price"].ffill()
    for c in ["signed_notional","total_notional","signed_qty","total_qty","signed_count","trade_count"]:
        out[c] = out[c].fillna(0.0)
    return out

def make_features(df: pd.DataFrame, bin_ms: int=500) -> pd.DataFrame:
    x = df.copy(); bars_per_s = int(round(1000/bin_ms)); w1,w5,w30=bars_per_s,5*bars_per_s,30*bars_per_s; eps=1e-12
    x["r1"] = np.log(x["last_price"]).diff()
    x["ret_5s_bps"] = np.log(x["last_price"]/x["last_price"].shift(w5))*1e4
    for name,w in [("1s",w1),("5s",w5)]:
        sn=x["signed_notional"].rolling(w,min_periods=w).sum(); tn=x["total_notional"].rolling(w,min_periods=w).sum()
        sc=x["signed_count"].rolling(w,min_periods=w).sum(); tc=x["trade_count"].rolling(w,min_periods=w).sum()
        x[f"fi_{name}"]=sn/(tn+eps); x[f"count_imb_{name}"]=sc/(tc+eps); x[f"notional_{name}"]=tn
    x["vol_30s_bps"] = x["r1"].rolling(w30,min_periods=w30).std()*1e4
    x["flow_accel"] = x["fi_1s"]-x["fi_5s"]
    s=np.sign(x["fi_1s"]); x["same_sign_3"] = s.eq(s.shift(1)) & s.eq(s.shift(2)) & s.ne(0)
    return x

def train_thresholds(train: pd.DataFrame) -> dict:
    t=train.dropna(subset=["fi_5s","fi_1s","ret_5s_bps","notional_5s"]).copy()
    q95=float(t["fi_5s"].abs().quantile(.95)); q75=float(t["fi_1s"].abs().quantile(.75)); med=float(t["notional_5s"].quantile(.5))
    cand=t[(t["fi_5s"].abs()>=q95)&(t["notional_5s"]>=med)]
    low=float((cand if len(cand)>=100 else t)["ret_5s_bps"].abs().quantile(.25))
    return {"abs_fi5_q95":q95,"abs_fi1_q75":q75,"notional5_median":med,"abs_ret5_lowimpact_q25":low}

def build_signals(x: pd.DataFrame, th: dict) -> pd.DataFrame:
    s=pd.DataFrame(index=x.index); dir5=np.sign(x["fi_5s"]).fillna(0.0); high=x["fi_5s"].abs()>=th["abs_fi5_q95"]
    s["flow_cont"]=np.where(high,dir5,0.0)
    absorb=high&(x["notional_5s"]>=th["notional5_median"])&(x["ret_5s_bps"].abs()<=th["abs_ret5_lowimpact_q25"])
    s["absorption_rev"]=np.where(absorb,-dir5,0.0)
    dir1=np.sign(x["fi_1s"]).fillna(0.0)
    pers=x["same_sign_3"].fillna(False)&(x["fi_1s"].abs()>=th["abs_fi1_q75"])&(x["fi_1s"].shift(1).abs()>=th["abs_fi1_q75"])&(x["fi_1s"].shift(2).abs()>=th["abs_fi1_q75"])
    s["flow_persistence"]=np.where(pers,dir1,0.0)
    return s

def nonoverlap_event_returns(price: pd.Series, sig: pd.Series, horizon_bars: int, latency_bars: int=1):
    p=price.to_numpy(dtype=float); s=sig.to_numpy(dtype=float); out=[]; i=0; n=len(p)
    while i<n:
        if s[i]==0 or not np.isfinite(s[i]): i+=1; continue
        ent=i+latency_bars; ex=ent+horizon_bars
        if ex>=n or not (np.isfinite(p[ent]) and np.isfinite(p[ex])): break
        out.append((i,int(np.sign(s[i])),s[i]*math.log(p[ex]/p[ent])*1e4)); i=ex+1
    return out

def summarize_strategy(name, events, cost_grid=(0.,.5,1.,2.,5.)):
    gross=np.array([e[2] for e in events],dtype=float); rows=[]
    if len(gross)==0:
        for c in cost_grid: rows.append(dict(strategy=name,per_side_cost_bps=c,n=0,mean_gross_bps=np.nan,mean_net_bps=np.nan,win_rate=np.nan,total_net_bps=np.nan,t_stat_gross=np.nan,breakeven_per_side_bps=np.nan))
        return rows
    sd=gross.std(ddof=1) if len(gross)>1 else np.nan; tstat=gross.mean()/(sd/math.sqrt(len(gross))) if np.isfinite(sd) and sd>0 else np.nan; be=gross.mean()/2
    for c in cost_grid:
        net=gross-2*c
        rows.append(dict(strategy=name,per_side_cost_bps=float(c),n=int(len(gross)),mean_gross_bps=float(gross.mean()),median_gross_bps=float(np.median(gross)),mean_net_bps=float(net.mean()),win_rate=float((net>0).mean()),total_net_bps=float(net.sum()),t_stat_gross=float(tstat) if np.isfinite(tstat) else np.nan,breakeven_per_side_bps=float(be)))
    return rows

def footprint_stats(x: pd.DataFrame, th: dict, bin_ms: int):
    y=x.dropna(subset=["fi_1s"]).copy(); f=y["fi_1s"]; strong=f.abs()>=th["abs_fi1_q75"]; s0=np.sign(f); s1=np.sign(f.shift(-1)); mask=strong&s0.ne(0)&s1.ne(0)
    rows=[{"metric":"strong_flow_next_same_sign_prob","value":float((s0[mask]==s1[mask]).mean()) if mask.sum() else np.nan},{"metric":"all_flow_next_same_sign_prob","value":float((s0[s0.ne(0)&s1.ne(0)]==s1[s0.ne(0)&s1.ne(0)]).mean())}]
    for seconds in [.5,1,2.5,5,10]:
        lag=max(1,int(round(seconds*1000/bin_ms))); rows.append({"metric":f"fi1_autocorr_{seconds:g}s","value":float(f.autocorr(lag=lag))})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--symbol",default="BTCUSDT"); ap.add_argument("--start",default="2026-08-10"); ap.add_argument("--end",default="2026-08-16"); ap.add_argument("--train-end",default="2026-08-13"); ap.add_argument("--bin-ms",type=int,default=500); ap.add_argument("--cache",default=".cache/binance"); ap.add_argument("--out",default="results"); args=ap.parse_args()
    frames=[]
    for day in daterange(args.start,args.end):
        d=download_and_aggregate(args.symbol,day,Path(args.cache),args.bin_ms); frames.append(d); print(day,len(d),flush=True)
    raw=complete_grid(pd.concat(frames).sort_index(),args.bin_ms); x=make_features(raw,args.bin_ms)
    x["day"]=pd.Series(pd.to_datetime(x.index,unit="ms",utc=True).date,index=x.index).astype(str)
    train=x[x["day"]<=args.train_end]; test=x[x["day"]>args.train_end]; th=train_thresholds(train); sig=build_signals(test,th)
    outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True); (outdir/"thresholds.json").write_text(json.dumps(th,indent=2),encoding="utf-8")
    bars_per_s=int(round(1000/args.bin_ms)); metrics=[]; event_records=[]
    for horizon_s in [1,5,30]:
        hb=horizon_s*bars_per_s
        for strat in sig.columns:
            ev=nonoverlap_event_returns(test["last_price"],sig[strat],hb,latency_bars=1); label=f"{strat}_{horizon_s}s"; metrics.extend(summarize_strategy(label,ev))
            for idx,direction,gross in ev: event_records.append({"strategy":strat,"horizon_s":horizon_s,"row":idx,"direction":direction,"gross_bps":gross})
    mdf=pd.DataFrame(metrics); mdf.to_csv(outdir/"metrics.csv",index=False); edf=pd.DataFrame(event_records)
    if len(edf):
        tidx=test.index.to_numpy(); edf["timestamp_ms"]=edf["row"].map(lambda i:int(tidx[int(i)])); edf["day"]=pd.to_datetime(edf["timestamp_ms"],unit="ms",utc=True).dt.strftime("%Y-%m-%d")
        daily=edf.groupby(["strategy","horizon_s","day"]).agg(n=("gross_bps","size"),mean_gross_bps=("gross_bps","mean"),win_rate_gross=("gross_bps",lambda s:float((s>0).mean()))).reset_index(); daily.to_csv(outdir/"daily_oos.csv",index=False); edf.to_csv(outdir/"events.csv",index=False)
    else: pd.DataFrame().to_csv(outdir/"daily_oos.csv",index=False)
    fp=footprint_stats(test,th,args.bin_ms); fp.to_csv(outdir/"footprint.csv",index=False)
    zero=mdf[mdf["per_side_cost_bps"]==0].copy(); one=mdf[mdf["per_side_cost_bps"]==1].copy()
    lines=["# BTCUSDT microstructure pilot","",f"- Data: Binance USDⓈ-M daily aggTrades, {args.start}..{args.end}",f"- Train: {args.start}..{args.train_end}; OOS: days after {args.train_end}",f"- Bar: {args.bin_ms} ms; execution latency proxy: {args.bin_ms} ms","- Price proxy: last traded price (not bid/ask mid).","- Costs: explicit sensitivity grid; spread/slippage are not separately reconstructed.","","## OOS strategy results (gross)","","| strategy | n | mean gross (bp) | gross t-stat | break-even cost/side (bp) |","|---|---:|---:|---:|---:|"]
    for _,r in zero.sort_values("mean_gross_bps",ascending=False).iterrows(): lines.append(f"| {r.strategy} | {int(r.n)} | {r.mean_gross_bps:.4f} | {r.t_stat_gross:.2f} | {r.breakeven_per_side_bps:.4f} |")
    lines += ["","## OOS at 1 bp per side","","| strategy | n | mean net (bp) | win rate | total net (bp) |","|---|---:|---:|---:|---:|"]
    for _,r in one.sort_values("mean_net_bps",ascending=False).iterrows(): lines.append(f"| {r.strategy} | {int(r.n)} | {r.mean_net_bps:.4f} | {r.win_rate:.3f} | {r.total_net_bps:.1f} |")
    lines += ["","## Flow-footprint diagnostics",""]
    for _,r in fp.iterrows(): lines.append(f"- {r.metric}: {r.value:.6f}")
    lines += ["","## Interpretation guardrails","","- This is a trade-flow pilot, not a full L2/L3 queue backtest.","- Positive gross edge is actionable only if break-even per-side cost exceeds realistic all-in execution cost.","- A footprint statistic indicates persistence/clustering, not identification of a specific counterparty or bot.",""]
    (outdir/"summary.md").write_text("\n".join(lines),encoding="utf-8"); print("\n".join(lines),flush=True)

if __name__=="__main__": main()
