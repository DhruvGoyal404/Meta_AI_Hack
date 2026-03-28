"""
Dataset Factory
===============
Generates deterministic dirty + expected DataFrames for each task.
Uses numpy default_rng(seed) — every (task_id, seed) pair is identical.

Task 4 is the novel one: also exposes generate_drift_batch() which the
environment calls every DRIFT_EVERY steps to inject fresh dirty rows
mid-episode, simulating a live streaming pipeline under data drift.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict


def make_task(task_id: str, seed: int) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    if task_id == "task1":
        return _task1(seed)
    elif task_id == "task2":
        return _task2(seed)
    elif task_id == "task3":
        return _task3(seed)
    elif task_id == "task4_data_drift":
        return _task4(seed)
    raise ValueError(f"Unknown task_id: {task_id!r}.")


# ── Task 1 ────────────────────────────────────────────────────────────────────

def _task1(seed: int):
    rng = np.random.default_rng(seed)
    n = 50
    ids   = list(range(1, n + 1))
    names = [f"Customer_{i:03d}" for i in range(n)]
    ages  = rng.integers(18, 75, size=n).tolist()
    sals  = np.round(rng.uniform(30_000, 120_000, size=n), 2).tolist()
    cities = rng.choice(["Mumbai","Delhi","Bangalore","Chennai","Pune"], size=n).tolist()

    null_age = set(rng.choice(n, size=10, replace=False).tolist())
    null_sal = set(rng.choice(n, size=8,  replace=False).tolist())
    markers  = ["", "N/A", "null", "missing", "NaN"]

    age_d = [str(ages[i]) if i not in null_age else str(rng.choice(markers)) for i in range(n)]
    sal_d = [sals[i] if i not in null_sal else None for i in range(n)]

    dirty = pd.DataFrame({"id": ids, "name": names, "age": age_d, "salary": sal_d, "city": cities})

    age_fill = int(np.median([ages[i] for i in range(n) if i not in null_age]))
    sal_fill = round(float(np.mean([sals[i] for i in range(n) if i not in null_sal])), 2)

    expected = pd.DataFrame({
        "id": ids, "name": names,
        "age":    pd.array([ages[i] if i not in null_age else age_fill for i in range(n)], dtype="int64"),
        "salary": pd.array([round(sals[i],2) if i not in null_sal else sal_fill for i in range(n)], dtype="float64"),
        "city": cities,
    })
    return {"main": dirty}, {"main": expected}


# ── Task 2 ────────────────────────────────────────────────────────────────────

def _task2(seed: int):
    rng = np.random.default_rng(seed)
    nu = 170

    ids  = list(range(1, nu + 1))
    cids = rng.integers(1001, 1200, size=nu).tolist()
    amts = np.round(rng.uniform(10, 5_000, size=nu), 2).tolist()
    null_a = set(rng.choice(nu, size=12, replace=False).tolist())
    stats  = rng.choice(["completed","pending","cancelled","refunded"], size=nu).tolist()
    cats   = rng.choice(["Electronics","Clothing","Food","Books","Sports"], size=nu).tolist()

    CV = {
        "USA":["USA","usa","U.S.A","United States","US"],
        "UK":["UK","uk","U.K.","United Kingdom"],
        "INDIA":["India","india","INDIA","IN"],
        "GERMANY":["Germany","germany","DE","GERMANY"],
        "FRANCE":["France","france","FR","FRANCE"],
    }
    ckeys   = list(CV.keys())
    cc      = rng.choice(ckeys, size=nu).tolist()
    cd      = [str(rng.choice(CV[c])) for c in cc]

    dr   = pd.date_range("2023-01-01","2024-12-31", periods=nu)
    diso = dr.strftime("%Y-%m-%d").tolist()
    dd   = [pd.Timestamp(d).strftime("%d/%m/%Y") if rng.random()<0.35 else d for d in diso]
    ad   = [amts[i] if i not in null_a else None for i in range(nu)]

    base = pd.DataFrame({"order_id":ids,"customer_id":cids,"country":cd,
                         "amount":ad,"order_date":dd,"status":stats,"product_category":cats})
    dups = base.iloc[rng.choice(nu, size=30, replace=True)].copy()
    dirty = (pd.concat([base, dups], ignore_index=True)
             .sample(frac=1, random_state=int(seed)).reset_index(drop=True))

    af = round(float(np.mean([amts[i] for i in range(nu) if i not in null_a])), 2)
    expected = pd.DataFrame({
        "order_id":ids,"customer_id":cids,"country":cc,
        "amount":pd.array([round(amts[i],2) if i not in null_a else af for i in range(nu)], dtype="float64"),
        "order_date":pd.to_datetime(diso),"status":stats,"product_category":cats,
    })
    return {"main": dirty}, {"main": expected}


# ── Task 3 ────────────────────────────────────────────────────────────────────

def _task3(seed: int):
    rng = np.random.default_rng(seed)
    nc, no = 100, 300

    cids  = list(range(1001, 1001+nc))
    cname = [f"Customer_{i:03d}" for i in range(nc)]
    ctry  = rng.choice(["USA","UK","India","Germany"], size=nc).tolist()
    ages  = rng.integers(18, 70, size=nc).tolist()
    nai   = set(rng.choice(nc, size=8, replace=False).tolist())
    ages_d = [str(ages[i]) if i not in nai else "N/A" for i in range(nc)]

    cust_dirty = pd.DataFrame({"customer_id":cids,"name":cname,"country":ctry,"age":ages_d})
    af = int(np.median([ages[i] for i in range(nc) if i not in nai]))
    cust_clean = pd.DataFrame({
        "customer_id":cids,"name":cname,"country":ctry,
        "age":pd.array([ages[i] if i not in nai else af for i in range(nc)], dtype="int64"),
    })

    oids = list(range(1, no+1))
    ocid = rng.choice(cids, size=no).tolist()
    amts = np.round(rng.uniform(10, 2_000, size=no), 2)
    for idx in rng.choice(no, size=20, replace=False):
        amts[idx] = float(rng.choice([0.01, -5.0, 50_000.0, 99_999.0]))
    dates = pd.date_range("2023-01-01","2024-12-31", periods=no).strftime("%Y-%m-%d").tolist()

    orders_dirty = pd.DataFrame({"order_id":oids,"customer_id":ocid,
                                 "amount":amts.tolist(),"order_date":dates})
    merged = pd.merge(orders_dirty, cust_clean, on="customer_id", how="inner")
    Q1, Q3 = merged["amount"].quantile(0.25), merged["amount"].quantile(0.75)
    IQR = Q3 - Q1
    mc = merged[(merged["amount"]>=Q1-1.5*IQR) & (merged["amount"]<=Q3+1.5*IQR)].copy().reset_index(drop=True)
    mc["order_year"] = pd.to_datetime(mc["order_date"]).dt.year

    return ({"orders": orders_dirty, "customers": cust_dirty}, {"main": mc})


# ── Task 4: Data Drift (Expert) ───────────────────────────────────────────────

def _task4(seed: int):
    """
    Live streaming transactions — 120 initial dirty rows.
    Env injects fresh dirty rows every DRIFT_EVERY=5 steps via generate_drift_batch().
    Agent must keep cleaning as new dirty data continuously arrives.

    Columns: txn_id, customer_id, amount, category, region, event_ts
    Dirty issues: nulls, wrong dtypes (amount as str), outliers, mixed timestamp formats.
    """
    rng = np.random.default_rng(seed)
    n = 120

    txn_ids  = [f"TXN_INIT_{i:04d}" for i in range(n)]
    cids     = rng.integers(1, 501, size=n).tolist()
    cats_c   = rng.choice(["Electronics","Clothing","Food","Books","Sports","Toys"], size=n).tolist()
    regs_c   = rng.choice(["North","South","East","West","Central"], size=n).tolist()
    amts_t   = np.round(rng.uniform(10, 3000, size=n), 2).tolist()

    amts_d = []
    for i in range(n):
        r = rng.random()
        if r < 0.15:   amts_d.append(None)
        elif r < 0.22: amts_d.append(str(round(amts_t[i], 2)))
        elif r < 0.27: amts_d.append(float(-rng.uniform(100, 5000)))
        elif r < 0.31: amts_d.append(float(rng.uniform(80000, 250000)))
        else:          amts_d.append(amts_t[i])

    def _ts(rng):
        base = (f"2024-{rng.integers(1,13):02d}-{rng.integers(1,29):02d} "
                f"{rng.integers(0,24):02d}:{rng.integers(0,60):02d}:00")
        r = rng.random()
        if r < 0.20: return base.split(" ")[0].replace("-", "/")
        if r < 0.30:
            p = base.split("-"); return f"{p[2][:2]}/{p[1]}/{p[0]}"
        return base

    ts_d   = [_ts(rng) for _ in range(n)]
    cats_d = [None if rng.random()<0.15 else cats_c[i] for i in range(n)]
    regs_d = [None if rng.random()<0.10 else regs_c[i] for i in range(n)]

    dirty = pd.DataFrame({
        "txn_id":txn_ids, "customer_id":cids,
        "amount":amts_d, "category":cats_d, "region":regs_d, "event_ts":ts_d,
    })

    # Expected: cleaned initial batch (outliers dropped, nulls filled, ts parsed)
    good_amts = [x for x in amts_t if 0 < x <= 10000]
    amt_fill  = round(float(np.mean(good_amts)), 2)
    amts_e = []
    for a in amts_d:
        if a is None:                               amts_e.append(amt_fill)
        elif isinstance(a, str):                    amts_e.append(float(a))
        elif isinstance(a, float) and (a<0 or a>10000): amts_e.append(None)
        else:                                       amts_e.append(round(a, 2))

    exp_df = pd.DataFrame({
        "txn_id":txn_ids, "customer_id":cids,
        "amount":pd.to_numeric(amts_e, errors="coerce"),
        "category":cats_c, "region":regs_c,
        "event_ts":pd.to_datetime(ts_d, errors="coerce"),
    }).dropna(subset=["amount"]).reset_index(drop=True)

    return {"stream": dirty}, {"stream": exp_df}


# ── Drift Batch Generator ─────────────────────────────────────────────────────

def generate_drift_batch(seed: int, batch_num: int, n_rows: int = 7) -> pd.DataFrame:
    """
    Generate a fresh batch of dirty rows injected mid-episode into task4.
    Called by DataCleanEnvironment.step() every DRIFT_EVERY steps.

    Fully deterministic: (seed, batch_num) always → same batch.
    Each batch introduces different dirty patterns so the agent faces novel problems.
    """
    rng = np.random.default_rng(seed * 1000 + batch_num)

    txn_ids  = [f"TXN_DRIFT_{batch_num:03d}_{i:02d}" for i in range(n_rows)]
    cids     = rng.integers(1, 501, size=n_rows).tolist()
    cats     = rng.choice(["Electronics","Clothing","Food","Books","Sports","Toys"], size=n_rows).tolist()
    regs     = rng.choice(["North","South","East","West","Central"], size=n_rows).tolist()

    amts = []
    for _ in range(n_rows):
        r    = rng.random()
        base = round(float(rng.uniform(10, 3000)), 2)
        if r < 0.20:   amts.append(None)
        elif r < 0.30: amts.append(str(base))
        elif r < 0.38: amts.append(float(-rng.uniform(100, 5000)))
        elif r < 0.44: amts.append(float(rng.uniform(80000, 250000)))
        else:          amts.append(base)

    ts = []
    for _ in range(n_rows):
        base = (f"2024-{rng.integers(1,13):02d}-{rng.integers(1,29):02d} "
                f"{rng.integers(0,24):02d}:{rng.integers(0,60):02d}:00")
        r = rng.random()
        if r < 0.20:   ts.append(base.split(" ")[0].replace("-", "/"))
        elif r < 0.35: p = base.split("-"); ts.append(f"{p[2][:2]}/{p[1]}/{p[0]}")
        else:          ts.append(base)

    for i in range(n_rows):
        if rng.random() < 0.18: cats[i] = None
        if rng.random() < 0.12: regs[i] = None

    return pd.DataFrame({
        "txn_id":txn_ids, "customer_id":cids,
        "amount":amts, "category":cats, "region":regs, "event_ts":ts,
    })