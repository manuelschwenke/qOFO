import os, glob, numpy as np, pandas as pd

BASE = r"Z:\Python_Projekte\qOFO_GH\experiments\results\006_cigre_mc"
TS = os.path.join(BASE, "timeseries")
files = sorted(glob.glob(os.path.join(TS, "run_*.npz")))
print(f"{len(files)} run files")

variants = ["V1", "V2", "V3", "V4", "V5"]

zones_set = set()
for f in files:
    d = np.load(f)
    for k in d.files:
        p = k.split("__")
        if len(p) == 4 and p[0] == "Vz":
            zones_set.add(int(p[2]))
zones = sorted(zones_set)
print("zones:", zones)

# Load every npz once into memory (keys -> arrays) to avoid reopening over the share.
print("loading npz files ...", flush=True)
data = []
for i, f in enumerate(files):
    d = np.load(f)
    data.append({k: np.asarray(d[k], float) for k in d.files})
    if (i + 1) % 20 == 0:
        print(f"  loaded {i+1}/{len(files)}", flush=True)
print("loaded all; aggregating ...", flush=True)

def tukey_whiskers(v):
    """matplotlib boxplot default whiskers: most extreme datum within Q1-1.5IQR / Q3+1.5IQR."""
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    whislo = float(v[v >= lo_fence].min())
    whishi = float(v[v <= hi_fence].max())
    return whislo, whishi


def summarize(V, z, arr_min, arr_max, pool):
    q1, med, q3 = (float(x) for x in np.percentile(pool, [25, 50, 75]))
    wlo, whi = tukey_whiskers(pool)
    return {
        "variant": V, "zone": z,
        "abs_min": float(arr_min.min()), "abs_max": float(arr_max.max()),
        "whisker_lo": wlo, "whisker_hi": whi,
        "q25": q1, "median": med, "q75": q3,
        "p0.1": float(np.percentile(pool, 0.1)),
        "p0.5": float(np.percentile(pool, 0.5)),
        "p2.5": float(np.percentile(pool, 2.5)),
        "p97.5": float(np.percentile(pool, 97.5)),
        "p99.5": float(np.percentile(pool, 99.5)),
        "p99.9": float(np.percentile(pool, 99.9)),
    }


rows = []
for V in variants:
    v_all_min, v_all_max, v_pool = [], [], []
    for z in zones:
        zmins, zmaxs, zpool = [], [], []
        for d in data:
            amin = d.get(f"Vz__{V}__{z}__min")
            amax = d.get(f"Vz__{V}__{z}__max")
            amean = d.get(f"Vz__{V}__{z}__mean")
            if amin is not None:
                a = amin[np.isfinite(amin)]; zmins.append(a); zpool.append(a)
            if amax is not None:
                a = amax[np.isfinite(amax)]; zmaxs.append(a); zpool.append(a)
            if amean is not None:
                a = amean[np.isfinite(amean)]; zpool.append(a)
        if not zmins:
            continue
        zmin = np.concatenate(zmins); zmax = np.concatenate(zmaxs); zp = np.concatenate(zpool)
        rows.append(summarize(V, z, zmin, zmax, zp))
        v_all_min.append(zmin); v_all_max.append(zmax); v_pool.append(zp)
    if v_all_min:
        vm = np.concatenate(v_all_min); vx = np.concatenate(v_all_max); vp = np.concatenate(v_pool)
        rows.append(summarize(V, "ALL", vm, vx, vp))

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
out_csv = os.path.join(BASE, "voltage_min_max_per_variant.csv")
df.to_csv(out_csv, index=False)
print("\nwrote:", out_csv)

# Per-variant worst/best across zones for the central-99.8% band.
# low   = worst (lowest)  p0.1  over the variant's zones
# high  = highest         p99.9 over the variant's zones
print("\n== Per-variant central-99.8% band (worst/best of all zones) ==")
pv = df[df["zone"] != "ALL"]
summ = []
for V in variants:
    sub = pv[pv["variant"] == V]
    if sub.empty:
        continue
    lo_idx = sub["p0.1"].idxmin()
    hi_idx = sub["p99.9"].idxmax()
    summ.append({
        "variant": V,
        "low (p0.1)": round(float(sub.loc[lo_idx, "p0.1"]), 4),
        "low_zone": sub.loc[lo_idx, "zone"],
        "high (p99.9)": round(float(sub.loc[hi_idx, "p99.9"]), 4),
        "high_zone": sub.loc[hi_idx, "zone"],
        "abs_min": round(float(sub["abs_min"].min()), 4),
        "abs_max": round(float(sub["abs_max"].max()), 4),
    })
sdf = pd.DataFrame(summ)
print(sdf.to_string(index=False))
sdf.to_csv(os.path.join(BASE, "voltage_central998_per_variant.csv"), index=False)
print("\nwrote:", os.path.join(BASE, "voltage_central998_per_variant.csv"))
