"""Is the controllers' model (shared_jac.net) the plant at the start of the run?  Compare the
Jacobian's own net with the plant net handed to pre_loop_hook (loop start, t = 0)."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import sys, contextlib, dataclasses, io, warnings
import psutil; psutil.Process().cpu_affinity([34])
sys.path.insert(0, r"F:\qOFO_ref"); os.chdir(r"F:\qOFO_ref")
import numpy as np
warnings.filterwarnings("ignore")
from experiments.ch_11_case2_multisystem import variants as v
from experiments.ch_11_case2_multisystem.ref import scenarios as sc
from experiments.runners.multi_tso_dso import run_multi_tso_dso
cfg = dataclasses.replace(v.resolved("O3", 360, sc.events("E2"), 20261027), result_dir="", verbose=0)
st = {}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    run_multi_tso_dso(cfg, pre_loop_hook=lambda s: st.update(s) or True)
plant = st["net"]
tso = st["tso_controllers"]
jn = {id(c.sensitivities): c.sensitivities.net for c in tso.values()}
print("distinct Jacobian objects among TSO controllers:", len(jn), "| shared_jac is theirs:",
      all(c.sensitivities is st["shared_jac"] for c in tso.values()))
m = st["shared_jac"].net
def cmp(tab, col):
    a, b = getattr(m, tab)[col], getattr(plant, tab)[col].reindex(getattr(m, tab).index)
    if a.dtype == bool or b.dtype == bool:
        d = (a.astype(bool) != b.astype(bool)); return f"{int(d.sum())} differ" + (f" {list(a.index[d])[:8]}" if d.any() else "")
    d = (a.astype(float) - b.astype(float)).abs(); return f"max |diff| {d.max():.4g}" + (f" at {d.idxmax()}" if d.max() > 0 else "")
for tab, col in (("gen", "in_service"), ("gen", "vm_pu"), ("gen", "p_mw"), ("trafo", "in_service"), ("trafo", "tap_pos"),
                 ("trafo3w", "tap_pos"), ("shunt", "step"), ("sgen", "p_mw"), ("sgen", "q_mvar"), ("load", "p_mw"),
                 ("load", "q_mvar"), ("load", "in_service")):
    try:
        print(f"  {tab}.{col}: {cmp(tab, col)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {tab}.{col}: n/a ({exc})")
for tab in ("res_bus",):
    a, b = m.res_bus.vm_pu, plant.res_bus.vm_pu.reindex(m.res_bus.index)
    print(f"  res_bus.vm_pu: max |diff| {np.nanmax(np.abs(a - b)):.4g} pu, mean {np.nanmean(np.abs(a - b)):.4g}")
print("  sum load P: model", round(float(m.load.p_mw[m.load.in_service].sum())), "plant", round(float(plant.load.p_mw[plant.load.in_service].sum())))
print("  sum sgen P: model", round(float(m.sgen.p_mw[m.sgen.in_service].sum())), "plant", round(float(plant.sgen.p_mw[plant.sgen.in_service].sum())))
print("  gens in service: model", list(m.gen.name[m.gen.in_service]), "| plant", list(plant.gen.name[plant.gen.in_service]))
print("--- per TSO controller")
for z, c in sorted(tso.items()):
    n = c.sensitivities.net
    common = n.res_bus.index.intersection(plant.res_bus.index)
    dv = np.abs(n.res_bus.vm_pu.loc[common] - plant.res_bus.vm_pu.loc[common])
    print(f"  zone {z}: {type(c.sensitivities).__name__}, buses {len(n.bus)} (plant {len(plant.bus)}), "
          f"gens in service {list(n.gen.name[n.gen.in_service])}, max|dV| on common buses {dv.max():.4g} pu, "
          f"sum load P {float(n.load.p_mw[n.load.in_service].sum()):.0f}, sgen P {float(n.sgen.p_mw[n.sgen.in_service].sum()):.0f}")
for name in ("coordinator",):
    co = st[name]
    print("  coordinator sensitivities:", type(getattr(co, "sensitivities", None)).__name__)
print("  config: local_sensitivities_tso", st["config"].local_sensitivities_tso, "| refresh_shared_jac_on_tso",
      st["config"].refresh_shared_jac_on_tso, "| sensitivity_update_interval", st["config"].sensitivity_update_interval,
      "| numerical_h", st["config"].numerical_h)
