"""Build joint shunt models from controller caches and communicated feedback."""
import numpy as np
from controller.shunt_hybrid import BankModel, HybridContext
from sensitivity.sensitivity_updater import bank_voltage_ratio


def build_hybrid_context(tso, banks, dsos, feedback, offsets, time_s, rescale=True):
    cfg, sens, meas = tso.config, tso.sensitivities, tso._last_measurement
    vb, pcc, lines = cfg.voltage_bus_indices, cfg.pcc_trafo_indices, cfg.current_line_indices
    nv, nq, ni, ng = len(vb), len(pcc), len(lines), len(cfg.gen_indices)
    mapping = cfg.der_mapping
    nd = mapping.n_der if mapping is not None else len(cfg.der_indices)
    models = []
    for bank in banks:
        c = bank.config
        signed_step = (-1 if c.kind == "MSC" else 1)*c.q_step_mvar
        ratio = bank_voltage_ratio(meas, sens.net, c.bus_idx) if rescale else 1.
        h = np.zeros(nv+nq+ni+ng)
        values, ids = sens.compute_dV_dQ_shunt(c.bus_idx, list(vb), signed_step)
        if set(ids) != set(vb):
            raise ValueError("Hybrid voltage sensitivity has missing observed buses")
        h[:nv] = [values[ids.index(b)] for b in vb]
        h[nv:nv+nq] = [sens.compute_dQtrafo3w_hv_dQ_shunt(
            trafo3w_idx=int(t), shunt_bus_idx=c.bus_idx, q_step_mvar=signed_step) for t in pcc]
        if ni:
            values, ids = sens.compute_dI_dQ_shunt(c.bus_idx, list(lines), signed_step)
            if set(ids) != set(lines):
                raise ValueError("Hybrid current sensitivity has missing observed lines")
            h[nv+nq:nv+nq+ni] = [values[ids.index(i)] for i in lines]
        if ng:
            hg, _, _ = sens.compute_dQgen_dQ_shunt_matrix(
                cfg.gen_bus_indices, [c.bus_idx], [signed_step])
            h[-ng:] = hg[:, 0]
        h *= ratio
        dso = dsos[c.dso_id]
        ids = np.asarray(dso.config.interface_trafo_indices, dtype=int)
        if any(int(i) not in pcc for i in ids):
            raise ValueError("All supported STS interfaces must belong to this TS zone")
        rs = bank_voltage_ratio(meas, dso.sensitivities.net, c.bus_idx) if rescale else 1.
        dq = np.array([dso.sensitivities.compute_dQtrafo3w_hv_dQ_shunt(
            trafo3w_idx=int(t), shunt_bus_idx=c.bus_idx, q_step_mvar=signed_step)
            for t in ids])*rs
        # The TS Q-row and communicated bank contribution must describe the
        # same physical interface step, despite independent reduced models.
        for t, value in zip(ids, dq):
            h[nv+pcc.index(int(t))] = value
        if not np.all(np.isfinite(dq)):
            raise ValueError("Non-finite shunt interface prediction")
        models.append(BankModel(bank, h, ids,
            np.array([nd+pcc.index(int(t)) for t in ids]),
            dq, np.array([offsets.get(int(t), 0.) for t in ids]),
            feedback.get(c.dso_id)))
    if cfg.v_setpoints_pu is None:
        raise ValueError("Hybrid support requires declared voltage references")
    # Supervisory TS fleet reserve: the machine state comes from the controller
    # itself, so it shares the capability source of its own reserve term. ng is
    # the length of the Q_gen tail of every bank column.
    return HybridContext(time_s, models, tso._compute_output_gradient(meas),
                         np.arange(nv), np.asarray(cfg.v_setpoints_pu),
                         tso.fleet_reserve_state(meas), ng, nd)
