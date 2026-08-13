# IEEE 39 network analyses

Run the DSO boundary-condition and OLTC investigation from the repository
root:

```powershell
python -m network.ieee39.analysis.probe_dso_boundary_conditions
```

The default 96-step study compares the fully coupled IEEE 39 network against
isolated DSOs supplied by three stiff 1.03 p.u. primary sources with equal
distributed-slack weights. It evaluates unity, 0.98, and 0.95 inductive DER
power factors and exports detailed and summary CSV files under
`results/ieee39_dso_boundary_condition_probe/`.
