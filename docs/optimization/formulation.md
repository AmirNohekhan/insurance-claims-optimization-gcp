# Workforce optimization

For market *m*, integer `x_m` is assigned adjusters; continuous `o_m`, `s_m`, and `e_m` are overtime
hours, uncovered workload/backlog, and excess capacity. For selected forecast quantile *q*:

`min Σ(c_x x_m + c_o o_m + (c_s+c_b)s_m + c_e e_m) + c_t transfers_in_m`

subject to `k x_m + k_o o_m + s_m - e_m = demand_m(q) + opening_backlog_m`, conservation
`Σx_m = Σcurrent_m`, market minimum/maximum staffing, transfer bounds, overtime bounds,
non-negativity, and integer staffing. The SciPy MILP/HiGHS implementation produces actual discrete
allocations. P50/P75/P90 runs expose the service-risk/cost trade-off. Assumptions are simulated and
configurable; recommendations support, rather than replace, claims leadership.

