# ADR 0001: Core forecasting and platform decisions

Status: accepted. Weekly grain matches staffing cadence and reduces daily noise. A global Poisson-loss
gradient boosting model shares signal across small markets; seasonal-naive remains the benchmark.
Split conformal residual quantiles provide transparent uncertainty without a fragile deep model.
Bottom-up reconciliation guarantees exact state→region→national coherence. Integer MILP is required
because adjusters are indivisible; risk quantiles are simpler and more auditable than a nominal
"stochastic" optimizer. BigQuery fits analytical market-week data, Vertex AI handles scheduled model
lifecycle, Pub/Sub handles events, Cloud Run serves stateless APIs, and Workflows coordinates services.
Scheduled retraining avoids coupling model lifecycle to individual events.

Alternatives considered: daily forecasting (too noisy for this decision), per-market models (weak in
small markets), Gaussian regression (poor count semantics), MinT (unnecessary while bottom forecasts
are authoritative), deep learning (complexity without demonstrated benefit), and per-event retraining
(unstable and operationally expensive).

