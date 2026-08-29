# Architecture

Pub/Sub accepts idempotent operational events continuously; it does not retrain models. BigQuery
stores partitioned market-week facts and clustered market dimensions. A weekly Workflow starts a
Vertex AI pipeline: validation → feature engineering → global count model → conformal calibration →
rolling backtest → statistical and decision-cost quality gates → Model Registry. Cloud Run serves
stateless forecast, scenario and optimization APIs. Plans and forecasts persist in BigQuery in GCP;
the in-memory API stores are deliberately local-demo adapters.

The quality gate requires challenger WAPE no worse than champion, acceptable bias, P90 coverage within
a configured band, and decision-backtest cost no worse than champion. Drift alerts (PSI, KS and
Wasserstein) require corroborating performance or business degradation before retraining.

