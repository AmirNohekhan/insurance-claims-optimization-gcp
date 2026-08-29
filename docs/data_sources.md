# Data sources and provenance

The executable demo is deterministic and offline: it ships no third-party records and synthesizes
weather-like covariates. The production ingestion design selects only the following public sources.

| Dataset | Provider and source | Coverage / variables | Terms | Retrieval and transformation | Limitations |
|---|---|---|---|---|---|
| Storm Events Database | NOAA NCEI, `ncei.noaa.gov/pub/data/swdi/stormevents/` | U.S., 1950–present; event type, state, dates, injuries/damage | U.S. federal government data are generally not copyrighted; NOAA asks users to cite NCEI. See the NCEI dataset documentation and disclaimer before redistribution. | Monthly bulk CSV; map event county/state and begin date to market-week; derive event counts and a capped severity proxy. | Reporting practices and damage fields vary over time; it is an event indicator, not a catastrophe-loss model. |
| Disaster Declarations Summaries | FEMA OpenFEMA, `fema.gov/openfema-data-page/disaster-declarations-summaries-v2` | U.S., 1953–present; declaration, incident type, state, incident dates | OpenFEMA data are public domain unless otherwise noted; API terms and attribution guidance apply. | JSON/CSV API; retain declaration information available as of each forecast origin and aggregate to state-week. | A declaration is administrative and may lag the event; never use its eventual value before publication. |
| Weekly weather summaries | NOAA Climate Data Online, `ncdc.noaa.gov/cdo-web/` | Station/daily precipitation, temperature and wind | Access is subject to NOAA/NCEI terms and rate limits; cite the dataset. | Token-authenticated API; station-to-market mapping, quality filtering, weekly aggregation. | Station coverage is uneven. Realized future observations are prohibited in backtests. |

Licensing statements above are intentionally narrow and link to provider documentation; deployment
owners must re-check current provider terms. Calendar features use Python's standard calendar and
require no external dataset. Claims, exposure, workforce and backlog are synthetic and contain no PII.

Future-feature policy: calendar and planned policy exposure are known; weather outlook variables are
forecasted exogenous inputs; realized precipitation, wind, storm severity and catastrophe labels are
unknown and excluded. The demo uses lagged observed weather as an explicitly imperfect outlook proxy.

