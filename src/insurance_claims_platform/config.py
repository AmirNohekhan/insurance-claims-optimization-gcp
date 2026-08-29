from dataclasses import dataclass

MARKETS = {
    "FL": "Southeast",
    "GA": "Southeast",
    "NC": "Southeast",
    "VA": "Southeast",
    "MD": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "TX": "South Central",
    "CA": "West",
}


@dataclass(frozen=True)
class CostConfig:
    regular_adjuster_week: float = 2_200.0
    overtime_hour: float = 95.0
    transfer_adjuster: float = 1_250.0
    shortage_unit: float = 600.0
    backlog_unit: float = 240.0
    overcapacity_unit: float = 15.0
    capacity_per_adjuster: float = 16.0
    overtime_capacity_per_hour: float = 0.4
    max_overtime_hours_per_adjuster: float = 10.0
    max_transfer_fraction: float = 0.35
