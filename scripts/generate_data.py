from pathlib import Path

from insurance_claims_platform.simulation import (
    generate_portfolio,
    generate_weekly_claims,
    generate_workforce,
)


def main() -> None:
    target = Path("data/generated")
    target.mkdir(parents=True, exist_ok=True)
    portfolio = generate_portfolio()
    generate_weekly_claims(portfolio).to_csv(target / "weekly_claims.csv", index=False)
    portfolio.to_csv(target / "portfolio.csv", index=False)
    generate_workforce(portfolio).to_csv(target / "workforce.csv", index=False)
    print(f"Wrote reproducible local data to {target}")


if __name__ == "__main__":
    main()
