from datetime import date
import pytest
from src.budget import budget_summary, money_to_cents


def test_current_month_formula():
    result = budget_summary(1000, 3100, '2026-01', date(2026, 1, 10))
    assert result['remaining_days'] == 22
    assert result['remaining_budget'] == 2100
    assert result['safe_to_spend_per_day'] == pytest.approx(2100 / 22)
    assert result['projected_month_end_spend'] == 3100
    assert result['drift'] == 0


def test_overspending():
    result = budget_summary(3500, 3100, '2026-01', date(2026, 1, 10))
    assert result['remaining_budget'] == result['safe_to_spend_per_day'] == 0
    assert result['overspend'] == 400
    assert 'exceeded' in result['alerts'][0]


def test_calendar_boundaries():
    result = budget_summary(100, 1000, '2024-02', date(2024, 2, 29))
    assert result['remaining_days'] == 1 and result['total_days'] == 29
    assert result['safe_to_spend_per_day'] == 900
    past = budget_summary(100, 1000, '2024-02', date(2024, 3, 1))
    assert past['safe_to_spend_per_day'] is None
    assert past['projected_month_end_spend'] == 100
    future = budget_summary(0, 3100, '2026-01', date(2025, 12, 31))
    assert future['safe_to_spend_per_day'] == 100
    assert future['projected_month_end_spend'] is None


def test_rounding_and_invalid():
    assert money_to_cents('1.005') == 101
    with pytest.raises(ValueError):
        budget_summary(-1, 10, '2026-01')
    with pytest.raises(ValueError):
        budget_summary(0, 10, '2026-13')
    with pytest.raises(ValueError):
        budget_summary(0, 10, '2026-1')
