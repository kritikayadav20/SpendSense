"""Calendar-aware arithmetic, independent of all machine learning."""
import calendar
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re


def money_to_cents(value, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError("Amount must be a number.")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("Amount must be a finite number.") from None
    if not amount.is_finite() or amount < 0 or amount > Decimal("1000000000"):
        raise ValueError("Amount must be finite, nonnegative, and at most 1 billion.")
    cents = int((amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if cents == 0 and not allow_zero:
        raise ValueError("Expense amount must be at least 0.01.")
    return cents


def validate_month(month):
    if not isinstance(month, str) or not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError("Month must use YYYY-MM.")
    date.fromisoformat(month + "-01")
    return month


def budget_summary(spent, budget, month, today=None):
    validate_month(month)
    spent_cents, budget_cents = money_to_cents(spent, True), money_to_cents(budget, True)
    today = today or date.today()
    year, month_number = map(int, month.split("-"))
    total_days = calendar.monthrange(year, month_number)[1]
    first, last = date(year, month_number, 1), date(year, month_number, total_days)
    if today < first:
        elapsed, remaining_days, period = 0, total_days, "future"
    elif today > last:
        elapsed, remaining_days, period = total_days, 0, "past"
    else:
        elapsed, remaining_days, period = today.day, total_days - today.day + 1, "current"
    s, b = spent_cents / 100, budget_cents / 100
    remaining, overspend = max(0, budget_cents - spent_cents) / 100, max(0, spent_cents - budget_cents) / 100
    projection = s / elapsed * total_days if elapsed else None
    expected = b * elapsed / total_days
    alerts = []
    if overspend:
        alerts.append(f"Budget exceeded by {overspend:,.2f}.")
    elif projection is not None and projection > b and period == "current":
        alerts.append(f"Current pace projects {projection - b:,.2f} over budget at month end.")
    return {"spent": s, "budget": b, "remaining_budget": remaining, "overspend": overspend,
            "remaining_days": remaining_days, "elapsed_days": elapsed, "total_days": total_days,
            "safe_to_spend_per_day": remaining / remaining_days if remaining_days else None,
            "projected_month_end_spend": projection, "expected_spend_to_date": expected,
            "drift": s - expected, "period": period, "alerts": alerts}
