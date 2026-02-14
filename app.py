import streamlit as st
import pandas as pd
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="CRA Corporate Instalment Interest & Penalty (Canada)", layout="wide")

# ----------------------------
# Helpers
# ----------------------------

def to_date(x) -> date:
    if isinstance(x, date):
        return x
    return datetime.strptime(str(x), "%Y-%m-%d").date()

def daterange_days(d1: date, d2: date) -> int:
    """Number of days between d1 and d2 (d2 - d1)."""
    return (d2 - d1).days

def daily_rate(annual_rate_percent: float) -> float:
    """Convert annual nominal percent to daily rate for daily compounding."""
    return (annual_rate_percent / 100.0) / 365.0

def compound_factor(annual_rate_percent: float, days: int) -> float:
    """Daily compounding factor over 'days' at annual_rate_percent."""
    if days <= 0:
        return 1.0
    r = daily_rate(annual_rate_percent)
    return (1.0 + r) ** days

@dataclass
class Event:
    d: date
    kind: str
    amount: float = 0.0  # positive increases balance, negative decreases

def quarter_start(d: date) -> date:
    """Return calendar quarter start for date d."""
    if d.month in (1,2,3):
        return date(d.year, 1, 1)
    if d.month in (4,5,6):
        return date(d.year, 4, 1)
    if d.month in (7,8,9):
        return date(d.year, 7, 1)
    return date(d.year, 10, 1)

def next_quarter_start(d: date) -> date:
    qs = quarter_start(d)
    return qs + relativedelta(months=3)

def build_quarter_boundaries(start: date, end: date):
    """All quarter boundary dates between start and end (inclusive start, exclusive end)."""
    boundaries = []
    cur = quarter_start(start)
    if cur < start:
        cur = next_quarter_start(cur)
    while cur < end:
        boundaries.append(cur)
        cur = next_quarter_start(cur)
    return boundaries

def fetch_cra_overdue_rate_by_quarter(year: int, quarter: int) -> float | None:
    """
    Scrape CRA 'Interest rates for the <quarter> calendar quarter <year>' page and extract
    the 'interest rate charged on overdue taxes' (commonly used for instalment/arrears interest).
    Example URL pattern:
      https://www.canada.ca/en/revenue-agency/services/tax/prescribed-interest-rates/2026-q1.html
    """
    url = f"https://www.canada.ca/en/revenue-agency/services/tax/prescribed-interest-rates/{year}-q{quarter}.html"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        # Find phrase and a percent near it
        marker = "interest rate charged on overdue taxes"
        idx = text.lower().find(marker)
        if idx == -1:
            return None
        window = text[idx:idx+250]
        # naive percent parse
        import re
        m = re.search(r"will be\s+(\d+(\.\d+)?)\s*%", window)
        if not m:
            m = re.search(r"(\d+(\.\d+)?)\s*%", window)
        return float(m.group(1)) if m else None
    except Exception:
        return None

def quarter_of_date(d: date) -> int:
    return (d.month - 1)//3 + 1

def build_rate_table_auto(from_date: date, to_date_: date) -> pd.DataFrame:
    """
    Build a table of calendar-quarter overdue-tax interest rates by scraping CRA pages.
    If scrape fails for a quarter, leave as NaN (user can fill manually).
    """
    rows = []
    q_start = quarter_start(from_date)
    cur = q_start
    while cur <= to_date_:
        y = cur.year
        q = quarter_of_date(cur)
        rate = fetch_cra_overdue_rate_by_quarter(y, q)
        rows.append({"quarter_start": cur, "year": y, "quarter": q, "annual_rate_percent": rate})
        cur = next_quarter_start(cur)
    return pd.DataFrame(rows)

def get_rate_for_day(rate_table: pd.DataFrame, d: date) -> float:
    """
    Get the annual rate (percent) applicable on date d, based on quarter_start entries.
    Assumes rate_table has quarter_start sorted and covers d; otherwise raises.
    """
    rt = rate_table.sort_values("quarter_start").reset_index(drop=True)
    applicable = rt[rt["quarter_start"] <= pd.Timestamp(d)]
    if applicable.empty:
        raise ValueError(f"No rate available for {d}. Add a rate for its quarter.")
    val = applicable.iloc[-1]["annual_rate_percent"]
    if pd.isna(val):
        raise ValueError(f"Rate missing for quarter starting {applicable.iloc[-1]['quarter_start'].date()}. Fill it in.")
    return float(val)

def generate_due_dates(tax_year_start: date, tax_year_end: date, freq: str) -> list[date]:
    """
    Monthly: last day of each month in the tax year.
    Quarterly: last day of each complete quarter in the tax year.
    CRA explains monthly last-day and quarterly quarter-end pattern. (T7B-Corp Due dates)
    """
    d = tax_year_start
    due_dates = []

    if freq == "Monthly":
        cur = tax_year_start
        while cur <= tax_year_end:
            month_end = (cur.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)
            if tax_year_start <= month_end <= tax_year_end:
                due_dates.append(month_end)
            cur = cur + relativedelta(months=1)
    else:
        # quarterly ends relative to tax year start
        cur = tax_year_start
        while cur <= tax_year_end:
            q_end = (cur + relativedelta(months=3)).replace(day=1) - timedelta(days=1)
            if tax_year_start <= q_end <= tax_year_end:
                due_dates.append(q_end)
            cur = cur + relativedelta(months=3)

    return sorted(set(due_dates))

def compute_instalment_interest(
    tax_year_start: date,
    balance_due_day: date,
    due_dates: list[date],
    required_amounts: dict[date, float],
    payments_df: pd.DataFrame,
    rate_table: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """
    Offset-method simulation:
    - On each due date: balance += required instalment
    - On each payment date: balance -= payment amount
    - Interest accrues between events with daily compounding at the quarter's annual rate.
    Returns: (total_interest, breakdown_df)
    """
    events: list[Event] = []

    # due events
    for dd in due_dates:
        amt = float(required_amounts.get(dd, 0.0))
        if amt != 0:
            events.append(Event(dd, "Instalment due", amt))

    # payment events
    if not payments_df.empty:
        for _, row in payments_df.iterrows():
            pdte = to_date(row["date"])
            amt = float(row["amount"])
            if pdte <= balance_due_day and amt != 0:
                events.append(Event(pdte, "Payment received", -amt))

    # rate-change boundaries as events (no amount)
    for b in build_quarter_boundaries(tax_year_start, balance_due_day + timedelta(days=1)):
        events.append(Event(b, "Rate change boundary", 0.0))

    # ensure start and end events
    events.append(Event(tax_year_start, "Tax year start", 0.0))
    events.append(Event(balance_due_day, "Balance-due day", 0.0))

    # sort and combine same-day events
    events.sort(key=lambda e: (e.d, 0 if e.kind == "Tax year start" else 1))

    combined = []
    i = 0
    while i < len(events):
        d0 = events[i].d
        same = []
        while i < len(events) and events[i].d == d0:
            same.append(events[i])
            i += 1
        # Order within day: due increases balance before interest period starts after this day,
        # payment reduces balance on that day. CRA’s exact intraday convention can differ;
        # this is a reasonable v1 approximation consistent with event-day posting.
        # We apply all same-day amounts before moving to next day interval.
        net_amt = sum(e.amount for e in same)
        kinds = ", ".join([e.kind for e in same])
        combined.append(Event(d0, kinds, net_amt))

    balance = 0.0
    total_interest = 0.0
    rows = []

    for j in range(len(combined) - 1):
        e = combined[j]
        next_e = combined[j + 1]

        # apply today's event amounts
        balance += e.amount

        days = daterange_days(e.d, next_e.d)
        if days > 0:
            rate = get_rate_for_day(rate_table, e.d)
            factor = compound_factor(rate, days)
            # interest for period on current balance:
            period_interest = balance * (factor - 1.0)
            total_interest += period_interest

            rows.append({
                "from": e.d,
                "to": next_e.d,
                "days": days,
                "annual_rate_%": rate,
                "balance_during_period": balance,
                "interest_for_period": period_interest,
                "event_applied_on_from": e.kind,
                "event_amount_on_from": e.amount,
            })

    breakdown = pd.DataFrame(rows)
    return float(total_interest), breakdown

def compute_penalty(instalment_interest: float, interest_if_no_payments: float) -> float:
    """
    Per CRA T7B-Corp:
      If instalment interest > 1000:
        penalty = 0.5 * (instalment_interest - max(1000, 0.25 * interest_if_no_payments))
      else 0
    """
    if instalment_interest <= 1000:
        return 0.0
    threshold = max(1000.0, 0.25 * interest_if_no_payments)
    diff = instalment_interest - threshold
    return 0.5 * diff if diff > 0 else 0.0


# ----------------------------
# UI
# ----------------------------

st.title("CRA Corporate Instalment Interest & Penalty Calculator (Canada)")

with st.expander("What this matches (CRA rules)", expanded=False):
    st.markdown(
        """
- Instalment interest is compounded daily and calculated using CRA’s **offset method**.  
- Instalment penalty applies only if instalment interest is **more than $1,000**, and uses CRA’s formula.  
"""
    )

colA, colB, colC = st.columns(3)

with colA:
    tax_year_start = st.date_input("Tax year start", value=date(2025, 1, 1))
with colB:
    tax_year_end_default = (tax_year_start + relativedelta(years=1)) - timedelta(days=1)
    tax_year_end = st.date_input("Tax year end", value=tax_year_end_default)
with colC:
    balance_due_day_default = tax_year_end + relativedelta(months=2)  # common for many corps, editable
    balance_due_day = st.date_input("Balance-due day (instalment interest stops here)", value=balance_due_day_default)

freq = st.radio("Instalment frequency", ["Monthly", "Quarterly"], horizontal=True)

due_dates = generate_due_dates(tax_year_start, tax_year_end, freq)
st.caption(f"Generated {len(due_dates)} due dates based on your tax year and frequency.")

st.subheader("Required instalments")
required_mode = st.selectbox("How will you enter required instalments?", ["Same amount each due date", "Enter each due date amount (table)"])

required_amounts = {}
if required_mode == "Same amount each due date":
    amt = st.number_input("Required instalment amount per due date", min_value=0.0, value=0.0, step=100.0)
    required_amounts = {d: float(amt) for d in due_dates}
else:
    req_df = pd.DataFrame({"due_date": due_dates, "required_amount": [0.0]*len(due_dates)})
    edited = st.data_editor(req_df, use_container_width=True, num_rows="fixed")
    required_amounts = {to_date(r["due_date"]): float(r["required_amount"]) for _, r in edited.iterrows()}

st.subheader("Actual payments made")
payments_df = pd.DataFrame({"date": [], "amount": []})
payments_df = st.data_editor(
    payments_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "date": st.column_config.DateColumn("Payment date"),
        "amount": st.column_config.NumberColumn("Amount", min_value=0.0, step=100.0),
    },
)

st.subheader("Interest rate table (quarterly)")
rate_source = st.selectbox("Rate source", ["Auto-load from CRA pages (best effort)", "Manual entry"])
if rate_source == "Auto-load from CRA pages (best effort)":
    rate_table = build_rate_table_auto(tax_year_start, balance_due_day)
    st.info("Auto-loaded rates where available. Fill any blanks manually below.")
else:
    rate_table = pd.DataFrame({"quarter_start": build_quarter_boundaries(tax_year_start, balance_due_day + timedelta(days=1))})
    rate_table["annual_rate_percent"] = None

rate_table = st.data_editor(
    rate_table,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "quarter_start": st.column_config.DateColumn("Quarter start"),
        "annual_rate_percent": st.column_config.NumberColumn("Annual rate (%)", step=0.25),
    },
)

if st.button("Calculate interest + penalty", type="primary"):
    try:
        # Ensure datetime types are consistent
        rt = rate_table.copy()
        rt["quarter_start"] = pd.to_datetime(rt["quarter_start"])
        rt = rt.sort_values("quarter_start")

        # Interest with actual payments
        inst_int, breakdown = compute_instalment_interest(
            tax_year_start=tax_year_start,
            balance_due_day=balance_due_day,
            due_dates=due_dates,
            required_amounts=required_amounts,
            payments_df=payments_df,
            rate_table=rt,
        )

        # Interest if NO payments (for penalty formula)
        empty_payments = pd.DataFrame({"date": [], "amount": []})
        int_no_pay, _ = compute_instalment_interest(
            tax_year_start=tax_year_start,
            balance_due_day=balance_due_day,
            due_dates=due_dates,
            required_amounts=required_amounts,
            payments_df=empty_payments,
            rate_table=rt,
        )

        penalty = compute_penalty(inst_int, int_no_pay)

        st.success("Done.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Instalment interest", f"${inst_int:,.2f}")
        c2.metric("Penalty (if applicable)", f"${penalty:,.2f}")
        c3.metric("Interest if no payments (penalty input)", f"${int_no_pay:,.2f}")

        st.subheader("Breakdown (audit trail)")
        st.dataframe(breakdown, use_container_width=True)

        st.caption(
            "Note: This is a practical v1 calculator. CRA’s internal posting conventions can be nuanced; "
            "for filings/objections, reconcile to CRA Statements of Account."
        )

    except Exception as e:
        st.error(str(e))
