# app.py
# Corporate Instalment Interest & Penalty Calculator — Streamlit
#
# Revised:
# - Option 1/2/3 required-instalment schedules match CRA descriptions
# - "Auto choose lowest" compares cumulative required-by-date (not just totals)
#   so Option 3 will be chosen when it produces lower early required instalments.
#
# Sources:
# - CRA instalment option descriptions (monthly & quarterly): CRA pages/FAQ
#   (see links in the user request)

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

# Optional auto-rate loading (best effort). Manual entry always works.
import requests
from bs4 import BeautifulSoup


st.set_page_config(page_title="Corporate Instalment Interest & Penalty Calculator", layout="centered")


# ----------------------------
# Helpers
# ----------------------------

def to_date(x) -> date:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        raise ValueError("Missing date value.")
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.date()
    s = str(x).strip()
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def daterange_days(d1: date, d2: date) -> int:
    return (d2 - d1).days


def daily_rate(annual_rate_percent: float) -> float:
    return (annual_rate_percent / 100.0) / 365.0


def compound_factor(annual_rate_percent: float, days: int) -> float:
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
    if d.month in (1, 2, 3):
        return date(d.year, 1, 1)
    if d.month in (4, 5, 6):
        return date(d.year, 4, 1)
    if d.month in (7, 8, 9):
        return date(d.year, 7, 1)
    return date(d.year, 10, 1)


def next_quarter_start(d: date) -> date:
    return quarter_start(d) + relativedelta(months=3)


def build_quarter_boundaries(start: date, end_exclusive: date) -> list[date]:
    boundaries = []
    cur = quarter_start(start)
    if cur < start:
        cur = next_quarter_start(cur)
    while cur < end_exclusive:
        boundaries.append(cur)
        cur = next_quarter_start(cur)
    return boundaries


def quarter_of_date(d: date) -> int:
    return (d.month - 1) // 3 + 1


def fetch_cra_overdue_rate_by_quarter(year: int, quarter: int) -> float | None:
    """
    Best-effort scrape CRA prescribed interest rate page for the calendar quarter.
    If it fails (network restrictions / format change), returns None.
    """
    url = f"https://www.canada.ca/en/revenue-agency/services/tax/prescribed-interest-rates/{year}-q{quarter}.html"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True).lower()

        marker = "interest rate charged on overdue taxes"
        idx = text.find(marker)
        if idx == -1:
            return None
        window = text[idx: idx + 500]

        import re
        m = re.search(r"will be\s+(\d+(\.\d+)?)\s*%", window)
        if not m:
            m = re.search(r"(\d+(\.\d+)?)\s*%", window)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def build_rate_table_auto(from_date: date, to_date_inclusive: date) -> pd.DataFrame:
    rows = []
    cur = quarter_start(from_date)
    while cur <= to_date_inclusive:
        y = cur.year
        q = quarter_of_date(cur)
        rate = fetch_cra_overdue_rate_by_quarter(y, q)
        rows.append({"quarter_start": cur, "annual_rate_percent": rate})
        cur = next_quarter_start(cur)
    return pd.DataFrame(rows)


def get_rate_for_day(rate_table_ts: pd.DataFrame, d: date) -> float:
    """
    Get the annual rate (%) applicable on date d, based on quarter_start entries.

    IMPORTANT: compares using DATE ONLY to avoid time-component bugs
    that can cause the first day of a quarter to use the previous quarter's rate.
    """
    rt = rate_table_ts.sort_values("quarter_start").reset_index(drop=True)

    # Compare using pure dates (drops any time component)
    qs_dates = rt["quarter_start"].dt.date
    applicable = rt[qs_dates <= d]

    if applicable.empty:
        raise ValueError(f"No interest rate available for {d}. Add the rate for its quarter.")

    val = applicable.iloc[-1]["annual_rate_percent"]
    if pd.isna(val):
        qs = applicable.iloc[-1]["quarter_start"].date()
        raise ValueError(f"Interest rate missing for quarter starting {qs}. Please fill it in.")

    return float(val)


def month_end(d: date) -> date:
    return (d.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)


def generate_due_dates(tax_year_start: date, tax_year_end: date, freq: str) -> list[date]:
    due_dates: list[date] = []
    if freq == "Monthly":
        cur = tax_year_start
        while cur <= tax_year_end:
            me = month_end(cur)
            if tax_year_start <= me <= tax_year_end:
                due_dates.append(me)
            cur = cur + relativedelta(months=1)
    else:
        cur = tax_year_start
        while cur <= tax_year_end:
            pe = (cur + relativedelta(months=3)).replace(day=1) - timedelta(days=1)
            if tax_year_start <= pe <= tax_year_end:
                due_dates.append(pe)
            cur = cur + relativedelta(months=3)
    return sorted(set(due_dates))


# ----------------------------
# Required instalment schedules (CRA options)
# ----------------------------

def build_required_amounts_from_option(
    due_dates: list[date],
    freq: str,
    option: int,
    base_opt1: float,
    base_opt2: float,
    base_opt3: float,
) -> dict[date, float]:
    """
    CRA options (per CRA description):
    Monthly:
      Option 1: 1/12 of current-year estimate each month
      Option 2: 1/12 of previous-year tax payable each month
      Option 3: 1/12 of year-before-previous tax payable for first 2 months,
                then 1/10 of (previous-year - total(first 2 payments)) for remaining 10 months
    Quarterly (eligible small CCPC):
      Option 1: 1/4 of current-year estimate each quarter
      Option 2: 1/4 of previous-year tax payable each quarter
      Option 3: 1/4 of year-before-previous tax payable for first quarter,
                then 1/3 of (previous-year - first payment) for remaining 3 quarters
    """
    n = len(due_dates)
    if n == 0:
        return {}

    # Map "base" by option number
    if option == 1:
        base = float(base_opt1)
    elif option == 2:
        base = float(base_opt2)
    elif option == 3:
        base = float(base_opt3)
    else:
        raise ValueError("Option must be 1, 2, or 3.")

    if freq == "Monthly":
        # Option 1/2 are even
        if option in (1, 2):
            p = float(base) / n
            return {d: p for d in due_dates}

        # Option 3 monthly (CRA): first 2 months based on base_opt3 / 12
        # Remaining 10 months based on (base_opt2 - first_two_total) / 10
        # If not a standard 12-month schedule, we still apply the CRA pattern
        # as closely as possible: first 2 instalments use base3/12; rest spread across remaining.
        first_two_amt = float(base_opt3) / 12.0
        k = min(2, n)
        first_total = first_two_amt * k
        remaining_count = n - k
        if remaining_count <= 0:
            return {due_dates[i]: first_two_amt for i in range(n)}

        remaining_total_basis = float(base_opt2) - first_total
        remaining_amt = remaining_total_basis / remaining_count

        req = {}
        for i, d in enumerate(due_dates):
            req[d] = first_two_amt if i < k else remaining_amt
        return req

    # Quarterly
    if option in (1, 2):
        p = float(base) / n
        return {d: p for d in due_dates}

    # Option 3 quarterly (CRA): first quarter based on base_opt3 / 4; remaining 3 quarters:
    first_amt = float(base_opt3) / 4.0
    k = 1 if n >= 1 else 0
    first_total = first_amt * k
    remaining_count = n - k
    if remaining_count <= 0:
        return {due_dates[0]: first_amt} if n == 1 else {}
    remaining_total_basis = float(base_opt2) - first_total
    remaining_amt = remaining_total_basis / remaining_count

    req = {}
    for i, d in enumerate(due_dates):
        req[d] = first_amt if i < k else remaining_amt
    return req


def cumulative_schedule(due_dates: list[date], req: dict[date, float]) -> list[float]:
    """Return cumulative required amounts aligned to due_dates order."""
    cum = []
    running = 0.0
    for d in due_dates:
        running += float(req.get(d, 0.0))
        cum.append(running)
    return cum


def choose_cra_lowest_option(due_dates: list[date], freq: str, base1: float, base2: float, base3: float) -> tuple[int, dict[date, float]]:
    """
    CRA says it will assess using the option that results in the least amount payable by instalments.
    For interest purposes, what matters is the *required cumulative amount by each due date*.
    So we compare schedules lexicographically by cumulative required-by-date:
      - lower at earlier due dates wins
      - if identical throughout, tie-breaker prefers Option 3 (more “CRA typical”)
    """
    req1 = build_required_amounts_from_option(due_dates, freq, 1, base1, base2, base3)
    req2 = build_required_amounts_from_option(due_dates, freq, 2, base1, base2, base3)
    req3 = build_required_amounts_from_option(due_dates, freq, 3, base1, base2, base3)

    c1 = cumulative_schedule(due_dates, req1)
    c2 = cumulative_schedule(due_dates, req2)
    c3 = cumulative_schedule(due_dates, req3)

    # Build a comparison key: tuple(cumulative amounts..., option preference)
    # Lower tuple is better. For ties, we prefer option 3, then 2, then 1.
    # (So we subtract a tiny epsilon via rank; in tuple comparison, smaller is better.)
    pref_rank = {3: 0, 2: 1, 1: 2}

    candidates = [
        (tuple(c1) + (pref_rank[1],), 1, req1),
        (tuple(c2) + (pref_rank[2],), 2, req2),
        (tuple(c3) + (pref_rank[3],), 3, req3),
    ]
    candidates.sort(key=lambda x: x[0])
    _, opt, req = candidates[0]
    return opt, req


# ----------------------------
# Interest & penalty
# ----------------------------

def compute_instalment_interest(
    tax_year_start: date,
    balance_due_day: date,
    due_dates: list[date],
    required_amounts: dict[date, float],
    payments_df: pd.DataFrame,
    rate_table_ts: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    events: list[Event] = []

    # Due events
    for dd in due_dates:
        amt = float(required_amounts.get(dd, 0.0))
        if abs(amt) > 1e-9:
            events.append(Event(dd, "Instalment due", amt))

    # Payment events
    if payments_df is not None and not payments_df.empty:
        for _, row in payments_df.iterrows():
            if pd.isna(row.get("date")) or pd.isna(row.get("amount")):
                continue
            pdte = to_date(row["date"])
            amt = float(row["amount"])
            if abs(amt) > 1e-9 and pdte <= balance_due_day:
                events.append(Event(pdte, "Payment received", -amt))

    # Quarter boundaries (rate change points)
    for b in build_quarter_boundaries(tax_year_start, balance_due_day + timedelta(days=1)):
        events.append(Event(b, "Rate change boundary", 0.0))

    # Start and end events
    events.append(Event(tax_year_start, "Tax year start", 0.0))
    events.append(Event(balance_due_day, "Balance-due day", 0.0))

    events.sort(key=lambda e: (e.d, 0 if e.kind == "Tax year start" else 1))

    # Combine same-day events
    combined: list[Event] = []
    i = 0
    while i < len(events):
        d0 = events[i].d
        same: list[Event] = []
        while i < len(events) and events[i].d == d0:
            same.append(events[i])
            i += 1
        net_amt = sum(e.amount for e in same)
        kinds = ", ".join(e.kind for e in same)
        combined.append(Event(d0, kinds, net_amt))

    balance = 0.0
    total_interest = 0.0
    rows = []

    for j in range(len(combined) - 1):
        e = combined[j]
        next_e = combined[j + 1]

        # Apply amounts at start of interval
        balance += e.amount

        days = daterange_days(e.d, next_e.d)
        if days <= 0:
            continue

        rate = get_rate_for_day(rate_table_ts, e.d)
        factor = compound_factor(rate, days)
        period_interest = balance * (factor - 1.0)
        total_interest += period_interest

        rows.append(
            {
                "from": e.d,
                "to": next_e.d,
                "days": days,
                "annual_rate_%": rate,
                "balance_during_period": balance,
                "interest_for_period": period_interest,
                "event_applied_on_from": e.kind,
                "event_amount_on_from": e.amount,
            }
        )

    return float(total_interest), pd.DataFrame(rows)


def compute_penalty(instalment_interest: float, interest_if_no_payments: float) -> float:
    # Keeps the CRA-style threshold logic used previously.
    if instalment_interest <= 1000.0:
        return 0.0
    threshold = max(1000.0, 0.25 * interest_if_no_payments)
    diff = instalment_interest - threshold
    return 0.5 * diff if diff > 0 else 0.0


# ----------------------------
# UI
# ----------------------------

st.title("CRA Corporate Instalment Interest & Penalty Calculator (Canada)")

colA, colB, colC = st.columns(3)
with colA:
    tax_year_start = st.date_input("Tax year start", value=date(2025, 1, 1))
with colB:
    tax_year_end_default = (tax_year_start + relativedelta(years=1)) - timedelta(days=1)
    tax_year_end = st.date_input("Tax year end", value=tax_year_end_default)
with colC:
    balance_due_day_default = tax_year_end + relativedelta(months=2)
    balance_due_day = st.date_input("Balance-due day (instalment interest stops here)", value=balance_due_day_default)

freq = st.radio("Instalment frequency", ["Monthly", "Quarterly"], horizontal=True)
due_dates = generate_due_dates(tax_year_start, tax_year_end, freq)
st.caption(f"Generated {len(due_dates)} due dates.")

st.subheader("Required instalments (CRA Options 1 / 2 / 3)")
entry_mode = st.selectbox(
    "How do you want to define required instalments?",
    [
        "Calculate from CRA instalment bases (Option 1/2/3)",
        "Manual table (enter required amount per due date)",
        "Same amount each due date",
    ],
)

required_amounts: dict[date, float] = {}

if entry_mode == "Calculate from CRA instalment bases (Option 1/2/3)":
    st.markdown(
        "Enter **instalment base** amounts (from your CRA worksheets / calculations). "
        "Option 3 uses the year-before-last base for the first 2 months (monthly) or first quarter (quarterly)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        base1 = st.number_input("Option 1 base (current-year estimate)", min_value=0.0, value=0.0, step=100.0)
    with c2:
        base2 = st.number_input("Option 2 base (previous year)", min_value=0.0, value=0.0, step=100.0)
    with c3:
        base3 = st.number_input("Option 3 base (two years ago)", min_value=0.0, value=0.0, step=100.0)

    auto_choose = st.checkbox("Auto-choose CRA lowest option (based on lowest cumulative required schedule)", value=True)

    if auto_choose:
        chosen_opt, required_amounts = choose_cra_lowest_option(due_dates, freq, base1, base2, base3)
        st.info(f"Chosen option: **Option {chosen_opt}** (based on lowest cumulative required-by-date schedule).")
    else:
        opt = st.radio("Pick option", [1, 2, 3], horizontal=True)
        required_amounts = build_required_amounts_from_option(due_dates, freq, opt, base1, base2, base3)

    req_preview = pd.DataFrame(
        {"due_date": list(required_amounts.keys()), "required_amount": list(required_amounts.values())}
    ).sort_values("due_date")
    req_preview["cumulative_required"] = req_preview["required_amount"].cumsum()
    st.dataframe(req_preview, use_container_width=True)

elif entry_mode == "Same amount each due date":
    amt = st.number_input("Required instalment per due date", min_value=0.0, value=0.0, step=100.0)
    required_amounts = {d: float(amt) for d in due_dates}

else:
    req_df = pd.DataFrame({"due_date": due_dates, "required_amount": [0.0] * len(due_dates)})
    edited = st.data_editor(
        req_df,
        use_container_width=True,
        num_rows="fixed",
        key="required_editor",
        column_config={
            "due_date": st.column_config.DateColumn("Due date"),
            "required_amount": st.column_config.NumberColumn("Required amount", min_value=0.0, step=100.0),
        },
    )
    required_amounts = {to_date(r["due_date"]): float(r["required_amount"] or 0.0) for _, r in edited.iterrows()}

st.subheader("Actual payments made")

# FIX: explicit dtypes so Streamlit can validate schema even when empty
payments_df = pd.DataFrame(
    {
        "date": pd.Series(dtype="datetime64[ns]"),
        "amount": pd.Series(dtype="float"),
    }
)

payments_df = st.data_editor(
    payments_df,
    use_container_width=True,
    num_rows="dynamic",
    key="payments_editor",
    column_config={
        "date": st.column_config.DateColumn("Payment date"),
        "amount": st.column_config.NumberColumn("Amount", min_value=0.0, step=100.0),
    },
)

st.subheader("Interest rate table (calendar quarters)")

rate_source = st.selectbox("Rate source", ["Auto-load from CRA pages (best effort)", "Manual entry"])

if rate_source == "Auto-load from CRA pages (best effort)":
    rate_table = build_rate_table_auto(tax_year_start, balance_due_day)
    st.info("If any rate cells are blank, fill them in manually below.")
else:
    qs = build_quarter_boundaries(tax_year_start, balance_due_day + timedelta(days=1))
    rate_table = pd.DataFrame({"quarter_start": qs, "annual_rate_percent": [None] * len(qs)})

rate_table = st.data_editor(
    rate_table,
    use_container_width=True,
    num_rows="fixed",
    key="rate_editor",
    column_config={
        "quarter_start": st.column_config.DateColumn("Quarter start"),
        "annual_rate_percent": st.column_config.NumberColumn("Annual rate (%)", step=0.25),
    },
)

st.divider()

if st.button("Calculate interest + penalty", type="primary"):
    try:
        # Normalize rate table (quarter_start must be datetime64[ns])
        rt = rate_table.copy()
        rt["quarter_start"] = rt["quarter_start"].apply(to_date)
        rt["quarter_start"] = pd.to_datetime(rt["quarter_start"])
        rt = rt.sort_values("quarter_start").reset_index(drop=True)

        # Normalize payments
        pay = payments_df.copy()
        if not pay.empty:
            pay["date"] = pd.to_datetime(pay["date"], errors="coerce")
            pay["amount"] = pd.to_numeric(pay["amount"], errors="coerce")

        inst_int, breakdown = compute_instalment_interest(
            tax_year_start=tax_year_start,
            balance_due_day=balance_due_day,
            due_dates=due_dates,
            required_amounts=required_amounts,
            payments_df=pay,
            rate_table_ts=rt,
        )

        # Interest if NO payments (for penalty formula)
        empty_payments = pd.DataFrame(
            {"date": pd.Series(dtype="datetime64[ns]"), "amount": pd.Series(dtype="float")}
        )
        int_no_pay, _ = compute_instalment_interest(
            tax_year_start=tax_year_start,
            balance_due_day=balance_due_day,
            due_dates=due_dates,
            required_amounts=required_amounts,
            payments_df=empty_payments,
            rate_table_ts=rt,
        )

        penalty = compute_penalty(inst_int, int_no_pay)

        st.success("Calculated successfully.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Instalment interest", f"${inst_int:,.2f}")
        c2.metric("Penalty (if applicable)", f"${penalty:,.2f}")
        c3.metric("Interest if no payments (penalty input)", f"${int_no_pay:,.2f}")

        st.subheader("Breakdown (audit trail)")
        st.dataframe(breakdown, use_container_width=True)

    except Exception as e:
        st.error(str(e))
