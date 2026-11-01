import argparse
import re
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib

if "--no-show" in sys.argv:          # must be chosen before pyplot is imported
    matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- look and feel
INK, TEAL, TAG, ALERT, MUTE, LINE = "#17212B", "#0E7C7B", "#F4B400", "#C0392B", "#6B7785", "#D9DDD5"
CYCLE = [TEAL, TAG, INK, ALERT, "#7FB7B5", "#B8C0C9", "#E3C77A"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": LINE,
    "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE, "axes.grid": True,
    "grid.color": "#ECEEE9", "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
    "figure.facecolor": "white",
})


def fmt(x):
    """Compact, currency-neutral number format: 1.2K, 3.45M."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:.2f}B"
    if a >= 1e6:
        return f"{x / 1e6:.2f}M"
    if a >= 1e3:
        return f"{x / 1e3:.1f}K"
    return f"{x:,.0f}"


money = FuncFormatter(lambda v, _: fmt(v))


# =========================================================================== 1. LOAD AND CLEAN
ROLE_ALIASES = {
    "date": ["date", "order date", "orderdate", "invoice date", "invoicedate", "transaction date",
             "sale date", "sales date", "day", "timestamp", "purchase date"],
    "product": ["product", "product name", "item", "item name", "sku", "description"],
    "category": ["category", "product category", "sub category", "department", "category name"],
    "region": ["region", "city", "state", "store", "branch", "location", "country", "area"],
    "customer": ["customer", "customer id", "customerid", "customer name", "client", "cust id"],
    "quantity": ["quantity", "qty", "units", "units sold", "quantity sold"],
    "price": ["price", "unit price", "unitprice", "rate", "selling price"],
    "amount": ["amount", "sales", "revenue", "total", "total sales", "total amount", "sale amount",
               "line total", "turnover"],
}


def _norm(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _to_number(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def _text(series):
    s = series.astype("string").str.strip().fillna("Unknown").replace("", "Unknown")
    return s.astype(object)


def detect_columns(df):
    """Map each role (date, amount, product ...) to the matching CSV column."""
    lookup = {_norm(c): c for c in df.columns}
    found = {}
    for role, names in ROLE_ALIASES.items():
        for name in names:
            col = lookup.get(_norm(name))
            if col is not None and col not in found.values():
                found[role] = col
                break
    if "date" not in found:                       # fall back: any column that parses as dates
        for col in df.columns:
            if col in found.values() or df[col].dtype != object:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if pd.to_datetime(df[col], errors="coerce").notna().mean() > 0.8:
                    found["date"] = col
                    break
    return found


def load_and_clean(path):
    """Read the CSV and return (clean dataframe, has-flags, quality report, matched columns)."""
    raw = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    raw.columns = [str(c).strip() for c in raw.columns]
    if raw.empty:
        raise ValueError("The CSV file has no rows.")
    cols = detect_columns(raw)
    if "date" not in cols:
        raise ValueError("No date column found. Name it 'Date' or 'Order Date'.")

    df = pd.DataFrame(index=raw.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["date"] = pd.to_datetime(raw[cols["date"]], errors="coerce")
    if "amount" in cols:
        df["amount"] = _to_number(raw[cols["amount"]])
    elif "quantity" in cols and "price" in cols:
        df["amount"] = _to_number(raw[cols["quantity"]]) * _to_number(raw[cols["price"]])
    else:
        raise ValueError("No sales column found. Name it 'Sales', 'Revenue' or 'Amount', "
                         "or include both 'Quantity' and 'Price'.")
    df["quantity"] = _to_number(raw[cols["quantity"]]) if "quantity" in cols else np.nan
    for role in ("product", "category", "region", "customer"):
        df[role] = _text(raw[cols[role]]) if role in cols else None

    dup = raw.duplicated()
    quality = {"rows in file": len(raw), "duplicates removed": int(dup.sum())}
    df = df[~dup]
    quality["bad dates removed"] = int(df["date"].isna().sum())
    quality["bad sales values removed"] = int(df["amount"].isna().sum())
    df = df.dropna(subset=["date", "amount"])
    quality["zero/negative sales removed"] = int((df["amount"] <= 0).sum())
    df = df[df["amount"] > 0]
    if df.empty:
        raise ValueError("No usable rows left after cleaning. Check the date and sales columns.")
    quality["rows used"] = len(df)
    has = {k: (k in cols) for k in ("product", "category", "region", "customer", "quantity")}
    return df.sort_values("date").reset_index(drop=True), has, quality, cols


# =========================================================================== 2. ANALYTICS
def forecast_monthly(m, periods=3):
    """Linear-trend forecast, adjusted for seasonality when 24+ months of data exist."""
    if len(m) < 4:
        return None
    seasonal = len(m) >= 24
    if seasonal:
        factors = m.groupby(m.index.month).mean()
        factors = factors / factors.mean()
        base = m / m.index.month.map(factors).to_numpy()
    else:
        factors, base = None, m
    y = base.to_numpy(dtype=float)[-24:]
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    sigma = float(np.std(y - (slope * x + intercept)))
    idx = pd.date_range(m.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    pred = slope * np.arange(len(y), len(y) + periods) + intercept
    scale = np.array([factors[d.month] for d in idx]) if seasonal else np.ones(periods)
    pred = np.maximum(pred * scale, 0)
    band = 1.5 * sigma * scale
    return pd.DataFrame({"pred": pred, "low": np.maximum(pred - band, 0), "high": pred + band}, index=idx)


def find_anomalies(daily):
    """Days that are unusual for their weekday (log scale, robust median/MAD z-score)."""
    empty = pd.DataFrame(columns=["date", "value", "kind", "score"])
    if len(daily) < 56:
        return empty
    weekday_median = daily.groupby(daily.index.dayofweek).transform("median")
    log_ratio = np.log(daily / weekday_median)
    local_level = log_ratio.rolling(31, center=True, min_periods=7).median()
    dev = (log_ratio - local_level).dropna()
    mad = (dev - dev.median()).abs().median()
    if not mad:
        return empty
    score = 0.6745 * (dev - dev.median()) / mad
    flagged = score[score.abs() > 4.5]
    if flagged.empty:
        return empty
    flagged = flagged.reindex(flagged.abs().sort_values(ascending=False).index).head(10)
    out = pd.DataFrame({"date": flagged.index, "value": daily.loc[flagged.index].to_numpy(),
                        "kind": np.where(flagged.to_numpy() > 0, "high", "low"), "score": flagged.to_numpy()})
    return out.sort_values("date").reset_index(drop=True)


def compute_rfm(df):
    """Recency / Frequency / Monetary scoring grouped into four plain segments."""
    g = df.groupby("customer").agg(last=("date", "max"), F=("date", "count"), M=("amount", "sum"))
    if len(g) < 8:
        return None
    g["R"] = (df["date"].max() + pd.Timedelta(days=1) - g["last"]).dt.days

    def score(s, ascending):
        return pd.qcut(s.rank(method="first", ascending=ascending), 4, labels=[1, 2, 3, 4]).astype(int)

    g["r"], g["f"], g["m"] = score(g["R"], False), score(g["F"], True), score(g["M"], True)

    def segment(row):
        if row.r >= 3 and row.f >= 3:
            return "Champions"
        if row.r >= 3:
            return "Recent buyers"
        if row.f >= 3 or row.m >= 3:
            return "At risk"
        return "Lost / low value"

    g["segment"] = g.apply(segment, axis=1)
    order = ["Champions", "Recent buyers", "At risk", "Lost / low value"]
    return g.groupby("segment").agg(customers=("M", "size"), revenue=("M", "sum")).reindex(order).fillna(0)


def analyse(df, has):
    r = {"has": has}
    revenue, n = float(df["amount"].sum()), len(df)
    monthly = df.set_index("date")["amount"].resample("MS").sum()
    daily = df.groupby(df["date"].dt.normalize())["amount"].sum()
    growth = None
    if len(monthly) >= 2 and monthly.iloc[-2] > 0:
        growth = float((monthly.iloc[-1] / monthly.iloc[-2] - 1) * 100)

    r.update(monthly=monthly, daily=daily, forecast=forecast_monthly(monthly), anomalies=find_anomalies(daily))
    r["kpis"] = {
        "revenue": revenue, "sales": n, "avg_sale": revenue / n,
        "units": float(df["quantity"].sum()) if has["quantity"] else None,
        "customers": int(df["customer"].nunique()) if has["customer"] else None,
        "growth": growth, "start": df["date"].min(), "end": df["date"].max(),
    }
    dd = daily.to_frame("v")
    dd["dow"], dd["mon"] = dd.index.dayofweek, dd.index.month
    r["heat"] = dd.pivot_table(index="dow", columns="mon", values="v", aggfunc="mean").reindex(index=range(7), columns=range(1, 13))
    r["weekday_avg"] = daily.groupby(daily.index.dayofweek).mean()

    r["breakdown"] = None
    if has["category"]:
        r["breakdown"] = ("Sales by category", df.groupby("category")["amount"].sum().nlargest(8))
    elif has["product"]:
        r["breakdown"] = ("Top products", df.groupby("product")["amount"].sum().nlargest(8))

    r["region"] = None
    if has["region"]:
        reg = df.groupby("region")["amount"].sum().sort_values(ascending=False)
        if len(reg) > 6:
            reg = pd.concat([reg.head(6), pd.Series({"Other": reg.iloc[6:].sum()})])
        r["region"] = reg

    r["pareto"] = None
    if has["product"] and df["product"].nunique() >= 5:
        p = df.groupby("product")["amount"].sum().sort_values(ascending=False)
        share = p.cumsum() / p.sum() * 100
        r["pareto"] = {"series": p, "cum": share, "n80": int((share < 80).sum() + 1), "total": len(p)}

    r["rfm"] = compute_rfm(df) if has["customer"] and df["customer"].nunique() >= 8 else None
    r["insights"] = build_insights(r)
    return r


def build_insights(r):
    """Plain-language findings as (label, value, note) tuples."""
    out, k, m = [], r["kpis"], r["monthly"]
    if k["growth"] is not None:
        out.append(("Month on month", f"{k['growth']:+.1f}%", f"{m.index[-1]:%b %Y} against {m.index[-2]:%b %Y}"))
    if r["breakdown"] is not None:
        title, s = r["breakdown"]
        out.append(("Best category" if "category" in title else "Best product", str(s.index[0]),
                    f"{s.iloc[0] / k['revenue'] * 100:.0f}% of all sales"))
    if r["pareto"]:
        p = r["pareto"]
        out.append(("Products that matter", f"{p['n80']} of {p['total']}", "bring in 80% of sales; keep these in stock"))
    wd = r["weekday_avg"]
    if len(wd) >= 5:
        out.append(("Busiest weekday", WEEKDAYS[int(wd.idxmax())],
                    f"averages {fmt(wd.max())} a day; {WEEKDAYS[int(wd.idxmin())]} is slowest at {fmt(wd.min())}"))
    if len(m) >= 6:
        by_month = m.groupby(m.index.month).mean()
        out.append(("Peak month", MONTHS[int(by_month.idxmax()) - 1],
                    f"averages {fmt(by_month.max())} against {fmt(by_month.mean())} for a typical month"))
    fc = r["forecast"]
    if fc is not None:
        nxt, last = fc.iloc[0], m.iloc[-1]
        change = (nxt["pred"] / last - 1) * 100 if last else 0
        out.append((f"Forecast {fc.index[0]:%b %Y}", fmt(nxt["pred"]),
                    f"{change:+.0f}% on last month; likely range {fmt(nxt['low'])} to {fmt(nxt['high'])}"))
    an = r["anomalies"]
    if len(an):
        out.append(("Unusual days", str(len(an)), f"latest: {an.iloc[-1]['date']:%d %b %Y}, unusually {an.iloc[-1]['kind']}"))
    if r["rfm"] is not None:
        risk = r["rfm"].loc["At risk"]
        if risk["customers"] > 0:
            out.append(("Customers at risk", str(int(risk["customers"])),
                        f"good buyers who have gone quiet; {fmt(risk['revenue'])} in past sales"))
    return out


# =========================================================================== 3. CHARTS
def _empty(ax, title, message):
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTE, transform=ax.transAxes)


def chart_trend(ax, r):
    m, fc = r["monthly"], r["forecast"]
    if len(m) < 2:
        return _empty(ax, "Monthly sales and forecast", "Needs at least two months of data")
    ax.plot(m.index, m.values, color=TEAL, lw=2, marker="o", ms=3.5, label="Actual sales")
    if fc is not None:
        ax.plot([m.index[-1], *fc.index], [m.values[-1], *fc["pred"]], ls="--", color="#C99700", lw=2, marker="o", ms=4, label="Forecast")
        ax.fill_between(fc.index, fc["low"], fc["high"], color=TAG, alpha=0.25, lw=0, label="Likely range")
    ax.yaxis.set_major_formatter(money)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_ylim(bottom=0)
    ax.set_title("Monthly sales and 3-month forecast")
    ax.legend(frameon=False, loc="upper left", ncol=3)


def chart_breakdown(ax, r):
    if r["breakdown"] is None:
        return _empty(ax, "Sales breakdown", "No category or product column found")
    title, s = r["breakdown"]
    s = s.sort_values()
    bars = ax.barh([str(i)[:18] for i in s.index], s.values, color=TEAL, height=0.62)
    ax.bar_label(bars, labels=[fmt(v) for v in s.values], padding=3, color=INK, fontsize=8)
    ax.xaxis.set_major_formatter(money)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=s.max() * 1.18)
    ax.set_title(title)


def chart_region(ax, r):
    s = r["region"]
    if s is None or len(s) < 2:
        return _empty(ax, "Share of sales by region", "No region column found")
    ax.grid(False)
    wedges, _ = ax.pie(s.values, startangle=90, counterclock=False, colors=CYCLE[:len(s)],
                       wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
    pct = s.values / s.values.sum() * 100
    ax.legend(wedges, [f"{str(n)[:16]}  {p:.0f}%" for n, p in zip(s.index, pct)],
              loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False)
    ax.set_title("Share of sales by region")


def chart_pareto(ax, r):
    p = r["pareto"]
    if not p:
        return _empty(ax, "Pareto (80/20)", "Needs a product column with 5+ products")
    top, cum = p["series"].head(15), p["cum"].head(15)
    ax.bar(range(len(top)), top.values, color=TEAL, width=0.7)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([str(i)[:14] for i in top.index], rotation=40, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(money)
    ax2 = ax.twinx()
    ax2.plot(range(len(top)), cum.values, color=INK, marker="o", ms=3, lw=1.5)
    ax2.axhline(80, color=ALERT, ls="--", lw=1)
    ax2.text(len(top) - 0.6, 81.5, "80% of sales", color=ALERT, ha="right", fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(LINE)
    ax.set_title(f"Pareto: {p['n80']} of {p['total']} products bring in 80% of sales")


def chart_heatmap(ax, r):
    heat = r["heat"]
    if heat.notna().sum().sum() < 7:
        return _empty(ax, "Average daily sales", "Not enough data")
    cmap = LinearSegmentedColormap.from_list("shop", ["#F1F4EE", "#9ACFCB", TEAL, "#0A4F4F"])
    ax.grid(False)
    im = ax.imshow(np.ma.masked_invalid(heat.to_numpy(dtype=float)), aspect="auto", cmap=cmap)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS, fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels(WEEKDAYS)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.yaxis.set_major_formatter(money)
    cb.outline.set_visible(False)
    ax.set_title("Average daily sales by weekday and month")


def chart_anomalies(ax, r):
    daily, an = r["daily"], r["anomalies"]
    if len(daily) < 28:
        return _empty(ax, "Unusual sales days", "Needs at least 28 days of data")
    ax.plot(daily.index, daily.values, color="#9AA5B1", lw=0.8, label="Daily sales")
    for kind, color, label in (("high", TEAL, "Unusually high"), ("low", ALERT, "Unusually low")):
        part = an[an["kind"] == kind]
        if len(part):
            ax.scatter(part["date"], part["value"], s=34, color=color, edgecolor=INK, zorder=3, label=label)
    ax.yaxis.set_major_formatter(money)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_ylim(bottom=0)
    ax.set_title("Unusual sales days")
    ax.legend(frameon=False, loc="upper left", ncol=3)


def chart_segments(ax, r):
    t = r["rfm"]
    if t is None:
        return _empty(ax, "Customer segments", "Needs a customer column with 8+ customers")
    t = t.iloc[::-1]
    colors = {"Champions": TEAL, "Recent buyers": "#7FB7B5", "At risk": TAG, "Lost / low value": "#B8C0C9"}
    bars = ax.barh(t.index, t["revenue"], color=[colors[i] for i in t.index], height=0.6)
    ax.bar_label(bars, labels=[f"{int(c)} customers" for c in t["customers"]], padding=4, color=INK, fontsize=8)
    ax.xaxis.set_major_formatter(money)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=t["revenue"].max() * 1.35)
    ax.set_title("Customer segments by lifetime sales")


def panel_findings(ax, r):
    """The 'receipt' of plain-language findings."""
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", edgecolor=LINE, lw=1))
    ax.text(0.06, 0.95, "What the data says", fontsize=14, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    y = 0.86
    step = 0.86 / max(len(r["insights"]), 1)
    for label, value, note in r["insights"]:
        ax.text(0.06, y, label, fontsize=10, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
        ax.text(0.94, y, value, fontsize=11, fontweight="bold", color="#0A5F5E", va="top", ha="right", transform=ax.transAxes)
        ax.text(0.06, y - 0.033, "\n".join(textwrap.wrap(note, 46)), fontsize=8, color=MUTE, va="top", transform=ax.transAxes)
        y -= min(step, 0.115)


def draw_kpis(fig, r, source):
    k = r["kpis"]
    fig.text(0.03, 0.955, "ShopSage sales report", fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.915, f"{source}  |  {k['start']:%d %b %Y} to {k['end']:%d %b %Y}  |  {k['sales']:,} sales",
             fontsize=9.5, color=MUTE, va="top")
    tiles = [("Revenue", fmt(k["revenue"])), ("Average sale", fmt(k["avg_sale"]))]
    if k["customers"] is not None:
        tiles.append(("Customers", f"{k['customers']:,}"))
    if k["units"] is not None:
        tiles.append(("Units sold", fmt(k["units"])))
    if k["growth"] is not None:
        tiles.append(("Month on month", f"{k['growth']:+.1f}%"))
    for i, (label, value) in enumerate(tiles):
        x = 0.50 + i * 0.095
        color = (TEAL if k["growth"] >= 0 else ALERT) if label == "Month on month" else INK
        fig.text(x, 0.955, value, fontsize=16, fontweight="bold", color=color, va="top")
        fig.text(x, 0.918, label, fontsize=8.5, color=MUTE, va="top")


def build_pages(r, source):
    """Page 1: overview. Page 2: deeper analysis."""
    f1 = plt.figure(figsize=(16, 9))
    g = f1.add_gridspec(2, 3, left=0.04, right=0.98, top=0.86, bottom=0.07, hspace=0.38, wspace=0.28, width_ratios=[1, 1, 0.85])
    chart_trend(f1.add_subplot(g[0, 0:2]), r)
    panel_findings(f1.add_subplot(g[:, 2]), r)
    chart_breakdown(f1.add_subplot(g[1, 0]), r)
    chart_region(f1.add_subplot(g[1, 1]), r)
    draw_kpis(f1, r, source)

    f2 = plt.figure(figsize=(16, 9))
    g = f2.add_gridspec(2, 3, left=0.05, right=0.97, top=0.92, bottom=0.10, hspace=0.45, wspace=0.28, width_ratios=[1, 1, 0.9])
    chart_pareto(f2.add_subplot(g[0, 0:2]), r)
    chart_heatmap(f2.add_subplot(g[0, 2]), r)
    chart_anomalies(f2.add_subplot(g[1, 0:2]), r)
    chart_segments(f2.add_subplot(g[1, 2]), r)
    return f1, f2


# =========================================================================== 4. DEMO DATA
def make_sample(path, seed=7):
    """Two years of demo sales with seasonality, weekend peaks and a few odd days."""
    rng = np.random.default_rng(seed)
    products = [("Rice 5kg", "Grocery", 320), ("Cooking Oil", "Grocery", 150), ("Tea 250g", "Grocery", 120),
                ("Cola 2L", "Beverages", 90), ("Fruit Juice", "Beverages", 110), ("Shampoo", "Personal Care", 180),
                ("Soap Pack", "Personal Care", 95), ("Earphones", "Electronics", 899), ("Power Bank", "Electronics", 1499),
                ("USB Cable", "Electronics", 249), ("Water Bottle", "Home & Kitchen", 349), ("Mixer Jar", "Home & Kitchen", 1899)]
    regions = ["North", "South", "East", "West", "Central"]
    dow = [0.85, 0.85, 0.9, 0.95, 1.15, 1.45, 1.3]
    month = [0.9, 0.85, 0.95, 1.0, 1.0, 0.9, 0.9, 1.0, 1.05, 1.25, 1.3, 1.5]
    special = {"2023-03-18": 5.0, "2023-10-21": 4.2, "2024-02-14": 4.0, "2024-08-15": 5.5, "2023-06-07": 0.08, "2024-05-09": 0.06}
    pw = 1 / np.arange(1, len(products) + 1) ** 0.9
    pw = rng.permutation(pw) / pw.sum()
    rows = []
    for i, day in enumerate(pd.date_range("2023-01-01", "2024-12-31")):
        lam = 22 * dow[day.dayofweek] * month[day.month - 1] * (1 + 0.2 * i / 730) * special.get(f"{day:%Y-%m-%d}", 1.0)
        for _ in range(rng.poisson(lam)):
            name, cat, price = products[rng.choice(len(products), p=pw)]
            qty = int(rng.choice([1, 1, 1, 2])) if price > 800 else int(rng.integers(1, 5))
            rows.append((f"{day:%Y-%m-%d}", f"C{rng.integers(1, 500):04d}", name, cat, rng.choice(regions), qty, price,
                         round(qty * price * float(rng.choice([1, 1, 1, 0.95, 0.9])), 2)))
    pd.DataFrame(rows, columns=["Order Date", "Customer ID", "Product", "Category", "Region", "Quantity", "Unit Price", "Sales"]
                 ).to_csv(path, index=False)
    return Path(path)


# =========================================================================== 5. MAIN
def pick_file():
    """Open a file-picker window (needs tkinter, which ships with Python on Windows and macOS)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(title="Choose your sales CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        root.destroy()
        return chosen or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="ShopSage: smart retail sales dashboard from a CSV file.")
    ap.add_argument("csv", nargs="?", help="path to your sales CSV (leave empty to pick a file)")
    ap.add_argument("--sample", action="store_true", help="create and use a demo dataset")
    ap.add_argument("--out", default="shopsage_output", help="folder for the saved files")
    ap.add_argument("--no-show", action="store_true", help="save files without opening chart windows")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.sample:
        path = make_sample(out / "sample_sales.csv")
        print(f"Demo data written to {path}")
    else:
        path = args.csv or pick_file()
        if not path:
            sys.exit("No file chosen. Run: python shopsage_dashboard.py your_sales.csv")
    path = Path(path)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    try:
        df, has, quality, cols = load_and_clean(path)
    except ValueError as err:
        sys.exit(f"Could not read the file: {err}")

    print("\nHow your file was read")
    print("  Columns matched: " + ", ".join(f"{role} = '{col}'" for role, col in cols.items()))
    for label, value in quality.items():
        print(f"  {label}: {value:,}")

    res = analyse(df, has)
    print("\nWhat the data says")
    for label, value, note in res["insights"]:
        print(f"  {label:<22}{value:<16}{note}")

    page1, page2 = build_pages(res, path.name)
    page1.savefig(out / "dashboard_overview.png", dpi=150)
    page2.savefig(out / "dashboard_details.png", dpi=150)
    with PdfPages(out / "shopsage_report.pdf") as pdf:
        pdf.savefig(page1)
        pdf.savefig(page2)
    print(f"\nSaved in '{out}': dashboard_overview.png, dashboard_details.png, shopsage_report.pdf")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()