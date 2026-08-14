"""
================================================================================
 HOTEL BOOKING DEMAND — REVENUE-AT-RISK DASHBOARD
 Python for Managers | IIM Rohtak
--------------------------------------------------------------------------------
 8 KPIs (2 rows x 4) + 8 visualisations, ordered as a 10-minute narrative:
 the problem -> where it concentrates -> why it happens -> when to see it
 coming -> where to target it -> the lever -> the geography -> pricing context.

 Data   : Kaggle "Hotel Booking Demand" (Antonio, Almeida & Nunes, 2019)
          https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
 Run    : streamlit run app.py   (see the README block at the bottom of this
          file, or the accompanying HOW_TO_RUN.md, for full setup steps)
================================================================================
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# 0. PAGE CONFIG + LOOK & FEEL
#    Neutral, "human" palette instead of a default AI-chart look:
#    navy (structure/text), gold (headline numbers), warm brown (secondary),
#    cream (backgrounds), with a restrained red/green pair reserved ONLY
#    for risk vs. safe signals so it stays meaningful, not decorative.
# --------------------------------------------------------------------------
st.set_page_config(page_title="Hotel Booking Demand | Revenue at Risk",
                    layout="wide", page_icon="🏨")

NAVY = "#1F2A44"
GOLD = "#C9A24B"
BROWN = "#8C5A3C"
CREAM = "#F6F1E7"
SAND = "#EFE3CC"
RISK = "#B44337"     # muted brick red — cancellations / loss
SAFE = "#4C7A5E"      # muted forest green — retained / realised
PALETTE = [NAVY, GOLD, BROWN, "#6E8B9E", "#A85751", "#D9B36C"]

st.markdown(f"""
<style>
.stApp {{ background-color: {CREAM}; }}
.kpi-card {{
    background-color: white; border-radius: 10px; padding: 14px 16px;
    border-left: 6px solid {NAVY}; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}}
.kpi-card.risk {{ border-left-color: {RISK}; }}
.kpi-card.safe {{ border-left-color: {SAFE}; }}
.kpi-label {{ font-size: 0.78rem; color: #555; text-transform: uppercase;
              letter-spacing: 0.03em; margin-bottom: 2px; }}
.kpi-value {{ font-size: 1.55rem; color: {NAVY}; font-weight: 700; }}
.kpi-sub {{ font-size: 0.75rem; color: #777; margin-top: 2px; }}
h1, h2, h3 {{ color: {NAVY}; }}
.section-note {{ background-color: {SAND}; padding: 10px 14px; border-radius: 8px;
                  font-size: 0.92rem; color: {NAVY}; margin-bottom: 10px; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 1. LOAD + CLEAN DATA
#    Cleaning choices are stated explicitly so the KPI numbers on this page
#    can be reconciled against the written spec (Final_Dashboard_Spec_8x8).
#    Adjust / remove any step below if your course spec defines cleaning
#    differently — every KPI is computed live from `df`, nothing is hard-coded.
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fill known-safe missing values instead of dropping rows for them
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)

    # Drop rows that cannot be real bookings: zero total guests, or a
    # negative / implausible ADR. Duplicates are deliberately RETAINED —
    # a guest can legitimately book the same room type twice.
    total_guests = df["adults"] + df["children"] + df["babies"]
    df = df[total_guests > 0]
    df = df[df["adr"] >= 0]
    df = df[df["adr"] < df["adr"].quantile(0.999)]  # trim extreme ADR outliers

    # Derived fields used throughout the dashboard
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df = df[df["total_nights"] > 0]
    df["revenue"] = df["adr"] * df["total_nights"]          # value per booking
    df["is_canceled"] = df["is_canceled"].astype(int)

    month_order = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
    df["arrival_date_month"] = pd.Categorical(df["arrival_date_month"],
                                               categories=month_order, ordered=True)
    return df


DATA_PATH = "hotel_bookings.csv"   # <- place the Kaggle CSV next to this script
try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Download it from the Kaggle dataset "
              "'Hotel Booking Demand' and place it in the same folder as app.py.")
    st.stop()


# --------------------------------------------------------------------------
# 2. SIDEBAR FILTERS (keeps the dashboard interactive, not just decorative)
# --------------------------------------------------------------------------
st.sidebar.header("Filters")
hotel_choice = st.sidebar.multiselect("Hotel type", options=df_raw["hotel"].unique().tolist(),
                                       default=df_raw["hotel"].unique().tolist())
year_choice = st.sidebar.multiselect("Arrival year", options=sorted(df_raw["arrival_date_year"].unique().tolist()),
                                      default=sorted(df_raw["arrival_date_year"].unique().tolist()))
segment_choice = st.sidebar.multiselect("Market segment", options=sorted(df_raw["market_segment"].unique().tolist()),
                                         default=sorted(df_raw["market_segment"].unique().tolist()))

df = df_raw[df_raw["hotel"].isin(hotel_choice) &
            df_raw["arrival_date_year"].isin(year_choice) &
            df_raw["market_segment"].isin(segment_choice)]

if df.empty:
    st.warning("No rows match the current filters — widen a selection in the sidebar.")
    st.stop()


# --------------------------------------------------------------------------
# 3. KPI COMPUTATION — every number below is derived live from `df`
# --------------------------------------------------------------------------
bookings = len(df)
cancelled = int(df["is_canceled"].sum())
cancel_rate = cancelled / bookings

mean_lead = df["lead_time"].mean()
median_lead = df["lead_time"].median()

direct_share_market = (df["market_segment"] == "Direct").mean()
direct_share_channel = (df["distribution_channel"] == "Direct").mean()

adr_mean = df["adr"].mean()
adr_median = df["adr"].median()
adr_city = df.loc[df["hotel"].str.contains("City", case=False), "adr"].mean()
adr_resort = df.loc[df["hotel"].str.contains("Resort", case=False), "adr"].mean()

rev_per_booking_mean = df["revenue"].mean()
rev_per_booking_median = df["revenue"].median()

potential_revenue = df["revenue"].sum()
lost_revenue = df.loc[df["is_canceled"] == 1, "revenue"].sum()
pct_of_potential_lost = lost_revenue / potential_revenue
realised_revenue = potential_revenue - lost_revenue
realisation_rate = realised_revenue / potential_revenue


def kpi_card(col, label, value, sub=None, kind=""):
    css_class = f"kpi-card {kind}".strip()
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(f"""
        <div class="{css_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 4. HEADER + KPI GRID (two rows of four, grouped by theme)
# --------------------------------------------------------------------------
st.title("🏨 Hotel Booking Demand — Revenue-at-Risk Dashboard")
st.caption("Hotel Booking Demand dataset · Python for Managers · IIM Rohtak")

st.markdown('<div class="section-note"><b>Row 1</b> answers: how much business, '
            'and how reliable is it? &nbsp; <b>Row 2</b> answers: what is it worth, '
            'and how much do we keep?</div>', unsafe_allow_html=True)

r1 = st.columns(4)
kpi_card(r1[0], "Bookings", f"{bookings:,}", f"{cancelled:,} cancelled")
kpi_card(r1[1], "Cancellation rate", f"{cancel_rate:.2%}", "The headline risk number", kind="risk")
kpi_card(r1[2], "Mean lead time", f"{mean_lead:,.1f} days", f"Median: {median_lead:,.0f} days")
kpi_card(r1[3], "Direct-booking share", f"{direct_share_market:.1%}",
         f"By channel field: {direct_share_channel:.2%}")

r2 = st.columns(4)
kpi_card(r2[0], "Average daily rate", f"€{adr_mean:,.2f}",
         f"Median €{adr_median:,.2f} · City €{adr_city:,.2f} vs Resort €{adr_resort:,.2f}")
kpi_card(r2[1], "Revenue per booking", f"€{rev_per_booking_mean:,.2f}",
         f"Median €{rev_per_booking_median:,.2f}")
kpi_card(r2[2], "Revenue at risk", f"€{lost_revenue/1e6:,.2f}M",
         f"{pct_of_potential_lost:.2%} of €{potential_revenue/1e6:,.2f}M potential", kind="risk")
kpi_card(r2[3], "Realisation rate", f"{realisation_rate:.2%}",
         "Realised ÷ potential revenue", kind="safe")

st.divider()


# --------------------------------------------------------------------------
# 5. CHART 1 — Potential -> Realised revenue (waterfall)
# --------------------------------------------------------------------------
st.subheader("1. Where the money goes: potential → realised revenue")
fig1 = go.Figure(go.Waterfall(
    orientation="v",
    measure=["absolute", "relative", "total"],
    x=["Potential revenue", "Lost to cancellations", "Realised revenue"],
    y=[potential_revenue, -lost_revenue, realised_revenue],
    text=[f"€{potential_revenue/1e6:.2f}M", f"−€{lost_revenue/1e6:.2f}M", f"€{realised_revenue/1e6:.2f}M"],
    textposition="outside",
    connector={"line": {"color": "#999"}},
    decreasing={"marker": {"color": RISK}},
    increasing={"marker": {"color": SAFE}},
    totals={"marker": {"color": NAVY}},
))
fig1.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                    yaxis_title="Euros")
st.plotly_chart(fig1, use_container_width=True)


# --------------------------------------------------------------------------
# 6. CHART 2 — Revenue-loss concentration (Pareto: bar + cumulative line)
# --------------------------------------------------------------------------
st.subheader("2. The loss is not spread — it has an address")
loss_by_segment = (df[df["is_canceled"] == 1]
                    .groupby("market_segment", observed=True)["revenue"].sum()
                    .sort_values(ascending=False))
cum_pct = loss_by_segment.cumsum() / loss_by_segment.sum()

fig2 = go.Figure()
fig2.add_bar(x=loss_by_segment.index, y=loss_by_segment.values, name="Revenue lost",
             marker_color=GOLD)
fig2.add_trace(go.Scatter(x=loss_by_segment.index, y=cum_pct.values, name="Cumulative %",
                           yaxis="y2", mode="lines+markers", line=dict(color=NAVY, width=3)))
fig2.update_layout(
    yaxis=dict(title="Revenue lost (€)"),
    yaxis2=dict(title="Cumulative share", overlaying="y", side="right", tickformat=".0%", range=[0, 1.05]),
    plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------------------------------
# 7. CHART 3 — Cancellation rate by deposit type (horizontal bar, emphasised)
# --------------------------------------------------------------------------
st.subheader("3. Deposit type is the single strongest cancellation signal")
dep = df.groupby("deposit_type", observed=True)["is_canceled"].mean().sort_values()
colors3 = [RISK if d == "Non Refund" else BROWN for d in dep.index]
fig3 = go.Figure(go.Bar(x=dep.values, y=dep.index, orientation="h",
                         marker_color=colors3,
                         text=[f"{v:.1%}" for v in dep.values], textposition="outside"))
fig3.update_layout(xaxis_title="Cancellation rate", xaxis_tickformat=".0%",
                    plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------------------------------
# 8. CHART 4 — Risk by booking lead time (line + area fill)
# --------------------------------------------------------------------------
st.subheader("4. Risk is visible the moment the booking is made")
bins = [-1, 7, 30, 90, 180, 365, 10000]
labels = ["0–7 days", "8–30 days", "31–90 days", "91–180 days", "181–365 days", "366+ days"]
df["lead_bin"] = pd.cut(df["lead_time"], bins=bins, labels=labels)
lead_risk = df.groupby("lead_bin", observed=True)["is_canceled"].mean()

fig4 = go.Figure(go.Scatter(x=lead_risk.index.astype(str), y=lead_risk.values,
                             mode="lines+markers", fill="tozeroy",
                             line=dict(color=RISK, width=3), fillcolor="rgba(180,67,55,0.18)"))
fig4.update_layout(yaxis_title="Cancellation rate", yaxis_tickformat=".0%",
                    xaxis_title="Lead time at booking", plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig4, use_container_width=True)


# --------------------------------------------------------------------------
# 9. CHART 5 — Lead time x market segment (heatmap, values in-cell)
# --------------------------------------------------------------------------
st.subheader("5. The overbooking-protection targeting grid")
heat = df.pivot_table(index="market_segment", columns="lead_bin",
                       values="is_canceled", aggfunc="mean", observed=True)
fig5 = px.imshow(heat, text_auto=".0%", color_continuous_scale=["#F6F1E7", GOLD, RISK],
                  aspect="auto", labels=dict(color="Cancellation rate"))
fig5.update_layout(plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig5, use_container_width=True)


# --------------------------------------------------------------------------
# 10. CHART 6 — Channel risk-value map (bubble scatter)
# --------------------------------------------------------------------------
st.subheader("6. Same rate, less risk: the case for direct bookings")
chan = df.groupby("market_segment", observed=True).agg(
    adr=("adr", "mean"), cancel_rate=("is_canceled", "mean"), volume=("is_canceled", "size")).reset_index()
fig6 = px.scatter(chan, x="cancel_rate", y="adr", size="volume", color="market_segment",
                   text="market_segment", size_max=60,
                   color_discrete_sequence=PALETTE,
                   labels={"cancel_rate": "Cancellation rate", "adr": "Average daily rate (€)"})
fig6.update_traces(textposition="top center")
fig6.update_layout(xaxis_tickformat=".0%", plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False)
st.plotly_chart(fig6, use_container_width=True)


# --------------------------------------------------------------------------
# 11. CHART 7 — Top source markets, coloured by risk
# --------------------------------------------------------------------------
st.subheader("7. Where the bookings — and the losses — come from")
top_n = 10
country_stats = df.groupby("country", observed=True).agg(
    bookings=("is_canceled", "size"), cancel_rate=("is_canceled", "mean")).sort_values(
    "bookings", ascending=False).head(top_n).reset_index()
fig7 = px.bar(country_stats.sort_values("bookings"), x="bookings", y="country", orientation="h",
              color="cancel_rate", color_continuous_scale=["#4C7A5E", GOLD, RISK],
              labels={"bookings": "Bookings", "country": "", "cancel_rate": "Cancellation rate"})
fig7.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                    coloraxis_colorbar_tickformat=".0%")
st.plotly_chart(fig7, use_container_width=True)


# --------------------------------------------------------------------------
# 12. CHART 8 — ADR seasonality by property type (two-series line)
# --------------------------------------------------------------------------
st.subheader("8. Two different businesses in one portfolio")
season = df.groupby(["arrival_date_month", "hotel"], observed=True)["adr"].mean().reset_index()
fig8 = px.line(season, x="arrival_date_month", y="adr", color="hotel", markers=True,
                color_discrete_sequence=[NAVY, GOLD],
                labels={"arrival_date_month": "", "adr": "Average daily rate (€)", "hotel": ""})
fig8.update_layout(plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig8, use_container_width=True)

st.divider()
st.caption("Built for the Python for Managers project · KPIs and charts follow the "
           "Final Dashboard Spec (8 KPIs, 8 visualisations) · Data: Kaggle Hotel Booking Demand")