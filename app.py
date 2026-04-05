import os
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai_lib import (
    incidents_df,
    industries_df,
    industry_long_df,
    severity_df,
    stakeholders_df,
    stakeholder_time_series_df,
)
from ai_lib.openai_api import get_article_component_data


st.set_page_config(
    page_title="Reported AI Cases Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

stakeholder_df = stakeholder_time_series_df.copy()
industry_df = industry_long_df.copy()
industry_df["Industry"] = industry_df["Industry"].replace(
    {
        "Government & security & defence": "Government, security, and defense",
        "Government / security / defence": "Government, security, and defense",
        "Government, security, and defence": "Government, security, and defense",
        "Media & social platforms & marketing": "Media, social platforms, and marketing",
        "Media / social platforms / marketing": "Media, social platforms, and marketing",
        "Robots & sensors & IT hardware": "Robots, sensors, and IT hardware",
        "Robots / sensors / IT hardware": "Robots, sensors, and IT hardware",
        "Healthcare & drugs & biotechnology": "Healthcare, drugs, and biotechnology",
        "Healthcare / drugs / biotechnology": "Healthcare, drugs, and biotechnology",
        "Arts & entertainment &": "Arts, entertainment, and recreation",
        "Arts & entertainment & recreation": "Arts, entertainment, and recreation",
        "Arts / entertainment / recreation": "Arts, entertainment, and recreation",
    }
)
industry_monthly_df = industries_df.copy()
industry_monthly_df = industry_monthly_df.rename(
    columns={
        "Government & security & defence": "Government, security, and defense",
        "Government / security / defence": "Government, security, and defense",
        "Government, security, and defence": "Government, security, and defense",
        "Media & social platforms & marketing": "Media, social platforms, and marketing",
        "Media / social platforms / marketing": "Media, social platforms, and marketing",
        "Robots & sensors & IT hardware": "Robots, sensors, and IT hardware",
        "Robots / sensors / IT hardware": "Robots, sensors, and IT hardware",
        "Healthcare & drugs & biotechnology": "Healthcare, drugs, and biotechnology",
        "Healthcare / drugs / biotechnology": "Healthcare, drugs, and biotechnology",
        "Arts & entertainment &": "Arts, entertainment, and recreation",
        "Arts & entertainment & recreation": "Arts, entertainment, and recreation",
        "Arts / entertainment / recreation": "Arts, entertainment, and recreation",
    }
)
industry_monthly_df["Date"] = pd.to_datetime(
    industry_monthly_df["Date"], format="%Y-%m", errors="coerce"
)
industry_monthly_df = industry_monthly_df.dropna(subset=["Date"]).copy()
industry_monthly_df["Year"] = industry_monthly_df["Date"].dt.year
industry_monthly_df = industry_monthly_df.melt(
    id_vars=["Date", "Year"],
    value_vars=[column for column in industry_monthly_df.columns if column not in ["Date", "Year"]],
    var_name="Industry",
    value_name="Count",
)
incidents_over_time_df = incidents_df.copy()
incidents_over_time_df["Date"] = pd.to_datetime(incidents_over_time_df["Date"])
severity_over_time_df = severity_df.copy()
severity_over_time_df["Date"] = pd.to_datetime(severity_over_time_df["Date"])
STAKEHOLDER_COLORS = ["#1f1f1f", "#4a4a4a", "#7b7b7b", "#8e3b46", "#b85c6b"]
INDUSTRY_SCALE = ["#f2f2f2", "#b5b5b5", "#2a2a2a"]
STACKED_AREA_COLORS = [
    "#1f1f1f",
    "#4a4a4a",
    "#7b7b7b",
    "#8e3b46",
    "#b85c6b",
    "#c78d96",
]
YEAR_FILTER_OPTIONS = {
    "All years": (2020, 2025),
    "2020-2022": (2020, 2022),
    "2023-2025": (2023, 2025),
}
STORY_GRADIENT = "linear-gradient(135deg, rgba(248,248,248,0.98), rgba(236,236,236,0.98))"
CASE_STUDY_CACHE_VERSION = 4


def inject_page_styles() -> None:
    """Apply a warmer editorial style so the app feels curated, not default."""
    stylesheet_path = Path(__file__).resolve().parent / "styles" / "front_end.css"
    st.markdown(f"<style>{stylesheet_path.read_text()}</style>", unsafe_allow_html=True)


def style_chart(fig: go.Figure, height: int = 420) -> go.Figure:
    """Use one shared chart style for a more cohesive visual identity."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.42)",
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(color="#262626", size=14),
        title_font=dict(size=20, color="#111111"),
        legend_font=dict(color="#262626"),
        legend_title_text="",
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#262626"),
        title_font=dict(color="#262626"),
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        tickfont=dict(color="#262626"),
        title_font=dict(color="#262626"),
    )
    return fig


@st.cache_data(show_spinner=False)
def load_case_study(
    stakeholder: str,
    case_index: int = 0,
    cache_version: int = CASE_STUDY_CACHE_VERSION,
) -> dict[str, str]:
    """Load one AI case study summary for the selected stakeholder."""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    _ = cache_version
    return get_article_component_data(stakeholder, case_index=case_index, api_key=api_key)


@st.cache_data(show_spinner=False)
def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Return one dataframe as UTF-8 CSV bytes for download buttons."""
    export_df = df.copy()
    if "Date" in export_df.columns:
        export_df["Date"] = export_df["Date"].astype(str)
    return export_df.to_csv(index=False).encode("utf-8")


def render_dataset_preview(
    title: str,
    copy: str,
    df: pd.DataFrame,
    download_name: str,
    preview_rows: int = 10,
) -> None:
    """Render one short dataset preview plus a CSV download."""
    preview_df = df.head(preview_rows).copy()
    if "Date" in preview_df.columns:
        preview_df["Date"] = preview_df["Date"].astype(str)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_intro("Dataset Notes", title, copy)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    st.download_button(
        label=f"Download {title} CSV",
        data=dataframe_to_csv(df),
        file_name=download_name,
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, copy: str) -> None:
    """Render one summary stat as a styled card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(label: str, text: str) -> None:
    """Render one short interpretive takeaway."""
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card_with_note(label: str, note: str, text: str) -> None:
    """Render one insight card with a scoped note under the label."""
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-note"><em>{note}</em></div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_stat_card(label: str, value: str, copy: str = "") -> None:
    """Render one compact rail stat."""
    copy_html = f'<div class="mini-stat-copy">{copy}</div>' if copy else ""
    st.markdown(
        f"""
        <div class="mini-stat-card">
            <div class="mini-stat-label">{label}</div>
            <div class="mini-stat-value">{value}</div>
            {copy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_intro(kicker: str, title: str, copy: str) -> None:
    """Render a compact section intro used throughout the story."""
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def compact_section_intro(kicker: str, title: str, copy: str) -> None:
    """Render a tighter section intro for companion cards."""
    st.markdown(
        f"""
        <div class="compact-section-kicker">{kicker}</div>
        <div class="compact-section-title">{title}</div>
        <div class="compact-section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def story_bridge_card(title: str, copy: str) -> None:
    """Render a short editorial bridge between chart and example."""
    st.markdown(
        f"""
        <div class="story-bridge-card">
            <div class="story-bridge-title">{title}</div>
            <div class="story-bridge-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def story_bridge_title(title: str) -> None:
    """Render only the editorial bridge title."""
    st.markdown(
        f"""
        <div class="story-bridge-card">
            <div class="story-bridge-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def story_bridge_copy(copy: str) -> None:
    """Render only the editorial bridge copy."""
    st.markdown(
        f"""
        <div class="story-bridge-copy standalone-story-bridge-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def build_story_metrics() -> dict[str, str]:
    """Compute headline figures for the opening section."""
    story_stakeholder_df = stakeholder_df[stakeholder_df["Year"].between(2020, 2025)].copy()
    in_depth_stakeholder_df = story_stakeholder_df[
        ~story_stakeholder_df["Stakeholder"].isin(["General public", "Other"])
    ].copy()
    story_industry_df = industry_df[industry_df["Year"].between(2020, 2025)].copy()
    story_incidents_df = incidents_over_time_df[
        incidents_over_time_df["Date"].dt.year.between(2020, 2025)
    ].copy()
    story_incidents_df["Year"] = story_incidents_df["Date"].dt.year

    total_incidents = int(story_incidents_df["Total Incidents & Hazards"].sum())

    stakeholder_totals = (
        in_depth_stakeholder_df.groupby("Stakeholder", as_index=False)["Count"].sum()
        .sort_values("Count", ascending=False)
    )
    top_stakeholder = stakeholder_totals.iloc[0]

    industry_totals = (
        story_industry_df.groupby("Industry", as_index=False)["Count"].sum()
        .sort_values("Count", ascending=False)
    )
    top_industry = industry_totals.iloc[0]

    yearly_totals = (
        story_incidents_df.groupby("Year", as_index=False)["Total Incidents & Hazards"]
        .sum()
        .sort_values("Year")
    )
    first_year = yearly_totals.iloc[0]
    last_year = yearly_totals.iloc[-1]
    growth_pct = (
        (
            last_year["Total Incidents & Hazards"]
            - first_year["Total Incidents & Hazards"]
        )
        / first_year["Total Incidents & Hazards"]
    ) * 100

    return {
        "total_incidents": f"{total_incidents:,}",
        "top_stakeholder": top_stakeholder["Stakeholder"],
        "top_stakeholder_count": f"{int(top_stakeholder['Count']):,}",
        "top_industry": top_industry["Industry"],
        "growth": f"{growth_pct:.0f}%",
        "first_year": str(int(first_year["Year"])),
        "last_year": str(int(last_year["Year"])),
    }


def build_story_figures() -> tuple[go.Figure, go.Figure]:
    """Create the charts used in the story tab."""
    story_stakeholder_df = stakeholder_df[stakeholder_df["Year"].between(2020, 2025)].copy()
    in_depth_stakeholder_df = story_stakeholder_df[
        ~story_stakeholder_df["Stakeholder"].isin(["General public", "Other"])
    ].copy()
    story_industry_df = industry_df[industry_df["Year"].between(2020, 2025)].copy()

    top_stakeholders = (
        in_depth_stakeholder_df.groupby("Stakeholder", as_index=False)["Count"].sum()
        .sort_values("Count", ascending=False)
        .head(5)["Stakeholder"]
        .tolist()
    )
    stakeholder_story_df = in_depth_stakeholder_df[
        in_depth_stakeholder_df["Stakeholder"].isin(top_stakeholders)
    ]

    stakeholder_fig = px.line(
        stakeholder_story_df,
        x="Year",
        y="Count",
        color="Stakeholder",
        markers=True,
        title=(
            "Reported cases by stakeholder group"
            "<br><span style='font-size:0.78em; font-style:italic; color:#666666;'>"
            "Other stakeholder groups included in this dataset are women,<br>"
            "civil society, and trade unions."
            "</span>"
        ),
        color_discrete_sequence=STAKEHOLDER_COLORS,
    )
    stakeholder_fig.update_traces(line=dict(width=3), marker=dict(size=9))
    stakeholder_fig.update_layout(hovermode="x unified")
    style_chart(stakeholder_fig, height=430)
    stakeholder_fig.update_layout(margin=dict(l=20, r=20, t=135, b=20))

    industry_story_df = (
        story_industry_df.groupby("Industry", as_index=False)["Count"].sum()
        .sort_values("Count", ascending=True)
    )
    industry_fig = px.bar(
        industry_story_df,
        x="Count",
        y="Industry",
        orientation="h",
        title="Reported cases by industry",
        color="Count",
        color_continuous_scale=INDUSTRY_SCALE,
    )
    industry_fig.update_layout(coloraxis_showscale=False)
    style_chart(industry_fig, height=460)

    return stakeholder_fig, industry_fig


def build_story_stakeholder_note() -> str:
    """Return the overview note for included and excluded stakeholder groups."""
    return (
        "Other stakeholder groups included in this dataset are Women, Civil society, and Trade unions."
    )


def build_selection_summary(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
    selected_year_label: str,
    include_2026: bool,
) -> tuple[str, str]:
    """Generate one concise narrative based on the active filters."""
    year_range_label = selected_year_label
    if selected_year_label == "All years" and include_2026:
        year_range_label = "All years plus partial 2026"

    if filtered_stakeholder_df.empty:
        return (
            f"No reported cases are available for {selected_stakeholder.lower()} in {year_range_label}.",
            "Try widening the time period or switching to all stakeholders to compare broader patterns.",
        )

    ranked = (
        filtered_stakeholder_df.groupby("Stakeholder", as_index=False)["Count"].sum()
        .sort_values("Count", ascending=False)
    )
    peak_row = filtered_stakeholder_df.sort_values("Count", ascending=False).iloc[0]

    if selected_stakeholder == "All stakeholders":
        lead = ranked.iloc[0]
        return (
            f"{lead['Stakeholder']} appears most often in the selected time window, with {int(lead['Count']):,} coded cases.",
            f"The sharpest point in the trend is {int(peak_row['Year'])}, when {peak_row['Stakeholder']} reached {int(peak_row['Count']):,} reported cases.",
        )

    total_cases = int(filtered_stakeholder_df["Count"].sum())
    return (
        f"{selected_stakeholder} appears in {total_cases:,} coded cases across {year_range_label.lower()}.",
        f"The highest annual level in this view is {int(peak_row['Year'])}, with {int(peak_row['Count']):,} reported cases affecting {selected_stakeholder.lower()}.",
    )


def build_year_window_label(selected_year_label: str, include_2026: bool) -> str:
    """Return one short label for the active time window."""
    if selected_year_label == "All years" and include_2026:
        return "Jan 2020 - Feb 2026"
    if selected_year_label == "All years":
        return "2020-2025"
    return selected_year_label


def build_fastest_growing_industry_summary(
    industry_trend_df: pd.DataFrame,
    monthly_industry_df: pd.DataFrame,
    include_2026: bool,
) -> str:
    """Return the industry with the strongest growth across the window."""
    if industry_trend_df.empty:
        return "No industry trend data is available for the selected window."

    base_trend_df = industry_trend_df[industry_trend_df["Year"] < 2026].copy()
    if base_trend_df.empty:
        base_trend_df = industry_trend_df.copy()

    year_bounds = sorted(base_trend_df["Year"].unique())
    start_year = year_bounds[0]
    end_year = year_bounds[-1]

    start_df = (
        base_trend_df[base_trend_df["Year"] == start_year][["Industry", "Count"]]
        .rename(columns={"Count": "Start Count"})
    )
    end_df = (
        base_trend_df[base_trend_df["Year"] == end_year][["Industry", "Count"]]
        .rename(columns={"Count": "End Count"})
    )
    growth_df = start_df.merge(end_df, on="Industry", how="outer").fillna(0)
    growth_df["Growth"] = growth_df["End Count"] - growth_df["Start Count"]
    growth_row = growth_df.iloc[growth_df["Growth"].argmax()]

    summary = (
        f"{growth_row['Industry']} shows the largest increase in reported cases over this period, "
        f"from {int(growth_row['Start Count']):,} cases in {start_year} "
        f"to {int(growth_row['End Count']):,} in {end_year}."
    )

    if include_2026 and not monthly_industry_df.empty and (monthly_industry_df["Year"] == 2026).any():
        latest_2026_date = monthly_industry_df.loc[
            monthly_industry_df["Year"] == 2026, "Date"
        ].max()
        comparison_month = int(latest_2026_date.month)
        ytd_df = monthly_industry_df[
            (
                (monthly_industry_df["Year"] == 2025)
                & (monthly_industry_df["Date"].dt.month <= comparison_month)
            )
            | (
                (monthly_industry_df["Year"] == 2026)
                & (monthly_industry_df["Date"].dt.month <= comparison_month)
            )
        ].copy()

        ytd_growth_df = (
            ytd_df.groupby(["Year", "Industry"], as_index=False)["Count"].sum()
            .pivot(index="Industry", columns="Year", values="Count")
            .fillna(0)
            .reset_index()
        )
        if {2025, 2026}.issubset(ytd_growth_df.columns):
            ytd_growth_df["Growth"] = ytd_growth_df[2026] - ytd_growth_df[2025]
            ytd_row = ytd_growth_df.iloc[ytd_growth_df["Growth"].argmax()]
            month_label = latest_2026_date.strftime("%B")
            summary += (
                "<br><br>"
                f"Comparing January through {month_label} in 2025 with the same months in 2026, "
                f"{ytd_row['Industry']} increases from {int(ytd_row[2025]):,} cases "
                f"to {int(ytd_row[2026]):,}."
            )
            if ytd_row["Industry"] != growth_row["Industry"]:
                summary += (
                    f" This is a different industry than the full-period leader, "
                    f"{growth_row['Industry']}."
                )

    return summary


def build_stakeholder_peak_summary(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
    include_2026: bool,
) -> str:
    """Return the peak year summary for incidents and hazards."""
    if filtered_stakeholder_df.empty:
        return "No incidents and hazards data is available for the selected view."

    year_min = int(filtered_stakeholder_df["Year"].min())
    year_max = int(filtered_stakeholder_df["Year"].max())
    yearly_totals = (
        incidents_over_time_df[
            incidents_over_time_df["Date"].dt.year.between(year_min, year_max)
        ]
        .assign(Year=lambda df: df["Date"].dt.year)
        .groupby("Year", as_index=False)["Total Incidents & Hazards"]
        .sum()
        .rename(columns={"Total Incidents & Hazards": "Count"})
    )
    yearly_totals = yearly_totals.sort_values("Year").reset_index(drop=True)
    peak_row = yearly_totals.iloc[yearly_totals["Count"].idxmax()]
    summary = (
        f"{int(peak_row['Year'])} had the highest annual total of incidents and hazards "
        f"({int(peak_row['Count']):,})."
    )

    if include_2026 and (yearly_totals["Year"] == 2026).any():
        current_2026 = float(yearly_totals.loc[yearly_totals["Year"] == 2026, "Count"].iloc[0])
        observed_months = 2
        projected_2026 = current_2026 / observed_months * 12

        prior_year_totals = yearly_totals[yearly_totals["Year"] < 2026]
        if not prior_year_totals.empty:
            prior_peak_row = prior_year_totals.iloc[prior_year_totals["Count"].idxmax()]
            if projected_2026 > float(prior_peak_row["Count"]):
                summary += (
                    "<br><br>"
                    f"If the remaining months in 2026 continue at the same pace, "
                    f"2026 will become the peak year at about {projected_2026:,.0f} cases."
                )
            else:
                summary += (
                    "<br><br>"
                    f"If the remaining months in 2026 continue at the same pace, "
                    f"2026 would still remain below the current peak of {int(prior_peak_row['Year'])}."
                )

    return summary


def build_stakeholder_peak_metric(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
    include_2026: bool,
) -> tuple[str, str]:
    """Return a compact metric-card version of the stakeholder peak summary."""
    if filtered_stakeholder_df.empty:
        return ("No data", "No stakeholder data is available for the selected view.")

    yearly_totals = (
        filtered_stakeholder_df.groupby("Year", as_index=False)["Count"].sum()
        if selected_stakeholder == "All stakeholders"
        else filtered_stakeholder_df[["Year", "Count"]].copy()
    )
    yearly_totals = yearly_totals.sort_values("Year").reset_index(drop=True)
    peak_row = yearly_totals.iloc[yearly_totals["Count"].idxmax()]

    value = f"{int(peak_row['Count']):,} total cases in {int(peak_row['Year'])}"
    return (value, "")


def build_stakeholder_total_metric(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
) -> tuple[str, str]:
    """Return a compact metric summarizing total coded cases in the active view."""
    if filtered_stakeholder_df.empty:
        return ("No data", "No stakeholder data is available for the selected view.")

    total_cases = int(filtered_stakeholder_df["Count"].sum())
    label = "All groups" if selected_stakeholder == "All stakeholders" else selected_stakeholder
    return (f"{total_cases:,}", f"{label} across the active time window.")


def build_stakeholder_change_metric(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
) -> tuple[str, str]:
    """Return a compact metric for change across the active stakeholder window."""
    if filtered_stakeholder_df.empty:
        return ("No data", "No stakeholder trend data is available for the selected view.")

    base_df = filtered_stakeholder_df[filtered_stakeholder_df["Year"] < 2026].copy()
    if base_df.empty:
        base_df = filtered_stakeholder_df.copy()

    yearly_totals = (
        base_df.groupby("Year", as_index=False)["Count"].sum()
        if selected_stakeholder == "All stakeholders"
        else base_df[["Year", "Count"]].copy()
    )
    yearly_totals = yearly_totals.sort_values("Year").reset_index(drop=True)

    if len(yearly_totals) < 2:
        only_row = yearly_totals.iloc[0]
        return (f"{int(only_row['Count']):,}", f"Only {int(only_row['Year'])} is available in this view.")

    first_row = yearly_totals.iloc[0]
    last_row = yearly_totals.iloc[-1]
    change = int(last_row["Count"] - first_row["Count"])
    direction = "up" if change >= 0 else "down"
    value = f"{abs(change):,} {direction}"
    copy = f"From {int(first_row['Year'])} to {int(last_row['Year'])}."
    return (value, copy)


def build_stakeholder_story_bridge(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
) -> tuple[str, str]:
    """Return a screenshot-friendly title and takeaway for the active stakeholder view."""
    if filtered_stakeholder_df.empty:
        return (
            "No stakeholder pattern is available in this view.",
            "Try widening the time period or selecting all stakeholders to compare a broader pattern.",
        )

    base_df = filtered_stakeholder_df[filtered_stakeholder_df["Year"] < 2026].copy()
    if base_df.empty:
        base_df = filtered_stakeholder_df.copy()

    yearly_totals = (
        base_df.groupby("Year", as_index=False)["Count"].sum()
        if selected_stakeholder == "All stakeholders"
        else base_df[["Year", "Count"]].copy()
    ).sort_values("Year").reset_index(drop=True)

    peak_row = yearly_totals.iloc[yearly_totals["Count"].idxmax()]
    total_cases = int(base_df["Count"].sum())

    if selected_stakeholder == "All stakeholders":
        title = "How reported AI harm shifts across stakeholder groups"
        lead_row = (
            base_df.groupby("Stakeholder", as_index=False)["Count"].sum()
            .sort_values("Count", ascending=False)
            .iloc[0]
        )
        copy = (
            f"{lead_row['Stakeholder']} appears most often in this window, with {int(lead_row['Count']):,} coded cases. "
            f"The highest annual level across the selected groups is {int(peak_row['Year'])}, at {int(peak_row['Count']):,} cases."
        )
        return (title, copy)

    title = f"AI Cases, 2020 - 2025: {selected_stakeholder}"

    if len(yearly_totals) < 2:
        copy = (
            f"{selected_stakeholder} appears in {total_cases:,} coded cases in this view. "
            f"The highest annual level shown is {int(peak_row['Year'])}, with {int(peak_row['Count']):,} cases."
        )
        return (title, copy)

    first_row = yearly_totals.iloc[0]
    last_row = yearly_totals.iloc[-1]
    change = int(last_row["Count"] - first_row["Count"])
    direction_phrase = (
        "rose over the selected period"
        if change > 0
        else "declined over the selected period"
        if change < 0
        else "stayed level over the selected period"
    )
    copy = (
        f"{selected_stakeholder} appears in {total_cases:,} coded cases and {direction_phrase}. "
        f"The highest annual level in this view is {int(peak_row['Year'])}, with {int(peak_row['Count']):,} cases."
    )
    return (title, copy)


def build_fastest_growing_industry_metric(
    industry_trend_df: pd.DataFrame,
    monthly_industry_df: pd.DataFrame,
    include_2026: bool,
) -> tuple[str, str]:
    """Return a compact metric-card version of the fastest-growing industry summary."""
    if industry_trend_df.empty:
        return ("No data", "No industry trend data is available for the selected window.")

    base_trend_df = industry_trend_df[industry_trend_df["Year"] < 2026].copy()
    if base_trend_df.empty:
        base_trend_df = industry_trend_df.copy()

    year_bounds = sorted(base_trend_df["Year"].unique())
    start_year = year_bounds[0]
    end_year = year_bounds[-1]
    start_df = (
        base_trend_df[base_trend_df["Year"] == start_year][["Industry", "Count"]]
        .rename(columns={"Count": "Start Count"})
    )
    end_df = (
        base_trend_df[base_trend_df["Year"] == end_year][["Industry", "Count"]]
        .rename(columns={"Count": "End Count"})
    )
    growth_df = start_df.merge(end_df, on="Industry", how="outer").fillna(0)
    growth_df["Growth"] = growth_df["End Count"] - growth_df["Start Count"]
    growth_row = growth_df.iloc[growth_df["Growth"].argmax()]
    value = str(growth_row["Industry"])
    return (value, "")


def prepare_explore_data(
    selected_years: tuple[int, int],
    selected_stakeholder: str,
) -> dict[str, pd.DataFrame]:
    """Prepare filtered stakeholder and industry data for the Explore tab."""
    explore_stakeholder_df = stakeholder_df[
        ~stakeholder_df["Stakeholder"].isin(["General public", "Other"])
    ].copy()

    filtered_stakeholder_df = explore_stakeholder_df[
        explore_stakeholder_df["Year"].between(selected_years[0], selected_years[1])
    ]
    if selected_stakeholder != "All stakeholders":
        filtered_stakeholder_df = filtered_stakeholder_df[
            filtered_stakeholder_df["Stakeholder"] == selected_stakeholder
        ]

    stakeholder_high_level_df = stakeholder_df[
        stakeholder_df["Year"].between(selected_years[0], selected_years[1])
    ].copy()
    filtered_industry_df = industry_df[
        industry_df["Year"].between(selected_years[0], selected_years[1])
    ]
    filtered_industry_monthly_df = industry_monthly_df[
        industry_monthly_df["Year"].between(selected_years[0], selected_years[1])
    ].copy()
    industry_trend_df = industry_df[
        industry_df["Year"].between(selected_years[0], selected_years[1])
    ].copy()

    return {
        "explore_stakeholder_df": explore_stakeholder_df,
        "filtered_stakeholder_df": filtered_stakeholder_df,
        "stakeholder_high_level_df": stakeholder_high_level_df,
        "filtered_industry_df": filtered_industry_df,
        "filtered_industry_monthly_df": filtered_industry_monthly_df,
        "industry_trend_df": industry_trend_df,
    }


def build_explore_industry_figure(
    industry_trend_df: pd.DataFrame,
    year_window_label: str,
) -> go.Figure:
    """Build the Explore-tab industry figure."""
    industry_legend_labels = {
        "IT infrastructure and hosting": "IT infrastructure",
        "Robots, sensors, and IT hardware": "Robots / sensors / hardware",
        "Mobility and autonomous vehicles": "Mobility / autonomous vehicles",
        "Media, social platforms, and marketing": "Media / social / marketing",
        "Digital security": "Digital security",
        "Government, security, and defense": "Government / security / defense",
    }
    top_industries = (
        industry_trend_df.groupby("Industry", as_index=False)["Count"].sum()
        .sort_values("Count", ascending=False)
        .head(6)["Industry"]
        .tolist()
    )
    industry_chart_trend_df = industry_trend_df[
        industry_trend_df["Industry"].isin(top_industries)
    ].copy()
    yearly_totals = industry_chart_trend_df.groupby("Year")["Count"].transform("sum")
    industry_chart_trend_df["Share"] = industry_chart_trend_df["Count"] / yearly_totals

    industry_fig = go.Figure()
    for index, industry in enumerate(top_industries):
        industry_df = industry_chart_trend_df[industry_chart_trend_df["Industry"] == industry]
        industry_fig.add_trace(
            go.Scatter(
                x=industry_df["Year"],
                y=industry_df["Share"],
                mode="lines",
                stackgroup="one",
                name=industry_legend_labels.get(industry, industry),
                line=dict(
                    width=1.2,
                    color=STACKED_AREA_COLORS[index % len(STACKED_AREA_COLORS)],
                ),
                customdata=[[industry]] * len(industry_df),
                hovertemplate="%{customdata[0]}<br>%{x}: %{y:.0%} of coded cases<extra></extra>",
            )
        )

    industry_fig.update_layout(
        title=f"AI cases by industry, {year_window_label}",
        hovermode="x unified",
        yaxis_tickformat=".0%",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
        ),
    )
    return style_chart(industry_fig, height=380)


def build_explore_stakeholder_figure(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
    year_window_label: str,
) -> go.Figure:
    """Build the Explore-tab stakeholder figure."""
    line_title = (
        f"Specific stakeholder coverage, {year_window_label}"
        if selected_stakeholder == "All stakeholders"
        else f"AI Cases, {year_window_label}: {selected_stakeholder}"
    )
    line_fig = px.line(
        filtered_stakeholder_df,
        x="Year",
        y="Count",
        color="Stakeholder" if selected_stakeholder == "All stakeholders" else None,
        markers=True,
        title=line_title,
        color_discrete_sequence=STAKEHOLDER_COLORS,
    )
    line_fig.update_traces(line=dict(width=3), marker=dict(size=8))
    line_fig.update_layout(hovermode="x unified")
    return style_chart(line_fig, height=390)


def load_case_study_content(selected_stakeholder: str, case_index: int) -> tuple[str, str, str, str, str, bool]:
    """Return case-study title, summary, relevance, source URL, source label, and preloaded flag."""
    try:
        if selected_stakeholder == "All stakeholders":
            return (
                "",
                "Choose one stakeholder to load a matching case summary from the OECD incident feed.",
                "",
                "",
                "",
                False,
            )
        with st.spinner("Loading AI case study..."):
            case_study_data = load_case_study(selected_stakeholder, case_index=case_index)
        source_label = (
            "View OECD article"
            if case_study_data.get("source_list_url")
            and case_study_data["source_url"] != case_study_data["source_list_url"]
            else "View OECD source list"
        )
        return (
            case_study_data.get("title", ""),
            case_study_data.get("summary", ""),
            case_study_data.get("relevance", ""),
            case_study_data["source_url"],
            source_label,
            bool(case_study_data.get("is_preloaded")),
        )
    except Exception as exc:
        return "", f"Unable to load AI case study: {exc}", "", "", "", False


def render_explore_time_filter(
    selected_year_label: str,
    include_2026_available: bool,
) -> None:
    """Render the Explore-tab time filter."""
    st.radio(
        "Time period in the data",
        options=list(YEAR_FILTER_OPTIONS.keys()),
        index=list(YEAR_FILTER_OPTIONS.keys()).index(selected_year_label),
        horizontal=True,
        key="explore_selected_year_label_top",
        on_change=sync_explore_year_from_top,
    )
    st.checkbox(
        "Include partial 2026 data",
        key="explore_include_2026",
        disabled=not include_2026_available,
    )


def render_explore_stakeholder_intro() -> None:
    """Render the explainer for the stakeholder graph below the time filter."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_intro(
        "Stakeholders",
        "Reported cases by stakeholder",
        "Use the controls below to compare stakeholder patterns over time and read a related case study.",
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_explore_stakeholder_row(
    line_fig: go.Figure,
    stakeholder_options: list[str],
    selected_stakeholder: str,
    filtered_stakeholder_df: pd.DataFrame,
    stakeholder_high_level_df: pd.DataFrame,
    include_2026: bool,
    case_study_title_text: str,
    case_study_summary: str,
    case_study_relevance: str,
    source_url: str,
    source_label: str,
    case_requested: bool,
    is_cached_case: bool,
) -> None:
    """Render the stakeholder top row plus a supporting detail row."""
    chart_row_col1, chart_row_col2 = st.columns([0.72, 1.28], gap="large")
    with chart_row_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_intro(
            "Stakeholders",
            "Reported cases by stakeholder",
            "Use the controls below to compare stakeholder patterns over time and read a related case study.",
        )
        st.markdown('<div class="overall-peak-card">', unsafe_allow_html=True)
        insight_card_with_note(
            "Highest annual total",
            "This includes total cases without an identified stakeholder, including those labeled General public.",
            build_stakeholder_peak_summary(
                stakeholder_high_level_df,
                "All stakeholders",
                include_2026,
            ),
        )
        st.markdown("</div>", unsafe_allow_html=True)
        filter_select_col, filter_clear_col = st.columns([0.76, 0.24], gap="small")
        with filter_select_col:
            st.selectbox(
                "Filter by specific stakeholder",
                options=stakeholder_options,
                index=(
                    stakeholder_options.index(selected_stakeholder)
                    if selected_stakeholder in stakeholder_options
                    else None
                ),
                placeholder="",
                key="explore_selected_stakeholder",
            )
        with filter_clear_col:
            st.markdown('<div class="stakeholder-clear-button">', unsafe_allow_html=True)
            st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
            st.button(
                "Clear",
                key="clear_stakeholder_filter",
                help="Clear stakeholder filter",
                type="secondary",
                use_container_width=True,
                on_click=clear_explore_stakeholder_filter,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        bridge_title, bridge_copy = build_stakeholder_story_bridge(
            filtered_stakeholder_df,
            selected_stakeholder,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with chart_row_col2:
        st.markdown('<div class="section-card stakeholder-chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="stakeholder-chart-intro-spacer"></div>', unsafe_allow_html=True)
        st.plotly_chart(line_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    detail_row_col1, detail_row_col2 = st.columns([0.72, 1.28], gap="large")
    with detail_row_col1:
        st.markdown('<div class="section-card stakeholder-linked-case-card">', unsafe_allow_html=True)
        case_study_title = (
            "Related Case Coverage"
            if selected_stakeholder == "All stakeholders"
            else f"Related Case Coverage: {selected_stakeholder}"
        )
        case_study_copy = (
            "Select a stakeholder to load related coverage from one reported case."
            if selected_stakeholder == "All stakeholders"
            else "Load related coverage about how AI affected this stakeholder group."
        )
        compact_section_intro(
            "Linked Case",
            case_study_title,
            case_study_copy,
        )
        if not case_requested:
            button_col, _ = st.columns([0.46, 0.54])
            with button_col:
                if st.button(
                    "Load related coverage",
                    key="load_case_study_card",
                    disabled=(selected_stakeholder == "All stakeholders"),
                ):
                    st.session_state["explore_case_requested"] = True
                    st.session_state["explore_case_index"] = 0
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with detail_row_col2:
        st.markdown('<div class="section-card stakeholder-case-study-card">', unsafe_allow_html=True)
        chip_text = "Specific stakeholder" if selected_stakeholder == "All stakeholders" else selected_stakeholder
        if is_cached_case and case_study_title_text and not case_study_title_text.startswith("Cached article: "):
            case_study_title_text = f"Cached article: {case_study_title_text}"
        headline_html = (
            f'<div class="case-study-headline">{escape(case_study_title_text)}</div>'
            if case_study_title_text
            else ""
        )
        case_html = (
            '<div class="case-study-shell">'
            f'<div class="case-study-chip">{chip_text}</div>'
            f"{headline_html}"
            f'<div class="case-study-body">{escape(case_study_summary)}</div>'
        )
        if case_study_relevance:
            case_html += f'<div class="case-study-body case-study-relevance">{escape(case_study_relevance)}</div>'
        if source_url:
            case_html += (
                '<div class="case-study-source">'
                '<div class="case-study-source-label">Source: OECD.AI</div>'
                f'<a href="{source_url}" target="_blank">{source_label or "View OECD source list"}</a>'
                "</div>"
            )
        case_html += "</div>"
        st.markdown(case_html, unsafe_allow_html=True)
        if selected_stakeholder != "All stakeholders" and case_requested:
            if st.button("Show another case", key=f"another_case_{selected_stakeholder}"):
                st.session_state["explore_case_index"] = st.session_state.get("explore_case_index", 0) + 1
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_explore_industry_row(
    industry_fig: go.Figure,
    industry_trend_df: pd.DataFrame,
    filtered_industry_monthly_df: pd.DataFrame,
    include_2026: bool,
) -> None:
    """Render the industry row with chart and corresponding text."""
    industry_row_col1, industry_row_col2 = st.columns([1.08, 0.68], gap="large")
    with industry_row_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(industry_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with industry_row_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_intro(
            "Industries",
            "Compare industry patterns over time",
            "This view shows which industries appear most often in reported AI harm cases across the selected time period.",
        )
        insight_card(
            "Largest increase in reported cases",
            build_fastest_growing_industry_summary(
                industry_trend_df,
                filtered_industry_monthly_df,
                include_2026,
            ),
        )
        st.markdown("</div>", unsafe_allow_html=True)


def sync_explore_year_from_top() -> None:
    """Sync the shared Explore year selection from the top control."""
    st.session_state["explore_selected_year_label"] = st.session_state["explore_selected_year_label_top"]


def clear_explore_stakeholder_filter() -> None:
    """Reset the Explore stakeholder filter and related case-study state."""
    st.session_state["explore_selected_stakeholder"] = None
    st.session_state["explore_case_requested"] = False
    st.session_state["explore_case_index"] = 0


inject_page_styles()
story_metrics = build_story_metrics()
story_stakeholder_fig, story_industry_fig = build_story_figures()
story_stakeholder_note = build_story_stakeholder_note()

story_tab, explore_tab, about_tab, downloads_tab = st.tabs(
    ["Overview", "Explore", "About the Data", "Downloads"]
)

with story_tab:
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-eyebrow">Based on OECD AI Incidents and Hazards Monitor data</div>
            <div class="hero-title">Reported AI cases and what they show</div>
            <div class="hero-copy">
                An overview of reported AI cases, showing who is affected,
                which industries are involved, and how incidents appear over time.
            </div>
            <div class="hero-note">
                This dashboard uses data from the OECD AI Incidents and Hazards Monitor, which compiles
                reported cases from international news coverage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4, gap="medium")
    with metric_col1:
        metric_card(
            "Incidents and hazards",
            story_metrics["total_incidents"],
            "Incidents and hazards between 2020-2025.",
        )
    with metric_col2:
        metric_card(
            "Most affected group",
            story_metrics["top_stakeholder"],
            f"{story_metrics['top_stakeholder_count']} reported cases across identified stakeholder groups.",
        )
    with metric_col3:
        metric_card(
            "Most exposed industry",
            story_metrics["top_industry"],
            "The industry with the largest concentration of reported cases between 2020-2025.",
        )
    with metric_col4:
        metric_card(
            "Incidents and hazards over time",
            story_metrics["growth"],
            f"Change from {story_metrics['first_year']} to {story_metrics['last_year']} in annual event totals.",
        )

    incidents_trend_fig = go.Figure()
    incidents_trend_fig.add_trace(
        go.Scatter(
            x=incidents_over_time_df["Date"],
            y=incidents_over_time_df["Total Incidents & Hazards"],
            mode="lines",
            name="Incidents and hazards",
            line=dict(color="#1f1f1f", width=3),
        )
    )
    incidents_trend_fig.add_trace(
        go.Scatter(
            x=incidents_over_time_df["Date"],
            y=incidents_over_time_df["6-month moving average"],
            mode="lines",
            name="6-month moving average",
            line=dict(color="#8e3b46", width=2, dash="dash"),
        )
    )
    incidents_trend_fig.update_layout(
        title="Reported AI cases over time",
        hovermode="x unified",
    )
    style_chart(incidents_trend_fig, height=360)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_intro(
        "Case trends over time",
        "Number of reported AI cases",
        "This tracks the total number of incidents and hazards in the source data over time, with a moving average to make longer-term changes easier to see.",
    )
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    st.plotly_chart(incidents_trend_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    story_col1, story_col2 = st.columns([1.15, 0.85], gap="large")
    with story_col1:
        with st.container(border=False):
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
            st.plotly_chart(story_stakeholder_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    with story_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_intro(
            "Stakeholder patterns",
            "Affected stakeholders",
            "This shows which groups appear most often in reported AI cases across the current overview window.",
        )
        insight_card(
            "Missing groups in the source data",
            "The source dataset does not separate LGBTQ+ people, people of color, or people with disabilities into their own stakeholder categories, though these groups are referenced in many of the underlying articles.",
        )
        insight_card(
            "Stakeholder categories can overlap",
            "A reported case can still involve more than one group. For example, women officials may also be part of government, and workers may also be consumers, so one article can be categorized under several stakeholder groups even when one is highlighted most clearly.",
        )
        insight_card(
            "Coverage versus occurrence",
            "These counts show which AI-related issues are reported most often in the OECD monitor. They reflect coverage and attention in the dataset, not a count of unique real-world events.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    story_col3, story_col4 = st.columns([0.88, 1.12], gap="large")
    with story_col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_intro(
            "Industry patterns",
            "Industries with the most reported cases",
            "This shows where reported AI cases appear most often across industries.",
        )
        insight_card(
            "Where AI harm is surfacing",
            "The industries at the top are not necessarily using the most AI. They are the ones where problems are being reported, noticed, and categorized most often.",
        )
        insight_card(
            "Visibility shapes the pattern",
            "Some industries rise in this view because their failures are more public-facing, more regulated, or more likely to be covered by the news.",
        )
        insight_card(
            "Risk is interpreted by people",
            "AI risk is not only technical. It is also shaped by how leaders, workers, users, and outside observers recognize harm, respond to warning signs, and decide what deserves attention.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with story_col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        st.plotly_chart(story_industry_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with explore_tab:
    st.markdown('<div class="explore-scope">', unsafe_allow_html=True)
    if st.session_state.get("explore_selected_year_label") not in YEAR_FILTER_OPTIONS:
        st.session_state["explore_selected_year_label"] = list(YEAR_FILTER_OPTIONS.keys())[0]

    selected_year_label = st.session_state["explore_selected_year_label"]
    include_2026_available = selected_year_label == "All years"
    if not include_2026_available:
        st.session_state["explore_include_2026"] = False
    include_2026 = bool(st.session_state.get("explore_include_2026", False))

    selected_years = YEAR_FILTER_OPTIONS[selected_year_label]
    if selected_year_label == "All years" and include_2026:
        selected_years = (selected_years[0], 2026)

    explore_data = prepare_explore_data(
        selected_years=selected_years,
        selected_stakeholder=st.session_state.get("explore_selected_stakeholder") or "All stakeholders",
    )
    explore_stakeholder_df = explore_data["explore_stakeholder_df"]
    stakeholder_options = sorted(explore_stakeholder_df["Stakeholder"].unique())
    selected_stakeholder = st.session_state.get("explore_selected_stakeholder") or "All stakeholders"
    previous_stakeholder_state = st.session_state.get("explore_case_stakeholder")
    if previous_stakeholder_state != selected_stakeholder:
        st.session_state["explore_case_index"] = 0
        st.session_state["explore_case_requested"] = False
        st.session_state["explore_case_stakeholder"] = selected_stakeholder

    explore_data = prepare_explore_data(
        selected_years=selected_years,
        selected_stakeholder=selected_stakeholder,
    )
    filtered_stakeholder_df = explore_data["filtered_stakeholder_df"]
    stakeholder_high_level_df = explore_data["stakeholder_high_level_df"]
    filtered_industry_monthly_df = explore_data["filtered_industry_monthly_df"]
    industry_trend_df = explore_data["industry_trend_df"]
    year_window_label = build_year_window_label(selected_year_label, include_2026)
    industry_fig = build_explore_industry_figure(industry_trend_df, year_window_label)
    line_fig = build_explore_stakeholder_figure(
        filtered_stakeholder_df,
        selected_stakeholder,
        year_window_label,
    )
    st.markdown('<div class="section-card explore-intro-card" style="background: ' + STORY_GRADIENT + ';">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">Explore Patterns in Reported AI Cases</div>
        """,
        unsafe_allow_html=True,
    )
    render_explore_time_filter(selected_year_label, include_2026_available)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.session_state.get("explore_case_requested") and selected_stakeholder != "All stakeholders":
        case_study_title_text, case_study_summary, case_study_relevance, source_url, source_label, is_cached_case = load_case_study_content(
            selected_stakeholder,
            st.session_state.get("explore_case_index", 0),
        )
    else:
        case_study_title_text, case_study_summary, case_study_relevance, source_url, source_label, is_cached_case = (
            "",
            (
                f"Load one reported case involving {selected_stakeholder.lower()}."
                if selected_stakeholder != "All stakeholders"
                else "Choose a specific stakeholder above to load one reported case."
            ),
            "",
            "",
            "",
            False,
        )
    render_explore_industry_row(
        industry_fig=industry_fig,
        industry_trend_df=industry_trend_df,
        filtered_industry_monthly_df=filtered_industry_monthly_df,
        include_2026=include_2026,
    )
    render_explore_stakeholder_row(
        line_fig=line_fig,
        stakeholder_options=stakeholder_options,
        selected_stakeholder=selected_stakeholder,
        filtered_stakeholder_df=filtered_stakeholder_df,
        stakeholder_high_level_df=stakeholder_high_level_df,
        include_2026=include_2026,
        case_study_title_text=case_study_title_text,
        case_study_summary=case_study_summary,
        case_study_relevance=case_study_relevance,
        source_url=source_url,
        source_label=source_label,
        case_requested=bool(st.session_state.get("explore_case_requested")),
        is_cached_case=is_cached_case,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with about_tab:
    st.markdown(
        """
        <div class="section-card" style="background: """
        + STORY_GRADIENT
        + """; margin-bottom: 1rem;">
            <div class="section-kicker">Method and Source Notes</div>
            <div class="section-title">About the Data</div>
            <div class="section-copy">
                A summary of the dashboard's source data, methods, and key limitations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    about_row1_col1, about_row1_col2 = st.columns([1.1, 0.9], gap="large")

    with about_row1_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_intro(
            "Data source",
            "How the data is collected",
            'This dashboard uses the <a href="https://oecd.ai/en/incidents" target="_blank">OECD AI Incidents and Hazards Monitor</a>. AIM scans reputable international news outlets for AI-related incidents and hazards, groups related news coverage into events, and structures the data. The data used in this dashboard includes both incidents and hazards.',
        )
        insight_card(
            "What is an incident?",
            "An incident is an event linked to harm that was reported to have already occurred."
        )
        insight_card(
            "What is a hazard?",
            "A hazard is an event linked to plausible potential harm, even if the harm has not yet occurred."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    severity_fig = go.Figure()
    severity_fig.add_trace(
        go.Scatter(
            x=severity_over_time_df["Date"],
            y=severity_over_time_df["AI incident"],
            stackgroup="one",
            mode="lines",
            name="AI incident",
            line=dict(color="#1f1f1f", width=2),
        )
    )
    severity_fig.add_trace(
        go.Scatter(
            x=severity_over_time_df["Date"],
            y=severity_over_time_df["AI hazard"],
            stackgroup="one",
            mode="lines",
            name="AI hazard",
            line=dict(color="#b85c6b", width=2),
        )
    )
    severity_fig.update_layout(
        title="Incident and hazard split over time",
        hovermode="x unified",
    )
    severity_fig.update_yaxes(showticklabels=False)
    style_chart(severity_fig, height=320)

    with about_row1_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_intro(
            "Incidents and hazards",
            "How the source data is split",
            "This chart shows how the source data is divided between incidents and hazards over time, which helps clarify the mix of already-reported harms and plausible potential harms in the dashboard.",
        )
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        st.plotly_chart(severity_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    about_row2_col1, about_row2_col2 = st.columns([1.1, 0.9], gap="large")

    with about_row2_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_intro(
            "How to read it",
            "What the counts mean",
            'A case in this dashboard refers to a reported AI-related issue that has been categorized in the OECD monitor. The counts show how reported cases are sorted in the dataset, rather than a count of unique real-world events. See the <a href="https://oecd.ai/en/incidents-methodology" target="_blank">OECD incidents methodology page</a> for more detail.',
        )
        insight_card(
            "Stakeholder analysis",
            'In this dashboard analysis, articles with a labeled stakeholder of "General public" are excluded from the in-depth comparison analysis because it is not clear which stakeholder(s) were most affected. They constitute 6,221 coded cases, or 38.2% of stakeholder-coded totals in the 2020-2025 overview window, so they are still included in the totals.'
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with about_row2_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_intro(
            "Limitations",
            "What this dashboard does not show",
            "This dashboard shows reported and coded cases in the OECD monitor, not every real-world AI harm event. What appears most often here is shaped by what gets covered, surfaced, and categorized in the source material.",
        )
        insight_card(
            "What gets covered",
            "Some industries, countries, harms, or stakeholder groups may appear more often because they receive more media attention or clearer public documentation."
        )
        insight_card(
            "What the counts mean",
            "A larger count means an issue appears more often in reported and coded cases in this dataset, not necessarily that it occurs more often in the world."
        )
        st.markdown("</div>", unsafe_allow_html=True)

with downloads_tab:
    st.markdown(
        """
        <div class="section-card" style="background: """
        + STORY_GRADIENT
        + """; margin-bottom: 1rem;">
            <div class="section-kicker">Dataset Access</div>
            <div class="section-title">Data Downloads</div>
            <div class="section-copy">
                Preview the source datasets used in this dashboard and download each one as a CSV.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_dataset_preview(
        "Stakeholders",
        "Reported AI cases by affected stakeholder over time.",
        stakeholders_df,
        "stakeholders_dataset.csv",
    )
    render_dataset_preview(
        "Industry",
        "Reported incidents and hazards by industry over time.",
        industries_df,
        "industry_dataset.csv",
    )
    render_dataset_preview(
        "Incidents and Hazards",
        "Monthly totals for reported incidents and hazards in the source data.",
        incidents_df,
        "incidents_and_hazards_dataset.csv",
    )
    render_dataset_preview(
        "Incident vs Hazard Split",
        "Monthly counts comparing incidents and hazards over time.",
        severity_df,
        "incident_vs_hazard_split_dataset.csv",
    )
