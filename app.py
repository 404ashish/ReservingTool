"""
Non-Life Insurance Reserving Tool
A Streamlit web application for actuarial loss reserving.
Methods: Chain Ladder (with Mack uncertainty), Bornhuetter-Ferguson, Cape Cod
"""

import io
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import chainladder as cl

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reserving Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card {
    background: #F0F4FF;
    border: 1px solid #C7D7FD;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.kpi-card .kpi-label { font-size: 0.78rem; color: #6B7280; font-weight: 500; margin-bottom: 4px; }
.kpi-card .kpi-value { font-size: 1.45rem; font-weight: 700; color: #1E3A5F; }
.kpi-card .kpi-sub   { font-size: 0.75rem; color: #9CA3AF; margin-top: 2px; }
.method-badge {
    display: inline-block; padding: 2px 10px;
    background: #EFF6FF; color: #1D4ED8;
    border-radius: 99px; font-size: 0.78rem; font-weight: 600;
    margin-right: 4px;
}
hr.sep { border: none; border-top: 1px solid #E5E7EB; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(v):
    """Format large numbers compactly."""
    if pd.isna(v) or v == 0:
        return "—"
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f}B"
    elif abs(v) >= 1e6:
        return f"{v/1e6:.2f}M"
    elif abs(v) >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:,.0f}"


def to_period_series(cl_triangle_attr, freq_label="A"):
    """
    Convert a chainladder triangle attribute to a pd.Series with a readable index.
    Annual   -> integer year  (2020, 2021 ...)
    Quarterly-> string label  (2022Q1, 2022Q2 ...)
    Drops phantom NaN rows chainladder occasionally appends.
    """
    df = cl_triangle_attr.to_frame()
    idx = df.index  # DatetimeIndex
    if freq_label == "Q":
        df.index = idx.to_period("Q").strftime("%YQ%q")
        df.index.name = "Accident Quarter"
    elif freq_label == "S":
        df.index = [f"{d.year}H{1 if d.month <= 6 else 2}" for d in idx]
        df.index.name = "Accident Half-Year"
    else:
        df.index = idx.year
        df.index.name = "Accident Year"
    s = df.iloc[:, 0]
    return s.dropna()


# Keep backward-compatible alias
def to_year_series(cl_triangle_attr):
    return to_period_series(cl_triangle_attr, "A")


def load_sample():
    return cl.load_sample("genins")



def _parse_origin_date(s):
    s = str(s).strip().upper().replace("-", "").replace(" ", "")
    if "Q" in s:
        if s.startswith("Q"):
            q = int(s[1]); year = int(s[2:])
        else:
            year = int(s[:4]); q = int(s[s.index("Q") + 1])
        month = (q - 1) * 3 + 1
        return pd.Timestamp(f"{year}-{month:02d}-01")
    else:
        return pd.Timestamp(str(int(s[:4])) + "-01-01")


def _detect_frequency(col_int_values):
    sorted_cols = sorted(col_int_values)
    if len(sorted_cols) < 2:
        return "A", 12
    diffs = [sorted_cols[i+1] - sorted_cols[i] for i in range(len(sorted_cols)-1)]
    min_diff = min(diffs)
    if min_diff <= 3:   return "Q", 3
    elif min_diff <= 6: return "S", 6
    else:               return "A", 12


def wide_df_to_triangle(df):
    """
    Convert a wide-format DataFrame to a chainladder Triangle.
    Supports ANNUAL and QUARTERLY triangles automatically.

    Row index formats:
      Annual    -> 2020, '2020'
      Quarterly -> '2022Q1', '2022-Q1', 'Q1-2022', 'Q12022'

    Column headers: integers, floats or strings representing months
      Annual:    12, 24, 36 ...
      Quarterly:  3,  6,  9, 12 ...
    """
    from dateutil.relativedelta import relativedelta

    df = df.copy()

    # Normalise column headers to integer months
    try:
        df.columns = (
            pd.Series(df.columns)
            .astype(str).str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .astype(int).values
        )
    except Exception:
        raise ValueError(
            "Column headers must be development periods in months "
            "(e.g. 3,6,9 for quarterly or 12,24,36 for annual)."
        )

    freq_label, _ = _detect_frequency(df.columns.tolist())

    # Parse origin index to Timestamps
    try:
        origins = [_parse_origin_date(i) for i in df.index]
    except Exception:
        raise ValueError(
            "Row index must be accident years (e.g. 2020) or "
            "accident quarters (e.g. 2022Q1)."
        )

    df.index = origins
    df.index.name = "origin"

    # Melt to long form, drop future (NaN) cells
    df_long = (
        df.reset_index()
        .melt(id_vars="origin", var_name="dev", value_name="loss")
        .dropna(subset=["loss"])
    )
    df_long["dev"] = df_long["dev"].astype(int)

    # Dev date = origin + dev months
    df_long["dev_date"] = df_long.apply(
        lambda r: r["origin"] + relativedelta(months=int(r["dev"])), axis=1
    )

    triangle = cl.Triangle(
        df_long,
        origin="origin",
        development="dev_date",
        columns=["loss"],
        cumulative=True,
    )
    return triangle, freq_label


def csv_to_triangle(f):
    df = pd.read_csv(f, index_col=0)
    return wide_df_to_triangle(df)


def excel_to_triangle(f):
    df = pd.read_excel(f, index_col=0)
    return wide_df_to_triangle(df)



# ── Model runners ─────────────────────────────────────────────────────────────
def run_models(triangle, apriori, use_bf, use_cc):
    dev      = cl.Development().fit(triangle)
    t_dev    = dev.transform(triangle)
    cl_model = cl.Chainladder().fit(t_dev)
    mack     = cl.MackChainladder().fit(t_dev)
    latest   = triangle.latest_diagonal

    bf_model = None
    cc_model = None
    if use_bf:
        bf_model = cl.BornhuetterFerguson(apriori=apriori).fit(t_dev, sample_weight=latest)
    if use_cc:
        cc_model = cl.CapeCod().fit(t_dev, sample_weight=latest)

    return dev, cl_model, mack, bf_model, cc_model


def build_summary(cl_model, mack, bf_model, cc_model, triangle, freq_label="A"):
    """Build a tidy per-accident-period summary DataFrame."""
    cl_ibnr  = to_period_series(cl_model.ibnr_, freq_label)
    cl_ult   = to_period_series(cl_model.ultimate_, freq_label)
    lat_diag = to_period_series(triangle.latest_diagonal, freq_label)

    # Mack std err: use last column of mack_std_err (ultimate column)
    mack_se_tmp = mack.mack_std_err_
    mack_se_frame = mack_se_tmp.to_frame()
    mack_se_idx = mack_se_frame.index
    if freq_label == "Q":
        mack_se_frame.index = mack_se_idx.to_period("Q").strftime("%YQ%q")
        mack_se_frame.index.name = "Accident Quarter"
    else:
        mack_se_frame.index = mack_se_idx.year
        mack_se_frame.index.name = "Accident Year"
    mack_se_series = mack_se_frame.iloc[:, -1]

    data = {
        "Latest Diagonal": lat_diag,
        "CL Ultimate":     cl_ult,
        "CL IBNR":         cl_ibnr,
        "Mack Std Error":  mack_se_series,
    }
    if bf_model is not None:
        data["BF IBNR"] = to_period_series(bf_model.ibnr_, freq_label)
        data["BF Ultimate"] = to_period_series(bf_model.ultimate_, freq_label)
    if cc_model is not None:
        data["CC IBNR"] = to_period_series(cc_model.ibnr_, freq_label)
        data["CC Ultimate"] = to_period_series(cc_model.ultimate_, freq_label)

    return pd.DataFrame(data).fillna(0).round(0)


# ── Excel export ──────────────────────────────────────────────────────────────
def build_excel(tri_df, ldf_df, summary_df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Reserve Summary")
        tri_df.to_excel(writer,     sheet_name="Loss Triangle")
        ldf_df.T.to_excel(writer,   sheet_name="Link Ratios")
        for ws_name in ["Reserve Summary", "Loss Triangle", "Link Ratios"]:
            ws = writer.sheets[ws_name]
            ws.set_column("A:A", 16)
            ws.set_column("B:N", 16)
    buf.seek(0)
    return buf.read()


# ── Charts ────────────────────────────────────────────────────────────────────
DARK_BLUE = "#1E3A5F"
SKY_BLUE  = "#0EA5E9"
TEAL      = "#10B981"
AMBER     = "#F59E0B"
LIGHT_BG  = "white"
GRID      = "#E5E7EB"


def base_layout(**kw):
    return dict(
        height=340,
        margin=dict(t=20, b=40, l=60, r=20),
        plot_bgcolor=LIGHT_BG,
        paper_bgcolor=LIGHT_BG,
        font=dict(size=12),
        yaxis=dict(gridcolor=GRID),
        **kw,
    )


def chart_ibnr_bar(summary_df):
    fig = go.Figure()
    cols = {
        "CL IBNR": ("Chain Ladder", DARK_BLUE),
        "BF IBNR": ("Bornhuetter–Ferguson", SKY_BLUE),
        "CC IBNR": ("Cape Cod", TEAL),
    }
    for col, (name, color) in cols.items():
        if col in summary_df.columns:
            fig.add_trace(go.Bar(
                name=name,
                x=summary_df.index.astype(str),
                y=summary_df[col],
                marker_color=color,
            ))
    fig.update_layout(barmode="group", xaxis_title="Accident Year",
                      yaxis_title="IBNR Reserve ($)", legend_title="Method",
                      **base_layout())
    return fig


def chart_triangle_heatmap(tri_df):
    z_clean = [[None if (isinstance(v, float) and np.isnan(v)) else v for v in row]
               for row in tri_df.values.tolist()]
    fig = go.Figure(go.Heatmap(
        z=z_clean,
        x=[str(c) for c in tri_df.columns],
        y=[str(i) for i in tri_df.index],
        colorscale=[[0, "#EFF6FF"], [1, DARK_BLUE]],
        colorbar=dict(title="Loss ($)"),
        hoverongaps=False,
    ))
    fig.update_layout(xaxis_title="Development Period", yaxis_title="Accident Year",
                      height=380, margin=dict(t=20, b=50, l=80, r=20),
                      paper_bgcolor=LIGHT_BG)
    return fig


def chart_ldfs(dev):
    ldf_vals = dev.ldf_.to_frame().iloc[0]
    fig = go.Figure(go.Bar(
        x=ldf_vals.index.astype(str),
        y=ldf_vals.values,
        marker_color=DARK_BLUE,
        text=[f"{v:.4f}" for v in ldf_vals.values],
        textposition="outside",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#9CA3AF")
    fig.update_layout(xaxis_title="Development Period",
                      yaxis_title="Link Development Factor",
                      **base_layout())
    return fig


def chart_ultimates(summary_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Latest Diagonal",
        x=summary_df.index.astype(str),
        y=summary_df["Latest Diagonal"],
        marker_color="#E5E7EB",
    ))
    ult_map = {
        "CL Ultimate": ("Chain Ladder", DARK_BLUE),
        "BF Ultimate": ("BF", SKY_BLUE),
        "CC Ultimate": ("Cape Cod", TEAL),
    }
    for col, (name, color) in ult_map.items():
        if col in summary_df.columns:
            fig.add_trace(go.Scatter(
                name=name, x=summary_df.index.astype(str), y=summary_df[col],
                mode="lines+markers",
                line=dict(color=color, width=2), marker=dict(size=7),
            ))
    fig.update_layout(barmode="overlay", xaxis_title="Accident Year",
                      yaxis_title="Ultimate Loss ($)", legend_title="Method",
                      **base_layout())
    return fig


def chart_mack(summary_df):
    df = summary_df[summary_df["CL IBNR"] > 0].copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index.astype(str),
        y=df["CL IBNR"],
        error_y=dict(type="data", array=df["Mack Std Error"].values,
                     visible=True, color="#E0472E", thickness=2),
        marker_color=DARK_BLUE,
        name="IBNR ± 1σ (Mack)",
    ))
    fig.update_layout(xaxis_title="Accident Year", yaxis_title="IBNR Reserve ($)",
                      **base_layout())
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📁 Data source")
        data_src = st.radio("", ["Built-in sample (genins)", "Upload my triangle"],
                            label_visibility="collapsed")

        triangle = None
        if data_src == "Upload my triangle":
            up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
            if up:
                try:
                    if up.name.endswith(".csv"):
                        triangle, freq_label = csv_to_triangle(up)
                    else:
                        triangle, freq_label = excel_to_triangle(up)
                    freq_name = {"Q": "Quarterly", "S": "Semi-annual", "A": "Annual"}.get(freq_label, "Annual")
                    st.success(f"Triangle loaded — {freq_name} development detected.")
                except Exception as e:
                    st.error(f"Parse error: {e}")
        else:
            triangle, freq_label = load_sample(), "A"
            st.caption("General insurance 10×10 triangle (annual development, paid losses).")

        st.markdown("---")
        st.markdown("## ⚙️ Methods")
        use_cl = st.checkbox("Chain Ladder  +  Mack uncertainty", value=True)
        use_bf = st.checkbox("Bornhuetter–Ferguson (BF)", value=True)
        use_cc = st.checkbox("Cape Cod (CC)", value=True)

        apriori = 0.65
        if use_bf:
            apriori = st.slider("BF a priori loss ratio", 0.30, 1.20, 0.65, 0.01,
                                help="Expected loss ratio used as the BF prior.")

        st.markdown("---")
        export_btn = st.button("📤 Build Excel report", use_container_width=True)

    # ── Guard ─────────────────────────────────────────────────────────────────
    st.title("📊 Non-Life Insurance Reserving Tool")

    if triangle is None:
        st.info("Upload a loss development triangle or select the built-in sample in the sidebar.")
        st.markdown("**Expected format:** rows = accident years, columns = development periods (12, 24, 36 …), values = cumulative paid losses.")
        demo = pd.DataFrame({"12":[357848,352118,290507],"24":[1124788,1236139,1292306],
                             "36":[1735330,2170033,2218525]},
                            index=pd.Index([2001,2002,2003], name="Accident Year"))
        st.dataframe(demo, use_container_width=True)
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    with st.spinner("Running reserving models…"):
        try:
            dev, cl_model, mack, bf_model, cc_model = run_models(
                triangle, apriori, use_bf, use_cc)
            tri_df  = triangle.to_frame().round(0)
            # Format tri_df index based on frequency
            if freq_label == "Q":
                tri_df.index = tri_df.index.to_period("Q").strftime("%YQ%q")
            elif hasattr(tri_df.index, 'year'):
                tri_df.index = tri_df.index.year
            ldf_df  = dev.ldf_.to_frame().round(4)
            summary = build_summary(cl_model, mack, bf_model, cc_model, triangle, freq_label)
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    active = (["CL"] if use_cl else []) + (["BF"] if use_bf else []) + (["CC"] if use_cc else [])
    badges = " ".join(f'<span class="method-badge">{m}</span>' for m in active)
    st.markdown(f"Active methods: {badges}", unsafe_allow_html=True)
    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    total_latest  = summary["Latest Diagonal"].sum()
    total_cl_ibnr = summary["CL IBNR"].sum()
    total_cl_ult  = summary["CL Ultimate"].sum()
    total_mack_se = float(mack.total_mack_std_err_.values.flatten()[0])
    cov = total_mack_se / total_cl_ibnr * 100 if total_cl_ibnr else 0

    c1, c2, c3, c4 = st.columns(4)
    def kpi_html(label, value, sub):
        return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'

    with c1: st.markdown(kpi_html("Total Paid to Date",  fmt(total_latest),  "latest diagonal"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_html("CL Total IBNR",       fmt(total_cl_ibnr), "chain ladder"),    unsafe_allow_html=True)
    with c3: st.markdown(kpi_html("CL Ultimate Reserve", fmt(total_cl_ult),  "paid + IBNR"),     unsafe_allow_html=True)
    with c4: st.markdown(kpi_html("Mack Std Error",      fmt(total_mack_se), f"CoV {cov:.1f}%"), unsafe_allow_html=True)

    if "BF IBNR" in summary.columns or "CC IBNR" in summary.columns:
        st.markdown("")
        extra_cols = st.columns(4)
        i = 0
        if "BF IBNR" in summary.columns:
            with extra_cols[i]:
                st.markdown(kpi_html("BF Total IBNR", fmt(summary["BF IBNR"].sum()),
                                     f"a priori LR = {apriori:.0%}"), unsafe_allow_html=True)
            i += 1
        if "CC IBNR" in summary.columns:
            with extra_cols[i]:
                st.markdown(kpi_html("Cape Cod Total IBNR", fmt(summary["CC IBNR"].sum()),
                                     "derived ELR"), unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["📋 Reserve Summary", "🔺 Triangle & LDFs", "📈 Charts", "🔬 Diagnostics"])

    # Tab 1 – Summary Table
    with tabs[0]:
        st.subheader("Reserve summary by accident year")
        disp = summary.copy()
        disp.index.name = "Accident Year"
        for col in disp.select_dtypes(include="number").columns:
            disp[col] = disp[col].apply(lambda x: f"{x:,.0f}" if x != 0 else "—")
        st.dataframe(disp, use_container_width=True, height=420)

        if "BF IBNR" in summary.columns:
            diff = summary["BF IBNR"].sum() - total_cl_ibnr
            sign = "+" if diff >= 0 else ""
            pct  = diff / total_cl_ibnr * 100 if total_cl_ibnr else 0
            st.caption(f"BF vs CL difference: **{sign}{fmt(diff)}** ({sign}{pct:.1f}%)")

    # Tab 2 – Triangle & LDFs
    with tabs[1]:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Cumulative loss development triangle")
            tri_disp = tri_df.copy()
            tri_disp.columns = [str(c) for c in tri_disp.columns]
            tri_disp.index   = [str(i) for i in tri_disp.index]
            st.dataframe(
                tri_disp.apply(pd.to_numeric, errors="coerce")
                        .style.format("{:,.0f}", na_rep="—")
                        .background_gradient(cmap="Blues", axis=None),
                use_container_width=True, height=400,
            )
        with right:
            st.subheader("Volume-weighted LDFs")
            ldf_disp = ldf_df.copy()
            ldf_disp.columns = [str(c) for c in ldf_disp.columns]
            ldf_disp.index = ["LDF"]
            st.dataframe(ldf_disp.T.style.format("{:.4f}"),
                         use_container_width=True, height=400)

        st.plotly_chart(chart_triangle_heatmap(
            tri_df.apply(pd.to_numeric, errors="coerce")),
            use_container_width=True, key="tri_heat")

    # Tab 3 – Charts
    with tabs[2]:
        r1l, r1r = st.columns(2)
        with r1l:
            st.subheader("IBNR by accident year")
            st.plotly_chart(chart_ibnr_bar(summary), use_container_width=True, key="ibnr_bar")
        with r1r:
            st.subheader("Ultimate loss comparison")
            st.plotly_chart(chart_ultimates(summary), use_container_width=True, key="ult_comp")

        r2l, r2r = st.columns(2)
        with r2l:
            st.subheader("Age-to-age link ratios")
            st.plotly_chart(chart_ldfs(dev), use_container_width=True, key="ldf_chart")
        with r2r:
            st.subheader("Mack reserve uncertainty (±1σ)")
            st.plotly_chart(chart_mack(summary), use_container_width=True, key="mack_chart")

    # Tab 4 – Diagnostics
    with tabs[3]:
        st.subheader("Model diagnostics")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Mack total uncertainty**")
            mack_diag = pd.DataFrame({
                "Metric": ["Total IBNR", "Total Mack Std Error", "CoV (%)",
                           "95% CI Lower", "95% CI Upper"],
                "Value": [
                    f"{total_cl_ibnr:,.0f}",
                    f"{total_mack_se:,.0f}",
                    f"{cov:.2f}%",
                    f"{max(0, total_cl_ibnr - 1.96 * total_mack_se):,.0f}",
                    f"{total_cl_ibnr + 1.96 * total_mack_se:,.0f}",
                ],
            }).set_index("Metric")
            st.dataframe(mack_diag, use_container_width=True)

        with d2:
            avail = [("Chain Ladder", total_cl_ibnr)]
            if "BF IBNR" in summary.columns:
                avail.append(("Bornhuetter–Ferguson", summary["BF IBNR"].sum()))
            if "CC IBNR" in summary.columns:
                avail.append(("Cape Cod", summary["CC IBNR"].sum()))
            if len(avail) > 1:
                st.markdown("**Method comparison — total IBNR**")
                base = avail[0][1]
                comp_df = pd.DataFrame([
                    {"Method": m, "Total IBNR": f"{v:,.0f}",
                     "vs CL (%)": f"{'+' if v >= base else ''}{(v-base)/base*100:.1f}%"}
                    for m, v in avail
                ]).set_index("Method")
                st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.markdown("**Note:** Tail factor = 1.000 (no tail applied). To add a tail, use `cl.TailCurve()` or `cl.TailConstant()` in your pipeline.")

    # ── Export ────────────────────────────────────────────────────────────────
    if export_btn:
        try:
            excel_bytes = build_excel(tri_df, ldf_df, summary)
            st.sidebar.download_button(
                label="⬇️ Download reserve_report.xlsx",
                data=excel_bytes,
                file_name="reserve_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.sidebar.success("Ready — click above to download.")
        except Exception as e:
            st.sidebar.error(f"Export error: {e}")


if __name__ == "__main__":
    main()
