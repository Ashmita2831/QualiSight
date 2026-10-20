import streamlit as st
import textwrap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from textwrap import dedent

def render_html(html):
    html = dedent(str(html)).strip()
    st.markdown(html, unsafe_allow_html=True)
# ============================================================
# QUALISIGHT - DATA QUALITY INTELLIGENCE SYSTEM
# ============================================================

from modules.profiling import (
    get_dataset_overview,
    get_column_profile,
    get_statistical_summary,
    get_categorical_summary
)

from modules.context_detection import detect_context

from modules.quality_checks import run_quality_checks

from modules.recommendations import (
    generate_recommendation,
    generate_insights
)

from modules.remediation import (
    fill_missing_values,
    remove_duplicates,
    standardize_category_column,
    handle_outliers,
    replace_invalid_values
)

from modules.scoring import (
    calculate_quality_score,
    get_score_label
)

from modules.anomaly_detection import detect_ml_anomalies


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="QualiSight | Data Quality Intelligence",
    page_icon="assets/qualisight_eye.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "qualisight_eye.png"

top_logo_html = ""
if LOGO_PATH.exists():
    import base64
    _top_logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    top_logo_html = f'<img src="data:image/png;base64,{_top_logo_b64}" alt="QualiSight logo">'


# ============================================================
# HTML CLEANUP HELPER
# ============================================================
def clean_html(html):
    """Return HTML as a single Markdown-safe line.

    Streamlit Markdown treats indented lines as code blocks.
    Collapsing all HTML/CSS whitespace onto one line prevents
    multiline HTML from ever being displayed as raw code.
    """
    if html is None:
        return ""
    return " ".join(
        line.strip()
        for line in str(html).splitlines()
        if line.strip()
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    clean_html("""
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background: #f4f8fd;
        color: #102a56;
    }

    .main {
        background: #f4f8fd;
    }

    .block-container { padding-top:.55rem; padding-left:1.15rem; padding-right:1.15rem; max-width:1500px; }

    /* Remove Streamlit top decoration */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] { background:radial-gradient(circle at 8% 82%,rgba(32,116,229,.42),transparent 28%),radial-gradient(circle at 100% 68%,rgba(71,63,203,.28),transparent 30%),linear-gradient(180deg,#071d47 0%,#0b2c65 52%,#06183a 100%); border-right:1px solid rgba(255,255,255,.08); position:relative; overflow:hidden; }
    section[data-testid="stSidebar"]::after { content:""; position:absolute; left:-12%; right:-12%; bottom:-54px; height:170px; background:radial-gradient(ellipse at 35% 25%,rgba(46,135,255,.42),transparent 57%); transform:rotate(-8deg); pointer-events:none; }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    .sidebar-logo { text-align:center; padding:2px 8px 12px 8px; position:relative; z-index:2; }

    .sidebar-logo img { width:225px; max-width:100%; filter:drop-shadow(0 8px 16px rgba(0,0,0,.16)); }

    .sidebar-subtitle { color:#a5cbff; font-size:12px; text-align:center; margin-top:-7px; margin-bottom:22px; letter-spacing:.25px; position:relative; z-index:2; }

    .sidebar-section { color:#8fc5ff; font-size:10px; font-weight:800; letter-spacing:1.7px; text-transform:uppercase; margin:12px 0 8px 7px; position:relative; z-index:2; }

    /* Sidebar buttons */

    section[data-testid="stSidebar"] .stButton > button { width:100%; border:1px solid transparent; border-radius:13px; background:transparent; color:#c5dbff; text-align:left; font-size:15px; font-weight:550; padding:12px 15px; margin:3px 0; transition:all .2s ease; position:relative; z-index:2; box-shadow:none; }

    section[data-testid="stSidebar"] .stButton > button:hover { background:rgba(39,126,244,.20); color:white; border-color:rgba(110,177,255,.12); transform:translateX(2px); }

    /* ======================================================
       TOP HEADER
       ====================================================== */

    .top-header { display:flex; align-items:center; justify-content:space-between; padding:0 7px 12px; margin-bottom:2px; }
    .top-brand { display:flex; align-items:center; gap:10px; }
    .top-brand img { width:82px; height:58px; object-fit:contain; }
    .top-brand-name { color:#102e69; font-size:30px; font-weight:800; letter-spacing:-1.2px; }

    .top-user {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #24477d;
        font-size: 14px;
    }

    .user-circle {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1269e8, #3155d9);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
    }

    .notification {
        font-size: 22px;
        color: #38649c;
        margin-right: 8px;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        position: relative;
        overflow: hidden;
        min-height:410px; border-radius:18px; padding:30px 24px 24px; margin-bottom:20px;

        background:
            radial-gradient(
                circle at 85% 45%,
                rgba(74, 151, 255, 0.22),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #eaf7ff 0%,
                #f5faff 48%,
                #dff2ff 100%
            );

        border: 1px solid #b9ddff;
        box-shadow: 0 10px 30px rgba(38, 102, 170, 0.08);
    }

    .hero-content { width:59%;
        position: relative;
        z-index: 2;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 16px;
        background: #d1efff;
        color: #0767ce;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .hero-title { font-size:41px;
        line-height: 1.12;
        font-weight: 800;
        color: #102e69;
        margin: 0;
        letter-spacing: -1.5px;
    }

    .hero-title span {
        color: #0875ed;
    }

    .hero-description {
        color: #42658f;
        font-size: 16px;
        line-height: 1.65;
        margin-top: 15px;
        max-width: 690px;
    }

    .hero-logo { position:absolute; right:12px; top:22px; width:47%; max-width:560px;
        z-index: 1;
    }

    .hero-logo img {
        width: 100%;
        height: auto;
        object-fit: contain;
        filter: drop-shadow(0px 15px 25px rgba(18, 76, 155, 0.12));
    }

    /* ======================================================
       FEATURE ROW
       ====================================================== */

    .feature-row {
        display: flex;
        margin-top: 25px;
        width: 100%;
    }

    .feature {
        flex: 1;
        padding: 5px 20px;
        border-right: 1px solid #b9d9f7;
    }

    .feature:first-child {
        padding-left: 0;
    }

    .feature:last-child {
        border-right: none;
    }

    .feature-title {
        color: #0d3371;
        font-size: 16px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .feature-text {
        color: #54749c;
        font-size: 12px;
        line-height: 1.5;
    }

    .feature-dot {
        display: inline-flex;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        align-items: center;
        justify-content: center;
        background: #d9edff;
        color: #0c73e8;
        margin-right: 8px;
        font-weight: 800;
    }

    /* ======================================================
       DASHBOARD CARDS
       ====================================================== */

    .metric-card { position:relative; overflow:hidden; background:linear-gradient(145deg,#fff 0%,#fbfdff 100%); border:1px solid #e1ebf7; border-radius:15px; min-height:145px; padding:18px 19px; box-shadow:0 8px 25px rgba(31,88,143,.09); }

    .metric-card::after {
        content: "";
        position: absolute;
        width: 160px;
        height: 70px;
        right: -30px;
        bottom: -28px;
        border-radius: 50%;
        background: rgba(35, 125, 235, 0.07);
    }

    .metric-icon {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 20px;
        font-weight: 800;
        float: left;
        margin-right: 14px;
    }

    .blue-icon {
        background: linear-gradient(135deg, #087df2, #3455dd);
    }

    .purple-icon {
        background: linear-gradient(135deg, #7049ef, #6d39d8);
    }

    .orange-icon {
        background: linear-gradient(135deg, #ffad36, #f28719);
    }

    .pink-icon {
        background: linear-gradient(135deg, #fa557a, #ec3970);
    }

    .metric-label {
        color: #183b70;
        font-size: 14px;
        font-weight: 650;
        padding-top: 3px;
    }

    .metric-value {
        color: #092e69;
        font-size: 29px;
        font-weight: 800;
        margin-top: 3px;
    }

    .metric-small {
        color: #57759c;
        font-size: 12px;
        margin-top: 4px;
    }

    .score-pill {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        background: #d5f8ef;
        color: #058d70;
        font-size: 11px;
        font-weight: 700;
        margin-top: 8px;
    }

    /* ======================================================
       SECTION CARDS
       ====================================================== */

    .section-card {
        background: white;
        border: 1px solid #e1ebf7;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 7px 22px rgba(31, 88, 143, 0.07);
        margin-top: 20px;
    }

    .section-title {
        color: #0a3474;
        font-size: 18px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #7189a8;
        font-size: 12px;
        margin-bottom: 15px;
    }

    /* ======================================================
       CONTEXT CARD
       ====================================================== */

    .context-card {
        background: linear-gradient(
            135deg,
            #f0f6ff,
            #f8fbff
        );
        border: 1px solid #cfe2ff;
        border-radius: 14px;
        padding: 18px;
        margin-top: 12px;
    }

    .context-name {
        color: #0c3476;
        font-size: 22px;
        font-weight: 800;
    }

    .confidence-pill {
        display: inline-block;
        background: #d7f7f0;
        color: #008a70;
        padding: 5px 11px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        margin-left: 10px;
    }

    .context-description {
        color: #53739d;
        font-size: 12px;
        margin-top: 10px;
    }

    .workspace-note { margin:245px 10px 0; padding-top:17px; border-top:1px solid rgba(130,184,255,.28); color:#a9c8f7; font-size:12px; line-height:1.55; position:relative; z-index:2; }
    .workspace-note strong { color:#fff; font-size:13px; }
    .workspace-note span { color:#79b7ff; font-style:italic; }

    /* ======================================================
       PAGE TITLES
       ====================================================== */

    .page-title {
        color: #0a2e69;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .page-subtitle {
        color: #607da2;
        font-size: 14px;
        margin-bottom: 25px;
    }

    /* ======================================================
       UPLOAD BOX
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        border: 1px dashed #75aeea;
        padding: 8px;
    }

    /* ======================================================
       TABLES
       ====================================================== */

    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #d2e3f6;
        color: #0a4c9c;
        background: white;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        border-color: #1877e8;
        color: #086ce0;
        background: #f0f7ff;
    }

    /* Primary button */

    .stDownloadButton > button {
        border-radius: 10px;
        background: linear-gradient(
            135deg,
            #0878eb,
            #3454d8
        );
        color: white;
        border: none;
        font-weight: 700;
    }

    /* ======================================================
       TABS
       ====================================================== */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #52729b;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #0874e5;
    }

    /* ======================================================
       ALERTS
       ====================================================== */

    .stAlert {
        border-radius: 12px;
    }

    /* ======================================================
       HIDE STREAMLIT BRANDING
       ====================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """),
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "df": None,
    "original_df": None,
    "issues": [],
    "context_result": None,
    "quality_score": 100,
    "anomaly_df": None,
    "anomaly_count": 0,
    "page": "Home",
    "uploaded_name": None,
    "insights": []
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dataset(uploaded_file):
    """Load CSV or Excel file."""

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            return pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            return pd.read_excel(uploaded_file)

        else:
            st.error("Please upload a CSV or Excel file.")
            return None

    except Exception as e:
        st.error(f"Unable to read the file: {e}")
        return None


def safe_value(dictionary, *keys, default=None):
    """Safely retrieve a value from a dictionary."""

    for key in keys:
        if key in dictionary:
            return dictionary[key]

    return default


# ============================================================
# QUALITY ISSUE HELPERS
# ============================================================

def get_issue_type(issue):
    return issue.get(
        "Type",
        "Data Quality Issue"
    )


def get_issue_column(issue):
    return issue.get(
        "Column",
        "Dataset"
    )


def get_issue_count(issue):
    try:
        return int(
            issue.get(
                "Count",
                0
            )
        )
    except:
        return 0


def get_issue_percentage(issue):
    try:
        return float(
            issue.get(
                "Percentage",
                0
            )
        )
    except:
        return 0.0


def get_issue_severity(issue):
    severity = str(
        issue.get(
            "Severity",
            "Low"
        )
    ).strip().title()

    if severity not in [
        "Critical",
        "High",
        "Medium",
        "Low"
    ]:
        severity = "Low"

    return severity


def severity_value(severity):
    mapping = {
        "Critical": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1
    }

    return mapping.get(str(severity), 1)


def analyze_dataset(df):

    # Context
    context_result = detect_context(df)

    # The context detector returns a dictionary containing the
    # detected context and confidence. The quality-check engine
    # expects only the context name (a string) for rule lookup.
    if isinstance(context_result, dict):
        context_name = (
            context_result.get("context")
            or context_result.get("detected_context")
            or "General"
        )
    else:
        context_name = context_result or "General"

    # Normalize known context names so they match the rule keys.
    context_aliases = {
        "academic": "Academic",
        "sales": "Sales",
        "hr": "HR",
        "human resources": "HR",
        "healthcare": "Healthcare",
        "general": "General"
    }
    context_name = context_aliases.get(
        str(context_name).strip().lower(),
        str(context_name).strip()
    )

    # Quality checks
    issues = run_quality_checks(
        df,
        context_name
    )

    # Quality score
    score = calculate_quality_score(
        df,
        issues
    )

    # ML anomaly detection
    anomaly_df = detect_ml_anomalies(
        df
    )

    if anomaly_df is not None and not anomaly_df.empty:
        anomaly_count = int(
            anomaly_df["Potential Anomaly"].sum()
        )
    else:
        anomaly_count = 0

    # Insights
    try:
        insights = generate_insights(
            df,
            context_result,
            issues,
            {
                "anomaly_count": anomaly_count,
                "anomaly_percentage": (
                    anomaly_count / len(df) * 100 if len(df) else 0
                )
            }
        )
    except Exception:
        insights = []

    return (
        context_result,
        issues,
        score,
        anomaly_df,
        anomaly_count,
        insights
    )


def process_uploaded_dataset(uploaded_file):

    df = load_dataset(uploaded_file)

    if df is None:
        return

    if df.empty:
        st.error("The uploaded dataset is empty.")
        return

    st.session_state.original_df = df.copy()
    st.session_state.df = df.copy()
    st.session_state.uploaded_name = uploaded_file.name

    (
        context_result,
        issues,
        score,
        anomaly_df,
        anomaly_count,
        insights
    ) = analyze_dataset(df)

    st.session_state.context_result = context_result
    st.session_state.issues = issues
    st.session_state.quality_score = score
    st.session_state.anomaly_df = anomaly_df
    st.session_state.anomaly_count = anomaly_count
    st.session_state.insights = insights


def rerun_analysis():

    if st.session_state.df is None:
        return

    (
        context_result,
        issues,
        score,
        anomaly_df,
        anomaly_count,
        insights
    ) = analyze_dataset(
        st.session_state.df
    )

    st.session_state.context_result = context_result
    st.session_state.issues = issues
    st.session_state.quality_score = score
    st.session_state.anomaly_df = anomaly_df
    st.session_state.anomaly_count = anomaly_count
    st.session_state.insights = insights


def get_context_name():

    result = st.session_state.context_result

    if not result:
        return "Not detected"

    if isinstance(result, dict):

        return (
            result.get("context")
            or result.get("detected_context")
            or "Unknown"
        )

    return str(result)


def get_context_confidence():

    result = st.session_state.context_result

    if not isinstance(result, dict):
        return 0

    confidence = result.get(
        "confidence",
        result.get("context_confidence", 0)
    )

    try:
        confidence = float(confidence)

        if confidence > 1:
            confidence = confidence / 100

        return min(max(confidence, 0), 1)

    except:
        return 0


def get_missing_count():

    if st.session_state.df is None:
        return 0

    return int(
        st.session_state.df.isna().sum().sum()
    )


def get_duplicate_count():

    if st.session_state.df is None:
        return 0

    return int(
        st.session_state.df.duplicated().sum()
    )


def get_outlier_count():

    count = 0

    for issue in st.session_state.issues:

        issue_type = get_issue_type(issue).lower()

        if (
            "outlier" in issue_type
            or "anomal" in issue_type
        ):
            count += get_issue_count(issue)

    return count


def quality_label():

    try:
        return get_score_label(
            st.session_state.quality_score
        )
    except:
        score = st.session_state.quality_score

        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Needs Attention"
        return "Poor"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if LOGO_PATH.exists():

        st.markdown(
            '<div class="sidebar-logo">',
            unsafe_allow_html=True
        )

        st.image(
            str(LOGO_PATH),
            width=190
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Data Quality Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">Workspace</div>',
        unsafe_allow_html=True
    )

    navigation = [("⌂", "Home"), ("▦", "Dashboard"), ("▤", "Data Profile"), ("△", "Quality Issues"), ("◈", "ML Anomalies"), ("◇", "Smart Insights"), ("⚒", "Fix Data")]

    for icon, item in navigation:
        if st.button(f"{icon}    {item}", key=f"nav_{item}", use_container_width=True):
            st.session_state.page = item
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        clean_html("""<div class="workspace-note"><strong>Rule Engine +</strong><br><strong>Machine Learning</strong><br><span>Isolation Forest</span></div>"""),
        unsafe_allow_html=True
    )


# ============================================================
# TOP HEADER
# ============================================================

st.markdown(
    clean_html(f"""<div class="top-header"><div class="top-brand">{top_logo_html}<div class="top-brand-name">QualiSight</div></div><div class="top-user"><div class="notification">♧</div><div class="user-circle">QS</div><div><div style="font-size:11px;color:#7188a5;">Welcome</div><b style="color:#173a70;">QualiSight User</b></div><span style="font-size:18px;color:#6c89ac;">⌄</span></div></div>"""),
    unsafe_allow_html=True
)


# ============================================================
# HOME
# ============================================================

def show_home():

    # Build the hero as ONE HTML block.
    # Keeping the image inside the same markdown block prevents
    # Streamlit from breaking the HTML container into separate elements.
    logo_html = ""
    if LOGO_PATH.exists():
        import base64
        logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        logo_html = (
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'alt="QualiSight logo">'
        )

    st.markdown(
        clean_html(f"""<div class="hero">
<div class="hero-content">
<div class="hero-badge">Data Quality Intelligence System</div>
<div class="hero-title">Understand your data.<br><span>Improve it smarter.</span></div>
<div class="hero-description">QualiSight analyzes datasets, detects data quality issues, identifies potential anomalies using machine learning, understands the likely dataset context, and provides recommendations for controlled improvement.</div>
<div class="feature-row">
<div class="feature"><div class="feature-title"><span class="feature-dot">01</span> Analyze</div><div class="feature-text">Profile your dataset and understand its structure and statistics.</div></div>
<div class="feature"><div class="feature-title"><span class="feature-dot">02</span> Understand</div><div class="feature-text">Detect the likely context and identify relevant data quality problems.</div></div>
<div class="feature"><div class="feature-title"><span class="feature-dot">03</span> Improve</div><div class="feature-text">Get recommendations and choose which corrections to apply.</div></div>
</div>
</div>
<div class="hero-logo">{logo_html}</div>
</div>"""),
        unsafe_allow_html=True
    )

    # Upload section
    # Keep this HTML flush-left inside the Markdown string.
    # Streamlit's Markdown parser treats 4+ leading spaces as a code block.
    st.markdown(
        clean_html("""<div class="section-card">
<div class="section-title">Analyze a Dataset</div>
<div class="section-subtitle">Upload a CSV or Excel file to begin your data quality analysis.</div>
</div>"""),
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        if (
            st.session_state.uploaded_name
            != uploaded_file.name
        ):

            with st.spinner(
                "Analyzing your dataset..."
            ):
                process_uploaded_dataset(
                    uploaded_file
                )

            st.success(
                f"{uploaded_file.name} analyzed successfully."
            )

    # Dashboard metrics
    if st.session_state.df is not None:

        st.markdown("<br>", unsafe_allow_html=True)

        display_dashboard_cards()
        show_dashboard(show_cards=False, show_title=False)


# ============================================================
# DASHBOARD CARDS
# ============================================================

def display_dashboard_cards():

    df = st.session_state.df

    score = st.session_state.quality_score
    missing = get_missing_count()
    rows = len(df)
    anomalies = st.session_state.anomaly_count

    columns = st.columns(4)

    card_data = [
        (
            "blue-icon",
            "Q",
            "Quality Score",
            f"{score:.0f}/100",
            quality_label()
        ),
        (
            "purple-icon",
            "R",
            "Total Rows",
            f"{rows:,}",
            f"{len(df.columns)} columns"
        ),
        (
            "orange-icon",
            "!",
            "Missing Values",
            f"{missing:,}",
            f"{(missing / max(df.size, 1) * 100):.2f}% of total"
        ),
        (
            "pink-icon",
            "A",
            "ML Anomalies",
            f"{anomalies:,}",
            f"{(anomalies / max(rows, 1) * 100):.2f}% of total"
        )
    ]

    for col, data in zip(columns, card_data):

        icon_class, icon_text, label, value, small = data

        with col:

            st.markdown(
                clean_html(f"""
                <div class="metric-card">

                    <div class="metric-icon {icon_class}">
                        {icon_text}
                    </div>

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-small">
                        {small}
                    </div>

                </div>
                """),
                unsafe_allow_html=True
            )


# ============================================================
# DASHBOARD
# ============================================================

def show_dashboard(show_cards=True, show_title=True):

    if show_title:
        st.markdown(
            '<div class="page-title">Dashboard</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="page-subtitle">A complete overview of your dataset quality.</div>',
            unsafe_allow_html=True
        )

    if st.session_state.df is None:

        st.info(
            "Upload a dataset from the Home page to view your dashboard."
        )
        return

    if show_cards:
        display_dashboard_cards()

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.55, 1])

    # --------------------------------------------------------
    # QUALITY ISSUES CHART
    # --------------------------------------------------------

    with left:

        st.markdown(
            clean_html("""
            <div class="section-card">

                <div class="section-title">
                    Quality Issues Overview
                </div>

                <div class="section-subtitle">
                    Distribution of detected rule-based data quality issues.
                </div>

            """),
            unsafe_allow_html=True
        )

                # --------------------------------------------------------
        # COUNT ISSUES BY SEVERITY
        # --------------------------------------------------------

        severity_counts = {
            "Low": 0,
            "Medium": 0,
            "High": 0,
            "Critical": 0
        }

        for issue in st.session_state.issues:

            severity = get_issue_severity(issue)

            if severity in severity_counts:
                severity_counts[severity] += 1

        # --------------------------------------------------------
        # CREATE SEVERITY CHART
        # --------------------------------------------------------

        if st.session_state.issues:

            severities = [
                "Low",
                "Medium",
                "High",
                "Critical"
            ]

            counts = [
                severity_counts["Low"],
                severity_counts["Medium"],
                severity_counts["High"],
                severity_counts["Critical"]
            ]

            severity_chart_colors = [
                "#38b87a",   # Low - green
                "#f6b73c",   # Medium - yellow
                "#ff7b54",   # High - orange
                "#ef476f"    # Critical - pink/red
            ]

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=severities,
                    y=counts,

                    marker_color=severity_chart_colors,

                    text=counts,
                    textposition="outside",

                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Issues: %{y}"
                        "<extra></extra>"
                    )
                )
            )

            fig.update_layout(

                height=330,

                margin=dict(
                    l=5,
                    r=5,
                    t=25,
                    b=10
                ),

                plot_bgcolor="white",
                paper_bgcolor="white",

                font=dict(
                    color="#244a7c"
                ),

                showlegend=False,

                xaxis=dict(
                    title="Severity",
                    categoryorder="array",
                    categoryarray=severities,
                    showgrid=False
                ),

                yaxis=dict(
                    title="Issues",
                    gridcolor="#edf3f9",
                    rangemode="tozero",
                    dtick=1
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

        else:

            st.success(
                "No rule-based quality issues were detected."
            )
            
        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    with right:

        context_name = get_context_name()
        confidence = get_context_confidence()

        st.markdown(
            clean_html(f"""
            <div class="section-card">

                <div class="section-title">
                    Detected Dataset Context
                </div>

                <div class="section-subtitle">
                    The detected context helps apply relevant
                    validation rules.
                </div>

                <div class="context-card">

                    <div>
                        <span class="context-name">
                            {context_name}
                        </span>

                        <span class="confidence-pill">
                            {confidence * 100:.1f}% Confidence
                        </span>
                    </div>

                    <div class="context-description">
                        Detected using dataset column names
                        and domain-related keywords.
                    </div>

                    <div style="
                        margin-top:16px;
                        height:8px;
                        border-radius:10px;
                        background:#dce9f7;
                    ">

                        <div style="
                            width:{confidence * 100}%;
                            height:8px;
                            border-radius:10px;
                            background:linear-gradient(
                                90deg,
                                #0878ee,
                                #7251e8
                            );
                        "></div>

                    </div>

                </div>

                <div style="
                    margin-top:14px;
                    padding:12px;
                    background:#edf7ff;
                    border:1px solid #c9e4ff;
                    border-radius:12px;
                    color:#2c609b;
                    font-size:12px;
                    line-height:1.5;
                ">
                    Context detection helps QualiSight apply
                    relevant validation rules to the dataset.
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )


# ============================================================
# DATA PROFILE
# ============================================================

def show_data_profile():

    st.markdown(
        '<div class="page-title">Data Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore the structure, statistics and characteristics of your dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.df is None:

        st.info(
            "Upload a dataset first."
        )
        return

    df = st.session_state.df

    overview = get_dataset_overview(df)

    # Overview cards
    cols = st.columns(5)

    overview_values = [
        ("Rows", overview.get("rows", len(df))),
        ("Columns", overview.get("columns", len(df.columns))),
        (
            "Missing",
            overview.get(
                "missing_values",
                get_missing_count()
            )
        ),
        (
            "Duplicates",
            overview.get(
                "duplicate_rows",
                get_duplicate_count()
            )
        ),
        (
            "Memory",
            overview.get(
                "memory_usage",
                "N/A"
            )
        )
    ]

    for col, (label, value) in zip(
        cols,
        overview_values
    ):

        with col:

            st.markdown(
                clean_html(f"""
                <div class="metric-card"
                     style="min-height:100px;">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value"
                         style="font-size:24px;">
                        {value}
                    </div>

                </div>
                """),
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Dataset Preview",
            "Column Profile",
            "Statistics",
            "Categorical Data"
        ]
    )

    with tab1:

        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=430
        )

    with tab2:

        profile = get_column_profile(df)

        st.dataframe(
            profile,
            use_container_width=True,
            height=430
        )

    with tab3:

        summary = get_statistical_summary(df)

        if summary is not None and not summary.empty:
            st.dataframe(
                summary,
                use_container_width=True,
                height=430
            )
        else:
            st.info(
                "No numerical columns available."
            )

    with tab4:

        categorical = get_categorical_summary(df)

        if categorical is not None and not categorical.empty:
            st.dataframe(
                categorical,
                use_container_width=True,
                height=430
            )
        else:
            st.info(
                "No categorical columns available."
            )


# ============================================================
# QUALITY ISSUES
# ============================================================

def show_quality_issues():

    # =========================================================
    # PAGE HEADER
    # =========================================================

    st.markdown(
        """
        <style>

        .qi-title {
            font-size: 38px;
            font-weight: 800;
            color: #0b3474;
            margin-bottom: 5px;
        }

        .qi-subtitle {
            font-size: 15px;
            color: #5c7da3;
            margin-bottom: 25px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="qi-title">Quality Issues</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="qi-subtitle">'
        'Review detected data quality problems, understand their impact, '
        'and inspect the recommended corrections.'
        '</div>',
        unsafe_allow_html=True
    )

    # =========================================================
    # GET DATA
    # =========================================================

    issues = st.session_state.get("issues", [])
    df = st.session_state.get("df", None)

    if not issues:

        st.success("No data quality issues were detected.")

        return

    # =========================================================
    # IMPORTANT:
    # YOUR ACTUAL ISSUE KEYS ARE:
    #
    # Type
    # Column
    # Count
    # Percentage
    # Severity
    # Description
    # Example
    # =========================================================

    # =========================================================
    # SEVERITY COUNTS
    # =========================================================

    severity_counts = {
        "Critical": 0,
        "High": 0,
        "Medium": 0,
        "Low": 0
    }

    for issue in issues:

        severity = str(
            issue.get("Severity", "Low")
        ).title()

        if severity in severity_counts:

            severity_counts[severity] += 1

    # =========================================================
    # TOTAL SUMMARY
    # =========================================================

    st.markdown("### Issue Summary")

    st.caption(
        f"{len(issues)} quality issues detected in the dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Critical",
            severity_counts["Critical"]
        )

    with c2:

        st.metric(
            "High",
            severity_counts["High"]
        )

    with c3:

        st.metric(
            "Medium",
            severity_counts["Medium"]
        )

    with c4:

        st.metric(
            "Low",
            severity_counts["Low"]
        )

    st.divider()

    # =========================================================
    # ISSUE TYPE SUMMARY
    # =========================================================

    st.markdown("### Issue Type Summary")

    type_summary = {}

    for issue in issues:

        issue_type = issue.get(
            "Type",
            "Unknown"
        )

        count = issue.get(
            "Count",
            0
        )

        try:
            count = int(count)
        except:
            count = 0

        if issue_type not in type_summary:

            type_summary[issue_type] = 0

        type_summary[issue_type] += count

    if type_summary:

        summary_df = pd.DataFrame(
            [
                {
                    "Issue Type": issue_type,
                    "Affected Records": count
                }

                for issue_type, count
                in type_summary.items()
            ]
        )

        summary_df = summary_df.sort_values(
            "Affected Records",
            ascending=False
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # =========================================================
    # AFFECTED COLUMN SUMMARY
    # =========================================================

    st.markdown("### Affected Columns")

    column_summary = {}

    for issue in issues:

        column = issue.get(
            "Column",
            "Dataset"
        )

        if column not in column_summary:

            column_summary[column] = 0

        column_summary[column] += 1

    column_df = pd.DataFrame(
        [
            {
                "Column": column,
                "Issues Detected": count
            }

            for column, count
            in column_summary.items()
        ]
    )

    column_df = column_df.sort_values(
        "Issues Detected",
        ascending=False
    )

    st.dataframe(
        column_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =========================================================
    # FILTERS
    # =========================================================

    st.markdown("### Detailed Issues")

    f1, f2, f3 = st.columns(3)

    issue_types = sorted(
        list(
            set(
                issue.get(
                    "Type",
                    "Unknown"
                )

                for issue in issues
            )
        )
    )

    columns = sorted(
        list(
            set(
                issue.get(
                    "Column",
                    "Dataset"
                )

                for issue in issues
            )
        )
    )

    with f1:

        selected_severity = st.selectbox(
            "Severity",
            [
                "All",
                "Critical",
                "High",
                "Medium",
                "Low"
            ],
            key="qi_severity"
        )

    with f2:

        selected_type = st.selectbox(
            "Issue Type",
            ["All"] + issue_types,
            key="qi_type"
        )

    with f3:

        selected_column = st.selectbox(
            "Column",
            ["All"] + columns,
            key="qi_column"
        )

    # =========================================================
    # FILTER ISSUES
    # =========================================================

    filtered_issues = []

    for issue in issues:

        severity = issue.get(
            "Severity",
            "Low"
        )

        issue_type = issue.get(
            "Type",
            "Unknown"
        )

        column = issue.get(
            "Column",
            "Dataset"
        )

        if (
            selected_severity != "All"
            and severity != selected_severity
        ):
            continue

        if (
            selected_type != "All"
            and issue_type != selected_type
        ):
            continue

        if (
            selected_column != "All"
            and column != selected_column
        ):
            continue

        filtered_issues.append(issue)

    st.caption(
        f"Showing {len(filtered_issues)} of {len(issues)} detected issues."
    )

    # =========================================================
    # DETAILED ISSUE CARDS
    # =========================================================

    for i, issue in enumerate(filtered_issues):

        issue_type = issue.get(
            "Type",
            "Unknown"
        )

        column = issue.get(
            "Column",
            "Dataset"
        )

        count = issue.get(
            "Count",
            0
        )

        percentage = issue.get(
            "Percentage",
            0
        )

        severity = issue.get(
            "Severity",
            "Low"
        )

        description = issue.get(
            "Description",
            "No description available."
        )

        example = issue.get(
            "Example",
            None
        )

        # -----------------------------------------------------
        # EXPANDER
        # -----------------------------------------------------

        with st.expander(
            f"{severity}  |  {issue_type}  |  {column}",
            expanded=False
        ):

            st.subheader(issue_type)

            st.caption(
                f"Severity: {severity}  |  Column: {column}"
            )

            m1, m2, m3 = st.columns(3)

            with m1:

                st.metric(
                    "Affected Records",
                    count
                )

            with m2:

                st.metric(
                    "Percentage",
                    f"{percentage:.2f}%"
                )

            with m3:

                if df is not None:

                    st.metric(
                        "Dataset Rows",
                        len(df)
                    )

            st.markdown("---")

            st.markdown("**What was detected**")

            st.write(description)

            if example:

                st.markdown("**Example**")

                st.code(
                    str(example)
                )

            st.markdown("**Recommended action**")

            if "Missing" in issue_type:

                st.info(
                    "Review the missing records and apply "
                    "an appropriate missing-value treatment."
                )

            elif "Duplicate" in issue_type:

                st.info(
                    "Review the duplicate records and remove "
                    "duplicates after confirmation."
                )

            elif "Categorical" in issue_type:

                st.info(
                    "Standardize equivalent categorical values "
                    "after reviewing the detected variations."
                )

            elif "Outlier" in issue_type:

                st.info(
                    "Review the unusual values before deciding "
                    "whether to cap, modify, or remove them."
                )

            elif "Invalid" in issue_type:

                st.info(
                    "Review the values against the applicable "
                    "validation rules."
                )

            else:

                st.info(
                    "Review the affected records before applying "
                    "a correction."
                )
# ============================================================
# ML ANOMALIES
# ============================================================

def show_ml_anomalies():

    st.markdown(
        '<div class="page-title">ML Anomalies</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Potentially unusual records detected using Isolation Forest.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.df is None:

        st.info("Upload a dataset first.")
        return

    anomaly_df = st.session_state.anomaly_df

    if anomaly_df is None or anomaly_df.empty:

        st.info(
            "No ML anomaly results are available for this dataset."
        )
        return

    anomaly_count = st.session_state.anomaly_count
    total_rows = len(st.session_state.df)

    cols = st.columns(3)

    with cols[0]:

        st.markdown(
            clean_html(f"""
            <div class="metric-card">

                <div class="metric-icon pink-icon">
                    A
                </div>

                <div class="metric-label">
                    Potential Anomalies
                </div>

                <div class="metric-value">
                    {anomaly_count}
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )

    with cols[1]:

        percentage = (
            anomaly_count /
            max(total_rows, 1)
        ) * 100

        st.markdown(
            clean_html(f"""
            <div class="metric-card">

                <div class="metric-label">
                    Dataset Percentage
                </div>

                <div class="metric-value">
                    {percentage:.2f}%
                </div>

                <div class="metric-small">
                    of total records
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )

    with cols[2]:

        st.markdown(
            clean_html("""
            <div class="metric-card">

                <div class="metric-label">
                    ML Method
                </div>

                <div class="metric-value"
                     style="font-size:23px;">
                    Isolation
                    Forest
                </div>

                <div class="metric-small">
                    Unsupervised anomaly detection
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        clean_html("""
        <div class="section-card">

            <div class="section-title">
                Potential Anomalous Records
            </div>

            <div class="section-subtitle">
                These records are statistically unusual according
                to the Isolation Forest model. An anomaly is not
                automatically an error.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )

    if "Potential Anomaly" in anomaly_df.columns:

        anomalous_records = anomaly_df[
            anomaly_df["Potential Anomaly"] == True
        ].copy()

        if not anomalous_records.empty:

            st.dataframe(
                anomalous_records,
                use_container_width=True,
                height=450
            )

        else:

            st.success(
                "No potential anomalies were identified."
            )

    else:

        st.dataframe(
            anomaly_df,
            use_container_width=True,
            height=450
        )


# ============================================================
# SMART INSIGHTS
# ============================================================

def get_comparison_snapshot(df):
    """Calculate comparable quality metrics without changing the dataframe."""
    if df is None:
        return None

    context_result = detect_context(df)
    if isinstance(context_result, dict):
        context_name = (
            context_result.get("context")
            or context_result.get("detected_context")
            or "General"
        )
    else:
        context_name = context_result or "General"

    context_aliases = {
        "academic": "Academic",
        "sales": "Sales",
        "hr": "HR",
        "human resources": "HR",
        "healthcare": "Healthcare",
        "general": "General"
    }
    context_name = context_aliases.get(
        str(context_name).strip().lower(),
        str(context_name).strip()
    )

    snapshot_issues = run_quality_checks(df, context_name)
    snapshot_score = calculate_quality_score(df, snapshot_issues)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "issues": len(snapshot_issues),
        "score": float(snapshot_score)
    }


def show_smart_insights():
    st.markdown(
        '<div class="page-title">Smart Insights</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">'
        'Interpret the findings and track the impact of cleaning actions.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.df is None:
        st.info("Upload a dataset first.")
        return

    current_df = st.session_state.df.copy()
    original_df = st.session_state.get("original_df")
    issues = st.session_state.get("issues", [])
    anomaly_count = int(st.session_state.get("anomaly_count", 0) or 0)
    row_count = len(current_df)
    column_count = len(current_df.columns)
    missing_count = int(current_df.isna().sum().sum())
    duplicate_count = int(current_df.duplicated().sum())
    high_priority_count = sum(
        1 for issue in issues
        if get_issue_severity(issue) in ("Critical", "High")
    )

    # Interpret the results instead of repeating raw issue/anomaly tables.
    insights = []
    context_name = get_context_name()
    confidence = get_context_confidence()
    if context_name and context_name != "Not detected":
        insights.append((
            "Dataset context",
            f"QualiSight classified this dataset as {context_name} with "
            f"{confidence * 100:.1f}% confidence. Treat this classification as guidance, "
            "not proof that every column has been interpreted correctly."
        ))

    if missing_count:
        insights.append((
            "Completeness needs attention",
            f"There are {missing_count:,} missing cell(s) in the current dataset. "
            "Choose a treatment column by column; mean or median filling is suitable only "
            "when it makes sense for that field."
        ))
    else:
        insights.append((
            "No missing cells detected",
            "The current dataset has no missing cells. This supports completeness, "
            "but does not by itself confirm that values are accurate or valid."
        ))

    if duplicate_count:
        insights.append((
            "Check repeated records",
            f"{duplicate_count:,} completely duplicated row(s) remain. Review whether "
            "they represent accidental repeats before removing them."
        ))
    else:
        insights.append((
            "No exact duplicate rows",
            "No completely duplicated rows were found. Similar records or duplicate IDs "
            "may still need separate checks."
        ))

    if high_priority_count:
        insights.append((
            "Prioritize higher-severity findings",
            f"{high_priority_count} detected issue(s) are marked High or Critical. "
            "Review these first because they may have a greater effect on downstream analysis."
        ))
    elif issues:
        insights.append((
            "Review findings before analysis",
            f"QualiSight currently reports {len(issues)} rule-based issue(s), with none "
            "marked High or Critical. Lower severity does not necessarily mean an issue "
            "is irrelevant to your use case."
        ))
    else:
        insights.append((
            "No rule-based issues reported",
            "The current rules did not report any issues. This does not prove the dataset "
            "is error-free; results depend on the checks available for the detected context."
        ))

    if anomaly_count:
        anomaly_pct = (anomaly_count / row_count * 100) if row_count else 0
        insights.append((
            "Investigate unusual records",
            f"Isolation Forest flagged {anomaly_count:,} potential anomalous record(s) "
            f"({anomaly_pct:.2f}% of rows). These are statistically unusual, not automatically "
            "incorrect; inspect them before changing or removing them."
        ))
    else:
        insights.append((
            "No potential ML anomalies reported",
            "The current anomaly detector did not flag records. This is not a guarantee "
            "that every value follows the expected business rules."
        ))

    if row_count == 0 or column_count == 0:
        readiness = "The dataset has no usable rows or columns, so analysis should wait until valid data is available."
    elif missing_count or duplicate_count or high_priority_count:
        readiness = "The dataset can be explored, but review outstanding completeness, duplication, or higher-severity findings before drawing conclusions."
    else:
        readiness = "The current checks show no missing cells, exact duplicates, or high-priority rule findings. Review domain validity and potential anomalies before treating the dataset as analysis-ready."
    insights.append(("Analysis readiness", readiness))

    st.markdown(
        '<div class="section-card"><div class="section-title">What the results mean</div>'
        '<div class="section-subtitle">Observations generated from the current dataset and analysis.</div></div>',
        unsafe_allow_html=True
    )
    for index, (heading, message) in enumerate(insights, start=1):
        st.markdown(
            clean_html(f"""
            <div class="section-card">
                <div style="display:flex;gap:14px;align-items:flex-start;">
                    <div style="width:35px;height:35px;min-width:35px;border-radius:50%;
                                background:#e4f2ff;color:#0874e5;display:flex;
                                align-items:center;justify-content:center;font-weight:800;">
                        {index}
                    </div>
                    <div style="color:#345a88;font-size:14px;line-height:1.6;">
                        <div style="font-weight:700;color:#173a70;margin-bottom:4px;">{heading}</div>
                        {message}
                    </div>
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

    # Before/after metrics use the same rule engine and scoring function.
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-card"><div class="section-title">Before vs After Comparison</div>'
        '<div class="section-subtitle">The uploaded dataset is compared with the current version after cleaning actions.</div></div>',
        unsafe_allow_html=True
    )

    if original_df is None:
        st.info("The original dataset snapshot is unavailable. Re-upload the dataset to enable this comparison.")
        return

    before = get_comparison_snapshot(original_df)
    after = get_comparison_snapshot(current_df)
    if before is None or after is None:
        st.info("Comparison metrics could not be calculated for this dataset.")
        return

    score_change = after["score"] - before["score"]
    if score_change > 0:
        st.success(f"Quality score increased by {score_change:.2f} points after the applied changes.")
    elif score_change < 0:
        st.warning(f"Quality score decreased by {abs(score_change):.2f} points. Review the changes and their effect on your data.")
    else:
        st.info("The quality score did not change after the applied changes.")

    comparison_rows = [
        {"Metric": "Quality score", "Before cleaning": f'{before["score"]:.2f}/100', "Current dataset": f'{after["score"]:.2f}/100', "Change": f'{score_change:+.2f} points'},
        {"Metric": "Rows", "Before cleaning": before["rows"], "Current dataset": after["rows"], "Change": f'{after["rows"] - before["rows"]:+,}'},
        {"Metric": "Columns", "Before cleaning": before["columns"], "Current dataset": after["columns"], "Change": f'{after["columns"] - before["columns"]:+,}'},
        {"Metric": "Missing cells", "Before cleaning": before["missing"], "Current dataset": after["missing"], "Change": f'{after["missing"] - before["missing"]:+,}'},
        {"Metric": "Exact duplicate rows", "Before cleaning": before["duplicates"], "Current dataset": after["duplicates"], "Change": f'{after["duplicates"] - before["duplicates"]:+,}'},
        {"Metric": "Rule-based issues", "Before cleaning": before["issues"], "Current dataset": after["issues"], "Change": f'{after["issues"] - before["issues"]:+,}'}
    ]
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
    st.caption("A lower issue count or higher score is not a substitute for validating whether the cleaned values are appropriate for your dataset.")


# ============================================================
# FIX DATA
# ============================================================

def show_fix_data():

    st.markdown(
        '<div class="page-title">Fix Data</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Select the corrections you want to apply. QualiSight does not '
        'blindly modify your dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.df is None:

        st.info("Upload a dataset first.")
        return

    df = st.session_state.df.copy()

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    st.markdown(
        clean_html("""
        <div class="section-card">

            <div class="section-title">
                Missing Values
            </div>

            <div class="section-subtitle">
                Choose how missing values should be handled.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )

    missing_columns = [
        col for col in df.columns
        if df[col].isna().sum() > 0
    ]

    if missing_columns:

        col1, col2 = st.columns([1, 1])

        with col1:

            selected_column = st.selectbox(
                "Column",
                missing_columns,
                key="missing_column"
            )

        with col2:

            method = st.selectbox(
                "Treatment",
                [
                    "median",
                    "mean",
                    "mode",
                    "forward_fill"
                ],
                key="missing_method"
            )

        if st.button(
            "Apply Missing Value Treatment",
            key="apply_missing"
        ):

            # The UI uses lowercase labels, while remediation.py expects
            # title-case method names. Map them before calling the function.
            method_map = {
                "median": "Median",
                "mean": "Mean",
                "mode": "Mode",
                "forward_fill": "Forward Fill",
            }
            method_for_cleaning = method_map.get(method, method)

            before_missing = int(st.session_state.df.isna().sum().sum())
            cleaned_df = fill_missing_values(
                st.session_state.df,
                selected_column,
                method_for_cleaning
            )
            after_missing = int(cleaned_df.isna().sum().sum())
            st.session_state.df = cleaned_df

            rerun_analysis()

            if after_missing < before_missing:
                st.success(
                    f"Treatment applied to '{selected_column}'. Missing cells decreased "
                    f"from {before_missing:,} to {after_missing:,}."
                )
            elif before_missing == 0:
                st.info("No missing cells were available to fill.")
            else:
                st.warning(
                    f"Missing cells remain at {after_missing:,}. This method may not "
                    "apply to this column's data type, or the selected column may not "
                    "have values that can be filled by this method."
                )

            st.rerun()

    else:

        st.success(
            "No missing values are currently present."
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        clean_html("""
        <div class="section-card">

            <div class="section-title">
                Duplicate Rows
            </div>

            <div class="section-subtitle">
                Remove completely duplicated records.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )

    duplicates = get_duplicate_count()

    if duplicates > 0:

        st.warning(
            f"{duplicates} duplicate rows detected."
        )

        if st.button(
            "Remove Duplicate Rows",
            key="remove_duplicates"
        ):

            st.session_state.df = remove_duplicates(
                st.session_state.df
            )

            rerun_analysis()

            st.success(
                "Duplicate rows removed."
            )

            st.rerun()

    else:

        st.success(
            "No duplicate rows are currently present."
        )

    # --------------------------------------------------------
    # CATEGORY STANDARDIZATION
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        clean_html("""
        <div class="section-card">

            <div class="section-title">
                Categorical Standardization
            </div>

            <div class="section-subtitle">
                Remove unwanted spaces and standardize categorical text.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_columns:

        category_column = st.selectbox(
            "Select categorical column",
            categorical_columns,
            key="category_column"
        )

        if st.button(
            "Standardize Category Values",
            key="standardize_category"
        ):

            st.session_state.df = standardize_category_column(
                st.session_state.df,
                category_column
            )

            rerun_analysis()

            st.success(
                "Categorical values standardized."
            )

            st.rerun()

    else:

        st.info(
            "No categorical columns detected."
        )

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        clean_html("""
        <div class="section-card">

            <div class="section-title">
                Statistical Outliers
            </div>

            <div class="section-subtitle">
                Handle numerical outliers using IQR-based processing.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numerical_columns:

        col1, col2 = st.columns(2)

        with col1:

            outlier_column = st.selectbox(
                "Numerical column",
                numerical_columns,
                key="outlier_column"
            )

        with col2:

            outlier_method = st.selectbox(
                "Treatment",
                [
                    "cap",
                    "remove"
                ],
                key="outlier_method"
            )

        if st.button(
            "Apply Outlier Treatment",
            key="apply_outlier"
        ):

            st.session_state.df = handle_outliers(
                st.session_state.df,
                outlier_column,
                outlier_method
            )

            rerun_analysis()

            st.success(
                "Outlier treatment applied."
            )

            st.rerun()

    else:

        st.info(
            "No numerical columns available."
        )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        clean_html("""
        <div class="section-card">

            <div class="section-title">
                Export Improved Dataset
            </div>

            <div class="section-subtitle">
                Download the currently improved dataset.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )

    csv_data = st.session_state.df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Improved CSV",
        data=csv_data,
        file_name="qualisight_improved_dataset.csv",
        mime="text/csv"
    )


# ============================================================
# ROUTING
# ============================================================

page = st.session_state.page

if page == "Home":

    show_home()

elif page == "Dashboard":

    show_dashboard()

elif page == "Data Profile":

    show_data_profile()

elif page == "Quality Issues":

    show_quality_issues()

elif page == "ML Anomalies":

    show_ml_anomalies()

elif page == "Smart Insights":

    show_smart_insights()

elif page == "Fix Data":

    show_fix_data()
