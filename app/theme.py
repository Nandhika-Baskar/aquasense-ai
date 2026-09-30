"""
Visual layer for AquaSense AI.

All custom CSS lives here so the pages stay readable. The stylesheet replaces
the default Streamlit look with a restrained dashboard theme: dark navigation
rail, light content canvas, flat cards, and consistent type scale.
"""

from __future__ import annotations

import streamlit as st

from app import config as cfg


def page_setup() -> None:
    """Call once, first thing in main()."""
    st.set_page_config(
        page_title=f"{cfg.APP_NAME} -- Water Quality Intelligence",
        page_icon=":material/water_drop:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


STYLESHEET = f"""
<style>
:root {{
  --qs-primary: {cfg.COLOR_PRIMARY};
  --qs-ink: {cfg.COLOR_INK};
  --qs-muted: {cfg.COLOR_MUTED};
  --qs-border: {cfg.COLOR_BORDER};
  --qs-surface: {cfg.COLOR_SURFACE};
  --qs-canvas: {cfg.COLOR_CANVAS};
  --qs-ok: {cfg.COLOR_OK};
  --qs-warn: {cfg.COLOR_WARN};
  --qs-danger: {cfg.COLOR_DANGER};
}}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [class*="css"] {{
  font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] > .main {{
  background: var(--qs-canvas);
}}

.block-container {{ padding: 1.4rem 2.1rem 4rem; max-width: 1500px; }}

/* ---------- typography ---------- */
h1, h2, h3 {{ color: var(--qs-ink); letter-spacing: -0.015em; }}
h1 {{ font-size: 1.55rem !important; font-weight: 700; margin: 0 0 .15rem 0 !important; }}
h2 {{ font-size: 1.12rem !important; font-weight: 650; }}
h3 {{ font-size: .95rem !important; font-weight: 650; }}
p, span, label {{ color: var(--qs-ink); }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
  background: {cfg.COLOR_SIDEBAR};
  border-right: 1px solid #14323F;
}}
section[data-testid="stSidebar"] * {{ color: #C6D5DC; }}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-top: 1.1rem; }}
section[data-testid="stSidebar"] hr {{ border-color: #1E3D4A; margin: .7rem 0; }}

.qs-brand {{ padding: 0 0 .55rem 0; }}
.qs-brand-name {{
  font-size: 1.18rem; font-weight: 700; color: #FFFFFF !important; letter-spacing: -0.01em;
}}
.qs-brand-sub {{
  font-size: .705rem; color: #7FA3B0 !important; text-transform: uppercase;
  letter-spacing: .085em; margin-top: .18rem; line-height: 1.35;
}}
.qs-nav-heading {{
  font-size: .66rem; text-transform: uppercase; letter-spacing: .12em;
  color: #6E909E !important; font-weight: 600; margin: .55rem 0 .4rem 0;
}}
section[data-testid="stSidebar"] .stButton > button {{
  background: transparent;
  border: 1px solid transparent;
  color: #C6D5DC;
  text-align: left;
  font-size: .875rem;
  font-weight: 500;
  padding: .46rem .6rem;
  border-radius: 8px;
  margin-bottom: 2px;
  transition: background 140ms ease, color 140ms ease;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
  background: #14323F; color: #FFFFFF; border-color: #1E3D4A;
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  background: rgba(53, 196, 216, .16);
  border-color: rgba(53, 196, 216, .45);
  color: #FFFFFF; font-weight: 600;
}}

.qs-status-box {{
  background: #102E3A; border: 1px solid #1E3D4A; border-radius: 10px;
  padding: .7rem .75rem; font-size: .745rem; line-height: 1.6;
}}
.qs-status-row {{ display: flex; justify-content: space-between; gap: .5rem; }}
.qs-status-row span:first-child {{ color: #8FAFBA; }}
.qs-status-row span:last-child {{ color: #DCE8EC; font-weight: 600; }}

/* ---------- cards ---------- */
.qs-grid {{ display: flex; gap: 14px; flex-wrap: wrap; }}
.qs-grid > .qs-cell {{ flex: 1 1 0; min-width: 168px; }}
.qs-card {{
  background: var(--qs-surface);
  border: 1px solid var(--qs-border);
  border-radius: 12px;
  padding: 14px 16px 13px;
  height: 100%;
  box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}}
.qs-card-accent {{ border-left: 3px solid var(--qs-primary); }}

.qs-eyebrow {{
  font-size: .655rem; letter-spacing: .1em;
  color: var(--qs-muted); font-weight: 650; margin-bottom: .42rem;
  display: flex; align-items: center; gap: .4rem;
}}
.qs-value {{
  font-size: 1.62rem; font-weight: 700; color: var(--qs-ink);
  line-height: 1.05; letter-spacing: -.02em;
}}
.qs-value-sm {{ font-size: 1.28rem; }}
.qs-unit {{ font-size: .755rem; color: var(--qs-muted); font-weight: 500; margin-left: .22rem; }}
.qs-desc {{ font-size: .755rem; color: var(--qs-muted); margin-top: .38rem; line-height: 1.35; }}

.qs-dot {{
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  margin-right: .32rem; vertical-align: middle;
}}
.qs-chip {{
  display: inline-block; font-size: .655rem; font-weight: 650;
  letter-spacing: .06em; text-transform: uppercase;
  padding: .17rem .45rem; border-radius: 5px;
}}

/* ---------- hero / risk ---------- */
.qs-hero {{
  background: var(--qs-surface); border: 1px solid var(--qs-border);
  border-radius: 14px; padding: 20px 22px; height: 100%;
  box-shadow: 0 1px 2px rgba(16, 24, 40, .05);
}}
.qs-hero-risk {{ font-size: 2.5rem; font-weight: 750; letter-spacing: -.025em; line-height: 1; }}
.qs-hero-bar {{ height: 8px; border-radius: 99px; background: #EAEFF2; overflow: hidden; margin-top: .7rem; }}
.qs-hero-bar > div {{ height: 100%; border-radius: 99px; }}
.qs-meta {{ display: flex; gap: 1.6rem; flex-wrap: wrap; margin-top: .9rem; }}
.qs-meta-item .qs-meta-k {{
  font-size: .63rem; text-transform: uppercase; letter-spacing: .09em; color: var(--qs-muted);
}}
.qs-meta-item .qs-meta-v {{ font-size: .93rem; font-weight: 650; color: var(--qs-ink); }}

/* ---------- misc ---------- */
.qs-panel {{
  background: var(--qs-surface); border: 1px solid var(--qs-border);
  border-radius: 12px; padding: 15px 17px;
}}
.qs-note {{
  background: #F0F6F8; border: 1px solid #D7E6EB; border-left: 3px solid var(--qs-primary);
  border-radius: 8px; padding: .68rem .8rem; font-size: .8rem; color: #33505A; line-height: 1.5;
}}
.qs-warn-note {{
  background: #FDF6EC; border: 1px solid #F2E0C4; border-left: 3px solid var(--qs-warn);
  border-radius: 8px; padding: .68rem .8rem; font-size: .8rem; color: #6B4C1C; line-height: 1.5;
}}
.qs-flow {{ display: flex; flex-direction: column; gap: 7px; }}
.qs-flow-step {{
  background: var(--qs-surface); border: 1px solid var(--qs-border); border-radius: 9px;
  padding: .58rem .8rem; font-size: .82rem; font-weight: 550; color: var(--qs-ink);
  display: flex; align-items: center; gap: .6rem;
}}
.qs-flow-step .qs-flow-idx {{
  font-size: .64rem; font-weight: 700; color: var(--qs-primary);
  background: #E7F2F5; border-radius: 5px; padding: .1rem .35rem;
}}
.qs-arrow {{ text-align: center; color: #9FB3BC; font-size: .8rem; line-height: .5; }}
.qs-badges {{ display: flex; flex-wrap: wrap; gap: 7px; }}
.qs-badge {{
  background: var(--qs-surface); border: 1px solid var(--qs-border); border-radius: 7px;
  padding: .35rem .6rem; font-size: .755rem; font-weight: 600; color: #33505A;
}}
.qs-statuslist {{ display: flex; flex-direction: column; gap: 9px; }}
.qs-statusitem {{
  display: flex; align-items: center; justify-content: space-between;
  border: 1px solid var(--qs-border); border-radius: 9px; padding: .6rem .8rem;
  font-size: .83rem; background: var(--qs-surface);
}}
.qs-statusitem .qs-sname {{ font-weight: 600; }}
.qs-statusitem .qs-sval {{ display: flex; align-items: center; gap: .38rem; font-weight: 600; }}
.qs-empty {{
  border: 1px dashed var(--qs-border); border-radius: 12px; padding: 2.2rem 1.2rem;
  text-align: center; color: var(--qs-muted); font-size: .87rem; background: #FBFCFD;
}}

/* ---------- widgets ---------- */
[data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 700; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--qs-border); border-radius: 10px; }}
section[data-testid="stSidebar"] .stDownloadButton > button,
section[data-testid="stSidebar"] .stTextInput > div > input {{ color: var(--qs-ink); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ padding: .35rem .7rem; font-size: .86rem; }}
hr {{ border-color: var(--qs-border); }}
</style>
"""


def inject_css() -> None:
    """Apply the stylesheet."""
    st.markdown(STYLESHEET, unsafe_allow_html=True)


def sidebar_css_note() -> None:
    """Small helper so the sidebar keeps its scrollbar subtle."""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] ::-webkit-scrollbar { width: 6px; }
        section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
            background: #23485A; border-radius: 99px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )