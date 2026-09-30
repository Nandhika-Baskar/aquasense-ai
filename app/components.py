"""
Reusable presentation components for AquaSense AI.

Everything here returns HTML strings (rendered with unsafe_allow_html) or calls
st.write once. Pages compose these instead of repeating markup, so the visual
language stays consistent across the dashboard.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st

from app import config as cfg


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def tint(hex_color: str, alpha: float = 0.12) -> str:
    """Convert #RRGGBB to rgba(...) with the given alpha, for chip backgrounds."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def eyebrow_text(label: str) -> str:
    """
    Small-caps style label for card headers.

    Sensor names keep their own notation -- uppercasing pH to "PH" would be
    chemically wrong -- so they are passed through untouched.
    """
    return str(label) if label in cfg.FEATURES else str(label).upper()


def now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def brand_block() -> None:
    st.markdown(
        f"""
        <div class="qs-brand">
          <div class="qs-brand-name">{escape(cfg.APP_NAME)}</div>
          <div class="qs-brand-sub">{escape(cfg.APP_SUBTITLE)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def nav_heading(text: str = "NAVIGATION") -> None:
    st.markdown(f'<div class="qs-nav-heading">{escape(text)}</div>', unsafe_allow_html=True)


def sidebar_status(ai_engine: str = "Online") -> None:
    rows = [
        ("AI Engine", ai_engine, cfg.COLOR_OK),
        ("Model", "Random Forest", "#35C4D8"),
        ("Data Source", "Demonstration Sensor Data", "#8FAFBA"),
    ]
    body = "".join(
        f'<div class="qs-status-row"><span>{escape(k)}</span>'
        f'<span><span class="qs-dot" style="background:{c}"></span>{escape(v)}</span></div>'
        for k, v, c in rows
    )
    st.markdown(f'<div class="qs-status-box">{body}</div>', unsafe_allow_html=True)
    st.caption(f"v{cfg.APP_VERSION}")


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"# {escape(title)}", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div style="color:{cfg.COLOR_MUTED};font-size:.855rem;margin:-.35rem 0 .9rem 0">'
            f"{escape(subtitle)}</div>",
            unsafe_allow_html=True,
        )


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {escape(title)}", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def metric_grid(items: list[dict]) -> None:
    """
    A row of equal-height metric cards.

    items: label, value, unit (opt), description (opt), status (opt), color (opt)
    """
    cells = []
    for item in items:
        unit = f'<span class="qs-unit">{escape(item["unit"])}</span>' if item.get("unit") else ""
        desc = f'<div class="qs-desc">{escape(item["description"])}</div>' if item.get("description") else ""
        color = item.get("color", cfg.COLOR_PRIMARY)
        status = item.get("status")
        if status:
            eyebrow = (
                f'<div class="qs-eyebrow"><span class="qs-dot" style="background:{color}"></span>'
                f"{escape(eyebrow_text(item['label']))}</div>"
            )
        else:
            eyebrow = f'<div class="qs-eyebrow">{escape(eyebrow_text(item["label"]))}</div>'
        chip = ""
        if status:
            chip = (
                f'<div style="margin-top:.5rem"><span class="qs-chip" '
                f'style="background:{tint(color, .13)};color:{color}">{escape(str(status).upper())}'
                "</span></div>"
            )
        cells.append(
            f'<div class="qs-cell"><div class="qs-card">{eyebrow}'
            f'<div class="qs-value">{escape(str(item["value"]))}{unit}</div>'
            f"{desc}{chip}</div></div>"
        )

    # The cells must be concatenated, not interpolated as a list: an f-string on a
    # list renders its repr, which leaks "['", "', '" and "']" into the page as
    # visible text between the cards.
    st.markdown(
        f'<div class="qs-grid">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def stat_grid(pairs: list[tuple[str, str]]) -> None:
    """A compact key/value strip, e.g. model name / sample counts."""
    cells = "".join(
        f'<div class="qs-cell"><div class="qs-card" style="padding:11px 14px">'
        f'<div class="qs-eyebrow" style="margin-bottom:.2rem">{escape(eyebrow_text(k))}</div>'
        f'<div style="font-size:1.02rem;font-weight:650">{escape(str(v))}</div></div></div>'
        for k, v in pairs
    )
    st.markdown(f'<div class="qs-grid">{cells}</div>', unsafe_allow_html=True)


def risk_hero(
    risk: str,
    confidence: float,
    probabilities: dict[str, float],
    timestamp: str,
    heading: str = "CONTAMINATION RISK",
) -> None:
    """The large risk card: class, confidence, probability bar, timestamp."""
    color = cfg.RISK_COLORS.get(risk, cfg.COLOR_NEUTRAL)
    remaining = [c for c in cfg.RISK_ORDER if c != risk]
    breakdown = "".join(
        f'<div style="margin-top:.42rem"><div style="display:flex;justify-content:space-between;'
        f'font-size:.735rem;color:{cfg.COLOR_MUTED}"><span>{escape(c)}</span>'
        f"<span>{pct(probabilities.get(c, 0.0))}</span></div>"
        f'<div class="qs-hero-bar" style="margin-top:.2rem;height:5px"><div style="width:'
        f'{min(probabilities.get(c, 0.0) * 100, 100):.1f}%;background:{cfg.RISK_COLORS[c]}">'
        "</div></div></div>"
        for c in remaining
    )

    st.markdown(
        f"""
        <div class="qs-hero" style="border-left:4px solid {color}">
          <div class="qs-eyebrow">{escape(heading)}</div>
          <div style="display:flex;align-items:baseline;gap:.75rem;flex-wrap:wrap">
            <div class="qs-hero-risk" style="color:{color}">{escape(risk)}</div>
            <div style="font-size:.95rem;color:{cfg.COLOR_MUTED};font-weight:550">
              {pct(confidence)} confidence</div>
          </div>
          <div class="qs-hero-bar"><div style="width:{min(confidence * 100, 100):.1f}%;background:{color}"></div></div>
          {breakdown}
          <div class="qs-meta">
            <div class="qs-meta-item">
              <div class="qs-meta-k">Probability</div>
              <div class="qs-meta-v">{pct(confidence)}</div>
            </div>
            <div class="qs-meta-item">
              <div class="qs-meta-k">Classification</div>
              <div class="qs-meta-v">{len(cfg.RISK_ORDER)} classes</div>
            </div>
            <div class="qs-meta-item">
              <div class="qs-meta-k">Timestamp</div>
              <div class="qs-meta-v">{escape(timestamp)}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_list(items: list[tuple[str, str, str]]) -> None:
    """Operational status list: (component, status, colour)."""
    rows = "".join(
        f'<div class="qs-statusitem"><span class="qs-sname">{escape(name)}</span>'
        f'<span class="qs-sval" style="color:{color}">'
        f'<span class="qs-dot" style="background:{color}"></span>{escape(status)}</span></div>'
        for name, status, color in items
    )
    st.markdown(f'<div class="qs-statuslist">{rows}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------
def note(text: str, kind: str = "info") -> None:
    css = "qs-warn-note" if kind == "warn" else "qs-note"
    st.markdown(f'<div class="{css}">{escape(text)}</div>', unsafe_allow_html=True)


def explanation_list(notes: list[dict]) -> None:
    """Renders the machine-generated explanation notes."""
    palette = {
        "danger": cfg.COLOR_DANGER,
        "warn": cfg.COLOR_WARN,
        "ok": cfg.COLOR_OK,
        "info": cfg.COLOR_PRIMARY,
    }
    blocks = []
    for item in notes:
        color = palette.get(item.get("level", "info"), cfg.COLOR_PRIMARY)
        blocks.append(
            f'<div style="border-left:2px solid {color};padding:.05rem 0 .05rem .7rem;margin-bottom:.72rem">'
            f'<div style="font-size:.8rem;font-weight:650;color:{cfg.COLOR_INK}">'
            f"{escape(item['title'])}</div>"
            f'<div style="font-size:.795rem;color:{cfg.COLOR_MUTED};line-height:1.5;margin-top:.1rem">'
            f"{escape(item['detail'])}</div></div>"
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def flow_diagram(steps: list[str]) -> None:
    """Vertical pipeline diagram with numbered stages."""
    parts = []
    for i, step in enumerate(steps, start=1):
        if i > 1:
            parts.append('<div class="qs-arrow">&#9660;</div>')
        parts.append(
            f'<div class="qs-flow-step"><span class="qs-flow-idx">{i:02d}</span>'
            f"<span>{escape(step)}</span></div>"
        )
    st.markdown(f'<div class="qs-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def badges(items: list[str]) -> None:
    chips = "".join(f'<span class="qs-badge">{escape(name)}</span>' for name in items)
    st.markdown(f'<div class="qs-badges">{chips}</div>', unsafe_allow_html=True)


def empty_state(title: str, detail: str, hint: str = "") -> None:
    hint_html = f'<div style="margin-top:.5rem;font-size:.78rem;color:#94A3AC">{escape(hint)}</div>' if hint else ""
    st.markdown(
        f'<div class="qs-empty"><div style="font-weight:650;color:{cfg.COLOR_INK}">{escape(title)}</div>'
        f'<div style="margin-top:.3rem">{escape(detail)}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )