"""Gradio theme and CSS.

Persian is right-to-left, so the Persian-facing inputs get explicit
``direction: rtl`` and a font stack that actually has Persian glyphs. Getting
this wrong is the most visible way a Persian tool looks unfinished.
"""

from __future__ import annotations

from typing import Any

#: Font stack: Persian faces first, then system UI fallbacks.
PERSIAN_FONTS = (
    '"Vazirmatn", "Vazir", "IRANSans", "Sahel", "Shabnam", '
    '"Noto Naskh Arabic", "Tahoma", -apple-system, "Segoe UI", sans-serif'
)

CSS = f"""
/* ---------- layout ---------- */
.gradio-container {{
    max-width: 1280px !important;
    margin: 0 auto !important;
}}

#app-header {{
    text-align: center;
    padding: 1.1rem 1rem 0.5rem;
}}
#app-header h1 {{
    margin: 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}}
#app-header p {{
    margin: 0.35rem 0 0;
    opacity: 0.72;
    font-size: 0.95rem;
}}

/* ---------- Persian text: right-to-left ---------- */
.rtl textarea, .rtl input[type="text"] {{
    direction: rtl !important;
    text-align: right !important;
    font-family: {PERSIAN_FONTS} !important;
    font-size: 1.02rem !important;
    line-height: 2 !important;
}}
.rtl label {{
    direction: rtl;
    text-align: right;
}}

/* ---------- reports ---------- */
.report textarea {{
    font-family: "SF Mono", "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.55 !important;
    direction: ltr !important;
    text-align: left !important;
    white-space: pre !important;
}}

/* ---------- emphasis ---------- */
.primary-action button {{
    font-weight: 600 !important;
    min-height: 46px !important;
}}

.hint {{
    font-size: 0.83rem;
    opacity: 0.68;
    margin: -0.35rem 0 0.5rem;
}}
.hint.rtl {{
    direction: rtl;
    text-align: right;
}}

/* ---------- quality badges ---------- */
.badge {{
    display: inline-block;
    padding: 0.12rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-inline-end: 0.35rem;
}}
.badge-good   {{ background: rgba(34,197,94,0.16);  color: #15803d; }}
.badge-warn   {{ background: rgba(234,179,8,0.18);  color: #a16207; }}
.badge-bad    {{ background: rgba(239,68,68,0.16);  color: #b91c1c; }}
.badge-neutral{{ background: rgba(100,116,139,0.16);color: #475569; }}

@media (prefers-color-scheme: dark) {{
    .badge-good    {{ color: #4ade80; }}
    .badge-warn    {{ color: #facc15; }}
    .badge-bad     {{ color: #f87171; }}
    .badge-neutral {{ color: #cbd5e1; }}
}}

/* ---------- diagnostics table ---------- */
.diagnostics table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}}
.diagnostics th, .diagnostics td {{
    text-align: left;
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid rgba(128,128,128,0.18);
}}
.diagnostics th {{ font-weight: 600; opacity: 0.8; }}
"""


def build_theme(name: str = "soft") -> Any:
    """Return a Gradio theme, tolerating older/newer Gradio versions."""
    import gradio as gr

    themes = {
        "soft": getattr(gr.themes, "Soft", None),
        "default": getattr(gr.themes, "Default", None),
        "glass": getattr(gr.themes, "Glass", None),
        "monochrome": getattr(gr.themes, "Monochrome", None),
        "base": getattr(gr.themes, "Base", None),
    }
    factory = themes.get(name.lower()) or themes.get("soft") or themes.get("default")
    if factory is None:  # pragma: no cover - very old Gradio
        return None

    try:
        return factory(
            primary_hue="teal",
            secondary_hue="blue",
            neutral_hue="slate",
        )
    except Exception:
        return factory()


def quality_badge(score: float, verdict: str) -> str:
    """HTML badge summarising a 0-100 quality score."""
    if score >= 70:
        css_class = "badge-good"
    elif score >= 45:
        css_class = "badge-warn"
    else:
        css_class = "badge-bad"
    return f'<span class="badge {css_class}">{verdict.upper()} &middot; {score:.0f}/100</span>'


def similarity_badge(score: float | None) -> str:
    """HTML badge for a speaker-similarity score."""
    if score is None:
        return '<span class="badge badge-neutral">SIMILARITY N/A</span>'
    if score >= 0.75:
        css_class = "badge-good"
    elif score >= 0.60:
        css_class = "badge-warn"
    else:
        css_class = "badge-bad"
    return f'<span class="badge {css_class}">SIMILARITY {score:.2f}</span>'


__all__ = ["CSS", "PERSIAN_FONTS", "build_theme", "quality_badge", "similarity_badge"]
