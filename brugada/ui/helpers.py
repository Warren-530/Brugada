import html
import re

import streamlit as st


def format_recommendation_tier(tier: str) -> str:
    """Replace underscores with spaces in recommendation tier names."""
    if not tier or not isinstance(tier, str):
        return "routine clinical correlation"
    return tier.replace("_", " ")


def render_metric_with_info(column, label: str, value: str, info_markdown: str, info_key: str) -> None:
    """Render metric as a custom card with an icon-only tooltip."""
    info_lines = []
    for raw_line in info_markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"\*\*|`", "", line)
        if line.startswith("- "):
            line = f"• {line[2:].strip()}"
        info_lines.append(html.escape(line))

    info_html = "<br>".join(info_lines) if info_lines else html.escape("No additional details.")

    with column:
        st.markdown(
            f"""
            <div class="metric-tile" id="{info_key}">
                <div class="metric-tile-header">
                    <span class="metric-tile-label">{html.escape(label)}</span>
                    <details class="metric-info-details">
                        <summary>
                            <span class="metric-info-icon" title="More info">ℹ</span>
                        </summary>
                        <div class="metric-info-panel">
                            <div class="metric-info-panel-title">MORE INFO</div>
                            {info_html}
                        </div>
                    </details>
                </div>
                <div class="metric-tile-value">{html.escape(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
