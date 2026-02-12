import concurrent.futures
import html
import json
import os
import re

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI


st.set_page_config(page_title="PM Angel vs Devil", page_icon=None, layout="wide")

CSS = """
<style>
:root {
  --bg: #f6f4f0;
  --ink: #1b1b1b;
  --muted: #6f6f6f;
  --angel-bg: linear-gradient(135deg, #f7fbff 0%, #e9f1ff 55%, #ffffff 100%);
  --angel-border: #c7d9ff;
  --devil-bg: linear-gradient(135deg, #2b0f12 0%, #3b1016 55%, #1a0b0c 100%);
  --devil-border: #7a1f2a;
  --summary-bg: linear-gradient(135deg, #10141a 0%, #15202b 60%, #0d0f14 100%);
  --summary-border: #2b3847;
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% 0%, #fff7ea 0%, rgba(246, 244, 240, 0.8) 55%, #f1f1f6 100%),
    radial-gradient(900px 450px at 90% 0%, #f0f6ff 0%, rgba(246, 244, 240, 0.7) 60%, transparent 100%);
  color: var(--ink);
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
}

.neutral-card {
  background: #ffffff;
  border: 1px solid #e6e1d8;
  border-radius: 16px;
  padding: 1.25rem 1.5rem;
  box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}

.neutral-card h2 {
  margin: 0 0 0.5rem 0;
  font-size: 1.4rem;
}

.subtle {
  color: var(--muted);
  margin-bottom: 0.75rem;
}

.hero {
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 1.5rem;
  align-items: center;
  margin-bottom: 1.5rem;
}

.hero h1 {
  margin: 0 0 0.25rem 0;
  font-size: 2.2rem;
}

.hero p {
  margin: 0.25rem 0 0 0;
  color: var(--muted);
  font-size: 1rem;
}

.hero-badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  background: #ffeccf;
  color: #8a5b17;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.04rem;
}

.mode-card {
  border-radius: 18px;
  padding: 1.25rem 1.5rem 1.75rem 1.5rem;
  min-height: 420px;
  position: relative;
  overflow: hidden;
}

.mode-card h3 {
  margin: 0 0 0.75rem 0;
  font-size: 1.3rem;
}

.mode-card .response {
  white-space: normal;
  line-height: 1.5;
  font-size: 1rem;
}

.angel-card {
  background: var(--angel-bg);
  border: 1px solid var(--angel-border);
  box-shadow: 0 12px 24px rgba(80, 120, 200, 0.15);
}

.angel-card::after {
  content: "ANGEL";
  position: absolute;
  right: 18px;
  top: 14px;
  font-size: 3.25rem;
  letter-spacing: 0.2rem;
  color: rgba(120, 150, 220, 0.12);
  font-weight: 700;
}

.devil-card {
  background: var(--devil-bg);
  border: 1px solid var(--devil-border);
  color: #f7d8dc;
  box-shadow: 0 12px 24px rgba(90, 10, 20, 0.35);
}

.devil-card::after {
  content: "DEVIL";
  position: absolute;
  right: 18px;
  top: 14px;
  font-size: 3.25rem;
  letter-spacing: 0.25rem;
  color: rgba(255, 120, 130, 0.12);
  font-weight: 700;
}

.devil-card h3 {
  color: #ffe6e8;
}

.response-empty {
  color: var(--muted);
  font-style: italic;
}

.response-block h4 {
  margin: 0.75rem 0 0.25rem 0;
  font-size: 1.05rem;
}

.response-block ul {
  padding-left: 1.1rem;
  margin: 0.25rem 0 0.75rem 0;
}

.response-block li {
  margin: 0.25rem 0;
}

.angel-illustration,
.devil-illustration {
  position: absolute;
  left: -10px;
  bottom: -10px;
  width: 190px;
  opacity: 0.9;
}

.devil-illustration {
  opacity: 0.85;
}

.scene-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.angel-clouds {
  position: absolute;
  right: -20px;
  bottom: 10px;
  width: 240px;
  opacity: 0.5;
}

.devil-flames {
  position: absolute;
  right: -10px;
  bottom: -20px;
  width: 260px;
  opacity: 0.45;
}

.summary-card {
  margin-top: 1.75rem;
  border-radius: 20px;
  padding: 1.5rem 1.75rem;
  background: var(--summary-bg);
  border: 1px solid var(--summary-border);
  color: #e6f0ff;
  box-shadow: 0 18px 32px rgba(10, 20, 30, 0.45);
}

.summary-card h3 {
  margin: 0 0 0.75rem 0;
  font-size: 1.4rem;
}

.summary-score {
  display: inline-flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.6rem 1rem;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.08);
  margin-bottom: 0.75rem;
}

.score-pill {
  font-size: 1.2rem;
  font-weight: 700;
}

.score-bar {
  height: 10px;
  background: linear-gradient(90deg, #c23b3b 0%, #d9a441 50%, #6bc27c 100%);
  border-radius: 999px;
  overflow: hidden;
  margin: 0.5rem 0 0.75rem 0;
}

.score-marker {
  height: 100%;
  width: 0%;
  background: #ffffff;
}

.footer-note {
  color: var(--muted);
  font-size: 0.9rem;
  margin-top: 0.5rem;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div>
        <span class="hero-badge">PLAYFUL DEMO</span>
        <h1>Angel vs Devil for PM Ideas</h1>
        <p>Two parallel takes on your product idea: one that lifts it up, and one that tears it down.</p>
      </div>
      <div class="neutral-card">
        <strong>How it works</strong>
        <div class="subtle">We run your idea through two prompts in parallel and format the output for quick scanning.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.warning("Missing OPENAI_API_KEY. Set it in your environment before running this app.")

client = OpenAI(api_key=api_key) if api_key else None

st.markdown(
    """
    <div class="neutral-card">
      <h2>Drop your idea</h2>
      <div class="subtle">We will spin it in two opposite directions. Angel mode will hype it up; Devil mode will roast it.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

idea = st.text_area(
    "Idea",
    height=140,
    placeholder="Example: A weekly AI-generated roadmap update that PMs can send to execs with one click.",
    label_visibility="collapsed",
)

model = st.sidebar.text_input("Model", value="gpt-5.1")
max_bullets = st.sidebar.slider("Bullets per side", min_value=3, max_value=7, value=5)
summary_detail = st.sidebar.select_slider(
    "Summary detail",
    options=["Tight", "Balanced", "More detail"],
    value="Balanced",
)

run = st.button("Summon both sides", type="primary", disabled=not idea.strip() or not api_key)


def build_prompt(mode: str, idea_text: str) -> str:
    bullet_count = max_bullets
    if mode == "angel":
        return (
            "You are Angel Mode: be wildly enthusiastic, over-the-top positive, and playfully gushing. "
            "Celebrate the idea as if it's brilliant and inevitable. Highlight unique strengths, why it is timely, "
            "and what makes it valuable. Keep it upbeat and a little silly, but avoid outright falsehoods.\n"
            "Return JSON only with keys: title, bullets, summary. Do not wrap in code fences.\n"
            f"- title: a short encouraging title\n"
            f"- bullets: exactly {bullet_count} concise bullets (max 12 words each)\n"
            "- summary: 2-3 short sentences with next-step momentum\n\n"
            f"Idea: {idea_text}"
        )
    return (
        "You are Devil Mode: be mercilessly critical, intensely skeptical, and borderline dramatic. "
        "Tear into weak assumptions, missing details, risks, feasibility, and market reality. "
        "No insults, slurs, or attacks on the person.\n"
        "Return JSON only with keys: title, bullets, summary. Do not wrap in code fences.\n"
        f"- title: a short critical title\n"
        f"- bullets: exactly {bullet_count} concise bullets (max 12 words each)\n"
        "- summary: 2-3 short sentences that sum up the core flaws\n\n"
        f"Idea: {idea_text}"
    )


def call_llm(mode: str, idea_text: str) -> str:
    prompt = build_prompt(mode, idea_text)
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()
    if getattr(response, "output", None):
        try:
            return response.output[0].content[0].text.strip()
        except (AttributeError, IndexError):
            pass
    return ""

def parse_response(raw: str) -> dict:
    if not raw:
        return {}
    try:
        cleaned = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1])
            else:
                cleaned = cleaned.strip("`")
        # Try to extract the first JSON object if there's extra text.
        if "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            cleaned = cleaned[start : end + 1]
        data = json.loads(cleaned)
        if isinstance(data, dict) and "bullets" in data:
            return data
    except json.JSONDecodeError:
        pass
    return {"title": "", "bullets": [], "summary": raw}


def render_response(
    title: str, body: str, css_class: str, illustration: str, scene: str, side: str
) -> None:
    data = parse_response(body)
    if not body:
        body_html = '<div class="response-empty">No response yet.</div>'
    else:
        bullet_items = ""
        if data.get("bullets"):
            items = [html.escape(b).replace("`", "&#96;") for b in data.get("bullets", [])]
            bullet_items = "<ul>" + "".join([f"<li>{b}</li>" for b in items]) + "</ul>"

        summary = html.escape(data.get("summary", "")).replace("`", "&#96;").replace("\n", "<br>")
        title_text = html.escape(data.get("title", "")).replace("`", "&#96;").strip()
        title_block = f"<h4>{title_text}</h4>" if title_text else ""

        body_html = (
            "<div class='response-block'>"
            f"{title_block}"
            f"{bullet_items}"
            f"<div class='response'>{summary}</div>"
            "</div>"
        )
        body_html = body_html.replace("```", "").replace("`", "&#96;")

    card_css = """
    <style>
      .card-inner { display: grid; height: 100%; }
      .card-inner.side-left { grid-template-columns: 78px 1fr; }
      .card-inner.side-right { grid-template-columns: 1fr 78px; }
      .side-strip {
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        align-items: center;
        padding: 10px 0;
      }
      .side-char { width: 64px; opacity: 0.9; }
      .content-col {
        display: flex;
        flex-direction: column;
        height: 100%;
        min-height: 0;
      }
      .mode-card {
        border-radius: 18px;
        padding: 1.4rem 1.7rem 2.3rem 1.7rem;
        height: 420px;
        position: relative;
        overflow: hidden;
        font-family: "Avenir", "Trebuchet MS", "Segoe UI", sans-serif;
      }
      .mode-card h3 { margin: 0 0 0.75rem 0; font-size: 1.3rem; }
      .mode-card .response { white-space: normal; line-height: 1.5; font-size: 1rem; }
      .response-empty { color: #6f6f6f; font-style: italic; }
      .response-block h4 { margin: 0.75rem 0 0.25rem 0; font-size: 1.05rem; }
      .response-block ul { padding-left: 1.1rem; margin: 0.25rem 0 0.75rem 0; }
      .response-block li { margin: 0.25rem 0; }
      .scroll-area { overflow-y: auto; padding-right: 6px; padding-bottom: 0.6rem; flex: 1; min-height: 0; }

      .angel-card {
        background: linear-gradient(135deg, #f7fbff 0%, #e9f1ff 55%, #ffffff 100%);
        border: 1px solid #c7d9ff;
        box-shadow: 0 12px 24px rgba(80, 120, 200, 0.15);
        color: #1b1b1b;
      }
      .angel-card::after {
        content: "ANGEL";
        position: absolute;
        right: 18px;
        top: 14px;
        font-size: 3.1rem;
        letter-spacing: 0.2rem;
        color: rgba(120, 150, 220, 0.12);
        font-weight: 700;
      }

      .devil-card {
        background: linear-gradient(135deg, #2b0f12 0%, #3b1016 55%, #1a0b0c 100%);
        border: 1px solid #7a1f2a;
        color: #f7d8dc;
        box-shadow: 0 12px 24px rgba(90, 10, 20, 0.35);
      }
      .devil-card::after {
        content: "DEVIL";
        position: absolute;
        right: 18px;
        top: 14px;
        font-size: 3.1rem;
        letter-spacing: 0.25rem;
        color: rgba(255, 120, 130, 0.12);
        font-weight: 700;
      }
      .devil-card h3 { color: #ffe6e8; }
      .mode-card h3,
      .response-block,
      .response-empty {
        position: relative;
        z-index: 3;
      }

      .scene-layer { position: absolute; inset: 0; pointer-events: none; }
      .angel-clouds { position: absolute; right: -20px; bottom: 10px; width: 240px; opacity: 0.5; }
      .devil-flames { position: absolute; right: -10px; bottom: -20px; width: 260px; opacity: 0.45; }
    </style>
    """
    side_class = "side-left" if side == "left" else "side-right"
    body_html = f"<div class='scroll-area'>{body_html}</div>"
    card_html = f"""
    {card_css}
    <div class="mode-card {css_class}">
      <div class="card-inner {side_class}">
        {f'<div class="side-strip">{illustration}</div>' if side == "left" else ''}
        <div class="content-col">
          <h3>{title}</h3>
          {body_html}
        </div>
        {f'<div class="side-strip">{illustration}</div>' if side == "right" else ''}
      </div>
      <div class="scene-layer">{scene}</div>
    </div>
    """
    components.html(card_html, height=560, scrolling=False)

def build_summary_prompt(idea_text: str, angel: dict, devil: dict) -> str:
    detail = summary_detail
    summary_len = "2-3 sentences" if detail == "Tight" else "3-4 sentences" if detail == "Balanced" else "4-5 sentences"
    return (
        "You are the Referee. Combine Angel and Devil into a final assessment. "
        "Return JSON only with keys: title, bullets, recommendation, score. Do not wrap in code fences.\n"
        "- title: short verdict\n"
        "- bullets: 3 concise bullets (max 12 words each)\n"
        f"- recommendation: {summary_len} with a clear next step\n"
        "- score: integer from -100 to 100 (negative = bad, positive = good)\n\n"
        f"Idea: {idea_text}\n"
        f"Angel: {json.dumps(angel)}\n"
        f"Devil: {json.dumps(devil)}"
    )


def call_summary(idea_text: str, angel: dict, devil: dict) -> dict:
    prompt = build_summary_prompt(idea_text, angel, devil)
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    output_text = getattr(response, "output_text", None)
    if output_text:
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            return {"title": "", "bullets": [], "recommendation": output_text, "score": 0}
    return {"title": "", "bullets": [], "recommendation": "", "score": 0}


angel_text = ""
devil_text = ""
summary_data = {}

if run and client:
    with st.spinner("Calling the angels and devils..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            angel_future = executor.submit(call_llm, "angel", idea)
            devil_future = executor.submit(call_llm, "devil", idea)
            angel_text = angel_future.result()
            devil_text = devil_future.result()
        angel_data = parse_response(angel_text)
        devil_data = parse_response(devil_text)
        summary_data = call_summary(idea, angel_data, devil_data)

left, right = st.columns(2, gap="large")

with left:
    angel_svg = """
    <svg class="side-char" viewBox="0 0 220 220" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="110" cy="110" r="80" fill="#f7e2c5"/>
      <ellipse cx="110" cy="60" rx="60" ry="16" stroke="#f0c17b" stroke-width="6" fill="none"/>
      <circle cx="90" cy="105" r="8" fill="#6b5a4a"/>
      <circle cx="130" cy="105" r="8" fill="#6b5a4a"/>
      <path d="M90 130 Q110 145 130 130" stroke="#6b5a4a" stroke-width="6" fill="none"/>
      <path d="M40 115 C10 95, 25 60, 65 70" fill="#e9f1ff" stroke="#c7d9ff" stroke-width="5"/>
      <path d="M180 115 C210 95, 195 60, 155 70" fill="#e9f1ff" stroke="#c7d9ff" stroke-width="5"/>
    </svg>
    <svg class="side-char" viewBox="0 0 220 220" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="110" cy="110" r="80" fill="#fbe9d3"/>
      <path d="M55 135 C60 170, 90 190, 110 190 C130 190, 160 170, 165 135" fill="#ffffff" stroke="#c7d9ff" stroke-width="5"/>
      <circle cx="90" cy="105" r="7" fill="#6b5a4a"/>
      <circle cx="130" cy="105" r="7" fill="#6b5a4a"/>
      <path d="M92 125 Q110 135 128 125" stroke="#6b5a4a" stroke-width="5" fill="none"/>
    </svg>
    <svg class="side-char" viewBox="0 0 220 220" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="110" cy="120" r="70" fill="#f5d7b1"/>
      <path d="M70 150 Q110 175 150 150" stroke="#6b5a4a" stroke-width="6" fill="none"/>
      <circle cx="92" cy="115" r="7" fill="#6b5a4a"/>
      <circle cx="128" cy="115" r="7" fill="#6b5a4a"/>
    </svg>
    """
    angel_scene = """
    <svg class="angel-clouds" viewBox="0 0 260 120" fill="none" xmlns="http://www.w3.org/2000/svg">
      <ellipse cx="60" cy="70" rx="50" ry="24" fill="#ffffff"/>
      <ellipse cx="110" cy="60" rx="60" ry="30" fill="#f4f9ff"/>
      <ellipse cx="170" cy="70" rx="45" ry="22" fill="#ffffff"/>
      <ellipse cx="210" cy="55" rx="35" ry="18" fill="#f4f9ff"/>
    </svg>
    """
    render_response("Angel Mode", angel_text, "angel-card", angel_svg, angel_scene, "left")

with right:
    devil_svg = """
    <svg class="side-char" viewBox="0 0 220 220" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="110" cy="110" r="80" fill="#f2b0b6"/>
      <path d="M75 70 L55 40" stroke="#f06c7f" stroke-width="8" stroke-linecap="round"/>
      <path d="M145 70 L165 40" stroke="#f06c7f" stroke-width="8" stroke-linecap="round"/>
      <circle cx="92" cy="105" r="7" fill="#3b1016"/>
      <circle cx="128" cy="105" r="7" fill="#3b1016"/>
      <path d="M90 132 Q110 120 130 132" stroke="#3b1016" stroke-width="6" fill="none"/>
    </svg>
    <svg class="side-char" viewBox="0 0 220 220" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="110" cy="110" r="80" fill="#f6a8b0"/>
      <path d="M70 135 C75 175, 105 195, 110 195 C120 195, 150 175, 150 135" fill="#3b1016" stroke="#7a1f2a" stroke-width="5"/>
      <circle cx="95" cy="110" r="7" fill="#3b1016"/>
      <circle cx="125" cy="110" r="7" fill="#3b1016"/>
      <path d="M90 130 Q110 120 130 130" stroke="#3b1016" stroke-width="6" fill="none"/>
    </svg>
    <svg class="side-char" viewBox="0 0 220 220" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="110" cy="120" r="70" fill="#f1a1aa"/>
      <path d="M95 110 L80 90" stroke="#7a1f2a" stroke-width="6" stroke-linecap="round"/>
      <path d="M125 110 L140 90" stroke="#7a1f2a" stroke-width="6" stroke-linecap="round"/>
      <circle cx="92" cy="122" r="6" fill="#3b1016"/>
      <circle cx="128" cy="122" r="6" fill="#3b1016"/>
      <path d="M92 140 Q110 130 128 140" stroke="#3b1016" stroke-width="6" fill="none"/>
    </svg>
    """
    devil_scene = """
    <svg class="devil-flames" viewBox="0 0 260 140" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M20 120 Q40 80 30 50 Q60 70 70 30 Q90 60 100 20 Q120 70 140 30 Q150 80 170 40 Q190 90 210 60 Q230 100 240 80 Q250 110 240 130 Z" fill="#7a1f2a" opacity="0.6"/>
      <path d="M40 120 Q60 90 55 70 Q75 85 85 55 Q100 90 120 55 Q130 90 150 65 Q170 95 190 70 Q210 105 220 90 Q230 115 220 130 Z" fill="#c23b3b" opacity="0.45"/>
    </svg>
    """
    render_response("Devil Mode", devil_text, "devil-card", devil_svg, devil_scene, "right")

st.markdown(
    "<div class='footer-note'>Tip: Keep it light. This is a playful demo, not a performance review.</div>",
    unsafe_allow_html=True,
)

if summary_data:
    score = int(summary_data.get("score", 0))
    score = max(-100, min(100, score))
    marker_pct = (score + 100) / 2
    bullets = summary_data.get("bullets", [])
    bullets_html = ""
    if bullets:
        bullets_html = "<ul>" + "".join([f"<li>{html.escape(b)}</li>" for b in bullets]) + "</ul>"
    recommendation_html = html.escape(summary_data.get("recommendation", "")).replace("\n", "<br>")

    summary_css = """
    <style>
      .summary-card {
        border-radius: 20px;
        padding: 1.75rem 2rem 2.2rem 2rem;
        background: linear-gradient(135deg, #10141a 0%, #15202b 60%, #0d0f14 100%);
        border: 1px solid #2b3847;
        color: #e6f0ff;
        box-shadow: 0 18px 32px rgba(10, 20, 30, 0.45);
        font-family: "Avenir", "Trebuchet MS", "Segoe UI", sans-serif;
        position: relative;
        height: 320px;
        display: flex;
        flex-direction: column;
      }
      .summary-icon {
        position: absolute;
        top: 16px;
        right: 16px;
        width: 64px;
        opacity: 0.8;
      }
      .summary-scroll { overflow-y: auto; padding-right: 6px; min-height: 0; }
      .summary-card h3 { margin: 0 0 0.75rem 0; font-size: 1.4rem; }
      .summary-score {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.6rem 1rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.08);
        margin-bottom: 0.75rem;
      }
      .score-pill { font-size: 1.2rem; font-weight: 700; }
      .score-bar {
        height: 10px;
        background: linear-gradient(90deg, #c23b3b 0%, #d9a441 50%, #6bc27c 100%);
        border-radius: 999px;
        overflow: hidden;
        margin: 0.5rem 0 0.75rem 0;
      }
      .score-marker { height: 100%; width: 0%; background: #ffffff; }
      .summary-card ul { padding-left: 1.1rem; margin: 0.25rem 0 0.75rem 0; }
      .summary-card li { margin: 0.25rem 0; }
    </style>
    """
    justice_svg = """
    <svg class="summary-icon" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M100 25 L100 155" stroke="#e6f0ff" stroke-width="6" stroke-linecap="round"/>
      <path d="M60 55 L140 55" stroke="#e6f0ff" stroke-width="6" stroke-linecap="round"/>
      <path d="M50 55 L25 95" stroke="#e6f0ff" stroke-width="4" stroke-linecap="round"/>
      <path d="M150 55 L175 95" stroke="#e6f0ff" stroke-width="4" stroke-linecap="round"/>
      <path d="M10 95 C25 125, 60 125, 75 95" stroke="#e6f0ff" stroke-width="4" fill="none"/>
      <path d="M125 95 C140 125, 175 125, 190 95" stroke="#e6f0ff" stroke-width="4" fill="none"/>
      <circle cx="100" cy="165" r="12" fill="#e6f0ff" opacity="0.6"/>
    </svg>
    """
    summary_html = f"""
    {summary_css}
    <div class="summary-card">
      {justice_svg}
      <h3>Final Recommendation</h3>
      <div class="summary-score">
        <div class="score-pill">{score:+d}%</div>
        <div>Composite from Angel + Devil</div>
      </div>
      <div class="score-bar">
        <div class="score-marker" style="width: {marker_pct}%;"></div>
      </div>
      <div class="summary-scroll">
        <strong>{html.escape(summary_data.get("title", ""))}</strong>
        {bullets_html}
        <div class="response">{recommendation_html}</div>
      </div>
    </div>
    """
    components.html(summary_html, height=360, scrolling=False)
