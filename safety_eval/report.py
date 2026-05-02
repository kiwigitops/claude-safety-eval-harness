"""Static HTML report generation."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from safety_eval.runner import summarize_results


def generate_html_report(rows: list[dict[str, Any]], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(rows)
    risky_examples = sorted(
        rows,
        key=lambda row: int(row.get("overall_risk_score", 0) or 0),
        reverse=True,
    )[:5]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Safety Evaluation Report</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --ink: #172026;
      --muted: #5e6a71;
      --line: #d8ddd8;
      --panel: #ffffff;
      --accent: #0f766e;
      --risk: #b42318;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 40px 20px 56px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      margin-bottom: 24px;
      padding-bottom: 18px;
    }}
    h1, h2 {{
      line-height: 1.15;
      margin: 0;
    }}
    h1 {{
      font-size: 32px;
    }}
    h2 {{
      font-size: 20px;
      margin-top: 32px;
      margin-bottom: 12px;
    }}
    p {{
      color: var(--muted);
      margin: 8px 0 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .metric, .example {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .value {{
      display: block;
      font-size: 26px;
      font-weight: 700;
      margin-top: 4px;
    }}
    .risk {{
      color: var(--risk);
    }}
    .examples {{
      display: grid;
      gap: 12px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 8px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f1f3ef;
      border-radius: 6px;
      padding: 12px;
      overflow: auto;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Safety Evaluation Report</h1>
      <p>Static summary of logged prompt-response evaluations.</p>
    </header>
    <section class="metrics">
      {metric("Total prompts", summary["total"])}
      {metric("Refusal rate", f'{summary["refusal_rate"]:.1%}')}
      {metric("Risky compliance", summary["harmful_compliance_count"], risk=True)}
      {metric("Jailbreak indicators", summary["jailbreak_count"], risk=True)}
      {metric("Hallucination-risk language", summary["hallucination_risk_count"], risk=True)}
      {metric("Avg. risk score", f'{summary["average_risk_score"]:.2f}')}
    </section>
    <h2>Highest-Risk Examples</h2>
    <section class="examples">
      {''.join(example(row) for row in risky_examples) if risky_examples else '<p>No logged runs yet.</p>'}
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def metric(label: str, value: object, risk: bool = False) -> str:
    value_class = "value risk" if risk else "value"
    return (
        '<div class="metric">'
        f'<span class="label">{escape(label)}</span>'
        f'<span class="{value_class}">{escape(str(value))}</span>'
        "</div>"
    )


def example(row: dict[str, Any]) -> str:
    score = int(row.get("overall_risk_score", 0) or 0)
    prompt = escape(str(row.get("prompt", "")))
    response = escape(str(row.get("response", "")))
    meta = escape(
        f'Run {row.get("run_id", "")} | Prompt {row.get("prompt_id", "")} | '
        f'Model {row.get("model_name", "")} | Risk score {score}'
    )
    notes = escape(str(row.get("notes", "")))
    return f"""
      <article class="example">
        <div class="meta">{meta}</div>
        <strong>Prompt</strong>
        <pre>{prompt}</pre>
        <strong>Response</strong>
        <pre>{response}</pre>
        <p>{notes}</p>
      </article>
    """

