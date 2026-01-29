# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from datetime import date, timedelta

# %%
def get_env_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Environment variable {name} is not set.")
        sys.exit(1)
    return value

# %%
import os
import sys
from databricks import sql
host = get_env_var("DATABRICKS_HOST")
http_path = get_env_var("DATABRICKS_HTTP_PATH")
token = get_env_var("DATABRICKS_TOKEN")

# Connect and run a simple query
try:
    with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * from hackathon.amer.academic_records limit 10")
            rows = cur.fetchall()
            print("Query result:")
            for r in rows:
                print(r)
except Exception as e:
    print("Failed to connect or run query:", e)
    sys.exit(2)

# %%
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # set the model you want in env
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def get_env_var(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Environment variable {name} is not set")
    return v

def _safe_id_literal(student_id):
    sid = str(student_id).strip()
    if sid.isdigit():
        return sid
    return "'" + sid.replace("'", "''") + "'"

def _date_range_months(months: int):
    end = date.today()
    # approximate months by days
    start = end - timedelta(days=30 * months)
    return start.isoformat(), end.isoformat()

def query_table(cur, sql_text):
    cur.execute(sql_text)
    cols = [d[0] for d in cur.description] if getattr(cur, "description", None) else []
    rows = cur.fetchall()
    list_of_dicts = []
    for r in rows:
        rowd = {}
        for i, c in enumerate(cols):
            rowd[c] = r[i] if i < len(r) else None
        list_of_dicts.append(rowd)
    return list_of_dicts

def generate_student_summary(student_id: str, months: int = 6):
    """
    Query Databricks and return a structured summary dict (attendance, skills, mentoring).
    """
    host = get_env_var("DATABRICKS_HOST")
    http_path = get_env_var("DATABRICKS_HTTP_PATH")
    token = get_env_var("DATABRICKS_TOKEN")

    start_date, end_date = _date_range_months(months)
    sid = _safe_id_literal(student_id)

    # Prefer student_skills for per-student progress; fallback to skills if needed.
    attendance_q = f"""
    SELECT attendance_date, attendance_status, participation_level, absence_reason, notes
    FROM hackathon.amer.attendance
    WHERE student_id = {sid}
      AND attendance_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
    ORDER BY attendance_date DESC
    """

    student_skills_q = f"""
    SELECT student_skill_id, skill_id, initial_proficiency_level, current_proficiency_level, last_assessment_date, progress_notes
    FROM hackathon.amer.student_skills
    WHERE student_id = {sid}
      AND (last_assessment_date BETWEEN DATE('{start_date}') AND DATE('{end_date}') OR last_assessment_date IS NULL)
    ORDER BY last_assessment_date DESC
    """

    skills_fallback_q = f"""
    SELECT skill_id, skill_name, proficiency_levels
    FROM hackathon.amer.skills
    WHERE skill_id IN (
      SELECT DISTINCT skill_id FROM hackathon.amer.student_skills WHERE student_id = {sid}
    )
    """

    mentoring_q = f"""
    SELECT mentoring_session_id, session_date, mentor_id, topics_discussed, mentor_notes, session_rating_by_student
    FROM hackathon.amer.mentoring_sessions
    WHERE student_id = {sid}
      AND (session_date BETWEEN DATE('{start_date}') AND DATE('{end_date}'))
    ORDER BY session_date DESC
    LIMIT 50
    """

    report = {
        "student_id": student_id,
        "date_range": {"start": start_date, "end": end_date},
        "attendance": {},
        "skills": [],
        "mentoring": []
    }

    try:
        with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
            with conn.cursor() as cur:
                attendance = query_table(cur, attendance_q)
                total = len(attendance)
                present = sum(1 for r in attendance if str(r.get("attendance_status", "")).lower() in ("present", "yes", "1", "true"))
                attendance_rate = round(100.0 * present / total, 2) if total > 0 else None
                report["attendance"] = {
                    "total_records": total,
                    "present_count": present,
                    "attendance_rate_percent": attendance_rate,
                    "recent": attendance[:30]
                }

                # Try student_skills first
                skills = query_table(cur, student_skills_q)
                if not skills:
                    # fallback: maybe only skills metadata available
                    skills = query_table(cur, skills_fallback_q)
                report["skills"] = skills

                mentoring = query_table(cur, mentoring_q)
                report["mentoring"] = mentoring

    except Exception as e:
        raise RuntimeError(f"Error querying Databricks: {e}")

    return report


# %%
def summarize_for_llm(structured_summary: dict) -> str:
    """
    Convert the structured summary into a concise text prompt for the LLM.
    Keep it short: 1) Key metrics 2) Top 5 recent attendance records 3) Top skills (first/current) 4) Top mentoring notes.
    """
    s = structured_summary
    parts = []
    parts.append(f"Student ID: {s.get('student_id')}")
    dr = s.get("date_range", {})
    parts.append(f"Date range: {dr.get('start')} to {dr.get('end')}")
    a = s.get("attendance", {})
    parts.append(f"Attendance: {a.get('total_records',0)} records; Present {a.get('present_count')}; Rate {a.get('attendance_rate_percent')}%")
    # sample attendance
    if a.get("recent"):
        parts.append("Recent attendance (most recent first):")
        for r in a["recent"][:5]:
            parts.append(f"- {r.get('attendance_date')} | {r.get('attendance_status')} | {r.get('participation_level')}")

    parts.append("Skills progress (sample):")
    for sk in s.get("skills", [])[:8]:
        # show key skill fields if present
        if "current_proficiency_level" in sk:
            parts.append(f"- {sk.get('skill_id') or sk.get('skill_name')}: {sk.get('initial_proficiency_level')} -> {sk.get('current_proficiency_level')} (last: {sk.get('last_assessment_date')})")
        else:
            parts.append(f"- {sk.get('skill_id') or sk.get('skill_name')}: metadata or fallback record")

    parts.append("Recent mentoring sessions (most recent first):")
    for m in s.get("mentoring", [])[:5]:
        parts.append(f"- {m.get('session_date')}: mentor {m.get('mentor_id')}, topics: {m.get('topics_discussed')}, notes: { (m.get('mentor_notes') or '')[:200] }")

    return "\n".join(parts)

# %%
def call_databricks_model(prompt: str | None = None,
                          messages: list | None = None,
                          endpoint_url: str | None = None,
                          token: str | None = None,
                          temperature: float = 0.2,
                          max_tokens: int = 700,
                          timeout: int = 60) -> str:
    """
    Call a Databricks Serving Endpoint and return the model text output.

    - Prefer passing `messages` (OpenAI chat format) when using chat-style endpoints.
    - If `messages` is None, falls back to trying several single-prompt payload shapes.
    """
    if endpoint_url is None:
        endpoint_url = os.getenv("DATABRICKS_MODEL_ENDPOINT")
    if not endpoint_url:
        raise RuntimeError("Databricks model endpoint URL is not set. Set DATABRICKS_MODEL_ENDPOINT.")

    if token is None:
        token = os.getenv("DATABRICKS_TOKEN")
    if not token:
        raise RuntimeError("Databricks token not set. Set DATABRICKS_TOKEN.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # If chat messages provided, send chat payload first (common requirement)
    if messages is not None:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
        if not resp.ok:
            raise RuntimeError(f"Databricks model call failed; status={resp.status_code}; body={resp.text}")
        # parse response
        try:
            j = resp.json()
        except Exception:
            return resp.text
        # Common chat response shapes
        if isinstance(j, dict):
            # OpenAI-like: {"choices":[{"message":{"content":"..."}}]}
            choices = j.get("choices")
            if choices and isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                # look for nested message.content
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("delta")
                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"]
                    # sometimes text is directly available
                    for k in ("text", "content", "generated_text"):
                        if k in first:
                            return first[k]
                # if choice is string
                if isinstance(first, str):
                    return first
            # Databricks may return {"outputs":[{"content": "..."}]} or {"outputs": ["..."]}
            outputs = j.get("outputs") or j.get("output") or j.get("data")
            if outputs:
                if isinstance(outputs, list) and len(outputs) > 0:
                    if isinstance(outputs[0], dict) and "content" in outputs[0]:
                        return outputs[0]["content"]
                    if isinstance(outputs[0], str):
                        return outputs[0]
            # Some endpoints return 'result' or 'response'
            for k in ("result", "response", "text", "generated_text"):
                if k in j and isinstance(j[k], str):
                    return j[k]
        if isinstance(j, list) and j:
            if isinstance(j[0], dict):
                for k in ("content", "text", "generated_text"):
                    if k in j[0]:
                        return j[0][k]
            if isinstance(j[0], str):
                return j[0]
        return json.dumps(j, ensure_ascii=False)

    # If no messages provided, try candidate single-prompt payloads (existing fallback)
    candidate_payloads = [
        {"input": prompt, "temperature": temperature, "max_tokens": max_tokens},
        {"inputs": prompt, "temperature": temperature, "max_tokens": max_tokens},
        {"instances": [prompt], "temperature": temperature},
        {"prompt": prompt, "temperature": temperature, "max_new_tokens": max_tokens},
        {"data": [prompt]},
    ]

    last_resp = None
    for payload in candidate_payloads:
        try:
            resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
        except Exception as e:
            last_resp = ("exception", str(e))
            continue

        last_resp = resp
        if not resp.ok:
            continue

        try:
            j = resp.json()
        except Exception:
            return resp.text

        # parse same as above
        if isinstance(j, dict):
            # try choices
            choices = j.get("choices")
            if choices and isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("delta")
                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"]
                    for k in ("text", "content", "generated_text"):
                        if k in first:
                            return first[k]
                if isinstance(first, str):
                    return first
            outputs = j.get("outputs") or j.get("predictions") or j.get("response")
            if outputs:
                if isinstance(outputs, list) and len(outputs) > 0:
                    if isinstance(outputs[0], dict) and "content" in outputs[0]:
                        return outputs[0]["content"]
                    if isinstance(outputs[0], str):
                        return outputs[0]
            for k in ("result", "text", "generated_text"):
                if k in j and isinstance(j[k], str):
                    return j[k]
        if isinstance(j, list) and j:
            if isinstance(j[0], dict):
                for k in ("generated_text", "text", "content"):
                    if k in j[0]:
                        return j[0][k]
            if isinstance(j[0], str):
                return j[0]

        return json.dumps(j, ensure_ascii=False)

    detail = last_resp.text if isinstance(last_resp, requests.Response) else str(last_resp)
    raise RuntimeError(f"Databricks model call failed; last response: {detail}")

# %%
def generate_narrative_report(student_id: str, months: int = 6, send_to_llm: bool = True):
    """
    Top-level function:
      - build structured summary,
      - create textual prompt,
      - (optionally) send to LLM and return textual report + summary.
    """
    summary = generate_student_summary(student_id, months=months)
    prompt_body = summarize_for_llm(summary)

    # LLM prompt template
    prompt = f"""
You are an educational analyst. Based on the structured summary below, produce a clear, empathetic, evidence-based student progress report (200-500 words) suitable for mentor. Include:
- One-paragraph plain-language summary of overall progress,
- Three bullet points: strengths, areas to improve, recommended actions,
- Any notable mentoring highlights.

Structured summary:
{prompt_body}

Give the narrative report backed by data in the three meaningful sections to help mentor understand student progress and next steps. it should be a well formatted complete report including introduction, body and conclusion."""
#Give output as JSON with keys: "for_guardian", "for_staff", "bullets" (list of {{"strengths":[],"improvements":[],"actions":[]}}).
    result_text = None
    if send_to_llm:
        # call OpenAI ChatCompletion (adjust to your client)
        db_endpoint = os.getenv("DATABRICKS_MODEL_ENDPOINT",
                                "https://adb-7405616962374519.19.azuredatabricks.net/serving-endpoints/databricks-claude-sonnet-4-5/invocations")
        token = os.getenv("DATABRICKS_TOKEN")
         # Build OpenAI-style chat messages (system + user)
        messages = [
            {"role": "system", "content": "You are a helpful assistant who writes education progress reports."},
            {"role": "user", "content": prompt}
        ]

        result_text = call_databricks_model(prompt=None, messages=messages, endpoint_url=db_endpoint, token=token,
                                           temperature=0.2, max_tokens=700)
        return result_text
    # return {
    #     "summary": summary,
    #     "prompt": prompt,
    #     "llm_output": result_text
    # }

# %%
!pip install reportlab


# %%
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus.flowables import HRFlowable

def _md_inline_to_para_html(text: str) -> str:
    # Convert simple markdown inline elements to ReportLab's mini-HTML:
    # bold **text**, italic *text*
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # escape any remaining angle brackets (safe guard)
    text = text.replace('< ', '&lt; ')
    return text

def markdown_to_pdf(markdown_text: str, output_path: str = "student_report.pdf"):
    styles = getSampleStyleSheet()
    # add/adjust styles
    styles.add(ParagraphStyle(name='Heading1Bold', parent=styles['Heading1'], spaceAfter=12))
    styles.add(ParagraphStyle(name='Heading2Bold', parent=styles['Heading2'], spaceAfter=8))
    normal = styles['BodyText']
    normal.spaceAfter = 6

    flowables = []
    lines = markdown_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            flowables.append(Spacer(1, 6))
            i += 1
            continue

        # Horizontal rule like '---'
        if re.match(r'^\s*-{3,}\s*$', line):
            flowables.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
            i += 1
            continue

        # Headings #, ##, ###
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            text = _md_inline_to_para_html(m.group(2).strip())
            style = 'Heading1Bold' if level == 1 else ('Heading2Bold' if level == 2 else styles['Heading3'])
            flowables.append(Paragraph(text, styles[style]))
            i += 1
            continue

        # Bullet list (continuous block starting with '- ')
        if re.match(r'^\s*-\s+', line):
            items = []
            while i < len(lines) and re.match(r'^\s*-\s+', lines[i]):
                item_text = re.sub(r'^\s*-\s+', '', lines[i]).strip()
                item_text = _md_inline_to_para_html(item_text)
                # Use Paragraph inside ListItem
                items.append(ListItem(Paragraph(item_text, normal), leftIndent=12))
                i += 1
            flowables.append(ListFlowable(items, bulletType='bullet', start='•'))
            continue

        # Numbered list (1. ...)
        if re.match(r'^\s*\d+\.\s+', line):
            items = []
            while i < len(lines) and re.match(r'^\s*\d+\.\s+', lines[i]):
                item_text = re.sub(r'^\s*\d+\.\s+', '', lines[i]).strip()
                item_text = _md_inline_to_para_html(item_text)
                items.append(ListItem(Paragraph(item_text, normal), leftIndent=12))
                i += 1
            flowables.append(ListFlowable(items, bulletType='1'))
            continue

        # Paragraph block: collect contiguous non-empty non-list lines into one paragraph
        para_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^\s*(#|-|\d+\.)\s+', lines[i]):
            para_lines.append(lines[i].rstrip())
            i += 1
        para_text = " ".join(l.strip() for l in para_lines)
        para_text = _md_inline_to_para_html(para_text)
        flowables.append(Paragraph(para_text, normal))

    # Build PDF
    doc = SimpleDocTemplate(output_path, rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=48)
    doc.build(flowables)
    return output_path

# Example usage:
# If your function returns the markdown string directly:
# md = generate_narrative_report("STU-000693", months=6, send_to_llm=True)
# If it returns a dict like {'llm_output': '...'}, adapt accordingly:
res = generate_narrative_report("STU-000357", months=6, send_to_llm=True)
if isinstance(res, dict):
    md = res.get('llm_output') or res.get('llm_output_text') or ''
else:
    md = res or ''

if not md:
    raise RuntimeError("No markdown content returned from generate_narrative_report.")

output_file = markdown_to_pdf(md, output_path="STU-000357_progress_report.pdf")
print("Wrote PDF:", output_file)

# %%
import os
import sys
import json
import requests
from databricks import sql

# %%
####investor_report(funding report helppy)
import os
import json
from datetime import date, timedelta
from databricks import sql
import requests

def _safe_literal(val):
    if val is None:
        return "NULL"
    s = str(val).strip()
    if s.isdigit():
        return s
    return "'" + s.replace("'", "''") + "'"

def _date_range_from_months(months: int):
    end = date.today()
    start = end - timedelta(days=30 * months)
    return start.isoformat(), end.isoformat()

def call_databricks_serving(messages, endpoint_url=None, token=None, timeout=60):
    """
    Minimal chat-style caller for Databricks serving endpoint.
    Expects OpenAI-like `messages` (list of {"role","content"}).
    Uses DATABRICKS_MODEL_ENDPOINT and DATABRICKS_TOKEN if not provided.
    """
    endpoint_url = endpoint_url or os.getenv("DATABRICKS_MODEL_ENDPOINT")
    if not endpoint_url:
        raise RuntimeError("DATABRICKS_MODEL_ENDPOINT not set")
    token = token or os.getenv("DATABRICKS_TOKEN")
    if not token:
        raise RuntimeError("DATABRICKS_TOKEN not set")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"messages": messages, "temperature": 0.2, "max_tokens": 700}
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    j = resp.json()
    # Try common shapes
    if isinstance(j, dict):
        choices = j.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            # nested message.content
            if isinstance(first, dict):
                msg = first.get("message") or first.get("delta")
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
                for k in ("text", "content", "generated_text"):
                    if k in first:
                        return first[k]
        outputs = j.get("outputs") or j.get("response") or j.get("result")
        if outputs:
            if isinstance(outputs, list):
                out0 = outputs[0]
                if isinstance(out0, dict) and "content" in out0:
                    return out0["content"]
                if isinstance(out0, str):
                    return out0
            if isinstance(outputs, str):
                return outputs
    # fallback
    return json.dumps(j, ensure_ascii=False)

def generate_investor_report(program_id: str = None,
                             months: int = 12,
                             call_model: bool = True):
    """
    Generate an investor-facing report summarizing outcomes and placements.

    - program_id: optional program identifier to filter placements/outcomes (uses program_connection_id column)
    - months: lookback window in months
    - call_model: if True, will call DATABRICKS_MODEL_ENDPOINT to produce a narrative (requires token)
    Returns: dict { summary: {...}, prompt: str, llm_output: str|None }
    """
    host = os.getenv("DATABRICKS_HOST")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")
    if not host or not http_path or not token:
        raise RuntimeError("Set DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN in env")

    start_date, end_date = _date_range_from_months(months)
    prog_filter = ""
    if program_id:
        prog_literal = _safe_literal(program_id)
        prog_filter = f" AND program_connection_id = {prog_literal} "

    # Aggregated placements query
    placements_q = f"""
    SELECT
      COUNT(*) AS total_placements,
      COUNT(DISTINCT student_id) AS unique_students,
      AVG(hourly_wage) AS avg_hourly_wage,
      SUM(CASE WHEN is_current THEN 1 ELSE 0 END) AS current_count,
      AVG(CASE WHEN student_satisfaction IS NOT NULL THEN student_satisfaction END) AS avg_satisfaction
    FROM hackathon.amer.employment_placements
    WHERE start_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
      {prog_filter}
    """

    # retention by status
    retention_q = f"""
    SELECT retention_status, COUNT(*) AS cnt
    FROM hackathon.amer.employment_placements
    WHERE start_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
      {prog_filter}
    GROUP BY retention_status
    ORDER BY cnt DESC
    LIMIT 20
    """

    # top industries
    industries_q = f"""
    SELECT industry, COUNT(*) AS cnt
    FROM hackathon.amer.employment_placements
    WHERE start_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
      {prog_filter}
    GROUP BY industry
    ORDER BY cnt DESC
    LIMIT 10
    """

    # sample anecdotes (notes)
    anecdotes_q = f"""
    SELECT student_id, employer_name, job_title, hourly_wage, start_date, student_satisfaction, notes
    FROM hackathon.amer.employment_placements
    WHERE start_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
      AND notes IS NOT NULL
      {prog_filter}
    ORDER BY start_date DESC
    LIMIT 6
    """

    # outcome metrics aggregation (how many metrics meet target)
    outcome_q = f"""
    SELECT
    om.metric_name,
    AVG(so.value) AS avg_value,
    AVG(CASE WHEN so.target_met THEN 1 ELSE 0 END) AS pct_target_met
    FROM hackathon.amer.student_outcomes so
    LEFT JOIN hackathon.amer.outcome_metrics om
    ON so.metric_id = om.metric_id
    WHERE so.measurement_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
    {('AND so.program_id = ' + _safe_literal(program_id)) if program_id else ''}
    GROUP BY om.metric_name
    ORDER BY pct_target_met DESC
    LIMIT 50
    """

    summary = {
        "program_id": program_id,
        "date_range": {"start": start_date, "end": end_date},
        "placements": {},
        "retention_breakdown": [],
        "top_industries": [],
        "anecdotes": [],
        "outcomes": []
    }

    try:
        with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
            with conn.cursor() as cur:
                cur.execute(placements_q)
                row = cur.fetchone() or ()
                summary["placements"] = {
                    "total_placements": int(row[0]) if len(row) > 0 and row[0] is not None else 0,
                    "unique_students": int(row[1]) if len(row) > 1 and row[1] is not None else 0,
                    "avg_hourly_wage": float(row[2]) if len(row) > 2 and row[2] is not None else None,
                    "current_count": int(row[3]) if len(row) > 3 and row[3] is not None else 0,
                    "avg_satisfaction": float(row[4]) if len(row) > 4 and row[4] is not None else None
                }

                cur.execute(retention_q)
                summary["retention_breakdown"] = [{ "retention_status": r[0], "count": int(r[1]) } for r in cur.fetchall()]

                cur.execute(industries_q)
                summary["top_industries"] = [{ "industry": r[0], "count": int(r[1]) } for r in cur.fetchall()]

                cur.execute(anecdotes_q)
                anecdotes = []
                for r in cur.fetchall():
                    anecdotes.append({
                        "student_id": r[0],
                        "employer": r[1],
                        "job_title": r[2],
                        "hourly_wage": float(r[3]) if r[3] is not None else None,
                        "start_date": str(r[4]) if r[4] is not None else None,
                        "satisfaction": r[5],
                        "notes": (r[6] or "")[:800]   # cap length
                    })
                summary["anecdotes"] = anecdotes

                cur.execute(outcome_q)
                summary["outcomes"] = [
                    {
                    "metric_name": r[0],
                    "avg_value": float(r[1]) if r[1] is not None else None,
                    "pct_target_met": round(100.0 * r[2], 2) if r[2] is not None else None
                    }
                    for r in cur.fetchall()
                ]

    except Exception as e:
        raise RuntimeError(f"Databricks query failed: {e}")

    # Build investor prompt
    prompt_lines = []
    prompt_lines.append(f"Create a persuasive investor-facing impact story for program: {program_id or 'ALL PROGRAMS'}")
    prompt_lines.append(f"Reporting period: {start_date} to {end_date}")
    p = summary["placements"]
    prompt_lines.append(f"- Total placements: {p.get('total_placements')}, unique students employed: {p.get('unique_students')}")
    if p.get("avg_hourly_wage") is not None:
        prompt_lines.append(f"- Average starting wage: ${p.get('avg_hourly_wage'):.2f} per hour")
    if p.get("avg_satisfaction") is not None:
        prompt_lines.append(f"- Average student satisfaction (placement): {p.get('avg_satisfaction'):.1f}/10 (if scaled)")
    prompt_lines.append(f"- Current placements still active: {p.get('current_count')}")
    if summary["top_industries"]:
        top = ", ".join([f'{t["industry"]} ({t["count"]})' for t in summary["top_industries"][:5]])
        prompt_lines.append(f"- Top industries by placements: {top}")
    if summary["outcomes"]:
        top_metric = summary["outcomes"][0]
        prompt_lines.append(f"- Example outcome metric: {top_metric['metric_name']} - {top_metric['pct_target_met']}% of measurements met target")

    prompt_lines.append("\nInclude three short human-interest anecdotes from placements (student_id, employer, job_title, short notes).")
    for a in summary["anecdotes"][:3]:
        note = a["notes"].replace("\n", " ")
        prompt_lines.append(f"- Anecdote: Student {a['student_id']} placed at {a['employer']} as {a['job_title']} (${a.get('hourly_wage')}) — {note[:240]}")

    prompt_lines.append("\nWrite a 300-500 word investor-facing narrative (tone: inspiring, evidence-backed). At the end include a concise 3-point 'how investors can help more' CTA (actionable asks) and a short data appendix listing the key numbers above.")
    prompt = "\n".join(prompt_lines)

    llm_output = None
    if call_model:
        # build chat messages
        messages = [
            {"role": "system", "content": "You are a concise, persuasive impact writer for nonprofit fundraisers."},
            {"role": "user", "content": prompt}
        ]
        # call Databricks serving endpoint
        try:
            llm_output = call_databricks_serving(messages)
        except Exception as e:
            llm_output = None
            summary["model_error"] = str(e)

    return {"summary": summary, "prompt": prompt, "llm_output": llm_output}

# %%
def build_investor_narrative_from_summary(summary: dict) -> str:
    """
    Build a 300-500 word investor-facing narrative from the structured summary.
    Returns a markdown string (headings + paragraphs + CTA).
    """
    p = summary.get("placements", {})
    start = summary.get("date_range", {}).get("start")
    end = summary.get("date_range", {}).get("end")
    total = p.get("total_placements", 0)
    unique = p.get("unique_students", 0)
    avg_wage = p.get("avg_hourly_wage")
    avg_sat = p.get("avg_satisfaction")
    current = p.get("current_count", 0)

    industries = summary.get("top_industries", [])
    top_inds = ", ".join([f"{i.get('industry')} ({i.get('count')})" for i in industries[:5]]) if industries else "Various sectors"

    outcomes = summary.get("outcomes", [])
    top_outcome = outcomes[0] if outcomes else None
    outcome_line = ""
    if top_outcome:
        outcome_line = f"Our strongest outcome is {top_outcome['metric_name']}, with {top_outcome.get('pct_target_met',0)}% of measurements meeting the target."

    # human-interest: keep for appendix only, but use a short anonymized highlight here (no IDs)
    anecdote_sample = summary.get("anecdotes", [])[:3]
    anecdote_lines = []
    for a in anecdote_sample:
        # short, anonymized mention
        job = a.get("job_title") or "role"
        employer = a.get("employer") or "an employer"
        wage = a.get("hourly_wage")
        wage_part = f" (${wage:.0f}/hr)" if isinstance(wage, (int, float)) else ""
        note = (a.get("notes") or "").split(".")[0][:140]  # single-sentence-ish snippet
        anecdote_lines.append(f"- A recent graduate secured a {job} at {employer}{wage_part} — {note}.")

    # Compose narrative (approx 350-420 words)
    parts = []
    parts.append(f"# Investor Impact Report")
    parts.append(f"**Reporting period:** {start} to {end}")
    parts.append("")
    parts.append("Over the last year we have scaled our employment outcomes while remaining laser-focused on long-term impact. "
                 f"In the reporting period, our programs supported approximately {total:,} placements across {unique:,} individuals. "
                 f"Of those placements, {current:,} were still active at time of reporting, demonstrating sustained employer engagement.")
    if avg_wage is not None:
        parts.append(f"The average starting wage for placements was ${avg_wage:.2f} per hour; this figure illustrates the program's role in helping participants secure living-wage work.")
    if avg_sat is not None:
        parts.append(f"Reported satisfaction for placement experiences was {avg_sat:.1f}/10, a useful indicator to guide targeted support and employer partnerships.")
    parts.append("")
    parts.append(f"Our placements concentrated in {top_inds}. {outcome_line}")
    parts.append("")
    parts.append("These results reflect programmatic strengths in training, employer matching, and post-placement support. "
                 "We combine practical, employer-aligned skill development with careful follow-through, which leads to faster onboarding and higher retention.")
    parts.append("")
    parts.append("Notable human-impact highlights (anonymized):")
    parts.extend(anecdote_lines)
    parts.append("")
    parts.append("Recommendations — how investors can help scale impact:")
    parts.append("1. Invest in employer partnership growth to expand sector diversity and increase starting wages through negotiated pathways.")
    parts.append("2. Fund targeted retention and up-skilling pilots (6–12 months) to improve long-term employment stability.")
    parts.append("3. Support technology and data infrastructure to measure outcomes more frequently and personalize post-placement supports.")
    parts.append("")
    parts.append("Data appendix (key figures):")
    parts.append(f"- Total placements: {total:,}")
    parts.append(f"- Unique students placed: {unique:,}")
    if avg_wage is not None:
        parts.append(f"- Average starting wage: ${avg_wage:.2f}/hr")
    if avg_sat is not None:
        parts.append(f"- Average placement satisfaction: {avg_sat:.2f}/10")
    if outcomes:
        # show top 3 outcome metrics briefly
        top_n = outcomes[:3]
        for o in top_n:
            parts.append(f"- {o.get('metric_name')}: {o.get('pct_target_met')}% measurements met target")

    # join into markdown style narrative
    md = "\n\n".join(parts)

    # ensure length bounds roughly in words (not exact; but ~350-450)
    # if too short or too long, adjust by adding/removing padding sentence
    words = len(md.split())
    if words < 280:
        md += "\n\nOur team is prepared to provide additional case studies and regional breakdowns on request."
    elif words > 600:
        # trim long anecdotes if any
        md = "\n\n".join(parts[:18])  # crude trim; fine for fallback
    return md

# %%
# Notebook cell: generate investor report, create charts, and write PDF
import os
import tempfile
import datetime
import math
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# Use your existing function defined earlier in the notebook
# from earlier cells: generate_investor_report(program_id=None, months=12, call_model=True)

def _simple_md_to_para_html(md_text: str) -> str:
    # Very small markdown -> ReportLab mini-HTML converter: headings and bold/italic and newlines.
    if not md_text:
        return ""
    text = md_text
    # Bold
    text = text.replace("**", "<b>").replace("<b><b>", "**")  # crude safeguard (rare)
    # Replace headings: '# ' -> <b><font size=14> heading </font></b>\n
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith("### "):
            out_lines.append(f"<b>{s[4:]}</b><br/>")
        elif s.startswith("## "):
            out_lines.append(f"<b>{s[3:]}</b><br/>")
        elif s.startswith("# "):
            out_lines.append(f"<b><font size=14>{s[2:]}</font></b><br/>")
        else:
            # preserve bullet lines
            if s.startswith("- "):
                out_lines.append(f"• {s[2:]}")
            else:
                out_lines.append(s)
    return "<br/>".join(out_lines)

def _save_retention_pie(retention_list, outpath):
    labels = [r.get("retention_status") or "Unknown" for r in retention_list]
    sizes = [r.get("count", 0) for r in retention_list]
    if not any(sizes):
        # placeholder pie
        plt.figure(figsize=(6,4))
        plt.text(0.5, 0.5, "No retention data", ha='center', va='center')
        plt.axis('off')
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
        return
    plt.figure(figsize=(6,4))
    plt.pie(sizes, labels=labels, autopct=lambda p: f"{p:.0f}%" if p>0 else "", startangle=90)
    plt.title("Retention Breakdown")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

def _save_industries_bar(industries_list, outpath, top_n=8):
    inds = industries_list[:top_n]
    if not inds:
        plt.figure(figsize=(6,4))
        plt.text(0.5, 0.5, "No industry data", ha='center', va='center')
        plt.axis('off')
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
        return
    names = [i.get("industry") or "Unknown" for i in inds]
    counts = [i.get("count", 0) for i in inds]
    fig, ax = plt.subplots(figsize=(6,4))
    y_pos = range(len(names))[::-1]
    ax.barh(y_pos, counts, color="#2b8cbe")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Placements")
    ax.set_title("Top Industries by Placements")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

def _save_outcomes_bar(outcomes_list, outpath, top_n=8):
    if not outcomes_list:
        plt.figure(figsize=(6,4))
        plt.text(0.5, 0.5, "No outcome metrics", ha='center', va='center')
        plt.axis('off')
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
        return
    # use pct_target_met measure if present
    top = [o for o in outcomes_list if o.get("pct_target_met") is not None][:top_n]
    if not top:
        plt.figure(figsize=(6,4))
        plt.text(0.5, 0.5, "No pct_target_met metric", ha='center', va='center')
        plt.axis('off')
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
        return
    names = [t["metric_name"] for t in top]
    vals = [t["pct_target_met"] for t in top]
    fig, ax = plt.subplots(figsize=(6,4))
    x = range(len(names))
    ax.bar(x, vals, color="#7fbf7b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, min(100, max(vals) * 1.1 if vals else 100))
    ax.set_ylabel("% Measurements Meeting Target")
    ax.set_title("Outcome Metrics - % Target Met")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

# Main orchestration
program_id = None   # set to a specific program id string if desired
months = 12
outfile = f"investor_report_{(program_id or 'ALL')}_{datetime.date.today().isoformat()}.pdf"
tmpdir = tempfile.mkdtemp(prefix="invest_report_")
# --- Updated orchestration: generate investor report, ensure narrative, create charts, and write PDF ---

print("Querying Databricks and generating structured summary...")
# If you want to avoid external model calls while debugging, set call_model=False
report_obj = generate_investor_report(program_id=program_id, months=months, call_model=True)
summary = report_obj.get("summary", {})
llm_text = (report_obj.get("llm_output") or "").strip()
prompt_text = report_obj.get("prompt") or ""

# If model failed or returned nothing, use the local fallback narrative builder
if not llm_text:
    try:
        llm_text = build_investor_narrative_from_summary(summary)
        summary.setdefault("model_fallback", True)
    except Exception as e:
        # Last resort: short synthetic narrative to avoid showing the prompt
        llm_text = ("# Investor Narrative\n\n"
                    "We have a robust set of outcomes this period. Key figures are available in the appendix. "
                    "Please contact the program team for the full dataset and student case studies.")
        summary.setdefault("model_fallback_error", str(e))

# 2) Create charts (same as before)
retention_fp = os.path.join(tmpdir, "retention.png")
industries_fp = os.path.join(tmpdir, "industries.png")
outcomes_fp = os.path.join(tmpdir, "outcomes.png")

_save_retention_pie(summary.get("retention_breakdown", []), retention_fp)
_save_industries_bar(summary.get("top_industries", []), industries_fp)
_save_outcomes_bar(summary.get("outcomes", []), outcomes_fp)

# 3) Compose PDF — narrative + visuals first, anonymized student info only in appendix
styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title', parent=styles['Title'], alignment=1, spaceAfter=12)
h2 = ParagraphStyle('H2', parent=styles['Heading2'], spaceAfter=8)
body = ParagraphStyle('Body', parent=styles['BodyText'], spaceAfter=6)

doc = SimpleDocTemplate(outfile, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
flow = []

# Header
flow.append(Paragraph("Investor Impact Report", title_style))
flow.append(Paragraph(f"Program: {program_id or 'ALL PROGRAMS'}", body))
flow.append(Paragraph(f"Reporting period: {summary.get('date_range',{}).get('start')} to {summary.get('date_range',{}).get('end')}", body))
flow.append(Spacer(1,12))

# LLM narrative (or fallback) — present as the main piece
# Convert markdown headings -> simple HTML lines
def md_to_paras(md):
    out = []
    blocks = md.split("\n\n")
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if b.startswith("# "):
            out.append(Paragraph(f"<b><font size=14>{b[2:].strip()}</font></b>", h2))
        elif b.startswith("## "):
            out.append(Paragraph(f"<b>{b[3:].strip()}</b>", body))
        else:
            # bullets handling: simple conversion
            lines = b.splitlines()
            if lines and lines[0].strip().startswith("- "):
                for ln in lines:
                    if ln.strip().startswith("- "):
                        out.append(Paragraph(f"• {ln.strip()[2:]}", body))
                    else:
                        out.append(Paragraph(ln, body))
            else:
                out.append(Paragraph(b.replace("\n", "<br/>"), body))
        out.append(Spacer(1,6))
    return out

flow.extend(md_to_paras(llm_text))
flow.append(Spacer(1,12))

# Charts section (images only)
flow.append(Paragraph("Key Visualizations", h2))
flow.append(Spacer(1,6))
def add_img(path, caption):
    if os.path.exists(path):
        img = Image(path, width=450, height=250)
        flow.append(img)
        flow.append(Paragraph(f"<i>{caption}</i>", body))
        flow.append(Spacer(1,12))

add_img(retention_fp, "Retention breakdown (count)")
add_img(industries_fp, "Top industries by placements")
add_img(outcomes_fp, "Top outcome metrics - % measurements meeting target")

# Start appendix on a new page
flow.append(PageBreak())

flow.append(Paragraph("Data Appendix & Selected Student References (anonymized)", h2))
flow.append(Spacer(1,6))

# Key numbers table (same as before)
k = summary.get("placements", {})
table_data = [
    ["Metric", "Value"],
    ["Total placements", str(k.get("total_placements", 0))],
    ["Unique students placed", str(k.get("unique_students", 0))],
    ["Average hourly wage", f"${k.get('avg_hourly_wage'):.2f}" if k.get('avg_hourly_wage') else "N/A"],
    ["Avg satisfaction", f"{k.get('avg_satisfaction'):.2f}" if k.get('avg_satisfaction') else "N/A"],
    ["Current placements active", str(k.get('current_count', 0))]
]
t = Table(table_data, colWidths=[260, 260])
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
    ('GRID',(0,0),(-1,-1),0.5, colors.grey),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
]))
flow.append(t)
flow.append(Spacer(1,12))

# Retention small table
if summary.get("retention_breakdown"):
    flow.append(Paragraph("Retention breakdown (counts):", body))
    rows = [["Status","Count"]] + [[r["retention_status"] or "Unknown", str(r["count"])] for r in summary["retention_breakdown"]]
    rt = Table(rows, colWidths=[260,260])
    rt.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0), colors.lightgrey)]))
    flow.append(rt)
    flow.append(Spacer(1,12))

# Top industries table
if summary.get("top_industries"):
    flow.append(Paragraph("Top industries (by placements):", body))
    rows = [["Industry","Count"]] + [[i["industry"] or "Unknown", str(i["count"])] for i in summary["top_industries"][:10]]
    it = Table(rows, colWidths=[260,260])
    it.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0), colors.lightgrey)]))
    flow.append(it)
    flow.append(Spacer(1,12))

# Anecdotes / student references: mask student IDs and keep only minimal info
flow.append(Paragraph("Selected placement anecdotes (anonymized):", h2))
for a in summary.get("anecdotes", [])[:10]:
    sid = a.get("student_id", "")
    # mask id: keep prefix and last 3 chars if available
    if sid and len(sid) > 6:
        masked = sid[:4] + "..." + sid[-3:]
    else:
        masked = "STU-xxxx"
    job = a.get("job_title") or "role"
    employer = a.get("employer") or "employer"
    wage = a.get("hourly_wage")
    wage_txt = f" (${wage:.0f}/hr)" if isinstance(wage, (int, float)) else ""
    notes = (a.get("notes") or "").replace("\n", " ")
    notes = notes[:300]  # cap
    flow.append(Paragraph(f"<b>{masked}</b> — {job} at {employer}{wage_txt}", body))
    flow.append(Paragraph(notes, body))
    flow.append(Spacer(1,6))

# Save PDF
doc.build(flow)
print("Wrote PDF:", os.path.abspath(outfile))
print("Temporary files in:", tmpdir)


# %%
#resume builder
import re
import html
from typing import List, Dict

def _keywords_from_text(text: str, min_len: int = 3) -> List[str]:
    if not text:
        return []
    text = re.sub(r"[^\w\s]", " ", text.lower())
    words = [w for w in text.split() if len(w) >= min_len]
    stop = {"the","and","for","with","from","that","this","their","have","will","are","our","in","on","to","a","an","of"}
    keywords = [w for w in words if w not in stop]
    return list(dict.fromkeys(keywords))  # keep order unique

def _join_preview(items: List[str], max_items: int = 6) -> str:
    return ", ".join(items[:max_items])

def _connect_and_query(sql_text: str):
    host = get_env_var("DATABRICKS_HOST")
    http_path = get_env_var("DATABRICKS_HTTP_PATH")
    token = get_env_var("DATABRICKS_TOKEN")
    try:
        with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
            with conn.cursor() as cur:
                return query_table(cur, sql_text)
    except Exception as e:
        raise RuntimeError(f"Databricks query error: {e}")

def fetch_student_core(student_id: str) -> Dict:
    sid = _safe_id_literal(student_id)
    q = f"""
    SELECT * FROM hackathon.amer.students
    WHERE student_id = {sid}
    LIMIT 1
    """
    rows = _connect_and_query(q)
    return rows[0] if rows else {}

def fetch_academic_records(student_id: str, limit=6) -> List[Dict]:
    sid = _safe_id_literal(student_id)
    q = f"""
    SELECT academic_year, semester, grade_level, school_name, gpa, attendance_rate, record_date, notes
    FROM hackathon.amer.academic_records
    WHERE student_id = {sid}
    ORDER BY record_date DESC
    LIMIT {limit}
    """
    return _connect_and_query(q)

def fetch_student_skills(student_id: str) -> List[Dict]:
    sid = _safe_id_literal(student_id)
    q = f"""
    SELECT ss.skill_id, ss.initial_proficiency_level, ss.current_proficiency_level, ss.last_assessment_date, s.skill_name
    FROM hackathon.amer.student_skills ss
    LEFT JOIN hackathon.amer.skills s ON ss.skill_id = s.skill_id
    WHERE ss.student_id = {sid}
    ORDER BY ss.last_assessment_date DESC
    """
    return _connect_and_query(q)

def fetch_certifications(student_id: str) -> List[Dict]:
    sid = _safe_id_literal(student_id)
    q = f"""
    SELECT sc.attempt_date, sc.passed, c.certification_name, c.issuing_organization, c.certification_type, sc.notes
    FROM hackathon.amer.student_certifications sc
    LEFT JOIN hackathon.amer.certifications c ON sc.certification_id = c.certification_id
    WHERE sc.student_id = {sid}
    ORDER BY sc.attempt_date DESC
    """
    return _connect_and_query(q)

def fetch_media_projects(student_id: str, limit: int = 8) -> list:
    """
    Robust fetch for media projects linked to a student.

    Strategy:
    1) Try the simple query using `created_by_student_id`.
    2) If that column is missing (SQL error), fall back to joining via `project_participants`.
    """
    sid = _safe_literal(student_id)
    # candidate 1: direct ownership column (works if present)
    q1 = f"""
    SELECT project_title, project_type, description, start_date, target_audience, views_count, likes_count
    FROM hackathon.amer.media_projects
    WHERE created_by_student_id = {sid}
    ORDER BY start_date DESC
    LIMIT {limit}
    """

    try:
        rows = _connect_and_query(q1)
        # if query succeeded but returned empty, still return it (student may have no direct-created projects)
        return rows
    except Exception as e:
        msg = str(e).lower()
        # if failure looks like unresolved column, try fallback join strategy
        if "created_by_student_id" in msg or "column" in msg or "cannot be resolved" in msg or "42703" in msg:
            q2 = f"""
            SELECT mp.project_title, mp.project_type, mp.description, mp.start_date, mp.target_audience, mp.views_count, mp.likes_count
            FROM hackathon.amer.media_projects mp
            JOIN hackathon.amer.project_participants pp
              ON mp.project_id = pp.project_id
            WHERE pp.student_id = {sid}
            ORDER BY mp.start_date DESC
            LIMIT {limit}
            """
            try:
                return _connect_and_query(q2)
            except Exception as e2:
                # fallback: try a very small sample from media_projects (best-effort)
                try:
                    return _connect_and_query(f"SELECT project_title, project_type, description, start_date, target_audience, views_count, likes_count FROM hackathon.amer.media_projects LIMIT {limit}")
                except Exception:
                    raise RuntimeError(f"Failed to fetch media projects (tried direct column and join): {e2}")
        else:
            # unknown error, re-raise for visibility
            raise

def fetch_employment_placements(student_id: str, limit=6) -> List[Dict]:
    sid = _safe_id_literal(student_id)
    q = f"""
    SELECT employer_name, industry, job_title, hourly_wage, start_date, end_date, is_current, student_satisfaction, notes
    FROM hackathon.amer.employment_placements
    WHERE student_id = {sid}
    ORDER BY start_date DESC
    LIMIT {limit}
    """
    return _connect_and_query(q)

def _score_relevance(texts: List[str], keywords: List[str]) -> int:
    if not texts or not keywords:
        return 0
    text = " ".join([t or "" for t in texts]).lower()
    score = 0
    for kw in keywords:
        if kw in text:
            score += 1
    return score

def generate_student_resume_and_linkedin(student_id: str,
                                        job_text: str | None = None,
                                        industry: str | None = None,
                                        months_recent: int = 36,
                                        write_files: bool = False,
                                        out_prefix: str | None = None) -> Dict:
    """
    Returns: { 'resume_md': str, 'linkedin_md': str, 'structured': {...} }
    Resume is a Markdown string suitable to copy/paste into doc editors or ATS-friendly tools.
    LinkedIn post is a short marketing blurb tailored to the job/industry.
    """
    # 1. Fetch data
    core = fetch_student_core(student_id)
    academics = fetch_academic_records(student_id)
    skills = fetch_student_skills(student_id)
    certs = fetch_certifications(student_id)
    projects = fetch_media_projects(student_id)
    placements = fetch_employment_placements(student_id)

    # 2. Build keyword set from job_text and industry
    job_kw = _keywords_from_text((job_text or "") + " " + (industry or ""))
    # fallback: include some skill names as keywords
    skill_names = [s.get("skill_name") or s.get("skill_id") or "" for s in skills]
    # 3. Score relevance for each section
    skill_relevance = sorted(skills, key=lambda s: -_score_relevance([s.get("skill_name","")], job_kw))
    project_relevance = sorted(projects, key=lambda p: -_score_relevance([p.get("project_title",""), p.get("description","")], job_kw))
    cert_relevance = sorted(certs, key=lambda c: -_score_relevance([c.get("certification_name",""), c.get("issuing_organization","")], job_kw))
    placement_relevance = sorted(placements, key=lambda e: -_score_relevance([e.get("job_title",""), e.get("notes",""), e.get("industry","")], job_kw))

    # 4. Compose resume sections (Markdown)
    # Header: use email/phone if name not available
    name_line = core.get("email") or core.get("student_id") or student_id
    headline = job_text or industry or "Candidate for " + (job_text or "roles")
    profile_lines = []
    profile_lines.append(f"# {name_line}")
    profile_lines.append(f"**{headline}**")
    profile_lines.append("")
    # Summary tailored:
    summary_parts = []
    if job_kw:
        top_k = ", ".join(job_kw[:6])
        summary_parts.append(f"Goal-oriented candidate with demonstrable experience in {industry or 'relevant sectors'}, focused on {top_k}.")
    else:
        summary_parts.append("Motivated candidate with practical experience in training, employer placements, and creative projects.")
    # highlight strongest recent placement/project if available
    if placement_relevance:
        p = placement_relevance[0]
        if p.get("job_title") and p.get("employer_name"):
            summary_parts.append(f"Most recently placed as {p.get('job_title')} at {p.get('employer_name')}, where they contributed to hands-on tasks and employer-valued outcomes.")
    elif project_relevance:
        pr = project_relevance[0]
        summary_parts.append(f"Recent portfolio work includes '{pr.get('project_title')}' — a {pr.get('project_type')} focused on {pr.get('target_audience') or 'creative output'}.")

    profile_lines.append(" ".join(summary_parts))
    profile_lines.append("")

    # Skills section
    profile_lines.append("## Key Skills")
    if skill_relevance:
        skill_items = []
        for s in skill_relevance[:12]:
            name = s.get("skill_name") or s.get("skill_id")
            prof = s.get("current_proficiency_level")
            prof_txt = f" — proficiency {prof}" if prof is not None else ""
            skill_items.append(f"- {name}{prof_txt}")
        profile_lines.extend(skill_items)
    else:
        profile_lines.append("- Training, soft skills, and employer-facing competencies")

    profile_lines.append("")

    # Experience / Placements (prioritize relevance)
    profile_lines.append("## Relevant Experience")
    if placement_relevance:
        for e in placement_relevance[:6]:
            title = e.get("job_title") or "Position"
            emp = e.get("employer_name") or e.get("industry") or "Employer"
            dates = ""
            if e.get("start_date"):
                dates = f" ({e.get('start_date')}{' - ' + str(e.get('end_date')) if e.get('end_date') else ' - present'})"
            wage = e.get("hourly_wage")
            wage_txt = f" — ${wage:.2f}/hr" if isinstance(wage, (int, float)) else ""
            notes = (e.get("notes") or "")
            notes = notes.replace("\n", " ")
            # keep a single-line bullet
            profile_lines.append(f"- **{title}**, {emp}{dates}{wage_txt} — {notes[:220]}")
    else:
        profile_lines.append("- No recorded placements in the database; see projects and certifications below.")

    profile_lines.append("")

    # Projects / Portfolio
    profile_lines.append("## Portfolio & Projects")
    if project_relevance:
        for pr in project_relevance[:6]:
            title = pr.get("project_title") or "Project"
            ptype = pr.get("project_type") or ""
            desc = (pr.get("description") or "")[:300].replace("\n"," ")
            views = pr.get("views_count")
            views_txt = f" — {views} views" if views else ""
            profile_lines.append(f"- **{title}** ({ptype}){views_txt} — {desc}")
    else:
        profile_lines.append("- No media projects found in the system.")

    profile_lines.append("")

    # Certifications
    profile_lines.append("## Certifications & Training")
    if cert_relevance:
        for c in cert_relevance[:8]:
            cname = c.get("certification_name") or ""
            org = c.get("issuing_organization") or ""
            passed = c.get("passed")
            when = c.get("attempt_date")
            status = "Passed" if passed else ("Attempted" if passed is not None else "Earned")
            profile_lines.append(f"- {cname} ({org}) — {status}{' on ' + str(when) if when else ''}")
    else:
        profile_lines.append("- No formal certifications recorded.")

    profile_lines.append("")

    # Education / academics
    profile_lines.append("## Education & Academic Highlights")
    if academics:
        for a in academics[:4]:
            school = a.get("school_name") or ""
            grade = a.get("grade_level")
            gpa = a.get("gpa")
            rd = a.get("record_date")
            line = f"- {school}"
            if grade:
                line += f", Grade {grade}"
            if gpa:
                line += f" — GPA {gpa}"
            if rd:
                line += f" ({rd})"
            profile_lines.append(line)
    else:
        profile_lines.append("- Academic records not available.")

    profile_lines.append("")

    # Contact / metadata
    profile_lines.append("## Contact & Links")
    email = core.get("email")
    phone = core.get("phone")
    contact_lines = []
    if email:
        contact_lines.append(f"- Email: {email}")
    if phone:
        contact_lines.append(f"- Phone: {phone}")
    # project links (if any include hint)
    if projects:
        contact_lines.append(f"- Portfolio highlights: { _join_preview([p.get('project_title') for p in projects if p.get('project_title')]) }")
    if contact_lines:
        profile_lines.extend(contact_lines)
    else:
        profile_lines.append("- Contact info not available in system; please add email/phone.")

    profile_lines.append("")
    # Tailored pitch at top for the job
    if job_kw:
        pitch = f"## Tailored Pitch for '{job_text or industry}'"
        profile_lines.append(pitch)
        bullet_reasons = []
        # pick top matching skill names
        top_skill_names = [s.get("skill_name") for s in skill_relevance if s.get("skill_name")]
        if top_skill_names:
            bullet_reasons.append(f"- Demonstrated skills: { _join_preview(top_skill_names) }")
        top_projects = [pr.get("project_title") for pr in project_relevance if pr.get("project_title")]
        if top_projects:
            bullet_reasons.append(f"- Portfolio: { _join_preview(top_projects) }")
        top_certs = [c.get("certification_name") for c in cert_relevance if c.get("certification_name")]
        if top_certs:
            bullet_reasons.append(f"- Certifications: { _join_preview(top_certs) }")
        if bullet_reasons:
            profile_lines.extend(bullet_reasons)
        profile_lines.append("")

    # 5. LinkedIn short profile
    # Build a tight, shareable LinkedIn post: 2-4 short paragraphs + CTA
    linkedin_parts = []
    name_display = core.get("student_id") or "Recent Graduate"
    linkedin_parts.append(f"{name_display} — Seeking opportunities in {industry or (job_text or 'related roles')}.")
    # One-sentence highlight
    key_strengths = []
    if skill_names:
        key_strengths.append(skill_names[0])
    if cert_relevance:
        key_strengths.append(cert_relevance[0].get("certification_name"))
    linkedin_parts.append("I bring hands-on experience in " + (", ".join([k for k in key_strengths if k]) or "skills and projects") + ", plus real-world placements with employer partners.")
    # Short anecdote line
    if placement_relevance:
        p = placement_relevance[0]
        linkedin_parts.append(f"Most recently worked as {p.get('job_title')} at {p.get('employer_name') or p.get('industry')}, where I focused on practical deliverables and employer feedback.")
    linkedin_parts.append("If your team is hiring or wants to see a short portfolio, DM me or email " + (core.get("email") or "[email]") + ".")
    linkedin_md = "\n\n".join(linkedin_parts)

    resume_md = "\n\n".join(profile_lines)

    result = {
        "resume_md": resume_md,
        "linkedin_md": linkedin_md,
        "structured": {
            "core": core,
            "academics": academics,
            "skills": skills,
            "certifications": certs,
            "projects": projects,
            "placements": placements,
            "relevance_keywords": job_kw
        }
    }

    # optionally write files
    if write_files:
        prefix = out_prefix or f"{student_id}_resume"
        with open(f"{prefix}.md", "w", encoding="utf-8") as f:
            f.write(resume_md)
        with open(f"{prefix}_linkedin.md", "w", encoding="utf-8") as f:
            f.write(linkedin_md)
        result["files_written"] = [f"{prefix}.md", f"{prefix}_linkedin.md"]

    return result

# %%
# Replace or add this helper near your resume generator so only passed certs appear
def _filter_passed_certifications(certs: list) -> list:
    """
    Return only certifications where `passed` is truthy.
    Accepts either the joined view (with 'passed' key) or simple cert rows.
    """
    if not certs:
        return []
    def passed_flag(c):
        # handle possible key names / types
        v = c.get("passed") if isinstance(c, dict) else None
        if v is None:
            return False
        # handle numeric or boolean
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.lower() in ("1","true","t","yes","y")
        return bool(v)
    return [c for c in certs if passed_flag(c)]

# %%
# Styled resume PDF generator — uses ReportLab to produce a neat two-column resume
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Frame, KeepInFrame
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

def build_styled_resume_pdf_from_structured(structured: dict, output_path: str, student_display_name: str | None = None):
    """
    Build a styled, ATS-friendly resume PDF from the `structured` data produced
    by `generate_student_resume_and_linkedin(...)`.
    - `structured` should include keys: core, academics, skills, certifications, projects, placements.
    - Only certifications with passed=True are shown.
    """
    styles = getSampleStyleSheet()
    # custom styles
    name_style = ParagraphStyle('Name', parent=styles['Heading1'], fontSize=20, leading=22, spaceAfter=6)
    headline_style = ParagraphStyle('Headline', parent=styles['Normal'], fontSize=11, textColor=colors.darkgray, spaceAfter=8)
    section_h = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=12, leading=14, spaceBefore=8, spaceAfter=6)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], leftIndent=6, bulletIndent=0, spaceBefore=2, spaceAfter=2)
    small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=9, textColor=colors.darkgray)

    core = structured.get('core', {}) or {}
    skills = structured.get('skills', []) or []
    certifications = _filter_passed_certifications(structured.get('certifications', []) or [])
    projects = structured.get('projects', []) or []
    placements = structured.get('placements', []) or []
    academics = structured.get('academics', []) or []

    # Document
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    # Header elements
    name = student_display_name or core.get('student_id') or core.get('email') or "Candidate"
    headline = core.get('primary_language') or core.get('school_type') or ""
    contact_parts = []
    if core.get('email'):
        contact_parts.append(core.get('email'))
    if core.get('phone'):
        contact_parts.append(core.get('phone'))
    contact_line = " | ".join(contact_parts)

    # Left column width (narrow) and right column (main)
    left_w = 2.1 * inch
    right_w = doc.width - left_w - 12

    # Build left column flow (skills, certifications, contact)
    left_flow = []
    left_flow.append(Paragraph(name, name_style))
    if headline:
        left_flow.append(Paragraph(headline, headline_style))
    if contact_line:
        left_flow.append(Paragraph(contact_line, small))
    left_flow.append(Spacer(1, 8))

    # Skills block
    left_flow.append(Paragraph("Skills", section_h))
    if skills:
        # prefer to show skill_name and level if available
        skill_lines = []
        for s in skills[:12]:
            nm = s.get('skill_name') or s.get('skill_id') or ""
            lvl = s.get('current_proficiency_level')
            if lvl is not None:
                skill_lines.append(f"{nm} ({lvl})")
            else:
                skill_lines.append(nm)
        # make a compact comma-separated paragraph
        left_flow.append(Paragraph(", ".join([str(x) for x in skill_lines if x]), bullet_style))
    else:
        left_flow.append(Paragraph("—", bullet_style))
    left_flow.append(Spacer(1,8))

    # Certifications block (only passed)
    left_flow.append(Paragraph("Certifications", section_h))
    if certifications:
        for c in certifications[:8]:
            cname = c.get('certification_name') or c.get('certification') or ""
            org = c.get('issuing_organization') or c.get('issuing_org') or ""
            when = c.get('attempt_date') or c.get('issue_date') or ""
            left_flow.append(Paragraph(f"<b>{cname}</b><br/><font size=9 color=grey>{org} {('· '+str(when)) if when else ''}</font>", bullet_style))
    else:
        left_flow.append(Paragraph("—", bullet_style))
    left_flow.append(Spacer(1,8))

    # Portfolio (small list)
    left_flow.append(Paragraph("Portfolio", section_h))
    if projects:
        for pr in projects[:6]:
            title = pr.get('project_title') or pr.get('project_id') or "Project"
            typ = pr.get('project_type') or ""
            left_flow.append(Paragraph(f"{title} <font size=9 color=grey>({typ})</font>", bullet_style))
    else:
        left_flow.append(Paragraph("—", bullet_style))

    # Build right column flow (experience, education, summary)
    right_flow = []
    # Professional summary
    right_flow.append(Paragraph("Professional Summary", section_h))
    summary_text = core.get('summary') or structured.get('summary_text') or ""
    if not summary_text:
        # build a short summary from placements/projects
        if placements:
            p = placements[0]
            title = p.get('job_title') or ""
            emp = p.get('employer_name') or p.get('industry') or ""
            summary_text = f"Practical experience as {title} at {emp}. Fast learner with employer-facing skills and demonstrated outcomes."
        elif projects:
            pr = projects[0]
            summary_text = f"Portfolio work includes '{pr.get('project_title')}'. Experienced with hands-on media and content production."
        else:
            summary_text = "Motivated candidate with training and placement experience; ready to contribute to employer teams."
    right_flow.append(Paragraph(summary_text, bullet_style))
    right_flow.append(Spacer(1,6))

    # Experience / Placements
    right_flow.append(Paragraph("Experience", section_h))
    if placements:
        for e in placements[:6]:
            title = e.get('job_title') or "Position"
            emp = e.get('employer_name') or e.get('industry') or "Employer"
            dates = ""
            if e.get('start_date'):
                dates = f"{e.get('start_date')}{' - '+str(e.get('end_date')) if e.get('end_date') else ' - present'}"
            wage = e.get('hourly_wage')
            wage_txt = f" · ${wage:.2f}/hr" if isinstance(wage, (int, float)) else ""
            # notes trimmed
            notes = (e.get('notes') or "").replace("\n"," ")
            right_flow.append(Paragraph(f"<b>{title}</b>, {emp} <font size=9 color=grey>({dates}){wage_txt}</font>", bullet_style))
            if notes:
                right_flow.append(Paragraph(f"{notes[:300]}", small))
            right_flow.append(Spacer(1,4))
    else:
        right_flow.append(Paragraph("No recorded placements.", bullet_style))
    right_flow.append(Spacer(1,6))

    # Education
    right_flow.append(Paragraph("Education", section_h))
    if academics:
        for a in academics[:4]:
            school = a.get('school_name') or ""
            grade = a.get('grade_level')
            gpa = a.get('gpa')
            rd = a.get('record_date')
            line = f"<b>{school}</b>"
            meta = []
            if grade:
                meta.append(f"Grade {grade}")
            if gpa:
                meta.append(f"GPA {gpa}")
            if rd:
                meta.append(str(rd))
            if meta:
                line += f" <font size=9 color=grey>({' · '.join(meta)})</font>"
            right_flow.append(Paragraph(line, bullet_style))
    else:
        right_flow.append(Paragraph("—", bullet_style))

    # Put flows into two-column table-like layout using KeepInFrame
    left_kif = KeepInFrame(left_w, 10*inch, left_flow, hAlign='LEFT')
    right_kif = KeepInFrame(right_w, 10*inch, right_flow, hAlign='LEFT')

    main_table = Table([[left_kif, right_kif]], colWidths=[left_w, right_w], hAlign='LEFT')
    main_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))

    elems = [main_table]
    doc.build(elems)

    return output_path

# %%
# 1) Generate structured data (no external model required)
res = generate_student_resume_and_linkedin(student_id="STU-000357",
                                           job_text="content creator / social media producer",
                                           industry="Media/Entertainment",
                                           write_files=False)

structured = res['structured']
# 2) Create a styled resume PDF (file path editable)
out_pdf = build_styled_resume_pdf_from_structured(structured, output_path="STU-000357_styled_resume.pdf", student_display_name="STU-000357")
print("Wrote:", out_pdf)
