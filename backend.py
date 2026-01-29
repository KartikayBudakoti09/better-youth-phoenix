"""
Cleaned backend logic for Better Youth reports.
Removed: notebook cells, duplicate functions, test code.
Kept: core functions only (student summary, narrative, investor report, PDF).
"""

import os
import re
import json
import requests
from datetime import date, timedelta
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import Databricks SQL connector
try:
    from databricks import sql
except ImportError:
    # Fallback import name
    import databricks.sql as sql


# ============================================================================
# CONFIG & UTILITIES
# ============================================================================

def get_env_var(name: str) -> str:
    """Get environment variable or raise error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return value


def safe_id_literal(student_id):
    """SQL-safe student ID literal."""
    sid = str(student_id).strip()
    if sid.isdigit():
        return sid
    return "'" + sid.replace("'", "''") + "'"


def safe_literal(val):
    """SQL-safe literal value."""
    if val is None:
        return "NULL"
    s = str(val).strip()
    if s.isdigit():
        return s
    return "'" + s.replace("'", "''") + "'"


def date_range_months(months: int):
    """Get (start_date, end_date) ISO strings for N months back."""
    end = date.today()
    start = end - timedelta(days=30 * months)
    return start.isoformat(), end.isoformat()


def query_table(cur, sql_text):
    """Execute query and return list of dicts."""
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


def db_connect():
    """Get Databricks connection."""
    return sql.connect(
        server_hostname=get_env_var("DATABRICKS_HOST"),
        http_path=get_env_var("DATABRICKS_HTTP_PATH"),
        access_token=get_env_var("DATABRICKS_TOKEN")
    )


# ============================================================================
# STUDENT REPORT FUNCTIONS
# ============================================================================

def generate_student_summary(student_id: str, months: int = 6) -> Dict:
    """Query Databricks and return structured summary (attendance, skills, mentoring)."""
    start_date, end_date = date_range_months(months)
    sid = safe_id_literal(student_id)

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
        with db_connect() as conn:
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

                skills = query_table(cur, student_skills_q)
                report["skills"] = skills

                mentoring = query_table(cur, mentoring_q)
                report["mentoring"] = mentoring

    except Exception as e:
        raise RuntimeError(f"Error querying Databricks: {e}")

    return report


def summarize_for_llm(structured_summary: Dict) -> str:
    """Convert structured summary into concise text for LLM."""
    s = structured_summary
    parts = []
    parts.append(f"Student ID: {s.get('student_id')}")
    dr = s.get("date_range", {})
    parts.append(f"Date range: {dr.get('start')} to {dr.get('end')}")
    a = s.get("attendance", {})
    parts.append(f"Attendance: {a.get('total_records',0)} records; Present {a.get('present_count')}; Rate {a.get('attendance_rate_percent')}%")
    
    if a.get("recent"):
        parts.append("Recent attendance (most recent first):")
        for r in a["recent"][:5]:
            parts.append(f"- {r.get('attendance_date')} | {r.get('attendance_status')} | {r.get('participation_level')}")

    parts.append("Skills progress (sample):")
    for sk in s.get("skills", [])[:8]:
        if "current_proficiency_level" in sk:
            parts.append(f"- {sk.get('skill_id') or sk.get('skill_name')}: {sk.get('initial_proficiency_level')} -> {sk.get('current_proficiency_level')} (last: {sk.get('last_assessment_date')})")
        else:
            parts.append(f"- {sk.get('skill_id') or sk.get('skill_name')}: metadata or fallback record")

    parts.append("Recent mentoring sessions (most recent first):")
    for m in s.get("mentoring", [])[:5]:
        notes = (m.get('mentor_notes') or '')[:200]
        parts.append(f"- {m.get('session_date')}: mentor {m.get('mentor_id')}, topics: {m.get('topics_discussed')}, notes: {notes}")

    return "\n".join(parts)


def call_databricks_model(
    prompt: str | None = None,
    messages: list | None = None,
    endpoint_url: str | None = None,
    token: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 700,
    timeout: int = 60
) -> str:
    """Call Databricks model endpoint and return text output."""
    endpoint_url = endpoint_url or os.getenv("DATABRICKS_MODEL_ENDPOINT")
    if not endpoint_url:
        raise RuntimeError("Databricks model endpoint URL not set")

    token = token or os.getenv("DATABRICKS_TOKEN")
    if not token:
        raise RuntimeError("Databricks token not set")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # If chat messages provided, send chat payload first
    if messages is not None:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
        if not resp.ok:
            raise RuntimeError(f"Databricks model call failed; status={resp.status_code}")
        
        try:
            j = resp.json()
        except Exception:
            return resp.text
        
        if isinstance(j, dict):
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
            outputs = j.get("outputs") or j.get("output")
            if outputs:
                if isinstance(outputs, list) and len(outputs) > 0:
                    if isinstance(outputs[0], dict) and "content" in outputs[0]:
                        return outputs[0]["content"]
                    if isinstance(outputs[0], str):
                        return outputs[0]
        if isinstance(j, list) and j:
            if isinstance(j[0], dict):
                for k in ("generated_text", "text", "content"):
                    if k in j[0]:
                        return j[0][k]
            if isinstance(j[0], str):
                return j[0]
        return json.dumps(j, ensure_ascii=False)

    # Fallback single-prompt payloads
    candidate_payloads = [
        {"input": prompt, "temperature": temperature, "max_tokens": max_tokens},
        {"inputs": prompt, "temperature": temperature, "max_tokens": max_tokens},
        {"prompt": prompt, "temperature": temperature, "max_new_tokens": max_tokens},
    ]

    for payload in candidate_payloads:
        try:
            resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
            if resp.ok:
                j = resp.json()
                if isinstance(j, dict):
                    for k in ("result", "text", "generated_text"):
                        if k in j and isinstance(j[k], str):
                            return j[k]
                if isinstance(j, list) and j and isinstance(j[0], str):
                    return j[0]
        except Exception:
            continue

    raise RuntimeError("Model call failed after all attempts")


def generate_narrative_report(student_id: str, months: int = 6, send_to_llm: bool = True) -> str:
    """Generate student progress report via LLM."""
    summary = generate_student_summary(student_id, months=months)
    prompt_body = summarize_for_llm(summary)

    prompt = f"""
You are an educational analyst. Based on the structured summary below, produce a clear, empathetic, evidence-based student progress report (200-500 words) suitable for a mentor.

Include:
- One-paragraph summary of overall progress
- Three bullet points: strengths, areas to improve, recommended actions
- Any notable mentoring highlights

Structured summary:
{prompt_body}

Provide a well-formatted complete report with introduction, body, and conclusion."""

    if send_to_llm:
        messages = [
            {"role": "system", "content": "You are a helpful assistant who writes education progress reports."},
            {"role": "user", "content": prompt}
        ]
        return call_databricks_model(messages=messages)
    
    return "Report generation skipped"


# ============================================================================
# INVESTOR REPORT FUNCTIONS
# ============================================================================

def generate_investor_report(
    program_id: str | None = None,
    months: int = 12,
    call_model: bool = True
) -> Dict:
    """Generate investor-facing report on placements & outcomes."""
    start_date, end_date = date_range_months(months)
    prog_filter = ""
    if program_id:
        prog_literal = safe_literal(program_id)
        prog_filter = f" AND program_connection_id = {prog_literal} "

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

    retention_q = f"""
    SELECT retention_status, COUNT(*) AS cnt
    FROM hackathon.amer.employment_placements
    WHERE start_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
      {prog_filter}
    GROUP BY retention_status ORDER BY cnt DESC LIMIT 20
    """

    industries_q = f"""
    SELECT industry, COUNT(*) AS cnt
    FROM hackathon.amer.employment_placements
    WHERE start_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
      {prog_filter}
    GROUP BY industry ORDER BY cnt DESC LIMIT 10
    """

    outcomes_q = f"""
    SELECT
      om.metric_name,
      AVG(so.value) AS avg_value,
      AVG(CASE WHEN so.target_met THEN 1 ELSE 0 END) AS pct_target_met
    FROM hackathon.amer.student_outcomes so
    LEFT JOIN hackathon.amer.outcome_metrics om ON so.metric_id = om.metric_id
    WHERE so.measurement_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
    {('AND so.program_id = ' + safe_literal(program_id)) if program_id else ''}
    GROUP BY om.metric_name ORDER BY pct_target_met DESC LIMIT 50
    """

    summary = {
        "program_id": program_id,
        "date_range": {"start": start_date, "end": end_date},
        "placements": {},
        "retention_breakdown": [],
        "top_industries": [],
        "outcomes": []
    }

    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(placements_q)
                row = cur.fetchone() or ()
                summary["placements"] = {
                    "total_placements": int(row[0]) if len(row) > 0 and row[0] else 0,
                    "unique_students": int(row[1]) if len(row) > 1 and row[1] else 0,
                    "avg_hourly_wage": float(row[2]) if len(row) > 2 and row[2] else None,
                    "current_count": int(row[3]) if len(row) > 3 and row[3] else 0,
                    "avg_satisfaction": float(row[4]) if len(row) > 4 and row[4] else None
                }

                cur.execute(retention_q)
                summary["retention_breakdown"] = [
                    {"retention_status": r[0], "count": int(r[1])} for r in cur.fetchall()
                ]

                cur.execute(industries_q)
                summary["top_industries"] = [
                    {"industry": r[0], "count": int(r[1])} for r in cur.fetchall()
                ]

                cur.execute(outcomes_q)
                summary["outcomes"] = [
                    {
                        "metric_name": r[0],
                        "avg_value": float(r[1]) if r[1] else None,
                        "pct_target_met": round(100.0 * r[2], 2) if r[2] else None
                    }
                    for r in cur.fetchall()
                ]

    except Exception as e:
        raise RuntimeError(f"Databricks query failed: {e}")

    return {"summary": summary}


def build_investor_narrative_from_summary(summary: Dict) -> str:
    """Build investor narrative from summary data."""
    p = summary.get("placements", {})
    start = summary.get("date_range", {}).get("start")
    end = summary.get("date_range", {}).get("end")
    total = p.get("total_placements", 0)
    unique = p.get("unique_students", 0)
    current = p.get("current_count", 0)
    avg_wage = p.get("avg_hourly_wage")

    industries = summary.get("top_industries", [])
    top_inds = ", ".join(
        [f"{i.get('industry')} ({i.get('count')})" for i in industries[:5]]
    ) if industries else "Various sectors"

    parts = [
        "# Investor Impact Report",
        f"**Reporting period:** {start} to {end}",
        "",
        f"Our programs supported approximately {total:,} placements across {unique:,} individuals. "
        f"Of those placements, {current:,} were still active at reporting time.",
    ]

    if avg_wage:
        parts.append(f"Average starting wage: ${avg_wage:.2f}/hr")

    parts.extend([
        f"Top industries: {top_inds}",
        "",
        "**How investors can help:**",
        "1. Expand employer partnerships to increase sector diversity",
        "2. Fund retention and up-skilling pilots",
        "3. Support data infrastructure for better measurement",
    ])

    return "\n\n".join(parts)


# ============================================================================
# PDF GENERATION
# ============================================================================

def markdown_to_pdf(markdown_text: str, output_path: str = "report.pdf") -> str:
    """Convert markdown text to PDF."""
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
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

        # Headings
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', m.group(2).strip())
            text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
            style_name = 'Heading1Bold' if level == 1 else 'Heading2Bold'
            flowables.append(Paragraph(text, styles[style_name]))
            i += 1
            continue

        # Bullet lists
        if re.match(r'^\s*-\s+', line):
            items = []
            while i < len(lines) and re.match(r'^\s*-\s+', lines[i]):
                item_text = re.sub(r'^\s*-\s+', '', lines[i]).strip()
                item_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', item_text)
                item_text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', item_text)
                items.append(ListItem(Paragraph(item_text, normal), leftIndent=12))
                i += 1
            flowables.append(ListFlowable(items, bulletType='bullet', start='•'))
            continue

        # Regular paragraph
        para_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
        para_text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', para_text)
        flowables.append(Paragraph(para_text, normal))
        i += 1

    doc = SimpleDocTemplate(output_path, rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=48)
    doc.build(flowables)
    return output_path


# ============================================================================
# RESUME & LINKEDIN BUILDER
# ============================================================================

def _keywords_from_text(text: str, min_len: int = 3) -> List[str]:
    """Extract keywords from text."""
    if not text:
        return []
    text = re.sub(r"[^\w\s]", " ", text.lower())
    words = [w for w in text.split() if len(w) >= min_len]
    stop = {"the","and","for","with","from","that","this","their","have","will","are","our","in","on","to","a","an","of"}
    keywords = [w for w in words if w not in stop]
    return list(dict.fromkeys(keywords))


def _join_preview(items: List[str], max_items: int = 6) -> str:
    """Join list items for display."""
    return ", ".join([str(i) for i in items[:max_items]])


def _score_relevance(texts: List[str], keywords: List[str]) -> int:
    """Score text relevance to keywords."""
    if not texts or not keywords:
        return 0
    text = " ".join([t or "" for t in texts]).lower()
    score = 0
    for kw in keywords:
        if kw in text:
            score += 1
    return score


def _filter_passed_certifications(certs: list) -> list:
    """Return only certifications where passed is truthy."""
    if not certs:
        return []
    def passed_flag(c):
        v = c.get("passed") if isinstance(c, dict) else None
        if v is None:
            return False
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.lower() in ("1","true","t","yes","y")
        return bool(v)
    return [c for c in certs if passed_flag(c)]


def fetch_student_core(student_id: str) -> Dict:
    """Fetch core student metadata."""
    sid = safe_id_literal(student_id)
    q = f"""
    SELECT * FROM hackathon.amer.students
    WHERE student_id = {sid}
    LIMIT 1
    """
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                rows = query_table(cur, q)
                return rows[0] if rows else {}
    except Exception as e:
        return {"error": str(e)}


def fetch_academic_records(student_id: str, limit=6) -> List[Dict]:
    """Fetch student academic records."""
    sid = safe_id_literal(student_id)
    q = f"""
    SELECT academic_year, semester, grade_level, school_name, gpa, attendance_rate, record_date, notes
    FROM hackathon.amer.academic_records
    WHERE student_id = {sid}
    ORDER BY record_date DESC
    LIMIT {limit}
    """
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                return query_table(cur, q)
    except Exception:
        return []


def fetch_student_skills(student_id: str) -> List[Dict]:
    """Fetch student skills with proficiency levels."""
    sid = safe_id_literal(student_id)
    q = f"""
    SELECT ss.skill_id, ss.initial_proficiency_level, ss.current_proficiency_level, ss.last_assessment_date, s.skill_name
    FROM hackathon.amer.student_skills ss
    LEFT JOIN hackathon.amer.skills s ON ss.skill_id = s.skill_id
    WHERE ss.student_id = {sid}
    ORDER BY ss.last_assessment_date DESC
    """
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                return query_table(cur, q)
    except Exception:
        return []


def fetch_certifications(student_id: str) -> List[Dict]:
    """Fetch student certifications."""
    sid = safe_id_literal(student_id)
    q = f"""
    SELECT sc.attempt_date, sc.passed, c.certification_name, c.issuing_organization, c.certification_type, sc.notes
    FROM hackathon.amer.student_certifications sc
    LEFT JOIN hackathon.amer.certifications c ON sc.certification_id = c.certification_id
    WHERE sc.student_id = {sid}
    ORDER BY sc.attempt_date DESC
    """
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                return query_table(cur, q)
    except Exception:
        return []


def fetch_media_projects(student_id: str, limit: int = 8) -> list:
    """Fetch media projects linked to student."""
    sid = safe_literal(student_id)
    q1 = f"""
    SELECT project_title, project_type, description, start_date, target_audience, views_count, likes_count
    FROM hackathon.amer.media_projects
    WHERE created_by_student_id = {sid}
    ORDER BY start_date DESC
    LIMIT {limit}
    """
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                return query_table(cur, q1)
    except Exception as e:
        # Try fallback join
        q2 = f"""
        SELECT mp.project_title, mp.project_type, mp.description, mp.start_date, mp.target_audience, mp.views_count, mp.likes_count
        FROM hackathon.amer.media_projects mp
        JOIN hackathon.amer.project_participants pp ON mp.project_id = pp.project_id
        WHERE pp.student_id = {sid}
        ORDER BY mp.start_date DESC
        LIMIT {limit}
        """
        try:
            with db_connect() as conn:
                with conn.cursor() as cur:
                    return query_table(cur, q2)
        except Exception:
            return []


def fetch_employment_placements(student_id: str, limit=6) -> List[Dict]:
    """Fetch student employment placements."""
    sid = safe_id_literal(student_id)
    q = f"""
    SELECT employer_name, industry, job_title, hourly_wage, start_date, end_date, is_current, student_satisfaction, notes
    FROM hackathon.amer.employment_placements
    WHERE student_id = {sid}
    ORDER BY start_date DESC
    LIMIT {limit}
    """
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                return query_table(cur, q)
    except Exception:
        return []


def generate_student_resume_and_linkedin(
    student_id: str,
    job_text: str | None = None,
    industry: str | None = None,
    months_recent: int = 36
) -> Dict:
    """Generate resume markdown and LinkedIn post from structured student data."""
    # Fetch all data
    core = fetch_student_core(student_id)
    academics = fetch_academic_records(student_id)
    skills = fetch_student_skills(student_id)
    certs = fetch_certifications(student_id)
    projects = fetch_media_projects(student_id)
    placements = fetch_employment_placements(student_id)

    # Build keyword set
    job_kw = _keywords_from_text((job_text or "") + " " + (industry or ""))
    
    # Score relevance
    skill_relevance = sorted(skills, key=lambda s: -_score_relevance([s.get("skill_name","")], job_kw))
    project_relevance = sorted(projects, key=lambda p: -_score_relevance([p.get("project_title",""), p.get("description","")], job_kw))
    cert_relevance = sorted(certs, key=lambda c: -_score_relevance([c.get("certification_name",""), c.get("issuing_organization","")], job_kw))
    placement_relevance = sorted(placements, key=lambda e: -_score_relevance([e.get("job_title",""), e.get("notes",""), e.get("industry","")], job_kw))

    # Build resume sections
    profile_lines = []
    name_line = core.get("email") or core.get("student_id") or student_id
    headline = job_text or industry or "Candidate for opportunities"
    
    profile_lines.append(f"# {name_line}")
    profile_lines.append(f"**{headline}**")
    profile_lines.append("")
    
    # Summary
    summary_parts = []
    if job_kw:
        top_k = ", ".join(job_kw[:6])
        summary_parts.append(f"Goal-oriented candidate with experience in {industry or 'relevant sectors'}, focused on {top_k}.")
    else:
        summary_parts.append("Motivated candidate with practical training and placement experience.")
    
    if placement_relevance:
        p = placement_relevance[0]
        if p.get("job_title") and p.get("employer_name"):
            summary_parts.append(f"Most recently placed as {p.get('job_title')} at {p.get('employer_name')}.")
    
    profile_lines.append(" ".join(summary_parts))
    profile_lines.append("")

    # Skills
    profile_lines.append("## Key Skills")
    if skill_relevance:
        skill_items = []
        for s in skill_relevance[:12]:
            nm = s.get("skill_name") or s.get("skill_id")
            prof = s.get("current_proficiency_level")
            prof_txt = f" — {prof}" if prof else ""
            skill_items.append(f"- {nm}{prof_txt}")
        profile_lines.extend(skill_items)
    else:
        profile_lines.append("- Training and professional skills")
    profile_lines.append("")

    # Experience
    profile_lines.append("## Experience")
    if placement_relevance:
        for e in placement_relevance[:6]:
            title = e.get("job_title") or "Position"
            emp = e.get("employer_name") or e.get("industry") or "Employer"
            dates = ""
            if e.get("start_date"):
                dates = f" ({e.get('start_date')}{' - ' + str(e.get('end_date')) if e.get('end_date') else ' - present'})"
            wage = e.get("hourly_wage")
            wage_txt = f" — ${wage:.2f}/hr" if isinstance(wage, (int, float)) else ""
            notes = (e.get("notes") or "").replace("\n", " ")[:220]
            profile_lines.append(f"- **{title}**, {emp}{dates}{wage_txt} — {notes}")
    else:
        profile_lines.append("- No recorded placements")
    profile_lines.append("")

    # Projects
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
        profile_lines.append("- No media projects recorded")
    profile_lines.append("")

    # Certifications
    profile_lines.append("## Certifications")
    passed_certs = _filter_passed_certifications(cert_relevance)
    if passed_certs:
        for c in passed_certs[:8]:
            cname = c.get("certification_name") or ""
            org = c.get("issuing_organization") or ""
            when = c.get("attempt_date")
            profile_lines.append(f"- {cname} ({org}){' — ' + str(when) if when else ''}")
    else:
        profile_lines.append("- No formal certifications recorded")
    profile_lines.append("")

    # Education
    profile_lines.append("## Education")
    if academics:
        for a in academics[:4]:
            school = a.get("school_name") or ""
            grade = a.get("grade_level")
            gpa = a.get("gpa")
            line = f"- {school}"
            if grade:
                line += f", Grade {grade}"
            if gpa:
                line += f" — GPA {gpa}"
            profile_lines.append(line)
    else:
        profile_lines.append("- Academic records not available")
    profile_lines.append("")

    # Contact
    profile_lines.append("## Contact")
    email = core.get("email")
    phone = core.get("phone")
    if email:
        profile_lines.append(f"- Email: {email}")
    if phone:
        profile_lines.append(f"- Phone: {phone}")
    if not email and not phone:
        profile_lines.append("- Contact info not available")

    resume_md = "\n\n".join(profile_lines)

    # LinkedIn post
    linkedin_parts = []
    name_display = core.get("student_id") or "Recent Graduate"
    linkedin_parts.append(f"{name_display} — Seeking opportunities in {industry or (job_text or 'related roles')}.")
    
    key_strengths = []
    if skill_relevance:
        key_strengths.append(skill_relevance[0].get("skill_name") or skill_relevance[0].get("skill_id"))
    if cert_relevance:
        key_strengths.append(cert_relevance[0].get("certification_name"))
    
    linkedin_parts.append("I bring hands-on experience in " + (", ".join([k for k in key_strengths if k]) or "skills and projects") + ", plus real-world placements with employer partners.")
    
    if placement_relevance:
        p = placement_relevance[0]
        linkedin_parts.append(f"Most recently worked as {p.get('job_title')} at {p.get('employer_name') or p.get('industry')}, where I focused on practical deliverables.")
    
    linkedin_parts.append("If your team is hiring or wants to see a portfolio, DM me or email " + (core.get("email") or "[email]") + ".")
    linkedin_md = "\n\n".join(linkedin_parts)

    return {
        "resume_md": resume_md,
        "linkedin_md": linkedin_md,
        "structured": {
            "core": core,
            "academics": academics,
            "skills": skills,
            "certifications": certs,
            "projects": projects,
            "placements": placements
        }
    }

