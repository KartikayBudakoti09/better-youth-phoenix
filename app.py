"""
Lightweight Streamlit UI for Better Youth Reports.
Tabs: Student Progress Report | Investor Impact Report | About
"""

import os
import sys
import streamlit as st
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file FIRST, before any backend imports
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# Verify Databricks credentials are loaded
if not os.getenv('DATABRICKS_HOST'):
    st.error('❌ DATABRICKS_HOST not found in .env file')
    st.stop()

from backend import (
    generate_student_summary,
    generate_narrative_report,
    generate_investor_report,
    build_investor_narrative_from_summary,
    generate_student_resume_and_linkedin,
    markdown_to_pdf,
)

st.set_page_config(page_title="Better Youth Reports", layout="wide", initial_sidebar_state="expanded")

st.title("🎓 Better Youth - Report Generator")
st.markdown("Generate student progress reports and investor impact summaries from Databricks.")

# Sidebar navigation
page = st.sidebar.radio(
    "Select Report Type",
    ["📊 Student Progress", "💰 Investor Impact", "🎓 Resume Builder", "ℹ️ About"],
    label_visibility="visible"
)

# ============================================================================
# PAGE: STUDENT PROGRESS REPORT
# ============================================================================

if page == "📊 Student Progress":
    st.header("Student Progress Report")
    st.markdown("Generate a comprehensive progress report for a student based on attendance, skills, and mentoring data.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        student_id = st.text_input("Student ID", value="STU-000357", help="Enter the student identifier")
    with col2:
        months = st.slider("Months to Review", 1, 24, value=6, help="Lookback window in months")

    if st.button("🚀 Generate Report", key="student_gen", use_container_width=True):
        try:
            with st.spinner("📖 Fetching data and generating report..."):
                narrative = generate_narrative_report(student_id, months=months, send_to_llm=True)
            
            st.success("✅ Report generated successfully!")
            st.markdown(narrative)
            
            # PDF export
            col1, col2 = st.columns(2)
            with col1:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf_path = markdown_to_pdf(narrative, tmp.name)
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📥 Download as PDF",
                            data=pdf_file.read(),
                            file_name=f"{student_id}_progress_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    os.unlink(pdf_path)
            with col2:
                st.info(f"📋 Report for: **{student_id}** | Period: **{months} months**")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# ============================================================================
# PAGE: INVESTOR IMPACT REPORT
# ============================================================================

elif page == "💰 Investor Impact":
    st.header("Investor Impact Report")
    st.markdown("Review placement metrics, retention data, and program outcomes for investor stakeholders.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        program_id = st.text_input("Program ID (optional)", value="", help="Leave blank for all programs")
    with col2:
        months = st.slider("Months to Review", 1, 36, value=12, help="Lookback window in months")

    if st.button("🚀 Generate Report", key="investor_gen", use_container_width=True):
        try:
            with st.spinner("📊 Querying data and building narrative..."):
                report = generate_investor_report(
                    program_id=program_id if program_id else None,
                    months=months,
                    call_model=False
                )
            
            summary = report.get("summary", {})
            
            with st.spinner("✍️ Building narrative..."):
                narrative = build_investor_narrative_from_summary(summary)
            
            st.success("✅ Report generated successfully!")
            st.markdown(narrative)
            
            # Key Metrics Display
            st.subheader("📈 Key Metrics")
            placements = summary.get("placements", {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Placements", placements.get("total_placements", 0))
            with col2:
                st.metric("Unique Students", placements.get("unique_students", 0))
            with col3:
                wage = placements.get("avg_hourly_wage")
                st.metric("Avg Wage/hr", f"${wage:.2f}" if wage else "N/A")
            with col4:
                sat = placements.get("avg_satisfaction")
                st.metric("Avg Satisfaction", f"{sat:.1f}/10" if sat else "N/A")
            
            # Retention breakdown
            if summary.get("retention_breakdown"):
                st.subheader("Retention Breakdown")
                ret_data = summary.get("retention_breakdown", [])
                ret_dict = {r["retention_status"]: r["count"] for r in ret_data}
                st.bar_chart(ret_dict)
            
            # Top industries
            if summary.get("top_industries"):
                st.subheader("Top Industries by Placements")
                ind_data = summary.get("top_industries", [])
                ind_dict = {i["industry"]: i["count"] for i in ind_data[:10]}
                st.bar_chart(ind_dict)
            
            # PDF export
            col1, col2 = st.columns(2)
            with col1:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf_path = markdown_to_pdf(narrative, tmp.name)
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📥 Download as PDF",
                            data=pdf_file.read(),
                            file_name=f"investor_report_{program_id or 'ALL'}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    os.unlink(pdf_path)
            with col2:
                st.info(f"📊 Program: **{program_id or 'ALL PROGRAMS'}** | Period: **{months} months**")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# ============================================================================
# PAGE: RESUME BUILDER
# ============================================================================

elif page == "🎓 Resume Builder":
    st.header("Student Resume & LinkedIn Builder")
    st.markdown("Generate a professional resume and LinkedIn post tailored to a specific job or industry.")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        student_id = st.text_input("Student ID", value="STU-000357", help="Enter the student identifier")
    with col2:
        job_text = st.text_input("Job/Role Target", value="", help="e.g., 'Data Analyst' or leave blank")
    with col3:
        industry = st.text_input("Industry Focus", value="", help="e.g., 'Tech' or 'Finance'")

    if st.button("🚀 Generate Resume & LinkedIn Post", key="resume_gen", use_container_width=True):
        try:
            with st.spinner("📋 Fetching student data..."):
                result = generate_student_resume_and_linkedin(
                    student_id,
                    job_text=job_text if job_text else None,
                    industry=industry if industry else None
                )
            
            st.success("✅ Resume and LinkedIn post generated!")
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📄 Resume", "🔗 LinkedIn Post", "📥 Downloads"])
            
            with tab1:
                st.subheader("Professional Resume (Markdown)")
                resume_md = result.get("resume_md", "")
                st.markdown(resume_md)
                st.text_area("Copy resume text:", value=resume_md, height=300, disabled=True, key="resume_copy")
            
            with tab2:
                st.subheader("LinkedIn Post")
                linkedin_md = result.get("linkedin_md", "")
                st.markdown(linkedin_md)
                st.text_area("Copy LinkedIn post:", value=linkedin_md, height=200, disabled=True, key="linkedin_copy")
            
            with tab3:
                st.subheader("Download Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Resume markdown download
                    st.download_button(
                        label="📄 Download Resume (Markdown)",
                        data=resume_md,
                        file_name=f"{student_id}_resume.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with col2:
                    # LinkedIn post download
                    st.download_button(
                        label="🔗 Download LinkedIn Post (Text)",
                        data=linkedin_md,
                        file_name=f"{student_id}_linkedin_post.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # PDF Resume
                st.markdown("---")
                st.subheader("Resume PDF")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    try:
                        pdf_path = markdown_to_pdf(resume_md, tmp.name)
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="📥 Download Resume PDF",
                                data=pdf_file.read(),
                                file_name=f"{student_id}_resume.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        os.unlink(pdf_path)
                    except Exception as e:
                        st.warning(f"Could not generate PDF: {e}")
                
                st.info(f"📊 Resume for: **{student_id}** | Target: **{job_text or industry or 'General'}**")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# ============================================================================
# PAGE: ABOUT
# ============================================================================

else:  # About
    st.header("About Better Youth Reports")
    
    st.markdown("""
    ### 📌 Overview
    Better Youth Reports generates comprehensive insights from your Databricks data warehouse. 
    Generate professional reports tailored for mentors, program teams, and investors.
    
    ---
    
    ### 🎯 Features
    
    **Student Progress Reports**
    - Attendance tracking and rates
    - Skills proficiency progression
    - Mentoring session summaries
    - LLM-powered narrative analysis
    - PDF export for distribution
    
    **Investor Impact Reports**
    - Employment placement metrics
    - Industry distribution analysis
    - Retention tracking
    - Outcome metric summaries
    - Visualization-ready data
    - Professional narrative
    
    ---
    
    ### ⚙️ Configuration
    
    Ensure these environment variables are set in your `.env` file:
    
    ```
    DATABRICKS_HOST=<your-host>
    DATABRICKS_HTTP_PATH=<your-http-path>
    DATABRICKS_TOKEN=<your-token>
    DATABRICKS_MODEL_ENDPOINT=<optional-endpoint-for-llm>
    ```
    
    ---
    
    ### 🚀 Getting Started
    
    1. **Install dependencies**: `pip install -r requirements.txt`
    2. **Set environment variables**: Create `.env` with your Databricks credentials
    3. **Run the app**: `streamlit run app.py`
    4. **Generate reports**: Use the tabs above to create student or investor reports
    5. **Export PDFs**: Download formatted reports for distribution
    
    ---
    
    ### 📊 Data Sources
    
    Reports are built from Databricks tables in the `hackathon.amer` schema:
    - `students` — Core student metadata
    - `attendance` — Session attendance records
    - `student_skills` — Skill proficiency tracking
    - `mentoring_sessions` — Mentor interaction logs
    - `employment_placements` — Job placement and retention data
    - `student_outcomes` — Outcome metric measurements
    
    ---
    
    ### 💡 Tips
    
    - **Student ID Format**: Typically `STU-000357` or similar numeric identifiers
    - **Program ID**: Optional; leave blank to aggregate across all programs
    - **Lookback Window**: Use 6–12 months for meaningful trends
    - **PDF Export**: Reports are automatically formatted for printing and sharing
    
    ---
    
    ### ❓ Questions?
    
    Contact the Better Youth team or check the [documentation](https://better-youth.org).
    """)
    
    with st.expander("🔧 System Information"):
        st.write(f"**Python Version**: {os.sys.version}")
        st.write(f"**Streamlit Version**: {st.__version__}")
        try:
            from backend import get_env_var
            host = get_env_var("DATABRICKS_HOST") if os.getenv("DATABRICKS_HOST") else "❌ Not set"
            st.write(f"**Databricks Host**: {host[:30]}..." if host != "❌ Not set" else f"**Databricks Host**: {host}")
        except:
            st.write("**Databricks Host**: ❌ Not configured")
