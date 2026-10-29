"""
SpellCV - tailors your resume to a job posting, with you in the loop.

Pipeline (each step is an LLM "agent" with its own role):
  1. Job Analyst     -> fetches the posting URL, extracts role, location, keywords
  2. Resume Writer   -> rewrites your resume for the role (truthful, no invented facts)
  3. YOU             -> approve or request changes (loops until approved)
  4. ATS Recruiter   -> scores the approved resume like a recruiter / ATS would
  5. Cover Letter    -> writes the letter from your resume + the JD
  6. PDF Builder     -> one-page resume in the locale's format + cover letter PDF

Run:  streamlit run app.py
"""
import io
import json
import os
import re

import requests
import streamlit as st
from anthropic import Anthropic
from bs4 import BeautifulSoup
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

# ----------------------------------------------------------------------------
# Locale rules: what a recruiter in that market expects on a one-page resume
# ----------------------------------------------------------------------------
LOCALES = {
    "US/CA": dict(page=letter, doc="Resume", date="Mon YYYY",
                  rules="No photo, no age, marital status or nationality. Reverse-chronological. "
                        "Quantified achievement bullets. 1 page. Skip 'References available'."),
    "UK/IE/AU/NZ": dict(page=A4, doc="CV", date="Mon YYYY",
                        rules="Called a CV. No photo or personal details. Short personal profile at top. "
                              "Reverse-chronological. British/Commonwealth spelling."),
    "India": dict(page=A4, doc="Resume", date="Mon YYYY",
                  rules="A4. Include a crisp profile summary, key skills, education with year and "
                        "grade/CGPA when strong. No photo for corporate roles. Include LinkedIn/GitHub."),
    "EU/DE/FR/NL": dict(page=A4, doc="CV", date="MM/YYYY",
                        rules="A4, tabular clarity, dates as MM/YYYY. Include languages with levels (CEFR). "
                              "No photo is generated here; add one manually if the country expects it."),
    "UAE/Gulf": dict(page=A4, doc="CV", date="Mon YYYY",
                     rules="A4. Include nationality/visa status line ONLY if the user's resume has it. "
                           "Professional summary, languages, concise bullets."),
    "Other": dict(page=A4, doc="Resume", date="Mon YYYY",
                  rules="Clean, ATS-friendly, reverse-chronological, 1 page."),
}


# ----------------------------------------------------------------------------
# LLM helpers
# ----------------------------------------------------------------------------
# provider -> (base_url, default model). Anthropic uses its own SDK; the rest are OpenAI-compatible.
PROVIDERS = {
    "Anthropic (paid)": (None, MODEL),
    "Google Gemini (free tier)": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"),
    "Groq (free tier)": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "Ollama (local, free)": ("http://localhost:11434/v1", "llama3.1:8b"),
}


def ask(system: str, user: str, max_tokens: int = 4000) -> str:
    S = st.session_state
    base_url, _ = PROVIDERS[S.provider]
    if base_url is None:
        r = Anthropic(api_key=S.api_key).messages.create(
            model=S.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")
    from openai import OpenAI
    r = OpenAI(api_key=S.api_key or "ollama", base_url=base_url).chat.completions.create(
        model=S.model, max_tokens=min(max_tokens, 3000),  # keeps free-tier per-minute token limits happy
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def ask_json(system: str, user: str, max_tokens: int = 4000) -> dict:
    text = ask(system + "\nReturn ONLY valid JSON. No markdown fences, no commentary.", user, max_tokens)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------
def fetch_job_text(url: str) -> str:
    """Fetch a job page and return readable text. Many boards (LinkedIn, Workday) block
    scrapers or render with JS - the UI offers a paste fallback for those."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SpellCV/1.0)"}
    html = requests.get(url, headers=headers, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    # Prefer schema.org JobPosting JSON-LD if present (very common, clean data)
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            items = data if isinstance(data, list) else [data]
            for it in items:
                if isinstance(it, dict) and it.get("@type") == "JobPosting":
                    desc = BeautifulSoup(it.get("description", ""), "html.parser").get_text("\n")
                    loc = json.dumps(it.get("jobLocation", ""))[:300]
                    return f"{it.get('title','')}\n{it.get('hiringOrganization',{}).get('name','')}\n{loc}\n{desc}"
        except Exception:
            pass
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        t.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
    return text[:15000]


def read_resume(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
    if name.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(file).paragraphs)
    return file.read().decode("utf-8", errors="ignore")


# ----------------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------------
def job_analyst(jd_text: str) -> dict:
    return ask_json(
        "You are a senior technical recruiter analysing a job posting.",
        f"""Extract from this posting:
{{"title": str, "company": str, "location": str,
  "locale": one of {list(LOCALES)},
  "must_have_keywords": [str], "nice_to_have_keywords": [str],
  "responsibilities": [str], "seniority": str, "tone": str}}

POSTING:
{jd_text}""")


RESUME_SCHEMA = """{
 "name": str, "headline": str,
 "contact": {"email": str, "phone": str, "location": str, "links": [str]},
 "summary": str,
 "skills": [{"category": str, "items": [str]}],
 "experience": [{"title": str, "company": str, "location": str, "dates": str, "bullets": [str]}],
 "projects": [{"name": str, "bullets": [str]}],
 "education": [{"degree": str, "school": str, "dates": str, "details": str}],
 "certifications": [str]
}"""


def resume_writer(resume_text: str, job: dict, jd_text: str, feedback: str = "", current: dict | None = None) -> dict:
    loc = LOCALES[job.get("locale", "Other")]
    system = f"""You are an expert resume writer and career coach.
HARD RULES:
- NEVER invent employers, titles, dates, degrees, metrics or skills the candidate does not have.
  You may reword, reorder, merge and emphasise existing facts. If a JD keyword is not supported by
  the resume, do NOT add it.
- Naturally weave in supported JD keywords (exact phrasing where truthful) for ATS matching.
- Must fit ONE page: max ~4 bullets for recent roles, 1-2 for older ones, each bullet <= 22 words,
  start with a strong verb, quantify where the source gives numbers.
- Market format ({job.get('locale')}): {loc['rules']} Date format: {loc['date']}.
- ATS-safe: plain section content, no tables/graphics in the text.
- Use only plain ASCII hyphens (-), straight quotes, and normal spaces. Never use non-breaking hyphens or special Unicode."""
    user = f"""TARGET JOB (analysis): {json.dumps(job)}

JOB DESCRIPTION:
{jd_text[:8000]}

ORIGINAL RESUME:
{resume_text}
"""
    if current and feedback:
        user += f"""
CURRENT DRAFT:
{json.dumps(current)}

CANDIDATE FEEDBACK TO APPLY (follow it exactly unless it requires inventing facts):
{feedback}
"""
    user += f"\nReturn the resume as JSON with this schema:\n{RESUME_SCHEMA}"
    return ask_json(system, user, 5000)


def ats_recruiter(resume: dict, job: dict, jd_text: str) -> dict:
    result = ask_json(
        "You are a veteran recruiter who also knows how ATS parsers (Workday, Greenhouse, Lever, "
        "Taleo) rank candidates. Be honest and calibrated; do not inflate scores.",
        f"""Score this resume against the job. Return:
{{"overall": 0-100,
  "breakdown": {{"keyword_match": 0-100, "role_fit": 0-100, "impact_and_metrics": 0-100,
                 "formatting_ats_safety": 0-100, "readability_skim_test": 0-100}},
  "matched_keywords": [str], "missing_keywords": [str],
  "strengths": [str], "gaps": [str],
  "recruiter_6_second_impression": str,
  "top_fixes": [str],
  "shortlist_likelihood": "Low|Medium|High"}}

JOB: {json.dumps(job)}
JD: {jd_text[:6000]}
RESUME: {json.dumps(resume)}""", 2500)
    # Deterministic cross-check so the score is not purely LLM opinion
    blob = json.dumps(resume).lower()
    musts = job.get("must_have_keywords", [])
    hits = [k for k in musts if k.lower() in blob]
    result["local_keyword_coverage"] = round(100 * len(hits) / len(musts)) if musts else None
    result["local_missing"] = [k for k in musts if k not in hits]
    return result


def cover_letter_writer(resume: dict, job: dict, jd_text: str, user_note: str = "") -> str:
    return ask(
        "You are an expert career coach writing cover letters. Rules: under 300 words, 3-4 short "
        "paragraphs, specific to the company/role, grounded ONLY in facts from the resume, no clichés "
        "('I am writing to apply', 'team player'), confident but humble, end with a clear call to action. "
        "Use the spelling/convention of the target market. Plain text only, no markdown.",
        f"JOB: {json.dumps(job)}\nJD: {jd_text[:5000]}\nRESUME: {json.dumps(resume)}\n"
        f"EXTRA NOTES FROM CANDIDATE: {user_note or 'none'}\n"
        "Write the letter body starting at the greeting and ending with the sign-off and the name.")


# ----------------------------------------------------------------------------
# PDF builders
# ----------------------------------------------------------------------------
_CHAR_FIXES = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-", "\u00ad": "",   # hyphens / minus / soft hyphen
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2002": " ", "\u2003": " ",  # odd spaces
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",                      # zero-width characters
    "\u2192": "->", "\u2190": "<-", "\u21d2": "=>", "\u2022": "\u2022",
    "\u25cf": "\u2022", "\u25aa": "\u2022", "\u2023": "\u2022", "\u2043": "-",
}


def clean(x):
    """Make text safe for the built-in PDF font (Latin-1/WinAnsi). Recurses into dicts and lists."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, str):
        for bad, good in _CHAR_FIXES.items():
            x = x.replace(bad, good)
        return x.encode("cp1252", "ignore").decode("cp1252")  # drops anything the font cannot draw
    return x

def _styles(base: float):
    f = "Helvetica"
    return dict(
        name=ParagraphStyle("name", fontName=f + "-Bold", fontSize=base + 9, leading=base + 11, alignment=TA_CENTER),
        sub=ParagraphStyle("sub", fontName=f, fontSize=base - 0.5, leading=base + 1.5, alignment=TA_CENTER,
                           textColor=colors.HexColor("#333333")),
        h=ParagraphStyle("h", fontName=f + "-Bold", fontSize=base + 0.5, leading=base + 3,
                         textColor=colors.HexColor("#1a1a1a"), spaceBefore=base * 0.5),
        body=ParagraphStyle("body", fontName=f, fontSize=base, leading=base + 2),
        row=ParagraphStyle("row", fontName=f, fontSize=base, leading=base + 2),
        bullet=ParagraphStyle("bullet", fontName=f, fontSize=base, leading=base + 1.8,
                              leftIndent=10, bulletIndent=1),
    )


def _build_resume(r: dict, page, base: float, margin: float):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=page, leftMargin=margin, rightMargin=margin,
                            topMargin=margin * 0.9, bottomMargin=margin * 0.8,
                            title=f"{r.get('name','')} - Resume", author=r.get("name", ""))
    s = _styles(base)
    e = lambda x: escape(str(x or ""))
    story = [Paragraph(e(r.get("name")), s["name"])]
    c = r.get("contact", {})
    bits = [c.get("email"), c.get("phone"), c.get("location"), *c.get("links", [])]
    story.append(Paragraph(" | ".join(e(b) for b in bits if b), s["sub"]))

    def section(title):
        story.append(Paragraph(title.upper(), s["h"]))
        story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#444444"),
                                spaceBefore=0, spaceAfter=base * 0.25))

    if r.get("summary"):
        section("Summary")
        story.append(Paragraph(e(r["summary"]), s["body"]))
    if r.get("skills"):
        section("Skills")
        for g in r["skills"]:
            story.append(Paragraph(f"<b>{e(g.get('category'))}:</b> {e(', '.join(g.get('items', [])))}", s["body"]))
    if r.get("experience"):
        section("Experience")
        for x in r["experience"]:
            story.append(Paragraph(
                f"<b>{e(x.get('title'))}</b>, {e(x.get('company'))}"
                f"{', ' + e(x.get('location')) if x.get('location') else ''} "
                f"<font color='#555555'>({e(x.get('dates'))})</font>", s["row"]))
            for b in x.get("bullets", []):
                story.append(Paragraph(e(b), s["bullet"], bulletText="•"))
            story.append(Spacer(1, base * 0.3))
    if r.get("projects"):
        section("Projects")
        for p in r["projects"]:
            story.append(Paragraph(f"<b>{e(p.get('name'))}</b>", s["row"]))
            for b in p.get("bullets", []):
                story.append(Paragraph(e(b), s["bullet"], bulletText="•"))
    if r.get("education"):
        section("Education")
        for x in r["education"]:
            line = f"<b>{e(x.get('degree'))}</b>, {e(x.get('school'))} <font color='#555555'>({e(x.get('dates'))})</font>"
            story.append(Paragraph(line, s["row"]))
            if x.get("details"):
                story.append(Paragraph(e(x["details"]), s["body"]))
    if r.get("certifications"):
        section("Certifications")
        story.append(Paragraph(e(" • ".join(r["certifications"])), s["body"]))
    doc.build(story)
    return buf.getvalue(), doc.page


def resume_pdf_one_page(r: dict, locale: str) -> bytes:
    """Shrinks font/margins step by step until the resume fits on exactly one page."""
    r = clean(r)
    page = LOCALES[locale]["page"]
    for base in (10.5, 10, 9.5, 9, 8.5, 8, 7.5):
        for margin in (16 * mm, 13 * mm, 11 * mm):
            pdf, pages = _build_resume(r, page, base, margin)
            if pages == 1:
                return pdf
    st.warning("Content is too long for one page even at the smallest size - ask me to shorten it.")
    return pdf


def cover_letter_pdf(text: str, r: dict, job: dict, locale: str) -> bytes:
    text, r = clean(text), clean(r)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LOCALES[locale]["page"], leftMargin=22 * mm, rightMargin=22 * mm,
                            topMargin=22 * mm, bottomMargin=20 * mm, title="Cover Letter")
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=9)
    head = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14, leading=18)
    c = r.get("contact", {})
    story = [Paragraph(escape(r.get("name", "")), head),
             Paragraph(escape(" | ".join(x for x in [c.get("email"), c.get("phone"), c.get("location")] if x)),
                       ParagraphStyle("s", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#444444"))),
             Spacer(1, 14)]
    for para in re.split(r"\n\s*\n", text.strip()):
        story.append(Paragraph(escape(para).replace("\n", "<br/>"), body))
    doc.build(story)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# UI / orchestration (human-in-the-loop state machine)
# ----------------------------------------------------------------------------
def init_state():
    for k, v in dict(stage="input", jd="", job=None, resume_text="", draft=None, history=[],
                     ats=None, letter="", api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                     provider="Anthropic (paid)", model=MODEL).items():
        st.session_state.setdefault(k, v)


def show_resume(r: dict):
    st.markdown(f"### {r.get('name','')}")
    st.caption(r.get("headline", ""))
    st.write(r.get("summary", ""))
    for g in r.get("skills", []):
        st.markdown(f"**{g['category']}:** {', '.join(g['items'])}")
    for x in r.get("experience", []):
        st.markdown(f"**{x['title']}**, {x['company']} _({x['dates']})_")
        for b in x.get("bullets", []):
            st.markdown(f"- {b}")
    for x in r.get("education", []):
        st.markdown(f"**{x['degree']}**, {x['school']} _({x['dates']})_")


def main():
    st.set_page_config(page_title="SpellCV", page_icon="🪄", layout="wide")
    init_state()
    st.title("🪄 SpellCV")
    S = st.session_state

    with st.sidebar:
        prev = S.provider
        S.provider = st.selectbox("AI provider", list(PROVIDERS), list(PROVIDERS).index(S.provider))
        if S.provider != prev:
            S.model = PROVIDERS[S.provider][1]
            S.api_key = ""
        S.api_key = st.text_input("API key (not needed for Ollama)", S.api_key, type="password")
        S.model = st.text_input("Model", S.model)
        if st.button("Start over"):
            for k in list(S.keys()):
                del S[k]
            st.rerun()

    # ---- Stage 1: inputs -----------------------------------------------------
    if S.stage == "input":
        url = st.text_input("Job posting URL")
        pasted = st.text_area("...or paste the job description (use this if the site blocks scraping)", height=120)
        up = st.file_uploader("Your current resume", type=["pdf", "docx", "txt"])
        if st.button("Tailor my resume", type="primary", disabled=not ((S.api_key or S.provider.startswith("Ollama")) and up and (url or pasted))):
            with st.status("Agents working...", expanded=True) as status:
                try:
                    st.write("🔎 Fetching the job posting")
                    S.jd = pasted.strip() or fetch_job_text(url)
                    if len(S.jd) < 300:
                        raise ValueError("Page returned too little text (likely blocked or JS-rendered). "
                                         "Paste the description instead.")
                    S.resume_text = read_resume(up)
                    st.write("🧠 Job Analyst: extracting role, location, keywords")
                    S.job = job_analyst(S.jd)
                    st.write("✍️ Resume Writer: tailoring your resume")
                    S.draft = resume_writer(S.resume_text, S.job, S.jd)
                    S.stage = "review"
                    status.update(label="Draft ready", state="complete")
                except Exception as ex:
                    status.update(label="Failed", state="error")
                    st.error(str(ex))
                    return
            st.rerun()

    # ---- Stage 2: human approval loop ---------------------------------------
    elif S.stage == "review":
        j = S.job
        st.info(f"**{j['title']}** at **{j['company']}** - {j['location']}  |  format: **{j['locale']}**")
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Draft resume - please review")
            show_resume(S.draft)
        with right:
            st.subheader("Your decision")
            st.caption("Must-have keywords: " + ", ".join(j.get("must_have_keywords", [])))
            S.locale = st.selectbox("Format", list(LOCALES), list(LOCALES).index(j.get("locale", "Other")))
            fb = st.text_area("Changes you want (leave empty if happy)", height=140,
                              placeholder="e.g. Move Projects above Education, shorten summary, "
                                          "emphasise the AWS migration, drop the 2016 internship...")
            c1, c2 = st.columns(2)
            if c1.button("🔁 Apply changes", disabled=not fb):
                with st.spinner("Revising..."):
                    S.history.append(fb)
                    S.draft = resume_writer(S.resume_text, S.job, S.jd, fb, S.draft)
                st.rerun()
            if c2.button("✅ Approve", type="primary"):
                with st.spinner("Recruiter scoring + writing cover letter..."):
                    S.job["locale"] = S.locale
                    S.ats = ats_recruiter(S.draft, S.job, S.jd)
                    S.letter = cover_letter_writer(S.draft, S.job, S.jd)
                S.stage = "final"
                st.rerun()

    # ---- Stage 3: score + downloads -----------------------------------------
    elif S.stage == "final":
        a, j = S.ats, S.job
        st.subheader(f"ATS score: {a['overall']}/100  ·  shortlist likelihood: {a['shortlist_likelihood']}")
        cols = st.columns(5)
        for col, (k, v) in zip(cols, a["breakdown"].items()):
            col.metric(k.replace("_", " ").title(), v)
        if a.get("local_keyword_coverage") is not None:
            st.progress(a["local_keyword_coverage"] / 100,
                        text=f"Must-have keyword coverage (exact match): {a['local_keyword_coverage']}%")
        st.markdown(f"**Recruiter's 6-second impression:** {a['recruiter_6_second_impression']}")
        c1, c2 = st.columns(2)
        c1.markdown("**Strengths**\n" + "\n".join(f"- {x}" for x in a["strengths"]))
        c2.markdown("**Gaps / fixes**\n" + "\n".join(f"- {x}" for x in a["top_fixes"]))
        if a.get("local_missing"):
            st.warning("JD keywords not in your resume (add only if true for you): " + ", ".join(a["local_missing"]))

        st.divider()
        st.subheader("Cover letter")
        note = st.text_input("Anything to add or change? (e.g. 'mention I'm relocating in March')")
        if st.button("Rewrite letter") and note:
            S.letter = cover_letter_writer(S.draft, j, S.jd, note)
        S.letter = st.text_area("Edit freely", S.letter, height=300)

        st.divider()
        name = re.sub(r"\W+", "_", S.draft.get("name", "candidate"))
        d1, d2, b = st.columns(3)
        d1.download_button("⬇️ Resume PDF (1 page)", resume_pdf_one_page(S.draft, j["locale"]),
                           f"{name}_Resume.pdf", "application/pdf")
        d2.download_button("⬇️ Cover letter PDF", cover_letter_pdf(S.letter, S.draft, j, j["locale"]),
                           f"{name}_Cover_Letter.pdf", "application/pdf")
        if b.button("← Back to editing resume"):
            S.stage = "review"
            st.rerun()


main()
