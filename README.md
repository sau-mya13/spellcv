# Resume Agent

1. `pip install -r requirements.txt`
2. `export ANTHROPIC_API_KEY=sk-ant-...`   (or paste it in the sidebar)
3. `streamlit run app.py`

Flow: job URL + resume -> tailored draft -> you approve / request changes (loop) ->
recruiter-style ATS score -> editable cover letter -> one-page resume PDF + cover letter PDF.

Notes
- LinkedIn/Workday/Indeed often block scrapers; paste the JD text in that case.
- The writer is instructed never to invent facts. Always read the draft before approving.
- Change the model with `CLAUDE_MODEL=...`.
