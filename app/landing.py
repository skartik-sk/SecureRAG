"""Landing page served at / — the resume link should land on something
presentable. Self-contained HTML (inline CSS, no external assets) so it stays
serverless-friendly."""

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SecureRAG — Multi-Workspace RAG Assistant</title>
<meta name="description" content="Hybrid-retrieval RAG assistant with citations: Telegram bot + REST API, powered by LangGraph, pgvector and PostgreSQL.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🛡️%3C/text%3E%3C/svg%3E">
<style>
  :root { --bg:#0b1020; --card:#121a30; --line:#223052; --text:#e8edf7; --dim:#93a1bd; --acc:#5eead4; --acc2:#818cf8; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:radial-gradient(1200px 600px at 80% -10%, #1b2a55 0%, var(--bg) 55%);
         color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
         min-height:100vh; display:flex; align-items:center; justify-content:center; padding:32px 20px; }
  .wrap { max-width:760px; width:100%; }
  .badge { display:inline-flex; align-items:center; gap:8px; font-size:13px; color:var(--dim);
           border:1px solid var(--line); border-radius:999px; padding:6px 14px; margin-bottom:22px; background:rgba(255,255,255,.02); }
  .dot { width:8px; height:8px; border-radius:50%; background:#f59e0b; }
  .dot.ok { background:var(--acc); }
  h1 { font-size:clamp(34px,6vw,52px); letter-spacing:-.02em; }
  h1 .grad { background:linear-gradient(90deg,var(--acc),var(--acc2)); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .tag { color:var(--dim); font-size:clamp(15px,2.4vw,18px); margin:14px 0 30px; line-height:1.6; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin-bottom:30px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 18px; }
  .card b { display:block; margin-bottom:6px; font-size:14.5px; }
  .card span { color:var(--dim); font-size:13.5px; line-height:1.55; }
  .actions { display:flex; gap:12px; flex-wrap:wrap; }
  a.btn { text-decoration:none; font-weight:600; font-size:15px; border-radius:12px; padding:13px 22px; transition:transform .12s ease; }
  a.btn:hover { transform:translateY(-1px); }
  .primary { background:linear-gradient(90deg,#14b8a6,#6366f1); color:#06131a; }
  .ghost { border:1px solid var(--line); color:var(--text); background:rgba(255,255,255,.03); }
  footer { margin-top:34px; color:var(--dim); font-size:12.5px; border-top:1px solid var(--line); padding-top:16px; line-height:1.7; }
  code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; color:var(--acc); }
</style>
</head>
<body>
<div class="wrap">
  <div class="badge"><span class="dot" id="st"></span><span id="stt">checking status…</span></div>
  <h1>Secure<span class="grad">RAG</span></h1>
  <p class="tag">A multi-workspace Retrieval-Augmented Generation assistant.
     Send it documents, ask questions on Telegram or over REST, and get answers
     with <b style="color:var(--text)">inline citations</b> — or a clean refusal when the
     documents can't answer.</p>

  <div class="grid">
    <div class="card"><b>🔎 Hybrid retrieval</b><span>Postgres full-text (BM25-style lexical) + pgvector dense search, fused with Reciprocal Rank Fusion.</span></div>
    <div class="card"><b>🎯 MMR + LLM rerank</b><span>Maximal Marginal Relevance diversification, then LLM relevance grading with query rewriting and retry.</span></div>
    <div class="card"><b>💬 Multi-turn memory</b><span>Agentic LangGraph pipeline backed by a PostgreSQL checkpointer — resume any conversation.</span></div>
    <div class="card"><b>🛡️ Workspace RBAC</b><span>Owner / editor / viewer roles, private &amp; public workspaces, a similarity guardrail against off-topic prompts.</span></div>
    <div class="card"><b>📎 Any document</b><span>PDF, DOCX, PPTX, XLSX, HTML, Markdown — upload via Telegram or the REST API.</span></div>
    <div class="card"><b>☁️ Serverless</b><span>Deployed on Vercel — API-driven LLM &amp; embeddings (Groq, Gemini) over Neon PostgreSQL with pgvector.</span></div>
  </div>

  <div class="actions">
    <a class="btn primary" href="https://t.me/SecureRAG_bot">💬 Chat on Telegram</a>
    <a class="btn ghost" href="/docs">📚 Interactive API docs</a>
    <a class="btn ghost" href="/health">❤️ Health</a>
  </div>

  <footer>
    Stack: <code>Python · FastAPI · LangGraph · pgvector · PostgreSQL · Telegram Bot API · Groq · Vercel</code><br>
    REST base: <code>/api/v1/workspaces</code> — authenticate with <code>X-API-Key</code>.
  </footer>
</div>
<script>
  fetch('/health').then(r=>r.json()).then(()=>{document.getElementById('st').classList.add('ok');document.getElementById('stt').textContent='operational';})
  .catch(()=>{document.getElementById('stt').textContent='status unknown';});
</script>
</body>
</html>"""
