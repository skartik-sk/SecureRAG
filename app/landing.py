"""Landing page served at / — the resume link should land on something
presentable. Design language: crisp, high-signal, editorial — pure white
canvas, hairline zinc borders, obsidian typography, solid black actions,
emerald verification accents. Self-contained HTML (inline CSS, no external
assets) so it stays serverless-friendly."""

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SecureRAG — Multi-Workspace RAG Assistant</title>
<meta name="description" content="Hybrid-retrieval RAG assistant with inline citations: Telegram bot + REST API, powered by LangGraph, pgvector and PostgreSQL.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2309090b' stroke-width='2'%3E%3Cpath d='M12 2l8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6z'/%3E%3C/svg%3E">
<style>
  :root {
    --z950:#09090b; --z900:#18181b; --z800:#27272a; --z500:#71717a; --z400:#a1a1aa;
    --z200:#e4e4e7; --z300:#d4d4d8; --z100:#f4f4f5; --z50:#fafafa;
    --emerald:#10b981; --emerald-bg:#ecfdf5; --emerald-text:#047857; --emerald-border:#a7f3d0;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html { scroll-behavior:smooth; }
  body { background:#ffffff; color:var(--z800);
         font-family:Inter,Geist,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
         -webkit-font-smoothing:antialiased; }
  .container { max-width:1080px; margin:0 auto; padding:0 24px; }

  /* header */
  header { position:sticky; top:0; z-index:50; background:rgba(255,255,255,.9);
           backdrop-filter:blur(12px); border-bottom:1px solid var(--z200); }
  .nav { height:64px; display:flex; align-items:center; justify-content:space-between; }
  .brand { display:flex; align-items:center; gap:10px; font-weight:700; color:var(--z950);
           font-size:16px; letter-spacing:-.01em; text-decoration:none; }
  .nav-right { display:flex; align-items:center; gap:8px; }
  .nav-link { font-size:13.5px; font-weight:500; color:var(--z800); text-decoration:none;
              padding:8px 12px; border-radius:8px; }
  .nav-link:hover { background:var(--z100); color:var(--z950); }
  .btn { display:inline-flex; align-items:center; gap:8px; font-weight:600; font-size:14px;
         border-radius:10px; padding:10px 18px; text-decoration:none; cursor:pointer;
         transition:background .15s ease, border-color .15s ease, transform .12s ease; }
  .btn-black { background:var(--z950); color:#fff; }
  .btn-black:hover { background:var(--z900); }
  .btn-ghost { background:#fff; color:var(--z950); border:1px solid var(--z200); }
  .btn-ghost:hover { background:var(--z50); border-color:var(--z300); }
  .btn-sm { padding:8px 14px; font-size:13px; }

  /* hero */
  .hero { padding:88px 0 56px; text-align:left; }
  .pill { display:inline-flex; align-items:center; gap:8px; font-size:12.5px; font-weight:500;
          color:var(--emerald-text); background:var(--emerald-bg); border:1px solid var(--emerald-border);
          border-radius:999px; padding:5px 14px; margin-bottom:24px; }
  .dot { width:7px; height:7px; border-radius:50%; background:var(--emerald);
         animation:pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { box-shadow:0 0 0 0 rgba(16,185,129,.45);} 50% { box-shadow:0 0 0 5px rgba(16,185,129,0);} }
  h1 { font-size:clamp(36px,6vw,56px); font-weight:700; letter-spacing:-.03em; color:var(--z950);
       line-height:1.05; }
  h1 .dim { color:var(--z400); }
  .sub { font-size:17px; line-height:1.65; color:var(--z500); max-width:640px; margin:20px 0 34px; }
  .sub b { color:var(--z800); font-weight:600; }
  .cta { display:flex; gap:12px; flex-wrap:wrap; }
  .mono-hint { margin-top:18px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
               font-size:12px; color:var(--z500); }

  /* telemetry strip */
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
           border:1px solid var(--z200); border-radius:14px; overflow:hidden; margin:56px 0 0; }
  .stat { padding:18px 22px; background:#fff; border-left:1px solid var(--z200); }
  .stat:first-child { border-left:none; }
  .stat .k { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px;
             text-transform:uppercase; letter-spacing:.08em; color:var(--z400); }
  .stat .v { font-size:17px; font-weight:600; color:var(--z950); margin-top:5px; letter-spacing:-.01em; }

  /* sections */
  section { padding:64px 0 8px; }
  .sec-head { font-size:24px; font-weight:600; letter-spacing:-.02em; color:var(--z900); margin-bottom:8px; }
  .sec-sub { font-size:14.5px; color:var(--z500); margin-bottom:28px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; }
  .card { background:#fff; border:1px solid var(--z200); border-radius:14px; padding:22px;
          transition:border-color .15s ease, transform .15s ease; }
  .card:hover { border-color:var(--z300); transform:translateY(-1px); }
  .card h3 { font-size:15.5px; font-weight:600; color:var(--z900); margin-bottom:8px;
             display:flex; align-items:center; gap:9px; }
  .card p { font-size:13.5px; line-height:1.65; color:var(--z500); }
  .chip { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:10.5px; padding:2px 8px;
          border-radius:999px; border:1px solid var(--z200); color:var(--z500); background:var(--z50); }
  .chip.sky { color:#0284c7; border-color:#bae6fd; background:#f0f9ff; }
  .chip.emerald { color:var(--emerald-text); border-color:var(--emerald-border); background:var(--emerald-bg); }

  /* how it works */
  .steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; }
  .step { border:1px solid var(--z200); border-radius:14px; padding:22px; background:var(--z50); }
  .step .n { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; color:var(--z400); }
  .step h3 { font-size:15.5px; font-weight:600; color:var(--z900); margin:10px 0 7px; }
  .step p { font-size:13.5px; line-height:1.6; color:var(--z500); }

  /* code sample */
  .terminal { background:var(--z100); border:1px solid var(--z200); border-radius:14px;
              padding:20px 22px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
              font-size:12.5px; line-height:1.8; overflow-x:auto; }
  .terminal .c1 { color:var(--z400); } .terminal .c2 { color:#0284c7; } .terminal .c3 { color:var(--z800); }

  /* chat demo */
  .chat { background:#fff; border:1px solid var(--z200); border-radius:14px; overflow:hidden; }
  .chat-head { display:flex; align-items:center; justify-content:space-between; gap:10px;
               padding:13px 18px; border-bottom:1px solid var(--z200); background:var(--z50); }
  .chat-head b { font-size:13.5px; color:var(--z900); }
  .msgs { padding:18px; display:flex; flex-direction:column; gap:10px; min-height:220px;
          max-height:380px; overflow-y:auto; }
  .msg { max-width:82%; padding:10px 14px; border-radius:12px; font-size:13.5px; line-height:1.6; }
  .msg.user { align-self:flex-end; background:var(--z950); color:#fff; border-bottom-right-radius:4px; }
  .msg.bot { align-self:flex-start; background:var(--z50); border:1px solid var(--z200);
             color:var(--z800); border-bottom-left-radius:4px; white-space:pre-wrap; }
  .msg.bot.err { color:#b45309; background:var(--z50); border-color:#fde68a; }
  .msg .src { display:block; margin-top:8px; padding-top:8px; border-top:1px solid var(--z200);
              font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; color:var(--z500); }
  .msg.typing span { display:inline-block; width:6px; height:6px; margin-right:3px; border-radius:50%;
                     background:var(--z400); animation:blink 1.2s infinite; }
  .msg.typing span:nth-child(2) { animation-delay:.2s; } .msg.typing span:nth-child(3) { animation-delay:.4s; }
  @keyframes blink { 0%,80%,100% { opacity:.25; } 40% { opacity:1; } }
  .chat-input { display:flex; gap:10px; padding:14px 18px; border-top:1px solid var(--z200); }
  .chat-input input { flex:1; background:#fff; border:1px solid var(--z200); border-radius:10px;
                      padding:11px 14px; font-size:14px; color:var(--z900); outline:none;
                      font-family:inherit; transition:border-color .15s ease; }
  .chat-input input:focus { border-color:var(--z950); }
  .chat-input input:disabled { background:var(--z100); color:var(--z400); }
  .sugs { display:flex; gap:8px; flex-wrap:wrap; padding:0 18px 14px; }
  .sug { font-size:12px; font-weight:500; color:var(--z800); background:#fff; border:1px solid var(--z200);
         border-radius:999px; padding:6px 13px; cursor:pointer; transition:border-color .15s ease; }
  .sug:hover { border-color:var(--z300); background:var(--z50); }
  .sug:disabled { opacity:.5; cursor:default; }

  /* footer */
  footer { border-top:1px solid var(--z200); margin-top:72px; padding:28px 0 44px; }
  .foot { display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; color:var(--z500); }
  a { color:inherit; }
  @media (max-width:640px) {
    .hero { padding-top:44px; }
    h1 { font-size:30px; }
    .sub { font-size:15px; }
    .stat { border-left:none; border-top:1px solid var(--z200); }
    .stat:first-child { border-top:none; }
    .nav { height:56px; }
    .nav-link { padding:7px 8px; font-size:12.5px; }
    .btn-sm { padding:8px 10px; font-size:12.5px; }
    .msgs { max-height:300px; }
  }
</style>
</head>
<body>

<header>
  <div class="container nav">
    <a class="brand" href="/">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#09090b" stroke-width="2" stroke-linejoin="round"><path d="M12 2l8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6z"/><path d="M9 12l2 2 4-4" stroke="#10b981"/></svg>
      SecureRAG
    </a>
    <nav class="nav-right">
      <a class="nav-link" href="/docs">API Docs</a>
      <a class="nav-link" href="/health">Health</a>
      <a class="btn btn-black btn-sm" href="https://t.me/SecureRAG_bot">Open Telegram Bot</a>
    </nav>
  </div>
</header>

<main class="container">

  <div class="hero">
    <span class="pill"><span class="dot" id="st"></span><span id="stt">operational — live on Vercel</span></span>
    <h1>Ask your documents.<br>Get <span class="dim">cited</span> answers.</h1>
    <p class="sub">SecureRAG is a multi-workspace Retrieval-Augmented Generation assistant.
       Upload PDFs, DOCX, spreadsheets or slides — then ask questions over <b>Telegram</b> or a
       <b>REST API</b>. Every answer carries inline citations, mapped back to file and section;
       if the documents can't answer, it says so.</p>
    <div class="cta">
      <a class="btn btn-black" href="https://t.me/SecureRAG_bot">Chat with the bot →</a>
      <a class="btn btn-ghost" href="/docs">Explore the API</a>
    </div>
    <p class="mono-hint">POST /api/v1/workspaces/{slug}/chat &nbsp;·&nbsp; authenticate with X-API-Key</p>

    <div class="stats">
      <div class="stat"><div class="k">Retrieval</div><div class="v">Hybrid · RRF · MMR</div></div>
      <div class="stat"><div class="k">Pipeline</div><div class="v">Agentic LangGraph</div></div>
      <div class="stat"><div class="k">Memory</div><div class="v">Multi-turn, resumable</div></div>
      <div class="stat"><div class="k">Access</div><div class="v">RBAC workspaces</div></div>
    </div>
  </div>

  <section>
    <div class="sec-head">How it works</div>
    <div class="sec-sub">Three steps from raw documents to cited answers.</div>
    <div class="steps">
      <div class="step"><div class="n">01</div><h3>Ingest</h3><p>Send a file to the Telegram bot or POST it to the API. Parsers extract structure, a splitter chunks by headings, and an embedding model indexes every chunk into pgvector.</p></div>
      <div class="step"><div class="n">02</div><h3>Retrieve</h3><p>Each question runs hybrid search — BM25-style lexical scoring and pgvector dense similarity, fused with Reciprocal Rank Fusion and diversified with MMR.</p></div>
      <div class="step"><div class="n">03</div><h3>Answer</h3><p>The agent grades retrieved chunks, rewrites and retries when retrieval is thin, generates with [n] citations — or refuses honestly when the corpus can't answer.</p></div>
    </div>
  </section>

  <section>
    <div class="sec-head">Built like a production system</div>
    <div class="sec-sub">Not a notebook demo — a deployed service with tests, roles and guardrails.</div>
    <div class="grid">
      <div class="card"><h3>Hybrid retrieval <span class="chip sky">RRF</span></h3><p>Postgres full-text search fused with pgvector dense vectors via Reciprocal Rank Fusion; Maximal Marginal Relevance keeps answers diverse, not repetitive.</p></div>
      <div class="card"><h3>Citations you can verify <span class="chip emerald">[n]</span></h3><p>Every claim maps to a source file and section. Cited numbers are extracted from the answer and rendered as a Sources block in Telegram and the API.</p></div>
      <div class="card"><h3>Guardrail against off-topic</h3><p>A similarity-score gate refuses questions the corpus can't support — one rewrite-and-retry pass first, then an honest refusal instead of hallucination.</p></div>
      <div class="card"><h3>Workspace RBAC</h3><p>Owner / editor / viewer roles with public and private workspaces. Private collections gate viewing, ingestion and management separately; invites are owner-gated.</p></div>
      <div class="card"><h3>Multi-turn memory</h3><p>Conversations persist in PostgreSQL via a LangGraph checkpointer — pronoun-aware query rewriting uses history, and /resume continues any past thread.</p></div>
      <div class="card"><h3>Serverless by design <span class="chip">Vercel</span></h3><p>LLM and embeddings are pure API calls; startup is lazy and idempotent; ingestion runs inline per request. Zero persistent local state beyond PostgreSQL.</p></div>
    </div>
  </section>

  <section>
    <div class="sec-head">Try the API</div>
    <div class="sec-sub">Any HTTP client works. Full schema in the <a href="/docs" style="color:#0284c7">interactive docs</a>.</div>
    <div class="terminal">
<span class="c1"># ask a workspace a question</span><br>
<span class="c2">curl</span> -X POST https://securerag.vercel.app/api/v1/workspaces/delivery-policy/chat <br>
&nbsp;&nbsp;-H <span class="c3">"X-API-Key: $KEY"</span> -H <span class="c3">"Content-Type: application/json"</span> <br>
&nbsp;&nbsp;-d <span class="c3">'{"message": "What is the delivery SLA for Zone B?"}'</span><br><br>
<span class="c1"># → {"answer": "Zone B: 2–3 business days [1]", "sources": [{"file": "delivery_policy.md", "section": "SLA"}]}</span>
    </div>
  </section>

  <section>
    <div class="sec-head">Try it live</div>
    <div class="sec-sub">This is the real pipeline — hybrid retrieval, reranking and the citation
       guardrail — answering from the seeded <b style="color:var(--z800)">delivery-policy</b> workspace.
       On Telegram it works the same way with your own documents.</div>
    <div class="chat">
      <div class="chat-head">
        <b>SecureRAG · delivery-policy</b>
        <span class="chip emerald" style="font-size:10.5px; padding:3px 10px;">● online</span>
      </div>
      <div class="msgs" id="msgs">
        <div class="msg bot">Hi! Ask me anything about the delivery policy — try a suggestion below.
Answers include [n] citations; if the documents can't answer, I'll tell you.</div>
      </div>
      <div class="sugs" id="sugs">
        <button class="sug">What is the delivery SLA for Zone B?</button>
        <button class="sug">Which goods are classified as special goods?</button>
        <button class="sug">Who won the football world cup?</button>
      </div>
      <div class="chat-input">
        <input id="q" type="text" maxlength="500" placeholder="Ask about the delivery policy…"
               autocomplete="off">
        <button class="btn btn-black" id="send">Ask</button>
      </div>
    </div>
  </section>

</main>

<footer>
  <div class="container foot">
    <div>Python · FastAPI · LangGraph · pgvector · PostgreSQL · Telegram Bot API · Groq · Gemini</div>
    <div>SecureRAG — built by <b style="color:var(--z800)">Singupalli Kartik</b></div>
  </div>
</footer>

<script>
  fetch('/health').then(r => r.json())
    .then(() => { document.getElementById('stt').textContent = 'operational — live on Vercel'; })
    .catch(() => { document.getElementById('stt').textContent = 'status unknown';
                   document.getElementById('st').style.background = '#f59e0b';
                   document.getElementById('st').style.animation = 'none'; });

  // --- live demo chat (ephemeral: in-memory session, nothing persisted) ---
  (function () {
    const msgs = document.getElementById('msgs'), q = document.getElementById('q'),
          send = document.getElementById('send'), sugs = document.getElementById('sugs');
    let sid = sessionStorage.getItem('securerag-demo-sid') || '';
    function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
    function render(text) { return esc(text).replace(/\\*\\*([^*\\n]+)\\*\\*/g, '<b>$1</b>'); }
    function bubble(cls, html) {
      const d = document.createElement('div'); d.className = 'msg ' + cls; d.innerHTML = html;
      msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight; return d;
    }
    let busy = false;
    async function ask(text) {
      text = (text || '').trim();
      if (!text || busy) return;
      busy = true; q.value = ''; q.disabled = true; send.disabled = true;
      [...sugs.children].forEach(b => b.disabled = true);
      bubble('user', esc(text));
      const typing = bubble('bot typing', '<span></span><span></span><span></span>');
      try {
        const r = await fetch('/api/v1/demo/chat', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, session_id: sid || null })
        });
        const data = await r.json();
        if (data.session_id) { sid = data.session_id; sessionStorage.setItem('securerag-demo-sid', sid); }
        typing.className = 'msg bot';
        if (!r.ok) { typing.classList.add('err'); typing.textContent = data.detail || 'Something went wrong.'; }
        else {
          typing.innerHTML = render(data.answer) + (data.refused ? '' :
            (data.sources || []).map((s, i) =>
              `<span class="src">[${i + 1}] ${esc(s.file || 'unknown')}${s.section ? ' — ' + esc(s.section) : ''}</span>`).join(''));
        }
      } catch (e) {
        typing.className = 'msg bot err'; typing.textContent = 'Network error — try again.';
      }
      busy = false; q.disabled = false; send.disabled = false;
      [...sugs.children].forEach(b => b.disabled = false); q.focus();
    }
    send.addEventListener('click', () => ask(q.value));
    q.addEventListener('keydown', e => { if (e.key === 'Enter') ask(q.value); });
    [...sugs.children].forEach(b => b.addEventListener('click', () => ask(b.textContent)));
  })();
</script>
</body>
</html>"""
