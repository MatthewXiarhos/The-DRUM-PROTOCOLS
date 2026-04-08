// ============================================================
// TDP CHAT — DROP-IN FRONTEND SCRIPT  v3 (styled + patched)
// tdp-chat.js
// ============================================================

(function () {
  "use strict";

  var WORKER_URL      = "https://tdp-chat.the-drum-protocols.workers.dev";
  var TURNSTILE_KEY   = "0x4AAAAAAC175HN3K_6IV1fG";
  var QUERY_TOKEN_CAP = 250;
  var EXCLUDE_PATHS   = ["/drum-protocols-admin-v5.html", "/admin"];

  var path = window.location.pathname;
  for (var i = 0; i < EXCLUDE_PATHS.length; i++) {
    if (path.indexOf(EXCLUDE_PATHS[i]) !== -1) return;
  }

  var C = {
    gold       : "#D4A017",
    goldDim    : "#B8860B",
    goldBorder : "rgba(184,134,11,0.35)",
    goldFaint  : "rgba(184,134,11,0.08)",
    goldGlow   : "rgba(212,160,23,0.18)",
    bg         : "#0d1720",
    surface    : "#131e28",
    surface2   : "#1a2633",
    surface3   : "#243345",
    text       : "#e8e4dc",
    text2      : "#9aabb8",
    text3      : "#607080",
    border     : "rgba(255,255,255,0.06)",
    borderMid  : "rgba(255,255,255,0.10)",
    healing    : "#52B788",
    danger     : "#c0392b",
    warn       : "#d4a017",
    mono       : "'DM Mono',monospace",
    sans       : "'DM Sans',system-ui,sans-serif",
    serif      : "'Cormorant Garamond',serif"
  };

  var SPHERE_LOGO_SM = '<svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:20px;height:20px">'
    + '<ellipse cx="10" cy="10" rx="8.5" ry="5" stroke="' + C.gold + '" stroke-width="1.2"/>'
    + '<line x1="1.5" y1="10" x2="18.5" y2="10" stroke="' + C.gold + '" stroke-width="1"/>'
    + '<path d="M5 10 Q6.5 6.5 10 6.5 Q13.5 6.5 15 10 Q13.5 13.5 10 13.5 Q6.5 13.5 5 10Z" stroke="' + C.goldDim + '" stroke-width="0.8" fill="none"/>'
    + '</svg>';

  var SPHERE_LOGO_LG = '<svg viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:28px;height:28px">'
    + '<ellipse cx="14" cy="14" rx="12" ry="7" stroke="' + C.gold + '" stroke-width="1.4"/>'
    + '<line x1="2" y1="14" x2="26" y2="14" stroke="' + C.gold + '" stroke-width="1"/>'
    + '<path d="M7 14 Q9 9 14 9 Q19 9 21 14 Q19 19 14 19 Q9 19 7 14Z" stroke="' + C.goldDim + '" stroke-width="1" fill="none"/>'
    + '<circle cx="14" cy="14" r="1.5" fill="' + C.gold + '"/>'
    + '</svg>';

  var FAB_LOGO = '<svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:26px;height:26px">'
    + '<ellipse cx="16" cy="16" rx="13" ry="8" stroke="' + C.bg + '" stroke-width="1.5"/>'
    + '<line x1="3" y1="16" x2="29" y2="16" stroke="' + C.bg + '" stroke-width="1.2"/>'
    + '<path d="M8 16 Q10.5 10 16 10 Q21.5 10 24 16 Q21.5 22 16 22 Q10.5 22 8 16Z" stroke="' + C.bg + '" stroke-width="1.2" fill="none"/>'
    + '<circle cx="16" cy="16" r="2" fill="' + C.bg + '"/>'
    + '</svg>';

  var CSS = [
    "#tdp-fab{position:fixed;bottom:24px;right:24px;z-index:9998;width:52px;height:52px;border-radius:50%;",
    "background:linear-gradient(135deg,",C.gold,",",C.goldDim,");",
    "border:none;cursor:pointer;box-shadow:0 4px 20px ",C.goldGlow,",0 2px 8px rgba(0,0,0,0.4);display:flex;align-items:center;justify-content:center;transition:transform .15s,box-shadow .15s;}",
    "#tdp-fab:hover{transform:scale(1.07);box-shadow:0 6px 28px ",C.goldGlow,",0 2px 10px rgba(0,0,0,0.5);}",
    "#tdp-panel{position:fixed;bottom:88px;right:24px;z-index:9999;width:430px;max-width:calc(100vw - 32px);height:610px;max-height:calc(100vh - 110px);background:",C.bg,";border:1px solid ",C.goldBorder,";border-radius:16px;display:flex;flex-direction:column;overflow:hidden;font-family:",C.sans,";font-size:14px;box-shadow:0 20px 60px rgba(0,0,0,.7),0 0 0 1px ",C.goldFaint,";transform:translateY(14px) scale(.97);opacity:0;pointer-events:none;transition:transform .2s cubic-bezier(.4,0,.2,1),opacity .2s;}",
    "#tdp-panel.tdp-open{transform:translateY(0) scale(1);opacity:1;pointer-events:all;}",
    "#tdp-hdr{flex-shrink:0;display:flex;align-items:center;justify-content:space-between;padding:11px 14px;border-bottom:1px solid ",C.goldBorder,";background:",C.surface,";border-top:2px solid ",C.gold,";}",
    "#tdp-hdr-l{display:flex;align-items:center;gap:10px;}",
    "#tdp-title{font-size:13px;font-weight:600;color:",C.text,";letter-spacing:.01em;font-family:",C.sans,";}",
    "#tdp-sub{font-size:10px;color:",C.text3,";font-family:",C.mono,";margin-top:1px;letter-spacing:.03em;}",
    "#tdp-close{width:26px;height:26px;border-radius:50%;border:1px solid ",C.border,";background:transparent;color:",C.text3,";cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;transition:all .12s;}",
    "#tdp-close:hover{border-color:",C.goldBorder,";color:",C.gold,";}",
    "#tdp-pipe{flex-shrink:0;display:none;padding:8px 13px;background:",C.surface,";border-bottom:1px solid ",C.border,";}",
    "#tdp-pipe.tdp-vis{display:block;}",
    ".tdp-plbl{font-family:",C.mono,";font-size:9px;color:",C.text3,";letter-spacing:.06em;text-transform:uppercase;margin-bottom:5px;}",
    "#tdp-pgrid{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-bottom:4px;}",
    ".tdp-pc{border:1px solid ",C.border,";border-radius:5px;padding:5px 6px;background:",C.bg,";}",
    ".tdp-pc.run{border-color:rgba(212,160,23,0.4);}.tdp-pc.done{border-color:rgba(82,183,136,0.4);}.tdp-pc.retry{border-color:rgba(212,160,23,0.6);background:rgba(212,160,23,0.05);}.tdp-pc.fail{border-color:rgba(192,57,43,0.4);}",
    ".tdp-pn{font-family:",C.mono,";font-size:8px;color:",C.text3,";}.tdp-pm{font-family:",C.mono,";font-size:8px;color:",C.text2,";white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px;}.tdp-ps{font-size:8px;margin-top:2px;color:",C.text3,";}",
    ".tdp-pc.run .tdp-ps{color:",C.gold,";}.tdp-pc.done .tdp-ps{color:",C.healing,";}.tdp-pc.retry .tdp-ps{color:",C.warn,";}.tdp-pc.fail .tdp-ps{color:",C.danger,";}",
    "#tdp-synth{display:flex;align-items:center;gap:7px;border:1px solid ",C.border,";border-radius:5px;padding:5px 8px;background:",C.bg,";}",
    "#tdp-sl{font-family:",C.mono,";font-size:9px;color:",C.text3,";min-width:68px;letter-spacing:.04em;}#tdp-sm{font-family:",C.mono,";font-size:9px;color:",C.gold,";flex:1;}#tdp-ss{font-family:",C.mono,";font-size:9px;color:",C.text3,";}",
    "#tdp-tier{flex-shrink:0;padding:4px 13px;background:",C.bg,";border-bottom:1px solid ",C.border,";display:none;}#tdp-tier.tdp-vis{display:flex;align-items:center;gap:8px;}",
    ".tdp-tbadge{font-family:",C.mono,";font-size:9px;padding:2px 7px;border-radius:20px;border:1px solid;letter-spacing:.03em;}",
    ".tdp-t-direct{color:",C.healing,";border-color:rgba(82,183,136,0.35);background:rgba(82,183,136,0.06);} .tdp-t-moe{color:",C.gold,";border-color:",C.goldBorder,";background:",C.goldFaint,";} .tdp-t-low{color:",C.warn,";border-color:rgba(212,160,23,0.35);background:rgba(212,160,23,0.05);} .tdp-t-scope{color:",C.text3,";border-color:",C.border,";background:transparent;}",
    ".tdp-tsrc{font-family:",C.mono,";font-size:9px;color:",C.text3,";}",
    "#tdp-msgs{flex:1;overflow-y:auto;padding:13px;display:flex;flex-direction:column;gap:11px;scroll-behavior:smooth;}#tdp-msgs::-webkit-scrollbar{width:3px;}#tdp-msgs::-webkit-scrollbar-thumb{background:rgba(184,134,11,0.25);border-radius:2px;}",
    ".tdp-mu{align-self:flex-end;max-width:78%;background:",C.surface3,";border:1px solid ",C.borderMid,";border-radius:12px 12px 3px 12px;padding:9px 13px;font-size:13px;line-height:1.6;color:",C.text,";white-space:pre-wrap;}",
    ".tdp-ma{align-self:flex-start;max-width:86%;}.tdp-mah{display:flex;align-items:center;gap:7px;margin-bottom:4px;}.tdp-man{font-family:",C.mono,";font-size:10px;color:",C.gold,";letter-spacing:.05em;}.tdp-mam{font-family:",C.mono,";font-size:10px;color:",C.text3,";}",
    ".tdp-mab{background:",C.surface,";border:1px solid ",C.borderMid,";border-radius:3px 12px 12px 12px;padding:11px 14px;font-size:13px;line-height:1.7;color:",C.text,";white-space:pre-wrap;}.tdp-mab.tdp-welcome{border-color:",C.goldBorder,";background:",C.goldFaint,";}.tdp-tip{font-family:",C.mono,";font-size:11px;color:",C.gold,";display:block;margin-top:7px;}",
    ".tdp-azw{align-self:flex-start;max-width:86%;width:100%;}.tdp-azh{font-family:",C.mono,";font-size:9px;color:",C.text3,";letter-spacing:.05em;margin-bottom:4px;}.tdp-azc{border:1px solid ",C.border,";border-radius:6px;margin-bottom:3px;overflow:hidden;}",
    ".tdp-azch{display:flex;align-items:center;justify-content:space-between;padding:5px 9px;background:",C.surface2,";cursor:pointer;user-select:none;}.tdp-azch:hover{background:",C.surface3,";}.tdp-azct{font-family:",C.mono,";font-size:9px;color:",C.text2,";}.tdp-azck{font-family:",C.mono,";font-size:9px;color:",C.text3,";}.tdp-azcv{font-size:9px;color:",C.text3,";transition:transform .12s;}.tdp-azcv.tdp-open{transform:rotate(180deg);} .tdp-azcb{padding:8px 10px;font-size:11.5px;line-height:1.65;color:",C.text2,";white-space:pre-wrap;display:none;max-height:160px;overflow-y:auto;background:",C.bg,";}.tdp-azcb.tdp-open{display:block;}",
    ".tdp-src{margin-top:5px;display:flex;flex-wrap:wrap;gap:4px;}.tdp-srctag{font-family:",C.mono,";font-size:9px;padding:2px 7px;border-radius:20px;border:1px solid ",C.goldBorder,";color:",C.gold,";background:",C.goldFaint,";}",
    ".tdp-fb{display:flex;gap:7px;margin-top:5px;}.tdp-fbb{font-size:13px;background:transparent;border:1px solid ",C.border,";border-radius:6px;padding:3px 7px;cursor:pointer;color:",C.text3,";transition:all .12s;}.tdp-fbb:hover{background:",C.surface2,";color:",C.text,";border-color:",C.borderMid,";}.tdp-fbb.tdp-voted{border-color:",C.goldBorder,";color:",C.gold,";}",
    ".tdp-dots{display:flex;gap:4px;align-items:center;padding:8px 0;}.tdp-dot{width:5px;height:5px;border-radius:50%;background:",C.text3,";animation:tdpBlink 1.2s infinite;}.tdp-dot:nth-child(2){animation-delay:.2s;}.tdp-dot:nth-child(3){animation-delay:.4s;}@keyframes tdpBlink{0%,80%,100%{opacity:.3}40%{opacity:1}}",
    ".tdp-err{align-self:flex-start;max-width:86%;background:rgba(192,57,43,.08);border:1px solid rgba(192,57,43,.3);border-radius:8px;padding:9px 13px;font-size:12.5px;color:#e07060;line-height:1.6;}",
    "#tdp-faq{flex-shrink:0;padding:7px 13px;border-top:1px solid ",C.border,";background:",C.surface,";}#tdp-faql{font-family:",C.mono,";font-size:9px;color:",C.text3,";letter-spacing:.06em;text-transform:uppercase;margin-bottom:5px;}#tdp-faqc{display:flex;flex-wrap:wrap;gap:4px;}.tdp-chip{font-family:",C.mono,";font-size:10px;padding:3px 9px;border-radius:20px;border:1px solid ",C.border,";background:transparent;color:",C.text2,";cursor:pointer;letter-spacing:.01em;transition:all .12s;}.tdp-chip:hover{background:",C.surface2,";color:",C.gold,";border-color:",C.goldBorder,";}",
    "#tdp-inp-area{flex-shrink:0;padding:9px 13px 11px;background:",C.surface,";border-top:1px solid ",C.goldBorder,";}#tdp-tok-row{display:flex;align-items:center;gap:8px;margin-bottom:6px;}#tdp-tok{font-family:",C.mono,";font-size:10px;color:",C.text3,";min-width:92px;transition:color .15s;}#tdp-tok.tdp-warn{color:",C.warn,";}#tdp-tok.tdp-over{color:",C.danger,";}#tdp-tbar{flex:1;height:2px;background:rgba(255,255,255,.08);border-radius:1px;overflow:hidden;}#tdp-tfill{height:100%;width:0%;border-radius:1px;background:",C.gold,";transition:width .15s,background .15s;}#tdp-tcap{font-family:",C.mono,";font-size:10px;color:",C.text3,";}#tdp-inrow{display:flex;gap:7px;align-items:flex-end;}#tdp-in{flex:1;background:",C.bg,";border:1px solid ",C.borderMid,";border-radius:9px;padding:8px 12px;font-family:",C.sans,";font-size:13px;color:",C.text,";resize:none;min-height:40px;max-height:110px;line-height:1.5;outline:none;transition:border-color .15s;}#tdp-in:focus{border-color:",C.goldBorder,";}#tdp-in::placeholder{color:",C.text3,";}#tdp-send{width:36px;height:36px;border-radius:50%;flex-shrink:0;border:none;background:linear-gradient(135deg,",C.gold,",",C.goldDim,");color:",C.bg,";cursor:pointer;font-size:13px;display:flex;align-items:center;justify-content:center;transition:opacity .15s,transform .1s;}#tdp-send:hover:not(:disabled){transform:scale(1.06);}#tdp-send:disabled{opacity:.3;cursor:default;}#tdp-hint{margin-top:5px;font-family:",C.mono,";font-size:9.5px;color:",C.text3,";line-height:1.5;}#tdp-hint span{color:",C.gold,";}#tdp-tswidget{margin-top:5px;}"
  ].join("");

  var HTML = '<button id="tdp-fab" aria-label="Ask Sphere about The DRUM Protocols">'
    + FAB_LOGO
    + '</button>'
    + '<div id="tdp-panel" role="dialog" aria-label="Ask Sphere">'
    +   '<div id="tdp-hdr"><div id="tdp-hdr-l">' + SPHERE_LOGO_LG + '<div><div id="tdp-title">Ask Sphere &middot; The DRUM Protocols</div><div id="tdp-sub">RAG-FIRST &middot; 5 ANALYZERS + SYNTHESIZER</div></div></div><button id="tdp-close" aria-label="Close">&#x2715;</button></div>'
    +   '<div id="tdp-pipe"><div class="tdp-plbl">MOE PIPELINE &middot; RUNNING</div><div id="tdp-pgrid">'
    +       [0,1,2,3,4].map(function(i){
              return '<div class="tdp-pc" id="tdp-pc-'+i+'"><div class="tdp-pn">ANALYZER '+(i+1)+'</div><div class="tdp-pm" id="tdp-pm-'+i+'">&mdash;</div><div class="tdp-ps" id="tdp-ps-'+i+'">waiting</div></div>';
            }).join('')
    +     '</div><div id="tdp-synth"><span id="tdp-sl">SYNTHESIZER</span><span id="tdp-sm">&mdash;</span><span id="tdp-ss">waiting</span></div></div>'
    +   '<div id="tdp-tier"><span id="tdp-tbadge" class="tdp-tbadge"></span><span id="tdp-tsrc" class="tdp-tsrc"></span></div>'
    +   '<div id="tdp-msgs"></div>'
    +   '<div id="tdp-faq"><div id="tdp-faql">suggested &middot; combine all your questions into one</div><div id="tdp-faqc"></div></div>'
    +   '<div id="tdp-inp-area"><div id="tdp-tok-row"><span id="tdp-tok">0 / 250 tokens</span><div id="tdp-tbar"><div id="tdp-tfill"></div></div><span id="tdp-tcap">250 tok limit</span></div><div id="tdp-inrow"><textarea id="tdp-in" rows="2" placeholder="Ask everything at once — what to expect, how to choose, how it works…"></textarea><button id="tdp-send" aria-label="Send">&#9654;</button></div><div id="tdp-hint"><span>&#8593; One rich question = same cost as five simple ones.</span> Each MOE run: 5 analyzers (75 tok each) + synthesizer (250 tok).</div><div id="tdp-tswidget"></div></div>'
    + '</div>';

  var DEFAULT_FAQS = [
    "What are The DRUM Protocols and how do they work?",
    "How does the PRC and phi architecture create regulation?",
    "What is the difference between HEALING, THRIVING, and TRANSFORMING?",
    "How do I choose the right series and intensity level?",
    "How does TDP compare to binaural beats and isochronic tones?",
    "What should I expect during and after a session?",
    "What is the Spectral Clock and why does it matter?",
    "Are there contraindications or precautions I should know?"
  ];

  var open = false;
  var loading = false;
  var tsWidget = null;
  var currentTSToken = null;

  function estimateTokens(text) {
    var w = (text || "").trim().split(/\s+/).filter(function (x) { return x.length > 0; });
    return Math.ceil(w.length * 1.3);
  }
  function el(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function scrollMsgs() { var m = el("tdp-msgs"); if (m) m.scrollTop = m.scrollHeight; }

  function formatCitation(source) {
    var authors = source.authors || "";
    var year = source.year || "";
    var dtype = source.doc_type || "";
    var lastName = authors.split(/\s+et\s+al/i)[0].split("&")[0].trim().split(/\s+/).pop() || authors;
    if (dtype === "tdp_primary") return "Xiarhos TDP " + year;
    return lastName + (year ? " " + year : "");
  }

  function normalizeAnalyzer(a) {
    var usage = a && a.usage ? a.usage : null;
    return {
      model: (a && (a.modelFull || a.model)) || "unknown",
      output: (a && (a.output || a.text || a.error)) || "No output",
      outputTokens: (a && a.outputTokens) || (usage && (usage.output_tokens || usage.completion_tokens)) || 0,
      latencyMs: (a && a.latencyMs) || 0,
      error: a && a.error ? a.error : (!(a && a.ok) ? "failed" : "")
    };
  }

  function normalizeSynth(s) {
    if (!s) return null;
    var usage = s.usage || null;
    return {
      model: s.modelFull || s.model || "unknown",
      outputTokens: s.outputTokens || (usage && (usage.output_tokens || usage.completion_tokens)) || 0
    };
  }

  function init() {
    var font = document.createElement("link");
    font.rel = "stylesheet";
    font.href = "https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Cormorant+Garamond:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap";
    document.head.appendChild(font);

    var style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    var wrap = document.createElement("div");
    wrap.innerHTML = HTML;
    document.body.appendChild(wrap);

    if (TURNSTILE_KEY && !/^YOUR_|^0x000/.test(TURNSTILE_KEY)) {
      var ts = document.createElement("script");
      ts.src = "https://challenges.cloudflare.com/turnstile/v0/api.js";
      ts.async = true;
      ts.defer = true;
      ts.onload = function () {
        if (window.turnstile) {
          tsWidget = window.turnstile.render("#tdp-tswidget", {
            sitekey: TURNSTILE_KEY,
            theme: "dark",
            size: "compact",
            callback: function (t) { currentTSToken = t; },
            "expired-callback": function () { currentTSToken = null; },
            "error-callback": function () { currentTSToken = null; }
          });
        }
      };
      document.head.appendChild(ts);
    }

    bindEvents();
    renderFAQs(DEFAULT_FAQS);
    addWelcome();
  }

  function bindEvents() {
    el("tdp-fab").addEventListener("click", togglePanel);
    el("tdp-close").addEventListener("click", togglePanel);
    el("tdp-send").addEventListener("click", send);
    var inp = el("tdp-in");
    inp.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    });
    inp.addEventListener("input", function () {
      autoResize(inp);
      updateTokenBar(inp.value);
    });
  }

  function togglePanel() {
    open = !open;
    el("tdp-panel").classList.toggle("tdp-open", open);
    if (open) setTimeout(function () { el("tdp-in").focus(); }, 220);
  }

  function autoResize(inp) {
    inp.style.height = "auto";
    inp.style.height = Math.min(inp.scrollHeight, 110) + "px";
  }

  function updateTokenBar(text) {
    var t = estimateTokens(text);
    var pct = Math.min(100, (t / QUERY_TOKEN_CAP) * 100);
    var te = el("tdp-tok"), fill = el("tdp-tfill");
    te.textContent = t + " / " + QUERY_TOKEN_CAP + " tokens";
    te.className = t > QUERY_TOKEN_CAP ? "tdp-over" : t > 200 ? "tdp-warn" : "";
    fill.style.width = pct + "%";
    fill.style.background = t > QUERY_TOKEN_CAP ? C.danger : t > 200 ? C.warn : C.gold;
  }

  function addWelcome() {
    var msgs = el("tdp-msgs");
    var div = document.createElement("div");
    div.className = "tdp-ma";
    div.innerHTML = '<div class="tdp-mah">' + SPHERE_LOGO_SM + '<span class="tdp-man">SPHERE</span><span class="tdp-mam">TDP scope locked &middot; no-harm enforced</span></div>'
      + '<div class="tdp-mab tdp-welcome">'
      + 'Welcome. I’m Sphere — the research and educational voice of The DRUM Protocols.\n\n'
      + 'I can answer your questions about how the protocols work, the neuroscience behind rhythmic entrainment, how to choose the right series and intensity level, what to expect when listening, and how to use the platform.\n\n'
      + '<span class="tdp-tip">➶ Best results: ask everything at once.</span>'
      + 'Each question runs through 5 independent AI analyzers plus a synthesizer. One rich multi-part question gets far better answers than five simple ones at the same cost.'
      + '</div>';
    msgs.appendChild(div);
  }

  function addUserMsg(text) {
    var div = document.createElement("div");
    div.className = "tdp-mu";
    div.textContent = text;
    el("tdp-msgs").appendChild(div);
    scrollMsgs();
  }

  function addTyping(label) {
    var div = document.createElement("div");
    div.id = "tdp-typing";
    div.className = "tdp-ma";
    div.innerHTML = '<div class="tdp-mah">' + SPHERE_LOGO_SM + '<span class="tdp-man">SPHERE</span><span class="tdp-mam" id="tdp-typing-lbl">' + (label || "retrieving from research library…") + '</span></div><div class="tdp-mab"><div class="tdp-dots"><div class="tdp-dot"></div><div class="tdp-dot"></div><div class="tdp-dot"></div></div></div>';
    el("tdp-msgs").appendChild(div);
    scrollMsgs();
  }

  function removeTyping() { var t = el("tdp-typing"); if (t) t.remove(); }

  function showTierBadge(tier, sources) {
    var tierEl = el("tdp-tier"), badge = el("tdp-tbadge"), srcEl = el("tdp-tsrc");
    if (!tierEl) return;
    var labels = {
      direct_answer: ["RAG DIRECT", "tdp-t-direct"],
      rag_direct: ["RAG DIRECT", "tdp-t-direct"],
      rag_augmented: ["RAG + MOE", "tdp-t-moe"],
      rag_augmented_moe: ["RAG + MOE", "tdp-t-moe"],
      moe_no_rag: ["MOE · NO RAG", "tdp-t-low"],
      low_evidence: ["LOW EVIDENCE · MOE", "tdp-t-low"],
      rag_direct_fallback: ["RAG FALLBACK", "tdp-t-low"],
      out_of_scope: ["OUT OF SCOPE", "tdp-t-scope"],
      temporary_unavailable: ["TEMPORARY ISSUE", "tdp-t-scope"]
    };
    var info = labels[tier] || ["SPHERE", "tdp-t-moe"];
    badge.textContent = info[0];
    badge.className = "tdp-tbadge " + info[1];
    if (sources && sources.length) {
      var cited = sources.slice(0, 5).map(function (s) { return formatCitation(s); }).filter(Boolean).join(", ");
      srcEl.textContent = cited ? "Sources: " + cited : "";
    } else {
      srcEl.textContent = "";
    }
    tierEl.classList.add("tdp-vis");
  }

  function addAssistantMsg(answer, analyzers, synth, tier, sources, runId) {
    var msgs = el("tdp-msgs");

    if (analyzers && analyzers.length) {
      var azWrap = document.createElement("div");
      azWrap.className = "tdp-azw";
      var azHTML = '<div class="tdp-azh">5 ANALYZER OUTPUTS · click to expand</div>';
      analyzers.forEach(function (raw, i) {
        var a = normalizeAnalyzer(raw);
        azHTML += '<div class="tdp-azc">'
          + '<div class="tdp-azch" onclick="window._tdpAz(\'' + runId + '\',' + i + ')">'
          + '<span class="tdp-azct">ANALYZER ' + (i + 1) + ' · ' + esc(a.model) + '</span>'
          + '<span class="tdp-azck">' + (a.error ? 'failed' : ((a.outputTokens || 0) + ' tok' + (a.latencyMs ? ' · ' + a.latencyMs + 'ms' : ''))) + '</span>'
          + '<span class="tdp-azcv" id="tdp-azcv-' + runId + '-' + i + '">▾</span>'
          + '</div>'
          + '<div class="tdp-azcb" id="tdp-azcb-' + runId + '-' + i + '">' + esc(a.output) + '</div>'
          + '</div>';
      });
      azWrap.innerHTML = azHTML;
      msgs.appendChild(azWrap);
    }

    var srcHTML = "";
    if (sources && sources.length) {
      srcHTML = '<div class="tdp-src">';
      sources.forEach(function (s) {
        var label = formatCitation(s);
        if (label) srcHTML += '<span class="tdp-srctag">' + esc(label) + '</span>';
      });
      srcHTML += '</div>';
    }

    var synthMeta = normalizeSynth(synth);
    var meta = synthMeta ? ('synthesizer · ' + esc(synthMeta.model) + ' · ' + (synthMeta.outputTokens || 0) + ' tok') : 'sphere · response';

    var div = document.createElement("div");
    div.className = "tdp-ma";
    div.innerHTML = '<div class="tdp-mah">' + SPHERE_LOGO_SM + '<span class="tdp-man">SPHERE</span><span class="tdp-mam">' + meta + '</span></div>'
      + '<div class="tdp-mab">' + esc(answer) + '</div>'
      + srcHTML
      + '<div class="tdp-fb">'
      + '<button class="tdp-fbb" onclick="window._tdpFb(this,\'up\',\'' + runId + '\')">&#128077;</button>'
      + '<button class="tdp-fbb" onclick="window._tdpFb(this,\'dn\',\'' + runId + '\')">&#128078;</button>'
      + '</div>';
    msgs.appendChild(div);
    scrollMsgs();
  }

  function addError(msg) {
    var div = document.createElement("div");
    div.className = "tdp-err";
    div.textContent = msg;
    el("tdp-msgs").appendChild(div);
    scrollMsgs();
  }

  function showPipeline() { el("tdp-pipe").classList.add("tdp-vis"); }
  function hidePipeline() {
    el("tdp-pipe").classList.remove("tdp-vis");
    [0, 1, 2, 3, 4].forEach(function (i) {
      el("tdp-pc-" + i).className = "tdp-pc";
      el("tdp-pm-" + i).innerHTML = "&mdash;";
      el("tdp-ps-" + i).textContent = "waiting";
    });
    el("tdp-sm").innerHTML = "&mdash;";
    el("tdp-ss").textContent = "waiting";
  }

  function setAnalyzer(i, modelFull, state, tokens, attempt) {
    var cell = el("tdp-pc-" + i), mdl = el("tdp-pm-" + i), st = el("tdp-ps-" + i);
    if (!cell) return;
    cell.className = "tdp-pc " + state;
    mdl.textContent = (modelFull || "").split("/").pop().slice(0, 22);
    if (state === "done") st.textContent = '✓ ' + (tokens || '') + ' tok';
    else if (state === "retry") st.textContent = 'retry ' + (attempt || 2) + '…';
    else if (state === "fail") st.textContent = 'fallback';
    else st.textContent = 'running…';
  }

  function setSynth(modelFull, state, tokens) {
    el("tdp-sm").textContent = (modelFull || "").split("/").pop().slice(0, 30);
    el("tdp-ss").textContent = state === "done" ? ('✓ ' + (tokens || '') + ' tok') : state === "running" ? 'synthesizing…' : 'waiting';
  }

  function renderFAQs(chips) {
    var c = el("tdp-faqc");
    c.innerHTML = "";
    chips.slice(0, 6).forEach(function (text) {
      var btn = document.createElement("button");
      btn.className = "tdp-chip";
      btn.textContent = text;
      btn.onclick = function () { el("tdp-in").value = text; updateTokenBar(text); el("tdp-in").focus(); };
      c.appendChild(btn);
    });
  }

  function send() {
    if (loading) return;
    var inp = el("tdp-in"), question = inp.value.trim();
    if (!question) return;

    if (estimateTokens(question) > QUERY_TOKEN_CAP) {
      addError('Your question is ~' + estimateTokens(question) + ' tokens — over the 250-token limit. Combine your questions into one message but keep the total concise.');
      return;
    }

    inp.value = '';
    inp.style.height = 'auto';
    updateTokenBar('');
    loading = true;
    el('tdp-send').disabled = true;

    addUserMsg(question);
    addTyping('retrieving from research library…');

    var payload = { question: question };
    if (currentTSToken) payload.turnstileToken = currentTSToken;

    fetch(WORKER_URL.replace(/\/$/, '') + '/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        var data = res.data;
        removeTyping();

        if (!res.ok) {
          addError(data.message || data.error || 'Something went wrong. Please try again.');
          return;
        }

        if (data.analyzers && data.analyzers.length > 0) {
          showPipeline();
          data.analyzers.forEach(function (raw, i) {
            var a = normalizeAnalyzer(raw);
            setAnalyzer(i, a.model, a.error ? 'fail' : 'done', a.outputTokens, null);
          });
          if (data.synthesizer) {
            var s = normalizeSynth(data.synthesizer);
            setSynth(s.model, 'done', s.outputTokens);
          }
          setTimeout(function () { hidePipeline(); }, 1400);
        }

        showTierBadge(data.tier, data.sources);
        addAssistantMsg(data.answer, data.analyzers, data.synthesizer, data.tier, data.sources, data.runId || String(Date.now()));

        if (window.turnstile && tsWidget !== null) {
          window.turnstile.reset(tsWidget);
          currentTSToken = null;
        }
      })
      .catch(function (err) {
        removeTyping();
        hidePipeline();
        addError('Connection error. Please check your connection and try again.');
        console.error('TDP Chat:', err);
      })
      .finally(function () {
        loading = false;
        el('tdp-send').disabled = false;
      });
  }

  window._tdpAz = function (runId, i) {
    var body = el('tdp-azcb-' + runId + '-' + i), chev = el('tdp-azcv-' + runId + '-' + i);
    if (!body) return;
    var isOpen = body.classList.toggle('tdp-open');
    if (chev) chev.classList.toggle('tdp-open', isOpen);
  };

  window._tdpFb = function (btn, direction, runId) {
    var siblings = btn.parentElement.querySelectorAll('.tdp-fbb');
    siblings.forEach(function (b) { b.classList.remove('tdp-voted'); });
    btn.classList.add('tdp-voted');
    fetch(WORKER_URL.replace(/\/$/, '') + '/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ runId: runId, signal: direction === 'up' ? 1 : -1 })
    }).catch(function () {});
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

}());
