#!/usr/bin/env python3
"""Rebuild RELATED_WORK.html with a proper reading layout."""
import json, re, html as H, pathlib, difflib

ROOT = pathlib.Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
src = (ROOT / ".related_work_source.html").read_text()  # immutable source
papers = json.load(open(ROOT / "outputs/metrics/litreview_data.json"))["papers"]

# ---------------------------------------------------------------- clean LaTeX
def clean(t: str) -> str:
    t = t.replace("\\ ", " ").replace("\\,", "\u202f")
    t = t.replace("---", "\u2014").replace("--", "\u2013")
    t = re.sub(r"\\emph\{(.*?)\}", r"<em>\1</em>", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    return re.sub(r"[ \t]{2,}", " ", t)

# ---------------------------------------------------------------- parse source
intro = clean(re.search(r'click any \[n\].*?</p>\s*<p>(.*?)</p>', src, re.S).group(1))
secs = []
for m in re.finditer(r'<h2>(\d+)&nbsp;&nbsp;(.*?)</h2>(.*?)(?=<h2>|<div class="refs">)', src, re.S):
    body = "".join(clean(p) for p in re.findall(r'<p>(.*?)</p>', m.group(3), re.S))
    paras = [clean(p) for p in re.findall(r'<p>(.*?)</p>', m.group(3), re.S)]
    secs.append({"n": m.group(1), "title": clean(m.group(2)), "paras": paras})

refs = []
for m in re.finditer(
    r'<li id="ref-(\d+)"><span class="rn">\[\d+\]</span>\s*(.*?)\s*"(.*?),"\s*<em>(.*?)</em>\s*(\d{4})\.\s*<a href="([^"]+)"', src, re.S):
    refs.append({"n": int(m.group(1)), "authors": clean(m.group(2)).strip(),
                 "title": clean(m.group(3)), "venue": clean(m.group(4)),
                 "year": m.group(5), "url": m.group(6)})

# strip meta-commentary that leaked into venue strings
for r in refs:
    r["venue"] = re.sub(r"\s*NOTE:.*", "", r["venue"], flags=re.S).strip(" .;")
    r["venue"] = re.sub(r"\s*\(open access\).*", "", r["venue"]).strip(" .;")
    if len(r["venue"]) > 150:
        r["venue"] = r["venue"][:147].rsplit(" ", 1)[0] + "…"

# ------------------------------------------------- match refs -> json metadata
def norm(s): return re.sub(r"[^a-z0-9]", "", H.unescape(s).lower())
by_title = {norm(p["title"]): p for p in papers}
for r in refs:
    key = norm(r["title"])
    hit = by_title.get(key)
    if not hit:
        c = difflib.get_close_matches(key, list(by_title), n=1, cutoff=0.72)
        hit = by_title[c[0]] if c else None
    r["meta"] = hit

# --------------------------------------------------------- compact table facts
def first_sentence(t, cap=110):
    if not t: return "—"
    s = re.split(r"(?<=[.;])\s", t.strip())[0]
    return (s[:cap].rsplit(" ", 1)[0] + "…") if len(s) > cap else s

# Validation scheme per reference — MANUALLY VERIFIED by reading each paper's
# methodology + limitations text (an automated keyword match is unsafe here:
# several papers say "NOT subject-independent LOSO", which naive matching
# would score as a positive).  ref-number -> (label, css class)
VALIDATION = {
  1:  ("no classifier",        "na"),    # Seifert 2020 — parameter validation study
  2:  ("review",               "na"),    # Gurbuz 2024 — survey
  3:  ("not stated",           ""),      # Alanazi 2022
  4:  ("no classifier",        "na"),    # López-Delgado 2026 — the dataset paper
  5:  ("random split",         "warn"),  # Hayashi 2021 — hold-out 7:3, explicitly not LOSO
  6:  ("not stated",           ""),      # Hoshiga 2021 — simulation
  7:  ("not stated",           ""),      # Papanastasiou 2020
  8:  ("subject-dependent",    "warn"),  # Ni & Huang 2020 — test shares subjects
  9:  ("subject-independent",  "ok"),    # Fard 2026 — leave-one-person-out
  10: ("k-fold",               "warn"),  # Nguyen 2024
  11: ("random split",         "warn"),  # Ha 2023 — random 72/8/20
  12: ("sample-level split",   "warn"),  # Seyfioğlu 2018 — not strict LOSO
  13: ("subject-independent",  "ok"),    # Park 2016 — each fold = one subject
  14: ("single-subject test",  "warn"),  # Erol 2020 — no cross-subject validation
  15: ("random split",         "warn"),  # Cai 2023 STRIDE — explicitly not LOSO
  16: ("k-fold",               "warn"),  # PD gait 2025 — 10-fold, not subject-independent
}

def validation_flag(refnum):
    return VALIDATION.get(refnum, ("not stated", ""))

def short_n(p):
    if not p: return "—"
    t = p.get("sample_size") or ""
    m = re.search(r"(\d[\d,]*)\s*(participants|subjects|people|volunteers)", t, re.I)
    if m: return m.group(1)
    m = re.search(r"^\D{0,24}(\d[\d,]*)", t)
    return m.group(1) if m else "—"

TAKEAWAY = {
 "1": "Radar is contactless, lighting-independent and privacy-preserving — which is exactly why it is attractive for continuous in-home gait monitoring.",
 "2": "Radar measures rhythm and pace reliably, but measures <strong>gait variability</strong> — a hallmark of parkinsonian gait — only poorly. The dataset paper for this thesis stops at parameter measurement and contains <strong>no classifier</strong>.",
 "3": "Two model families dominate: CNNs on spectrograms and RNNs on velocity envelopes. Crucially, micro-Doppler identifies <strong>individuals</strong> at &gt;93% — which dictates how disease classifiers must be validated.",
 "4": "Transfer learning and GAN augmentation are the field's two standard answers to small radar datasets — both directly relevant to a 58-subject study.",
 "5": "Radar for neurodegenerative disease is <strong>sparse and recent</strong>. No published work does subject-independent Parkinson's-vs-control classification from micro-Doppler.",
 "6": "The field's headline accuracies are undermined by three recurring failures: <strong>subject leakage</strong>, <strong>shortcut learning on artifacts</strong>, and <strong>uncontrolled confounds — above all age</strong>.",
 "7": "This thesis targets the gap: the first subject-independent PD classifier from radar micro-Doppler, with the confound controls the literature says are necessary but rarely applies together.",
}

# ------------------------------------------------------------------ build HTML
def esc(s): return s

toc = "".join(
    f'<li><a href="#s{s["n"]}"><span class="tn">{s["n"]}</span>{s["title"]}</a></li>' for s in secs)

sections_html = []
for s in secs:
    paras = "".join(f"<p>{p}</p>" for p in s["paras"])
    tk = TAKEAWAY.get(s["n"])
    kt = f'<aside class="key"><span class="klabel">Key point</span><p>{tk}</p></aside>' if tk else ""
    sections_html.append(f"""
<section id="s{s['n']}" class="sec">
  <h2><span class="secnum">{s['n']}</span>{s['title']}</h2>
  {paras}
  {kt}
</section>""")

rows = []
for r in refs:
    p = r["meta"]
    vf, vc = validation_flag(r["n"])
    modality = first_sentence(p.get("modality") if p else "", 46)
    rows.append(f"""<tr>
  <td class="c-n"><a href="#ref-{r['n']}">{r['n']}</a></td>
  <td class="c-y">{r['year']}</td>
  <td class="c-t">{r['title']}</td>
  <td class="c-m">{modality}</td>
  <td class="c-s">{short_n(p)}</td>
  <td class="c-v"><span class="vflag {vc}">{vf}</span></td>
</tr>""")

def block(label, text, cls=""):
    return f'<div class="fld {cls}"><span class="flabel">{label}</span><p>{text}</p></div>' if text else ""

cards = []
for r in refs:
    p = r["meta"]
    body = ""
    if p:
        body = (block("Method", p.get("methodology"))
              + block("Key findings", p.get("key_findings"))
              + block("Reported performance", p.get("reported_performance"), "perf")
              + block("Strengths", p.get("pros"), "pro")
              + block("Limitations", p.get("cons_limitations"), "con")
              + block("Lesson for this thesis", p.get("lesson_for_us"), "lesson"))
    det = (f'<details><summary>Details, strengths &amp; limitations</summary>'
           f'<div class="dbody">{body}</div></details>') if body else ""
    cards.append(f"""
<article class="ref" id="ref-{r['n']}">
  <div class="rhead"><span class="rnum">{r['n']}</span>
    <div><h3>{r['title']}</h3>
      <p class="rmeta">{r['authors']} · <strong>{r['year']}</strong></p>
      <p class="rven">{r['venue']}</p>
      <p class="rlink"><a href="{r['url']}" target="_blank" rel="noopener">{r['url']}</a></p>
    </div></div>
  {det}
</article>""")

CSS = """
:root{
 --bg:#fbfaf8; --panel:#fff; --ink:#1a1a1a; --muted:#6b6b6b; --line:#e6e2dc;
 --accent:#b5471f; --accent-soft:#fdf1ec; --link:#1a5f9e;
 --ok:#1f7a4d; --ok-bg:#eaf6ef; --warn:#9a5b00; --warn-bg:#fdf3e2;
 --key-bg:#f4f7fb; --key-line:#2f6fae;
 --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
 --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif;
 --serif:"Iowan Old Style",Georgia,"Times New Roman",serif;
}
@media (prefers-color-scheme:dark){
 :root{--bg:#15171a;--panel:#1c1f23;--ink:#e6e6e6;--muted:#9aa0a6;--line:#2c3137;
  --accent:#ff8a5c;--accent-soft:#2a1d17;--link:#7fb6e8;
  --ok:#6bd39a;--ok-bg:#16281f;--warn:#e8b061;--warn-bg:#2a2113;
  --key-bg:#1a2230;--key-line:#4a90d9;}
}
:root[data-theme="dark"]{--bg:#15171a;--panel:#1c1f23;--ink:#e6e6e6;--muted:#9aa0a6;--line:#2c3137;
 --accent:#ff8a5c;--accent-soft:#2a1d17;--link:#7fb6e8;--ok:#6bd39a;--ok-bg:#16281f;
 --warn:#e8b061;--warn-bg:#2a2113;--key-bg:#1a2230;--key-line:#4a90d9;}
:root[data-theme="light"]{--bg:#fbfaf8;--panel:#fff;--ink:#1a1a1a;--muted:#6b6b6b;--line:#e6e2dc;
 --accent:#b5471f;--accent-soft:#fdf1ec;--link:#1a5f9e;--ok:#1f7a4d;--ok-bg:#eaf6ef;
 --warn:#9a5b00;--warn-bg:#fdf3e2;--key-bg:#f4f7fb;--key-line:#2f6fae;}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--serif);
 line-height:1.72;font-size:17px;-webkit-font-smoothing:antialiased}

#bar{position:fixed;top:0;left:0;height:3px;width:0;background:var(--accent);z-index:100;transition:width .1s}

.wrap{max-width:1180px;margin:0 auto;padding:0 22px 90px;
 display:grid;grid-template-columns:230px minmax(0,1fr);gap:44px}
@media(max-width:900px){.wrap{grid-template-columns:1fr;gap:0}}

/* ---------- header ---------- */
header.hero{grid-column:1/-1;padding:52px 0 26px;border-bottom:1px solid var(--line);margin-bottom:34px}
.kicker{font-family:var(--sans);font-size:12px;letter-spacing:.11em;text-transform:uppercase;
 color:var(--accent);font-weight:700;margin:0 0 10px}
h1{font-family:var(--sans);font-size:clamp(28px,4.4vw,42px);line-height:1.15;
 font-weight:700;margin:0 0 12px;letter-spacing:-.02em}
.sub{font-family:var(--sans);color:var(--muted);font-size:15px;margin:0 0 22px;max-width:60ch}
.stats{display:flex;flex-wrap:wrap;gap:10px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:9px 14px}
.stat b{font-family:var(--sans);font-size:19px;display:block;line-height:1.1}
.stat span{font-family:var(--sans);font-size:11.5px;color:var(--muted);
 text-transform:uppercase;letter-spacing:.05em}

/* ---------- toc ---------- */
nav.toc{position:sticky;top:26px;align-self:start;font-family:var(--sans);max-height:calc(100vh - 52px);overflow:auto}
@media(max-width:900px){nav.toc{position:static;margin-bottom:34px;
 border:1px solid var(--line);border-radius:11px;padding:14px 16px;background:var(--panel)}}
nav.toc h4{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
 margin:0 0 12px;font-weight:700}
nav.toc ol{list-style:none;margin:0;padding:0}
nav.toc li{margin:0 0 2px}
nav.toc a{display:flex;gap:9px;text-decoration:none;color:var(--muted);font-size:13.5px;
 line-height:1.4;padding:6px 9px;border-radius:7px;border-left:2px solid transparent}
nav.toc a:hover{color:var(--ink);background:var(--panel)}
nav.toc a.on{color:var(--accent);background:var(--accent-soft);border-left-color:var(--accent);font-weight:600}
.tn{font-variant-numeric:tabular-nums;opacity:.6;min-width:12px}
.tocx{margin-top:16px;padding-top:14px;border-top:1px solid var(--line)}

/* ---------- sections ---------- */
main{min-width:0}
.sec{margin:0 0 46px;scroll-margin-top:24px}
h2{font-family:var(--sans);font-size:23px;font-weight:700;letter-spacing:-.01em;
 margin:0 0 16px;display:flex;gap:13px;align-items:baseline;line-height:1.25}
.secnum{font-size:13px;color:var(--accent);border:1.5px solid var(--accent);border-radius:6px;
 min-width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;
 flex:none;font-weight:700;transform:translateY(-2px)}
.sec p{margin:0 0 16px;max-width:68ch}
em{font-style:italic}

.key{background:var(--key-bg);border-left:3px solid var(--key-line);border-radius:0 9px 9px 0;
 padding:14px 18px;margin:22px 0 0;max-width:68ch}
.klabel{font-family:var(--sans);font-size:10.5px;font-weight:700;letter-spacing:.09em;
 text-transform:uppercase;color:var(--key-line)}
.key p{margin:5px 0 0;font-size:15.5px;line-height:1.6}

/* ---------- table ---------- */
.tblwrap{overflow-x:auto;border:1px solid var(--line);border-radius:11px;background:var(--panel);margin:0 0 46px}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:13px;min-width:660px}
th{text-align:left;padding:11px 13px;border-bottom:1.5px solid var(--line);
 font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
 font-weight:700;white-space:nowrap;background:var(--panel);position:sticky;top:0}
td{padding:10px 13px;border-bottom:1px solid var(--line);vertical-align:top;line-height:1.45}
tr:last-child td{border-bottom:0}
tbody tr:hover{background:var(--accent-soft)}
.c-n a{font-weight:700;color:var(--accent);text-decoration:none}
.c-y,.c-s{font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--muted)}
.c-t{font-weight:600;min-width:210px}
.c-m{color:var(--muted);min-width:150px}
.vflag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;
 white-space:nowrap;background:var(--line);color:var(--muted)}
.vflag.ok{background:var(--ok-bg);color:var(--ok);font-weight:600}
.vflag.warn{background:var(--warn-bg);color:var(--warn);font-weight:600}
.vflag.na{background:transparent;color:var(--muted);border:1px dashed var(--line)}

/* ---------- references ---------- */
.ref{background:var(--panel);border:1px solid var(--line);border-radius:11px;
 padding:18px 20px;margin:0 0 14px;scroll-margin-top:24px}
.ref:target{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.rhead{display:flex;gap:14px}
.rnum{font-family:var(--sans);font-weight:700;font-size:13px;color:var(--accent);
 background:var(--accent-soft);border-radius:7px;min-width:30px;height:30px;flex:none;
 display:inline-flex;align-items:center;justify-content:center}
.ref h3{font-family:var(--sans);font-size:16px;font-weight:700;margin:2px 0 6px;line-height:1.35}
.rmeta{font-family:var(--sans);font-size:13px;color:var(--ink);margin:0 0 3px}
.rven{font-family:var(--sans);font-size:12.5px;color:var(--muted);margin:0 0 5px;font-style:italic}
.rlink{margin:0;font-size:12px}
.rlink a{color:var(--link);word-break:break-all;font-family:var(--mono);font-size:11.5px}
details{margin:13px 0 0}
summary{font-family:var(--sans);font-size:12.5px;font-weight:600;color:var(--link);
 cursor:pointer;padding:7px 0;list-style:none;user-select:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"▸ ";display:inline-block;transition:transform .15s}
details[open] summary::before{content:"▾ "}
.dbody{border-top:1px solid var(--line);padding-top:13px;margin-top:3px}
.fld{margin:0 0 13px}
.flabel{font-family:var(--sans);font-size:10.5px;font-weight:700;letter-spacing:.08em;
 text-transform:uppercase;color:var(--muted)}
.fld p{margin:4px 0 0;font-size:14.5px;line-height:1.6}
.fld.pro .flabel{color:var(--ok)} .fld.con .flabel{color:var(--warn)}
.fld.perf .flabel{color:var(--accent)}
.fld.lesson{background:var(--key-bg);border-left:3px solid var(--key-line);
 padding:11px 15px;border-radius:0 8px 8px 0}
.fld.lesson .flabel{color:var(--key-line)}

h2.big{font-size:26px;margin:56px 0 8px;padding-top:26px;border-top:2px solid var(--ink)}
.lede{font-family:var(--sans);color:var(--muted);font-size:14px;margin:0 0 22px;max-width:66ch}

#top{position:fixed;right:20px;bottom:20px;width:42px;height:42px;border-radius:50%;
 background:var(--panel);border:1px solid var(--line);color:var(--ink);cursor:pointer;
 font-size:17px;display:none;align-items:center;justify-content:center;z-index:50}
#top.show{display:flex}
a{color:var(--link)}
sup a{font-family:var(--sans);font-size:.78em;font-weight:700;text-decoration:none;
 padding:0 1px;vertical-align:super}
p a[href^="#ref"]{font-family:var(--sans);font-size:.85em;font-weight:600;text-decoration:none}
p a[href^="#ref"]:hover{text-decoration:underline}

@media print{
 nav.toc,#top,#bar{display:none}
 .wrap{display:block;max-width:100%}
 body{font-size:10.5pt;background:#fff;color:#000}
 .ref,.tblwrap{break-inside:avoid} details{display:none}
}
"""

JS = """
const bar=document.getElementById('bar'),top=document.getElementById('top');
const links=[...document.querySelectorAll('nav.toc a[href^="#s"]')];
const secs=links.map(a=>document.getElementById(a.getAttribute('href').slice(1)));
function upd(){
 const h=document.documentElement;
 const p=h.scrollTop/(h.scrollHeight-h.clientHeight||1);
 bar.style.width=(p*100)+'%';
 top.classList.toggle('show',h.scrollTop>600);
 let i=0; secs.forEach((s,k)=>{ if(s && s.getBoundingClientRect().top<160) i=k; });
 links.forEach((a,k)=>a.classList.toggle('on',k===i));
}
addEventListener('scroll',upd,{passive:true}); addEventListener('resize',upd); upd();
top.onclick=()=>scrollTo({top:0,behavior:'smooth'});
"""

nrefs = len(refs)
out = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Related Work — Radar micro-Doppler & Parkinson's</title>
<style>{CSS}</style></head><body>
<div id="bar"></div>
<div class="wrap">

<header class="hero">
  <p class="kicker">Master's thesis · Chapter 2</p>
  <h1>Related Work</h1>
  <p class="sub">Contactless radar sensing, micro-Doppler signal processing and machine-learning
  gait analysis — and where a subject-independent Parkinson's classifier fits among them.</p>
  <div class="stats">
    <div class="stat"><b>{nrefs}</b><span>sources verified</span></div>
    <div class="stat"><b>2 / 16</b><span>use subject-independent validation</span></div>
    <div class="stat"><b>0</b><span>prior subject-independent PD classifiers</span></div>
    <div class="stat"><b>94.9%</b><span>young-vs-elderly from the same modality</span></div>
  </div>
</header>

<nav class="toc">
  <h4>Contents</h4>
  <ol>{toc}</ol>
  <div class="tocx"><ol>
    <li><a href="#glance"><span class="tn">▦</span>Evidence at a glance</a></li>
    <li><a href="#refs"><span class="tn">▤</span>References</a></li>
  </ol></div>
</nav>

<main>
  <p class="lede">{intro}</p>
  {''.join(sections_html)}

  <h2 class="big" id="glance">Evidence at a glance</h2>
  <p class="lede">All {nrefs} sources, with the validation scheme each used — the column that
  explains why headline accuracies in this field are often not comparable to ours.</p>
  <div class="tblwrap"><table>
    <thead><tr><th>#</th><th>Year</th><th>Title</th><th>Modality</th><th>n</th><th>Validation</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>

  <h2 class="big" id="refs">References</h2>
  <p class="lede">Every entry was retrieved and verified from its live source. Expand any card for
  method, findings, strengths, limitations and the specific lesson taken into this thesis.</p>
  {''.join(cards)}
</main>
</div>
<button id="top" title="Back to top">↑</button>
<script>{JS}</script>
</body></html>"""

(ROOT / "RELATED_WORK.html").write_text(out)
print(f"sections={len(secs)} refs={len(refs)} matched={sum(1 for r in refs if r['meta'])}")
print(f"bytes={len(out):,}")
