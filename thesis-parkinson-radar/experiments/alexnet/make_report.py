#!/usr/bin/env python3
"""Figures + ALEXNET.html for the dataset-or-methodology experiment."""
import base64, io, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT / "tools"))
import vizstyle as V                                     # noqa: E402
V.apply()

HERE = ROOT / "experiments/alexnet"
FIG = HERE / "outputs/figures"; FIG.mkdir(exist_ok=True)
RND = json.load(open(HERE / "outputs/alexnet_random.json"))
LOSO = json.load(open(HERE / "outputs/alexnet_loso_summary.json"))["summary"]
EXTRA = json.load(open(HERE / "outputs/alexnet_loso_extra.json"))
FOLDS = pd.read_csv(HERE / "outputs/alexnet_loso_folds.csv")
CAMP = json.load(open(ROOT / "outputs/runs/campaign_summary.json"))

subj_rand = [r["subject"]["auc"] for r in RND]
win_rand = [r["window"]["auc"] for r in RND]

# ── fig 1: two protocols, one architecture ──
fig, ax = plt.subplots(figsize=(10.6, 4.2), constrained_layout=True)
rows = [("Random split\n(subjects seen in training)", subj_rand, V.PD),
        ("Leave-one-subject-out\n(never-seen subjects)", [EXTRA["pooled"]], V.CONTROL)]
for (lab, vals, col), y in zip(rows, (1, 0)):
    if len(vals) > 1:
        ax.plot([min(vals), max(vals)], [y, y], color=V.GRID, lw=3, zorder=1)
    else:
        ci = EXTRA["ci"]
        ax.plot(ci, [y, y], color=V.GRID, lw=3.2, zorder=1, solid_capstyle="round")
        for xv in ci:
            ax.plot([xv, xv], [y - 0.07, y + 0.07], color=V.INK_3, lw=1.4)
    ax.plot(vals, [y] * len(vals), "o", ms=11, color=col, zorder=3, **V.ring())
    # Label every point; drop a label below its dot when its neighbour is
    # close enough that two labels above would collide.
    offs = {v: (0, 14) for v in vals}
    sv = sorted(vals)
    for a, b in zip(sv, sv[1:]):
        if b - a < 0.03:
            offs[a] = (0, -24)
    for v in vals:
        ax.annotate(f"{v:.2f}", (v, y), textcoords="offset points",
                    xytext=offs[v], ha="center", fontsize=10.5,
                    fontweight="bold", color=col)
for x, lab, col, ty, ha in [(0.5, "chance", V.INK_3, -0.62, "right"),
                            (0.610, "duration alone", V.ACCENT, -0.62, "left")]:
    ax.axvline(x, color=col, lw=1.4, alpha=0.75, zorder=0)
    ax.text(x - 0.006 if ha == "right" else x + 0.006, ty, lab, ha=ha,
            va="center", fontsize=9.2, color=col)
ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
ax.set_ylim(-0.85, 1.6); ax.set_xlim(0.42, 0.97)
ax.set_xlabel("subject-level AUC")
ax.grid(True, axis="x"); ax.grid(False, axis="y")
V.title(ax, "One architecture, two protocols",
        "AlexNet on the same data: dots are seeds (random split) and the pooled "
        "LOSO score with its 95% confidence interval")
fig.savefig(FIG / "alex_ladder.png"); plt.close(fig)
print("   alex_ladder.png")

# ── fig 2: confusions ──
cm_rand = np.sum([np.array(r["window"]["confusion"]) for r in RND], axis=0)
cm_loso = np.array(LOSO["confusion_matrix"])
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
for a, (name, c, note) in zip(axes, [
        ("Random split, window level\n(pooled over 3 seeds)", cm_rand, ""),
        ("LOSO, subject level\n(threshold = prevalence)", cm_loso, "")]):
    frac = c / c.sum(axis=1, keepdims=True)
    a.imshow(frac, cmap=V.SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{c[i,j]:,}\n{frac[i,j]:.0%}", ha="center", va="center",
                   fontsize=10.5, fontweight="bold",
                   color="white" if frac[i, j] > 0.55 else V.INK)
    a.set_xticks([0, 1]); a.set_yticks([0, 1])
    a.set_xticklabels(["pred control", "pred PD"], fontsize=9.5)
    a.set_yticklabels(["control", "PD"], fontsize=9.5)
    a.set_title(name, loc="left", fontsize=10.5); a.grid(False)
axes[0].set_ylabel("true class")
fig.suptitle("AlexNet's confusion matrices under the two protocols",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=V.INK)
fig.savefig(FIG / "alex_confusion.png"); plt.close(fig)
print("   alex_confusion.png")

# ── fig 3: LOSO per-subject scatter ──
f = FOLDS.sort_values("subject_prob").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11.2, 4.4), constrained_layout=True)
thr = LOSO["threshold"]
for lab, col, name in [(0, V.CONTROL, "control"), (1, V.PD, "PD")]:
    d = f[f.subject_label == lab]
    ax.plot(d.index, d.subject_prob, "o", ms=8, color=col, label=name, **V.ring())
wrong = f[(f.subject_prob >= thr).astype(int) != f.subject_label]
ax.plot(wrong.index, wrong.subject_prob, "x", ms=12, color=V.INK, mew=2,
        label="misclassified", ls="")
ax.axhline(thr, color=V.ACCENT, lw=1.6)
V.note(ax, 1, thr + 0.02, f"threshold = prevalence = {thr:.2f}", ha="left")
ax.set_xlabel("subjects, sorted by AlexNet's LOSO probability")
ax.set_ylabel("P(PD)")
V.title(ax, "AlexNet under LOSO: every subject's probability",
        "the two classes interleave along the ramp, as with every other model")
ax.legend(fontsize=9.5, loc="upper left")
fig.savefig(FIG / "alex_scatter.png"); plt.close(fig)
print("   alex_scatter.png")

# ══════════════ HTML ══════════════
def img(name, maxw=1400, q=84):
    p = FIG / name
    im = Image.open(p).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" alt="{name}">'

def fig_html(name, caption, insight=None):
    ins = (f'<div class="insight"><span class="ilab">Reading the figure</span>'
           f'<p>{insight}</p></div>') if insight else ""
    return (f'<figure>{img(name)}<figcaption><span class="fname">{name}</span> '
            f'{caption}</figcaption>{ins}</figure>')

def key(t):  return f'<aside class="key"><span class="klabel">Key point</span><p>{t}</p></aside>'
def warn(t): return f'<aside class="warn"><span class="klabel">Caution</span><p>{t}</p></aside>'
def defn(term, body): return f'<div class="defn"><span class="dterm">{term}</span><p>{body}</p></div>'
def intro(t): return f'<div class="intro"><p>{t}</p></div>'

CSS = (ROOT / "tools/eda.css").read_text() + """
.intro{border-left:3px solid var(--accent);padding:2px 0 2px 16px;margin:22px 0 6px}
.intro p{margin:0;font-size:15.5px;line-height:1.62;max-width:none;color:var(--ink)}
"""

rand_rows = "".join(
    f"<tr><td>seed {r['seed']}</td>"
    f"<td class=\"num\">{r['window']['auc']:.3f}</td>"
    f"<td class=\"num\">{r['window']['accuracy']*100:.1f} %</td>"
    f"<td class=\"num\">{r['recording']['auc']:.3f}</td>"
    f"<td class=\"num\">{r['subject']['auc']:.3f}</td>"
    f"<td class=\"num\">{r['test_subjects_seen_in_train']:.0%}</td></tr>"
    for r in RND)

(tn, fp), (fn, tp) = LOSO["confusion_matrix"]

S = []
S.append(("Why this experiment exists", f"""
<p>Every model in the main campaign was designed by us, which leaves one
objection standing: perhaps the modest numbers reflect our choices rather than
the dataset. This experiment removes that objection by running <strong>the
field&rsquo;s own architecture</strong>, AlexNet, the network behind the
literature&rsquo;s cautionary 97.8&nbsp;% spectrogram result, on the same
window cache, under both validation designs. If the dataset were the problem,
AlexNet should fail under both protocols. If the methodology gap is the story,
AlexNet should post literature-grade numbers under the literature&rsquo;s
protocol and fall back into the honest band under ours.</p>

{defn("The setup, in one paragraph", "ImageNet-pretrained AlexNet; the first convolution&rsquo;s colour filters averaged and tiled to the two radar channels, exactly as the thesis adapted ResNet-18; the final layer replaced for two classes; convolutional features frozen and the full 54.5-million-parameter classifier block trained at the literature&rsquo;s learning rate of 10<sup>-4</sup>; the same v2 window cache and augmentation as the main campaign. Two runs: a recording-level random 70/15/15 split repeated with three seeds (the literature&rsquo;s hold-out, with window-level early stopping and no window weighting, as the literature does), and the full 58-fold leave-one-subject-out protocol.")}
"""))

S.append(("Their protocol: literature-grade numbers appear on demand", f"""
<div class="tblwrap"><table>
<thead><tr><th>Run</th><th>Window AUC</th><th>Accuracy</th><th>Per recording</th><th>Per subject</th><th>Test subjects seen in training</th></tr></thead>
<tbody>{rand_rows}</tbody></table></div>

{key("Under the literature&rsquo;s validation, AlexNet reaches <strong>window AUC up to 0.86</strong> and <strong>subject-level AUC up to 0.89</strong> on this very dataset, with accuracies around 70 to 78&nbsp;%. Every test subject was seen during training, in every seed. The dataset is perfectly capable of producing publishable numbers; it only requires not asking whether the model works on people it has never met.")}
"""))

S.append(("Our protocol: the same architecture, honestly measured", f"""
<div class="tblwrap"><table>
<thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody>
<tr><td>Pooled subject-level AUC</td><td class="num"><strong>{EXTRA['pooled']:.3f}</strong></td></tr>
<tr><td>95&nbsp;% confidence interval</td><td class="num">[{EXTRA['ci'][0]:.3f}, {EXTRA['ci'][1]:.3f}]</td></tr>
<tr><td>Balanced accuracy at the prevalence threshold</td><td class="num">{LOSO['balanced_accuracy']:.3f}</td></tr>
<tr><td>PD subjects caught / missed</td><td class="num">{tp} / {fn}</td></tr>
<tr><td>Controls falsely flagged</td><td class="num">{fp} of {tn+fp}</td></tr>
<tr><td>Correlation of subject scores with window count</td><td class="num">+{EXTRA['corr_nwin']:.2f}</td></tr>
<tr><td>Mean best epoch</td><td class="num">{LOSO['mean_best_epoch']:.1f}</td></tr>
</tbody></table></div>

{fig_html("alex_ladder.png",
  "AlexNet's subject-level AUC under the two protocols: three random-split seeds against the pooled leave-one-subject-out score with its confidence interval, with chance and the duration floor marked.",
  "<strong>What the figure shows.</strong> The same architecture on the same data moves from a mean of 0.83 to 0.69 the moment the test subjects become strangers, and the honest score's confidence interval still contains the duration floor.<br><br><strong>What it means.</strong> The 0.2 gap is manufactured by the split alone; nothing about the model or the data changed.")}

{fig_html("alex_scatter.png",
  "Every subject's LOSO probability from AlexNet, sorted, with the prevalence threshold and misclassified subjects marked.",
  "<strong>What the figure shows.</strong> The familiar picture: the two classes interleave along the whole ramp, so no threshold separates them cleanly.<br><br><strong>What it means.</strong> AlexNet under honest validation behaves like every model of the main campaign, not like its own random-split self.")}

{fig_html("alex_confusion.png",
  "AlexNet's confusion matrices under the two protocols.",
  "<strong>What the figure shows.</strong> Under the random split both classes are classified well. Under LOSO the model buys its 84&nbsp;% PD recall by flagging over half the controls.<br><br><strong>What it means.</strong> Moving to honest validation does not just lower a number; it changes the character of the errors to those of a weak ranker forced through a threshold.")}

{warn("An honest caveat cuts both ways. AlexNet&rsquo;s LOSO score of 0.69 is the highest pooled number of any protocol-compliant run in this project, and it deserves scrutiny rather than celebration: its confidence interval [0.554, 0.833] still contains the duration floor, its subject scores track window count at <strong>+0.25</strong>, the strongest duration residue of any compliant model (our campaign models sit near +0.05), and it is a single configuration added after the campaign, exactly the situation where one run out of many looks good by chance. By the standards this thesis pre-registered, it does not change the verdict; a duration-matched re-scoring would be the next check if this number were ever to be leaned on.")}
"""))

S.append(("The verdict: dataset or methodology", f"""
<div class="tblwrap"><table>
<thead><tr><th>Model &amp; protocol</th><th>Subject-level AUC</th></tr></thead><tbody>
<tr><td>AlexNet, random split (subjects seen in training), best seed</td><td class="num">0.891</td></tr>
<tr><td>AlexNet, random split, mean of 3 seeds</td><td class="num">{np.mean(subj_rand):.3f}</td></tr>
<tr><td><strong>AlexNet, leave-one-subject-out</strong></td><td class="num"><strong>{EXTRA['pooled']:.3f}</strong></td></tr>
<tr><td>ResNet-18 probe, leave-one-subject-out (campaign)</td><td class="num">{CAMP['resnet18_fc_v2']['pooled']:.3f}</td></tr>
<tr><td>SmallCNN, leave-one-subject-out (campaign)</td><td class="num">{CAMP['smallcnn_v2']['pooled']:.3f}</td></tr>
<tr><td>Recording duration alone (floor)</td><td class="num">0.610</td></tr>
</tbody></table></div>

{key("<strong>The answer to the question this experiment was built for.</strong> The dataset is not what separates this thesis from the literature: the field&rsquo;s own architecture produces literature-grade numbers on it whenever the validation allows subjects to appear on both sides of the split, and falls back into the same honest band as every other model the moment it is scored on people it has never seen. The problem the thesis reports is therefore not a property of our models, and not a defect of the corpus as data; it is that <strong>the question the literature answers is easier than the question a screening tool must answer</strong>, and this corpus, like theirs, only supports impressive numbers for the easy question.")}
"""))

toc = "".join(f'<li><a href="#s{i}"><span class="tn">{i}</span>{t}</a></li>'
              for i, (t, _) in enumerate(S, 1))
body = "".join(f'<section id="s{i}" class="sec"><h2><span class="secnum">{i}</span>{t}</h2>{c}</section>'
               for i, (t, c) in enumerate(S, 1))
JS = """const bar=document.getElementById('bar'),top=document.getElementById('top');
const links=[...document.querySelectorAll('nav.toc a[href^="#s"]')];
const secs=links.map(a=>document.getElementById(a.getAttribute('href').slice(1)));
function upd(){const h=document.documentElement;
 bar.style.width=(h.scrollTop/(h.scrollHeight-h.clientHeight||1)*100)+'%';
 top.classList.toggle('show',h.scrollTop>600);
 let i=0;secs.forEach((s,k)=>{if(s&&s.getBoundingClientRect().top<160)i=k;});
 links.forEach((a,k)=>a.classList.toggle('on',k===i));}
addEventListener('scroll',upd,{passive:true});addEventListener('resize',upd);upd();
top.onclick=()=>scrollTo({top:0,behavior:'smooth'});"""

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The AlexNet Experiment &middot; Radar micro-Doppler Parkinson's</title>
<style>{CSS}</style></head><body>
<div id="bar"></div><div class="wrap">
<header class="hero">
  <p class="kicker">Master's thesis &middot; Dataset or methodology?</p>
  <h1>The AlexNet Experiment</h1>
  <p class="sub">The field's own architecture, run on this dataset under both
  validation designs, to settle whether the thesis's modest numbers come from
  the data or from the protocol.</p>
  <div class="statlead">Where it ends:</div>
  <div class="stats">
    <div class="stat"><b>0.89</b><span>best subject AUC, their protocol<i>subjects seen in training</i></span></div>
    <div class="stat"><b>{EXTRA['pooled']:.2f}</b><span>subject AUC, our protocol<i>never-seen subjects</i></span></div>
    <div class="stat"><b>+0.25</b><span>duration residue under LOSO<i>strongest of any compliant run</i></span></div>
    <div class="stat"><b>0.610</b><span>still inside the CI<i>the duration floor</i></span></div>
  </div>
</header>
<nav class="toc"><h4>Contents</h4><ol>{toc}</ol></nav>
<main>{body}</main></div>
<button id="top" title="Back to top">&uarr;</button>
<script>{JS}</script></body></html>"""

out = HERE / "ALEXNET.html"
out.write_text(html)
print(f"wrote {out} ({len(html)/1e6:.2f} MB)")
print("missing:", html.count("missing figure"), "emdash:", html.count(chr(8212)))
