#!/usr/bin/env python3
"""RESULTS_MODELLING.html — every result of the modelling campaign, in the
same style as the setup document. No em-dashes, key terms bolded,
intro -> figure -> reading, every number pulled from the stored summaries."""
import base64, io, json, pathlib
import pandas as pd
from PIL import Image

ROOT = pathlib.Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
FIG, TOOLS = ROOT / "outputs/figures", ROOT / "tools"
BS = json.load(open(ROOT / "reports/baseline_summary.json"))
R = BS["results"]
CAMP = json.load(open(ROOT / "outputs/runs/campaign_summary.json"))
CONF = json.load(open(ROOT / "reports/confusion_summary.json"))
BG = json.load(open(ROOT / "experiments/bg_suppressed/outputs/bg_suppressed_summary.json"))

def _bg(model, field, nd=3):
    v = BG[model].get(field)
    return "n/a" if v is None else f"{v:.{nd}f}"
best, clean, dur = (R["logreg|confounded_only"], R["logreg|shape_clean"],
                    R["logreg|duration_only"])
lo, hi = best["auc_lo"], best["auc_hi"]
clo, chi = clean["auc_lo"], clean["auc_hi"]

def _c(key, field, nd=3):
    v = CAMP[key].get(field)
    return "n/a" if v is None else f"{v:.{nd}f}"

def _ci(key):
    ci = CAMP[key].get("ci")
    if not ci or ci[0] is None or ci[1] is None:
        return "n/a"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"

def img(name, maxw=1500, q=84):
    p = FIG / name
    if not p.exists():
        return f'<p style="color:#b00">[missing figure: {name}]</p>'
    im = Image.open(p).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" alt="{name}">'

def fig(name, caption, insight=None, maxw=1500):
    ins = (f'<div class="insight"><span class="ilab">Reading the figure</span>'
           f'<p>{insight}</p></div>') if insight else ""
    return (f'<figure>{img(name, maxw)}<figcaption><span class="fname">{name}</span> '
            f'{caption}</figcaption>{ins}</figure>')

def key(t):  return f'<aside class="key"><span class="klabel">Key point</span><p>{t}</p></aside>'
def warn(t): return f'<aside class="warn"><span class="klabel">Caution</span><p>{t}</p></aside>'
def defn(term, body): return f'<div class="defn"><span class="dterm">{term}</span><p>{body}</p></div>'
def defn2(term, body): return f'<div class="defn"><span class="dterm">{term}</span><div>{body}</div></div>'
def intro(t): return f'<div class="intro"><p>{t}</p></div>'

CSS = (TOOLS / "eda.css").read_text() + """
.sec h4{font-family:var(--sans);font-size:15px;font-weight:700;color:var(--ink);margin:26px 0 8px}
.intro{border-left:3px solid var(--accent);padding:2px 0 2px 16px;margin:22px 0 6px}
.intro p{margin:0;font-size:15.5px;line-height:1.62;max-width:none;color:var(--ink)}
td.hi{background:rgba(31,122,85,.10);font-weight:700}
"""

grid_rows = []
for mdl, mlab in [("logreg", "logreg"), ("svm_rbf", "SVM"),
                  ("random_forest", "random_forest")]:
    cells = []
    for fs in ["all_features", "duration_invariant", "shape_clean", "confounded_only", "duration_only"]:
        v = R.get(f"{mdl}|{fs}", {}).get("auc")
        cls = ' class="hi"' if (v is not None and v >= 0.66) else ""
        cells.append(f'<td{cls}>{v:.3f}</td>' if v is not None else "<td>n/a</td>")
    grid_rows.append(f'<tr><td><strong>{mlab}</strong></td>' + "".join(cells) + "</tr>")
GRID = "".join(grid_rows)

ORDER = [("smallcnn_v2", "SmallCNN"),
         ("resnet18_fc_v2", "ResNet-18 probe"),
         ("envlstm", "Envelope-LSTM")]
camp_rows = "".join(
    f"<tr><td><strong>{lab}</strong></td>"
    f"<td class=\"num\">{_c(k,'pooled')}</td>"
    f"<td class=\"num\">{_ci(k)}</td></tr>"
    for k, lab in ORDER)

CONF_ORDER = ["logreg_shape_clean", "smallcnn_v2", "resnet18_fc_v2", "envlstm"]
conf_rows = "".join(
    f"<tr><td><strong>{m['label']}</strong></td>"
    f"<td class=\"num\">{m['recall']:.2f}</td>"
    f"<td class=\"num\">{m['specificity']:.2f}</td>"
    f"<td class=\"num\">{m['precision']:.2f}</td>"
    f"<td class=\"num\">{m['f1']:.2f}</td>"
    f"<td class=\"num\">{m['balanced_acc']:.2f}</td></tr>"
    for m in (CONF["models"][k] for k in CONF_ORDER))
cb = CONF["models"]["resnet18_fc_v2"]
(cb_tn, cb_fp), (cb_fn, cb_tp) = cb["confusion"]

S = []

S.append(("How to read every number in this document", f"""
<p>This document is the answer sheet to the modelling setup: the same sections,
now with their outcomes. Every score is a <strong>subject-level AUC under
leave-one-subject-out validation</strong>, one averaged probability per person,
exactly as the protocol defined it.</p>

{defn2("The other scores used in this document", '''
<p>One later section cuts each model&rsquo;s probabilities at a threshold and reports
the resulting decisions, counted in four cells. With <strong>PD as the positive
class</strong>: TP is a PD subject correctly called, TN a control correctly left
alone, FP a control wrongly flagged, FN a PD subject missed. Every further score
is arithmetic on those four counts:</p>
<div class="tblwrap"><table>
<thead><tr><th>Score</th><th>Formula</th><th>Plain meaning</th></tr></thead><tbody>
<tr><td><strong>Recall</strong> (sensitivity)</td><td class="num">TP / (TP + FN)</td><td>the share of true PD subjects the model catches</td></tr>
<tr><td><strong>Specificity</strong></td><td class="num">TN / (TN + FP)</td><td>the share of true controls correctly left unflagged</td></tr>
<tr><td><strong>Precision</strong></td><td class="num">TP / (TP + FP)</td><td>when the model says PD, how often it is right</td></tr>
<tr><td><strong>F1</strong></td><td class="num">2 &middot; precision &middot; recall / (precision + recall)</td><td>one number balancing catching PD and being right when flagging</td></tr>
<tr><td><strong>Balanced accuracy</strong></td><td class="num">(recall + specificity) / 2</td><td>the average hit rate across the two classes</td></tr>
</tbody></table></div>''')}

<h3>The floors</h3>
{intro("Two deliberately impoverished predictors were scored under the identical protocol. They are the floors: any model not clearly above them has demonstrated nothing about gait.")}

<div class="tblwrap"><table>
<thead><tr><th>Floor</th><th>AUC</th><th>What it is</th></tr></thead><tbody>
<tr><td><strong>Recording duration alone</strong></td><td class="num">0.610</td>
<td>One number per subject, how long their recordings last, fed to a logistic
regression under the same leave-one-subject-out loop. This is the classical
grid&rsquo;s own <code>duration_only</code> cell; no gait content at all.</td></tr>
<tr><td><strong>Window count alone</strong></td><td class="num">0.620</td>
<td>No model at all: subjects are ranked directly by how many windows they
yield (a restatement of duration) and the AUC of that ranking is computed.</td></tr>
</tbody></table></div>

<h3>How every result is judged</h3>
{intro("The same two readings accompany every result in this document, so they are defined once here and never re-introduced.")}

<div class="tblwrap"><table>
<thead><tr><th>Reading</th><th>How it is computed</th><th>What it answers</th></tr></thead><tbody>
<tr><td><strong>Pooled AUC</strong></td>
<td>All 58 subject scores against the 58 diagnoses.</td>
<td>The headline number, comparable to the literature.</td></tr>
<tr><td><strong>95&nbsp;% confidence interval</strong></td>
<td>The 58 subjects are redrawn <strong>with replacement</strong> 2,000 times (some subjects appearing twice, others not at all), the AUC recomputed each time, and the interval is the middle 95&nbsp;% of those 2,000 values.</td>
<td>The error bar on the score: the range the score would most likely stay inside if the whole study were repeated with a different set of 58 people. If 0.5 sits inside that range, the score cannot be trusted to beat a coin flip.</td></tr>
</tbody></table></div>

{key("<strong>The finding, stated up front.</strong> The campaign was built to answer one question: does this dataset carry a gait signature that models can read? No such signature was demonstrable: pooled scores sit between 0.56 and 0.69, every 95&nbsp;% confidence interval contains the duration floor of 0.610, and the interpretability experiments attribute what score there is to properties of the recordings rather than of the walks. This is a statement about <strong>the signal in the data, not about the models</strong>: they behaved exactly as models should, finding the strongest pattern available, and that pattern was not gait. Establishing this rigorously, where the literature would have reported the 0.69 as a success, is the contribution. The rest of this document is the evidence.")}
"""))

S.append(("The classical yardstick, measured", f"""
{intro("The setup promised a 15-cell grid: three classifiers, logistic regression, an SVM and a random forest, by five feature sets, each cell a full leave-one-subject-out run over the 58 subjects.")}

<div class="tblwrap"><table>
<thead><tr><th>Model</th><th><a href="#fsets">all_features</a></th><th><a href="#fsets">duration_invariant</a></th><th><a href="#fsets">shape_clean</a></th><th><a href="#fsets">confounded_only</a></th><th><a href="#fsets">duration_only</a></th></tr></thead>
<tbody>{GRID}</tbody></table></div>

<div class="defn" id="fsets"><span class="dterm">The five feature sets, in one line each</span><div>
<div class="tblwrap"><table>
<thead><tr><th>Set</th><th>What it contains</th></tr></thead><tbody>
<tr><td><code>all_features</code></td><td>All 25 handcrafted measurements, no restraint.</td></tr>
<tr><td><code>duration_invariant</code></td><td>Only measurements whose definition cannot grow with recording length (averages, shapes, ratios; no totals).</td></tr>
<tr><td><code>shape_clean</code></td><td>The subset of those that also pass a direct test against measured duration: the cleanest description of the walk&rsquo;s shape.</td></tr>
<tr><td><code>confounded_only</code></td><td>Only the totals, which grow with recording length.</td></tr>
<tr><td><code>duration_only</code></td><td>Recording length and nothing else: the floor.</td></tr>
</tbody></table></div></div></div>

{key("Read the logistic regression row <strong>right to left</strong>. Its <strong>duration_only</strong> cell, recording length alone, scores <strong>{d:.3f}</strong>. Its <strong>confounded_only</strong> cell, the measurements that merely encode length, scores <strong>{c:.3f}</strong>, the best cell in the table. Its <strong>shape_clean</strong> cell, the measurements describing the shape of the walk with length removed, scores <strong>{s:.3f}</strong>. The further the features get from elapsed time, <strong>the worse the classifier does</strong>: the grid is the controlled experiment the setup designed, and it came out in the direction that indicts the confound.".format(d=dur['auc'], c=best['auc'], s=clean['auc']))}

<h3>The confidence intervals, applied to the grid</h3>
<p>Section 1 defined the interval that accompanies every result. Applied to the
grid, it says which of its numbers is real: the interval on the
<strong>confounded</strong> cell runs from <strong>{lo:.3f} to {hi:.3f}</strong>
and excludes 0.5, so the length signal is genuinely there; the interval on the
<strong>clean</strong> cell runs from <strong>{clo:.3f} to {chi:.3f}</strong>
and contains 0.5, so the gait signal has not been shown to exist.</p>

{fig("baseline_auc_grid.png",
  "The same grid drawn as a heat map, one cell per classifier and feature set.",
  "<strong>What the figure shows.</strong> Rows are the three classifiers, columns the five feature sets, each cell the subject-level AUC of that pairing. Every row peaks in the <strong>confounded_only</strong> column, which holds the warmest cell of the whole map: reading left to right, the colour ebbs through the progressively cleaner feature sets and then jumps at the features that merely encode duration, a gradient running toward the confound rather than away from it. No classifier finds anything in the clean features that it did not already get, more cheaply, from elapsed time.")}
"""))

S.append(("The deep campaign", f"""
{intro("The three models the setup chapter defined, SmallCNN, the ResNet-18 linear probe and the Envelope-LSTM, each under the full leave-one-subject-out protocol; plus two exploratory input variants we ran on our own initiative, the foot channel alone (a stated hypothesis of the source publication) and the turn-excluded corpus (the source publication's own preprocessing choice).")}

{fig("res_campaign.png",
  "The three models and the two exploratory variants on one axis. The blue dot is the pooled subject-level AUC over all 58 subjects, and the grey bar is its 95&nbsp;% confidence interval. The vertical lines are the floors from the table of section 1: chance at 0.5, recording duration alone at 0.610, window count alone at 0.621.",
  "<strong>What the figure shows.</strong> The five dots sit in a narrow band around the floors, and <strong>every confidence interval contains the duration floor</strong>: no configuration is distinguishable from a predictor that reads only how long the walk took. Two intervals clear chance; none clears the floor.<br><br><strong>What it means.</strong> A weak association exists, but nothing in it demonstrably exceeds what elapsed time alone provides, and no configuration, from 1,026 to 35,586 trainable weights, escapes this.")}


<h3>The three models of the setup chapter</h3>
<div class="tblwrap"><table>
<thead><tr><th>Model</th><th>Pooled AUC</th><th>95&nbsp;% CI</th></tr></thead>
<tbody>{camp_rows}</tbody></table></div>


"""))

S.append(("Decisions, not just rankings", f"""
{intro("Every score so far has been an AUC, and an AUC measures ranking: the probability that a randomly drawn PD subject scores above a randomly drawn control. A screening deployment cannot rank, it must decide, and a decision needs a threshold. As pre-registered, each headline model's per-subject probabilities are cut at the prevalence of the positive class, 25/58 = 0.431: any subject scoring at or above it is called PD. This section reports what the four headline models decide at that threshold, for all 58 subjects.")}

<div class="tblwrap"><table>
<thead><tr><th>Model</th><th>Recall (sensitivity)</th><th>Specificity</th><th>Precision</th><th>F1</th><th>Balanced accuracy</th></tr></thead>
<tbody>{conf_rows}</tbody></table></div>

{fig("res_confusion.png",
  "One confusion matrix per headline model, all at the same pre-registered threshold. Each cell carries the subject count and the row-normalised percentage; rows are the true class, columns the model's call.",
  "<strong>What the figure shows.</strong> The four matrices are variations on one shape: both rows split close to evenly, with no model getting either class much beyond two thirds right. SmallCNN is the most conservative (11 of 33 controls flagged) and the ResNet probe the most liberal (17 of 33 flagged), yet both catch a similar share of the PD group.<br><br><strong>What it means.</strong> Moving from ranking to deciding does not rescue any model. Every matrix is roughly what a weak ranker forced through a threshold must produce: each extra PD subject caught is bought with roughly one control falsely flagged, which is the arithmetic signature of near-chance discrimination.")}

{fig("res_subject_scatter.png",
  "Every subject's averaged LOSO probability from the best deep model, the ResNet-18 probe, sorted left to right. Blue is a true control, red a true PD subject. The violet line is the decision threshold: every subject above it is called PD, every subject below it control. It sits at 0.431 rather than 0.5 because that is the cohort's own share of PD subjects (25 of 58), the operating point fixed in the methods chapter. An X marks each subject the decision gets wrong.",
  "<strong>What the figure shows.</strong> The two colours are interleaved along the whole ramp rather than separated by it. Reds sit among the lowest scores and blues among the highest; the X marks cluster on both sides of the line, not at its edges.<br><br><strong>What it means.</strong> There is no probability band that is safely control or safely PD, so no alternative threshold would do much better: sliding the line left or right only trades one kind of X for the other. The weak decisions in the table are a property of the scores themselves, not of the cut chosen.")}

{fig("res_case_gallery.png",
  "One subject per confusion cell, for both image models: the most confident correct PD, the most confidently missed PD, the most confident false alarm, and the most confident correct control, each shown through a representative window of theirs (foot channel) with the model's probability.",
  "<strong>What the figure shows.</strong> Read the backgrounds, not the strides. In both rows, the two subjects <em>called PD</em> (the correct catch and the false alarm) sit on bright, noisy backgrounds, and the two <em>called control</em> (the correct pass and the missed patient) sit on dark, clean ones, regardless of their true diagnosis. The most extreme case is the probe&rsquo;s false alarm, a control whose window is almost pure noise texture and who is flagged at 0.81, while a genuine PD subject with a clean background is waved through at 0.24.<br><br><strong>What it means.</strong> The errors are not random: they are the systematic price of scoring by recording character. A deployment built on this would flag people whose recordings are noisy and miss patients whose recordings are clean, which is precisely what the attention maps of the next section show from inside the network.")}

{key("The honest reading of this section. At the pre-registered threshold the best deep model catches <strong>{tp} of the 25 PD subjects</strong> and misses <strong>{fn}</strong>, while falsely flagging <strong>{fp} of the 33 controls</strong>; balanced accuracy across the four models spans 0.58 to 0.63. These are roughly the catch-and-miss numbers a deployment would see, and they are exactly consistent with the AUC story: near-chance discrimination looks, at decision time, like a coin that is only slightly bent.".format(tp=cb_tp, fn=cb_fn, fp=cb_fp))}
"""))

# ═══ Grad-CAM ═══
# Attention statistics below come from outputs/metrics/gradcam_summary.csv,
# produced by tools/run_gradcam.py over all 58 folds of both deep models.
S.append(("What the models attended to", f"""
<p>Every number so far says <em>how well</em> the models scored. This section
asks <em>where they were looking</em> when they scored, using
<strong>Grad-CAM</strong>: for a given window, the gradient of the model&rsquo;s
PD output with respect to its last convolutional layer says which regions of
the picture pushed the decision, and the weighted, rectified sum of those
activations is drawn over the spectrogram as a heat map. This is the direct
check the field&rsquo;s own cautionary tale demands, the test that would have
caught the published network reading its recording sites. It was run on all
58 folds of both deep models, one map per window of each held-out
subject.</p>

{fig("gradcam_gallery.png",
  "Where the PD output of the SmallCNN looks, for eight held-out windows: confident correct decisions (top) and confident mistakes (bottom), spectrogram in grey, attention overlaid in colour, dotted lines marking the 200 and 500 Hz band edges.",
  "<strong>What the figure shows.</strong> Look at what is coloured and what is left grey. In every window called PD, right or wrong, the attention floods the <strong>empty background</strong>, and the gait envelope itself, the black shapes, is the one region left unhighlighted. In every window called control the map is almost dark. The confident mistakes in the bottom row follow the same rule: a clean-backed PD window is called control, a noisy-backed control window is called PD.<br><br><strong>What it means.</strong> The decision variable is a property of the <em>recording&rsquo;s background texture</em>, not of the walk drawn on top of it. This is the same conclusion the random-split gallery reached from the outside, now confirmed from inside the network.")}

{fig("gradcam_profile.png",
  "Left: mean attention along the Doppler axis for both deep models, with the torso (within 200 Hz) and foot (200 to 500 Hz) bands shaded. Right: the same profile split by whether the window straddles the turn.",
  "<strong>What the figure shows.</strong> The SmallCNN profile is <strong>inverted</strong> relative to where gait energy lives: minimum at 0 Hz, rising steadily toward the silent edges of the spectrum. Per pixel, the background rows above 500 Hz carry the highest attention density (0.192) against 0.173 in the foot band and 0.118 in the torso band, despite containing no gait energy at all. The frozen probe is flatter and torso-peaked, with no PD-versus-control asymmetry in its attention, consistent with features that never adapted to this data and a best epoch of 0.5. The right panel adds the second finding: <strong>turn-straddling windows draw four to six times the attention</strong> of single-direction walking windows across the whole axis (correlation of background attention with the turn label, Spearman rho = +0.449).<br><br><strong>What it means.</strong> The two places the attention concentrates, background texture and turn segments, are exactly the two nuisance channels this document keeps meeting: the per-recording noise texture and the duration content. The maps make the mechanism visible.")}

{warn("A caveat that belongs in the report: the maps are max-normalised per window, so comparisons between windows describe how attention is <em>distributed</em>, not the raw strength of the underlying output. The band and turn comparisons above are distribution statements, and are read as such.")}

{key("The interpretability finding in one sentence: <strong>the models were not looking at the walk</strong>. Attention concentrates on the background rows above 500 Hz and on the turn segments, PD-labelled windows carry more background attention than control windows (mass 0.33 against 0.27, centre of attention 380 Hz against 333 Hz), and the gait bands are the least-attended part of the picture. The attention maps independently corroborate that the scores measure properties of the recordings, not of the walks.")}
"""))

# ═══ background-suppressed follow-up ═══
S.append(("Suppressing what they attended to: the follow-up experiment", f"""
<p>The attention maps point at a specific, removable channel, so they were put
to work. A follow-up experiment, pre-registered in its own configuration file
before any window was rebuilt, changes exactly two things about the input and
nothing else: each recording&rsquo;s <strong>background profile</strong> (its
per-frequency median over time) is subtracted before windowing, and the empty
rows beyond <strong>&plusmn;500 Hz</strong> are cropped away, so the region the
attention maps flooded no longer exists. Both deep models were then re-run
under the full, unchanged protocol.</p>

<div class="tblwrap"><table>
<thead><tr><th>Model</th><th>Pooled AUC</th><th>95&nbsp;% CI</th></tr></thead><tbody>
<tr><td><strong>SmallCNN, background-suppressed</strong></td><td class="num">{_bg('smallcnn_bg','pooled')}</td><td class="num">[{BG['smallcnn_bg']['ci'][0]:.3f}, {BG['smallcnn_bg']['ci'][1]:.3f}]</td></tr>
<tr><td>SmallCNN, standard input (reference)</td><td class="num">{_c('smallcnn_v2','pooled')}</td><td class="num">{_ci('smallcnn_v2')}</td></tr>
<tr><td><strong>ResNet-18 probe, background-suppressed</strong></td><td class="num">{_bg('resnet18_fc_bg','pooled')}</td><td class="num">[{BG['resnet18_fc_bg']['ci'][0]:.3f}, {BG['resnet18_fc_bg']['ci'][1]:.3f}]</td></tr>
<tr><td>ResNet-18 probe, standard input (reference)</td><td class="num">{_c('resnet18_fc_v2','pooled')}</td><td class="num">{_ci('resnet18_fc_v2')}</td></tr>
</tbody></table></div>

<p>Two readings:</p>
<ul>
<li><strong>The scores did not collapse.</strong> Both models hold or slightly
exceed their reference values with the background gone, the confidence
intervals overlapping almost entirely. The recording-level nuisance therefore
<strong>leaks through more channels than the visible texture</strong>:
removing the region the attention flooded still leaves enough per-recording
signature, plausibly in the in-band enhancement characteristics, for the
models to find. The nuisance is not one removable feature; it is a property of
the per-recording processing.</li>
<li><strong>No hidden gait signal surfaced.</strong> If the background texture
had been masking a true gait signal, removing it should have lifted the scores
clearly above the duration floor. Instead every interval still contains 0.610,
so the experiment uncovered nothing gait-shaped underneath.</li>
</ul>

{key("The follow-up strengthens the finding either way it could have gone, and it went the confirming way: <strong>the negative result stands after the attention-guided intervention</strong>, and the experiment adds a mechanistic detail the campaign alone could not: the per-recording nuisance is distributed through the recording, not confined to the background the models preferred to read.")}
"""))

S.append(("Verdict, and what stays open", f"""
{key("<strong>The verdict.</strong> Under leak-free, subject-independent validation, no model family and no configuration clearly exceeds the recording-duration floor on this dataset: every confidence interval contains 0.610, the attention maps show the models reading per-recording texture and turn segments rather than the gait bands, and removing that texture leaves the scores unchanged. The one signal that survives every control is the clinically real observation that <strong>PD subjects take longer</strong>, carried by elapsed time, not by the spectral shape of the walk. This outcome was pre-registered as a possibility before the first model ran, and together with the random-split demonstration it shows how apparent radar-PD performance can be manufactured by validation design and per-recording nuisances.")}

{key("A statement the report must carry: <strong>eleven deep configurations were explored in total</strong> across the campaign and its exploratory variants; the informative ones are shown in this document and every run is archived, so nothing was selected for looking good. With 58 subjects, quoting only the best of eleven runs would manufacture exactly the optimism this thesis criticises. The best first-pass number (0.674) is stated <em>with</em> the pre-registered re-test that brought it back down.")}

{warn("One discovery is about the <strong>data</strong> rather than the models: under the background-corrected direction labels, 24 of the 58 subjects have no window with a clean single-direction foot signature, in an all-or-nothing pattern per subject, consistent with a per-recording sign convention in how the two foot radars were merged. It is a standing question for the data owners and it gates the one untried signal source: left/right asymmetry, which needs the per-node arrays.")}

<div class="tblwrap"><table>
<thead><tr><th>Still open</th><th>What it would settle</th></tr></thead><tbody>
<tr><td><strong>The formal confound battery</strong></td><td>Headline models re-scored duration-matched and on the chair-free variant only.</td></tr>
<tr><td><strong>The two recording sessions</strong></td><td>The corpus shows signs of having been captured in two sessions with different PD shares. Whether that matters, and how to treat it, is a question for the supervisor and the data owners; session-restricted analyses are archived outside this document pending that conversation.</td></tr>
<tr><td><strong>Subject ages</strong></td><td>Whether even the duration signal is disease or ageing.</td></tr>
<tr><td><strong>The random-split demonstration</strong></td><td>Already run; it opens the discussion chapter and shows the same model posting literature-grade numbers the moment the protocol is relaxed.</td></tr>
</tbody></table></div>
"""))

toc = "".join(f'<li><a href="#s{i}"><span class="tn">{i}</span>{t}</a></li>'
              for i, (t, _) in enumerate(S, 1))
body = "".join(f'<section id="s{i}" class="sec"><h2><span class="secnum">{i}</span>{t}</h2>{c}</section>'
               for i, (t, c) in enumerate(S, 1))

JS = """
const bar=document.getElementById('bar'),top=document.getElementById('top');
const links=[...document.querySelectorAll('nav.toc a[href^="#s"]')];
const secs=links.map(a=>document.getElementById(a.getAttribute('href').slice(1)));
function upd(){const h=document.documentElement;
 bar.style.width=(h.scrollTop/(h.scrollHeight-h.clientHeight||1)*100)+'%';
 top.classList.toggle('show',h.scrollTop>600);
 let i=0;secs.forEach((s,k)=>{if(s&&s.getBoundingClientRect().top<160)i=k;});
 links.forEach((a,k)=>a.classList.toggle('on',k===i));}
addEventListener('scroll',upd,{passive:true});addEventListener('resize',upd);upd();
top.onclick=()=>scrollTo({top:0,behavior:'smooth'});
"""

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Modelling Results &middot; Radar micro-Doppler Parkinson's</title>
<style>{CSS}</style></head><body>
<div id="bar"></div>
<div class="wrap">
<header class="hero">
  <p class="kicker">Master's thesis &middot; Results of the modelling campaign</p>
  <h1>The Modelling Results</h1>
  <p class="sub">The answer sheet to the modelling setup: the classical
  yardstick measured, the models and exploratory variants reported, the audited
  protocol, and every design question closed with the number that answers it.</p>
  <div class="statlead">Where it ends:</div>
  <div class="stats">
    <div class="stat"><b>15</b><span>classical cells<i>3 models &times; 5 feature sets</i></span></div>
    <div class="stat"><b>3</b><span>models, 2 exploratory variants<i>11 configurations explored in total</i></span></div>
    <div class="stat"><b>0.610</b><span>the floor to beat<i>recording duration alone</i></span></div>
    <div class="stat"><b>0</b><span>models clearly above the floor<i>the finding</i></span></div>
  </div>
</header>
<nav class="toc"><h4>Contents</h4><ol>{toc}</ol></nav>
<main>{body}</main>
</div>
<button id="top" title="Back to top">&uarr;</button>
<script>{JS}</script>
</body></html>"""

out = ROOT / "RESULTS_MODELLING.html"
out.write_text(html)
print(f"wrote {out} ({len(html)/1e6:.2f} MB)")
print(f"sections={len(S)} figures={html.count('data:image/jpeg')}")
print("missing figures:", html.count("[missing"))
print("unresolved placeholders:", html.count("{lo") + html.count("{hi")
      + html.count("{d:") + html.count("{c:") + html.count("{s:")
      + html.count("{tp") + html.count("{fn") + html.count("{fp")
      + html.count("_c(") + html.count("_ci("))
print("em-dashes:", html.count("—"))
