#!/usr/bin/env python3
"""CLASSICAL_SPLIT.html — the random-split detour, written up.

Same style rules as the main modelling document: no em-dashes, bold key terms,
intro -> figure -> reading, no source-file names in prose.
"""
import base64, io, json, pathlib, sys
import numpy as np
import pandas as pd
from PIL import Image

ROOT = pathlib.Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
CS = ROOT / "classical_split"
FIG = CS / "outputs/figures"
RES = json.loads((CS / "outputs/results.json").read_text())
CAMP = json.load(open(ROOT / "outputs/runs/campaign_summary.json"))
LOSO = CAMP["resnet18_fc_v2"]

def rows(exp): return [r for r in RES if r["experiment"] == exp]

def agg(exp, *path, tag="test"):
    vals = []
    for r in rows(exp):
        v = r[tag]
        for k in path: v = v[k]
        vals.append(v)
    a = np.array(vals, float)
    return a.mean(), a.std()

def cell(exp, *path, tag="test", pct=False):
    m, s = agg(exp, *path, tag=tag)
    if pct: return f"{m*100:.1f} &plusmn; {s*100:.1f} %"
    return f"{m:.3f} &plusmn; {s:.3f}"

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

HAYASHI = ('<span class="src"><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8197185/" '
           'target="_blank" rel="noopener">Hayashi et al. (2021)</a></span>')

CSS = (ROOT / "tools/eda.css").read_text() + """
.dterm .src{font-weight:400;margin-left:7px;letter-spacing:0}
.sec h4{font-family:var(--sans);font-size:15px;font-weight:700;color:var(--ink);margin:26px 0 8px}
.src a{color:inherit;text-decoration:none;border-bottom:1px dotted currentColor}
.defn>div{margin:0}
.intro{border-left:3px solid var(--accent);padding:2px 0 2px 16px;margin:22px 0 6px}
.intro p{margin:0;font-size:15.5px;line-height:1.62;max-width:none;color:var(--ink)}
td.hi{background:rgba(192,57,43,.10);font-weight:700}
"""

S = []

# ═══ 1 ═══
S.append(("Why this detour exists", f"""
<p>Everything in the main modelling document is validated leave-one-subject-out,
and under that protocol nothing on this dataset beats the confound floors. The
literature reports 90 to 99 percent. This detour closes the loop between those
two facts <strong>experimentally</strong>: we validate the way the literature
does, on our own data, with our own model, and watch what the numbers do.</p>

{key("<strong>The design principle: one variable.</strong> Same window cache, same transfer-learning model, same optimiser, same augmentation, same hardware as the thesis's corrected protocol. The <em>only</em> thing that changes in each experiment below is <strong>how the data is split</strong> into training, validation and test. Whatever the numbers do, the split did it.")}

<p>The detour also has a second job: a model scoring high under a random split
must be reading <em>something</em>. The experiments are built to answer
<strong>what</strong>: the disease, or the person.</p>
"""))

# ═══ 2 ═══
S.append(("Three splits, in increasing order of honesty", f"""
{intro("Each experiment mimics a validation practice found in the reviewed literature. All use the same 70/15/15 proportions for train, validation and test.")}

<div class="tblwrap"><table>
<thead><tr><th>Experiment</th><th>What is shuffled</th><th>What it mimics</th><th>The leak it allows</th></tr></thead><tbody>
<tr><td><strong>Window-level split</strong><br>3 seeds</td>
<td>All 1,673 windows individually, stratified by label.</td>
<td>Frame-level and &ldquo;segment-level&rdquo; splits, common in the deep-learning end of the field.</td>
<td>Total. Overlapping windows of the <em>same walk</em> sit in train and test: the model can score by matching nearly identical pictures.</td></tr>
<tr><td><strong>Recording-level split</strong><br>5 seeds</td>
<td>Whole recordings; a recording&rsquo;s windows stay together.</td>
<td>The random hold-out of {HAYASHI} and most of the reviewed work.</td>
<td>Subject identity. Each person has about six recordings, so nearly every test recording is of a person the model trained on.</td></tr>
<tr><td><strong>Collapse test</strong><br>3 seeds</td>
<td>Recording-level split as above, but <strong>8 subjects (5 control, 3 PD, the cohort&rsquo;s own proportions) are first removed entirely</strong>.</td>
<td>Nothing in the literature. It is the diagnostic.</td>
<td>The same trained model is scored twice: on test recordings of subjects it saw, and on the 8 it never saw. The gap between those two numbers <em>is</em> the leak, isolated.</td></tr>
</tbody></table></div>

{defn("Why the collapse test settles the question", "A model that learned <strong>disease</strong> should score roughly the same on both groups: Parkinson&rsquo;s looks like Parkinson&rsquo;s in people it never met. A model that learned <strong>people</strong> scores high on the familiar group, where it can recognise the walker and recall the label, and falls toward chance on the strangers. The two hypotheses make opposite predictions, so one experiment separates them.")}

{defn("On the number of repeats", "The careful end of the literature repeats its random split up to <strong>30 times</strong> with re-initialisation and reports the mean, because a single random split is noisy; our own seeds confirm that noise, spanning more than 0.1 of AUC within one design. We run 3 to 5 seeds per design rather than 30: the effect being measured here is several times larger than the seed noise, and each extra seed costs a full training.")}
"""))

# ═══ 3 ═══
S.append(("The setup", f"""
{intro("The model and training recipe are held identical to the thesis's corrected protocol, with two deliberate exceptions that copy the literature's habits. Both exceptions are stated here rather than discovered later.")}

<div class="tblwrap"><table>
<thead><tr><th>Choice</th><th>Value</th><th>Why</th></tr></thead><tbody>
<tr><td><strong>Model</strong></td><td>ResNet-18, ImageNet-pretrained, all frozen except the final linear layer (1,026 trainable weights)</td>
<td><strong>Transfer learning</strong>, as requested and as the field prefers: the frozen network is a fixed description generator, so results are stable across seeds and cheap to repeat. It is also the exact model whose leave-one-subject-out numbers we already have, which makes the comparison clean.</td></tr>
<tr><td><strong>Optimiser</strong></td><td>Adam, learning rate 10<sup>-3</sup>, weight decay 10<sup>-4</sup></td>
<td>Identical to the thesis protocol. With only the final layer training, a moderate learning rate converges in a few epochs and nothing needed tuning.</td></tr>
<tr><td><strong>Budget</strong></td><td>Up to 20 epochs, batch 32, early stopping with patience 5</td>
<td>Identical to the thesis protocol.</td></tr>
<tr><td><strong>Augmentation</strong></td><td>Doppler flip, noise, time and Doppler masking, training split only</td>
<td>Identical to the thesis protocol.</td></tr>
<tr><td><strong>Input</strong></td><td>The same cached 2 &times; 224 &times; 224 windows; standardisation computed from the training split only</td>
<td>Identical cache; the scale is still fit on training data only, so the inflation measured here cannot be blamed on that classic error.</td></tr>
<tr><td><strong>Exception 1: early stopping metric</strong></td><td>Window-level validation AUC</td>
<td>The literature stops on window or sample metrics. The thesis's subject-level stopping does not exist in a world without subject-aware splits.</td></tr>
<tr><td><strong>Exception 2: no window weighting</strong></td><td>Every window counts equally</td>
<td>The inverse-count weighting is a confound control the literature does not use; keeping it would make the mimicry less faithful.</td></tr>
<tr><td><strong>Decision threshold</strong></td><td>0.5 for accuracy, F1, precision, recall and confusion matrices</td>
<td>The literature default. AUC needs no threshold and stays the primary number.</td></tr>
</tbody></table></div>
"""))

# ═══ 4 ═══
conf_rows = ""
for exp, label in [("window_split", "Window-level split"),
                   ("recording_split", "Recording-level split")]:
    conf_rows += (
        f"<tr><td><strong>{label}</strong></td>"
        f"<td class=\"num hi\">{cell(exp, 'window', 'auc')}</td>"
        f"<td class=\"num\">{cell(exp, 'window', 'accuracy', pct=True)}</td>"
        f"<td class=\"num\">{cell(exp, 'window', 'f1')}</td>"
        f"<td class=\"num\">{cell(exp, 'window', 'precision')}</td>"
        f"<td class=\"num\">{cell(exp, 'window', 'recall')}</td>"
        f"<td class=\"num\">{cell(exp, 'recording', 'auc')}</td>"
        f"<td class=\"num\">{cell(exp, 'subject', 'auc')}</td></tr>")
seen_pct = agg("recording_split", "test_subjects_seen_in_train")[0] * 100

S.append(("What the random splits produced", f"""
{intro("Eleven trainings in total: three seeds of the window split, five of the recording split, three of the collapse test. Every number below is the mean and standard deviation over those seeds.")}

{fig("cs_ladder.png",
  "Every validation design on one axis, least honest at the top. Dots are individual seeds; the printed number is the mean. The bottom two rows are the same model's leave-one-subject-out results from the main campaign, for reference.",
  "<strong>What the figure shows.</strong> A staircase. The leakiest split sits at the top; each step of added honesty removes score; the floor is chance. The same model that manages only {loso:.2f} under leave-one-subject-out, and roughly 0.5 within an acquisition batch, jumps far above both the moment the split lets it see the test subjects during training.<br><br><strong>What it means.</strong> Since the model, data, and training recipe never changed, the entire staircase is <strong>validation design</strong>, not machine learning. This is the experimental version of the review chapter's argument: the field's headline numbers are produced by the split.".format(loso=LOSO["pooled"]))}

<h3>The headline metrics</h3>
<div class="tblwrap"><table>
<thead><tr><th>Experiment</th><th>AUC (window)</th><th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th><th>AUC per recording</th><th>AUC per subject</th></tr></thead>
<tbody>{conf_rows}</tbody></table></div>

<p>One column deserves its own sentence: in the recording-level split,
<strong>{seen_pct:.0f} % of test windows come from subjects the model trained
on</strong>. That is not an accident of one seed, it is a structural property of
splitting recordings when every person contributes six of them.</p>

{fig("cs_confusion.png",
  "Window-level confusion matrices, pooled over seeds: counts and row percentages. Left and centre are the two leaky splits; right is the never-seen group of the collapse test.",
  "<strong>What the figure shows.</strong> On the leaky splits both classes are classified well: high diagonals, balanced errors, exactly the healthy-looking tables the literature prints. On the never-seen subjects the same model's table degrades sharply, the errors spreading toward a coin flip.<br><br><strong>What it means.</strong> A clean confusion matrix is not evidence of a working disease detector. It is evidence the test set was answerable, and with familiar subjects it is answerable by recognition.")}

{fig("cs_curves.png",
  "Validation AUC per epoch for every run.",
  "<strong>What the figure shows.</strong> With familiar subjects in the validation set the curve leaps upward within the first epochs and keeps climbing. Compare the corrected leave-one-subject-out runs in the main document, where the same probe's best epoch averaged 0.5: there, with honest validation, there was nothing to climb toward.<br><br><strong>What it means.</strong> Fast, smooth learning curves feel like success. Here they are the signature of an easy shortcut being found early.")}

{fig("cs_gallery.png",
  "Test windows from a recording-level split: the four the model was most confidently right about (top) and the four it was most confidently wrong about (bottom), foot channel, with true class and predicted probability of PD.",
  "<strong>What the figure shows.</strong> Look at the backgrounds, not the strides. Every window the model calls PD with high confidence, right or wrong, has a <strong>bright, noisy background texture</strong>; every window it calls control has a <strong>dark, clean one</strong>. The confident mistakes in the bottom row are precisely the cases where the background disagrees with the label: a clean-backed PD window is called control at 0.09, a noisy-backed control window is called PD at 0.88.<br><br><strong>What it means.</strong> The model's confidence tracks a property of the <em>recording</em>, its background and enhancement texture, not a property of the <em>walk</em>. That is the same failure the literature's own cautionary tale documented, reproduced here in eight thumbnails.")}
"""))

# ═══ 5 ═══
drop_w = agg("collapse_test", "window", "auc")[0] - agg("collapse_test", "window", "auc", tag="never_seen")[0]
S.append(("The collapse test: person, not disease", f"""
{intro("The diagnostic experiment. One model per seed, trained on a leaky recording-level split of 50 subjects, then scored on both familiar and never-seen people.")}

{fig("cs_collapse.png",
  "Each line is one seed: the same trained model scored on test recordings of familiar subjects (left) and on the 8 subjects excluded from training entirely (right).",
  "<strong>What the figure shows.</strong> Every line falls. On familiar subjects the model posts the inflated numbers of the previous section, window AUC {a:.2f} on average. On the never-seen subjects it loses on average {d:.2f} of AUC, landing at {b:.2f}, consistent with the leave-one-subject-out campaign. The three repeats do not fall equally: two land at chance, one octet of strangers held at 0.74. With only 8 test subjects that spread is expected, and it is the small-sample counterpart of the foot-only batch-B episode in the main campaign; the leave-one-subject-out protocol, which averages over all 58 possible strangers, is the trustworthy version of this number.<br><br><strong>What it means.</strong> The two-hypotheses test from section 2 has its answer. A disease detector would have held its score on strangers. This model did not, so what it learned in the leaky regime was dominated by <strong>who people are</strong>, not <strong>what they have</strong>.".format(
    a=agg("collapse_test", "window", "auc")[0],
    b=agg("collapse_test", "window", "auc", tag="never_seen")[0],
    d=drop_w))}

{key("This is the thesis's central claim, demonstrated rather than argued: <strong>the same model is simultaneously a 0.8-class classifier under the literature's validation and a chance-level classifier on unseen people</strong>. Both numbers are real. Only one of them describes a screening tool.")}
"""))

# ═══ 6 ═══
S.append(("What a human could look for", f"""
{intro("Set the models aside. If a clinician were shown one of these spectrograms, what visible characteristics would distinguish Parkinsonian gait? The literature names several; our own exploratory analysis tested each on this dataset. The figure shows one median-length recording per group, then the table says what to look for and what this dataset actually delivers.")}

{fig("cs_examples.png",
  "One whole recording per group, foot channel, both from the chair variant, each of median length for its group, drawn on the same time axis.",
  "<strong>What the figure shows.</strong> First the anatomy, which is the same in both: spiky excursions to high frequencies are the feet, one spike per step; the walk out sits below the 0 Hz line (motion away), the walk back sits above it; the sign flip in the middle is the turn; the quiet, noisy stretches at the edges are sitting and standing. Now the two honest observations. The walking passes themselves look strikingly alike. And this median control recording is actually slightly <em>longer</em> than the median PD one, 10.4 against 9.8 seconds.<br><br><strong>What it means.</strong> The one confirmed cue, PD taking about 10 % longer, is a <strong>group average</strong>, not a rule readable off a single pair: the two distributions overlap heavily, and a typical member of each group can order either way. That is exactly why looking at one spectrogram, human or machine, does not diagnose anyone on this data. Note also that the two backgrounds differ in texture; that is per-recording enhancement and noise, not disease, and it is precisely what the gallery above showed the leaky model exploiting.")}

<h3>The characteristics, one by one</h3>
<div class="tblwrap"><table>
<thead><tr><th>What to look for</th><th>How it appears in the spectrogram</th><th>What this dataset shows</th></tr></thead><tbody>
<tr><td><strong>Slower peak foot speed</strong> (bradykinesia, the cardinal motor sign)</td>
<td>The foot spikes reach <em>less far</em> from the 0 Hz line; less energy at the highest frequencies.</td>
<td><strong>Reversed.</strong> The PD group&rsquo;s foot-band centre of mass sits <em>higher</em>, not lower, and the per-recording contrast enhancement saturates the peaks of both groups, so peak speed is not even readable from these images.</td></tr>
<tr><td><strong>Fainter, smaller foot bursts</strong> (short, shuffling steps)</td>
<td>A dimmer, less busy picture.</td>
<td><strong>Confounded.</strong> PD recordings carry <em>more</em> total energy, purely because they last longer; and after per-recording enhancement, brightness is not comparable across recordings at all.</td></tr>
<tr><td><strong>The walk takes longer</strong> (slowness, hesitation, harder sit-to-stand)</td>
<td>The whole picture is <em>wider</em>, with more slow-speed content around the standing and turning moments.</td>
<td><strong>Confirmed, and alone.</strong> PD trials run 10 % longer in the chair variant, the difference holds inside both acquisition batches, and it is the only cue in this table that survives every control. It is also visible to the naked eye.</td></tr>
<tr><td><strong>Irregular rhythm</strong> (disturbed stride-to-stride timing)</td>
<td>Unevenly spaced foot spikes; missing &ldquo;teeth&rdquo;.</td>
<td><strong>Reversed.</strong> Our regularity measure came out <em>lower</em> in PD, meaning a flatter, less modulated image, opposite to the clinical prior.</td></tr>
<tr><td><strong>Wider spread of speeds</strong></td>
<td>A broader vertical smear around the centre line.</td>
<td><strong>Reversed and confounded.</strong> PD images are broader and noisier, consistent with background texture differing between recordings, not with faster PD gait.</td></tr>
<tr><td><strong>Less foot swing relative to the trunk</strong></td>
<td>The fast band shrinks relative to the slow band.</td>
<td><strong>Reversed.</strong> The foot-to-torso ratio is <em>higher</em> in PD, and it tracks recording length.</td></tr>
<tr><td><strong>Trunk stops answering the feet</strong> (weakened heel-strike coupling, the source publication&rsquo;s marker)</td>
<td>The torso band&rsquo;s small pulses lose their timing with the foot spikes.</td>
<td><strong>Untested.</strong> No coupling measure has been computed yet, and the deep campaign&rsquo;s best channel was the foot alone, which gives no indirect support either.</td></tr>
<tr><td><strong>Left/right asymmetry</strong> (an established PD marker)</td>
<td>Alternating foot bursts of unequal reach or brightness.</td>
<td><strong>Invisible in this data.</strong> The two foot radars are merged into one channel before we ever see them; the per-node arrays would be needed, which is an open question to the data owners.</td></tr>
</tbody></table></div>

{warn("The scorecard: of the eight characteristics the literature and clinical intuition predict, this dataset <strong>confirms one</strong> (the walk takes longer), <strong>reverses four</strong>, finds one <strong>confounded beyond use</strong>, and leaves two <strong>unmeasurable</strong> with the current data. So a human reading these spectrograms honestly has one reliable cue, elapsed time, which is exactly what every quantitative result in the thesis keeps finding. Two standing caveats apply to the whole table: ages are unknown, and radar separates young from elderly walkers at 94.9 %, so even the surviving cue could be age; and the per-recording enhancement makes brightness comparisons between recordings unsafe in principle.")}
"""))

# ═══ 7 ═══
S.append(("Verdict, and what this buys the thesis", f"""
{key("<strong>Validated the literature's way, this dataset produces literature-grade numbers: window AUC {w:.2f} on a window split and {r:.2f} on a recording split, with accuracies in the {acc:.0f} % range. Validated honestly, the same model, data and recipe sit at the confound floors.</strong> The gap is not a property of Parkinson's, radar, or the network. It is a property of the split.".format(
    w=agg("window_split", "window", "auc")[0],
    r=agg("recording_split", "window", "auc")[0],
    acc=agg("recording_split", "window", "accuracy")[0]*100))}

<div class="tblwrap"><table>
<thead><tr><th>What this detour contributes</th><th>Where it lands in the report</th></tr></thead><tbody>
<tr><td>An <strong>experimental measurement of the inflation</strong> the review chapter could only allege from reading papers: same everything, split changed, score jumps.</td><td>Results chapter, beside the leave-one-subject-out campaign; one figure and one table.</td></tr>
<tr><td>The <strong>collapse test</strong>, direct evidence that the inflated score is person recognition.</td><td>Discussion, as the empirical backbone of the validation-rigour argument.</td></tr>
<tr><td>The <strong>human-readable characteristics</strong> section, connecting the spectrograms to clinical gait signs and to what this dataset actually shows.</td><td>Introduction or dataset chapter, as reader orientation.</td></tr>
</tbody></table></div>

<p>None of these numbers is a claim of performance. They are the opposite: a
controlled demonstration of how performance is manufactured, run so the thesis
can say it with evidence instead of citation.</p>
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
<title>The Random-Split Detour &middot; Radar micro-Doppler Parkinson's</title>
<style>{CSS}</style></head><body>
<div id="bar"></div>
<div class="wrap">
<header class="hero">
  <p class="kicker">Master's thesis &middot; A controlled demonstration</p>
  <h1>The Random-Split Detour</h1>
  <p class="sub">What happens when this dataset is validated the way the
  literature validates: the same transfer-learning model, the same windows, the
  same training recipe, and only the split changed. Plus: what a human could
  look for in these spectrograms.</p>
  <div class="statlead">Where it ends:</div>
  <div class="stats">
    <div class="stat"><b>{agg("window_split","window","auc")[0]:.2f}</b><span>window-split AUC<i>the leakiest design</i></span></div>
    <div class="stat"><b>{agg("recording_split","window","auc")[0]:.2f}</b><span>recording-split AUC<i>the literature's hold-out</i></span></div>
    <div class="stat"><b>{agg("collapse_test","window","auc",tag="never_seen")[0]:.2f}</b><span>on never-seen subjects<i>the same trained model</i></span></div>
    <div class="stat"><b>{LOSO["pooled"]:.2f}</b><span>honest LOSO reference<i>main campaign</i></span></div>
  </div>
</header>
<nav class="toc"><h4>Contents</h4><ol>{toc}</ol></nav>
<main>{body}</main>
</div>
<button id="top" title="Back to top">&uarr;</button>
<script>{JS}</script>
</body></html>"""

out = CS / "CLASSICAL_SPLIT.html"
out.write_text(html)
print(f"wrote {out} ({len(html)/1e6:.2f} MB)")
print(f"sections={len(S)} figures={html.count('data:image/jpeg')}")
if "__HUMAN_TABLE__" in html:
    print("NOTE: __HUMAN_TABLE__ placeholder still present")
