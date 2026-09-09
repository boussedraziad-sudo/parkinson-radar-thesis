#!/usr/bin/env python3
"""MODELLING_SETUP.html — everything decided before a single model is trained.

Style rules: no em-dashes, key terms bolded, lists over prose, nothing phrased
as a question the author asks himself, no source-file names in the prose.
Every step follows the same shape: what we are about to do, the picture, what
the picture shows, and what it means for the work.
"""
import base64, io, json, pathlib, sys
import pandas as pd
from PIL import Image

ROOT = pathlib.Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
FIG, TOOLS = ROOT / "outputs/figures", ROOT / "tools"
sys.path.insert(0, str(ROOT))

BS = json.load(open(ROOT / "reports/baseline_summary.json"))
MAN = pd.read_csv(ROOT / "outputs/preprocessed/manifest.csv")
R, FS = BS["results"], BS["feature_sets"]
NW = len(MAN)
n_all, n_inv, n_cln, n_cnf = (len(FS["all_features"]), len(FS["duration_invariant"]),
                              len(FS["shape_clean"]), len(FS["confounded_only"]))
best, clean, dur = (R["logreg|confounded_only"], R["logreg|shape_clean"],
                    R["logreg|duration_only"])
lo, hi = best["auc_lo"], best["auc_hi"]
CAMP = json.load(open(ROOT / "outputs/runs/campaign_summary.json"))
def _c(key, field, nd=3):
    v = CAMP[key].get(field)
    return "n/a" if v is None else f"{v:.{nd}f}"


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
def later(t): return f'<div class="later"><span class="ltag">Reported again after training</span><p>{t}</p></div>'
def step(n, t): return f'<h3><span class="stepn">{n}</span>{t}</h3>'

PAPER = ('<span class="src"><a href="https://doi.org/10.1109/TBME.2025.3583785" '
         'target="_blank" rel="noopener">L&oacute;pez-Delgado et al. (2026)</a></span>')
HAYASHI = ('<span class="src"><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8197185/" '
           'target="_blank" rel="noopener">Hayashi et al. (2021)</a></span>')

CSS = (TOOLS / "eda.css").read_text() + """
.dterm .src{font-weight:400;margin-left:7px;letter-spacing:0}
.sec h4{font-family:var(--sans);font-size:15px;font-weight:700;color:var(--ink);margin:26px 0 8px}
.src a{color:inherit;text-decoration:none;border-bottom:1px dotted currentColor}
.defn>div{margin:0}
.stepn{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;
 border-radius:50%;background:var(--accent-soft);color:var(--accent);font-size:14px;
 font-weight:700;margin-right:11px;vertical-align:-4px}
.math{font-family:var(--mono);font-size:14.5px;line-height:1.85;background:var(--panel);
 border:1px solid var(--line);border-radius:8px;padding:13px 16px;margin:13px 0;
 text-align:center;overflow-x:auto}
.math .mnote{display:block;font-family:var(--sans);font-size:12.5px;color:var(--muted);
 margin-top:7px;font-style:italic}
pre.code{font-family:var(--mono);font-size:12.5px;line-height:1.6;background:var(--panel);
 border:1px solid var(--line);border-left:3px solid var(--key-line);border-radius:0 8px 8px 0;
 padding:13px 16px;margin:14px 0;overflow-x:auto;white-space:pre}
.intro{border-left:3px solid var(--accent);padding:2px 0 2px 16px;margin:22px 0 6px}
.intro p{margin:0;font-size:15.5px;line-height:1.62;max-width:none;color:var(--ink)}
td.hi{background:rgba(31,122,85,.10);font-weight:700}
td.lo{background:rgba(192,57,43,.10)}
"""


def intro(t): return f'<div class="intro"><p>{t}</p></div>'


S = []

# ═══════════════════════════ 1 ═══════════════════════════
S.append(("From a recording to a tensor", f"""
<p>The dataset is 348 recordings. A network needs inputs of one fixed size, and
348 examples is not enough to train one. Preprocessing solves both problems in a
single pass, and this section follows a recording all the way through it.</p>

{fig("pipe_overview.png",
  "The whole deep-learning path from a stored recording to one score per person. The upper row runs once and writes a cache to disk; the lower row runs inside every fold of the validation loop.",
  "Read the shapes along the bottom of each box: they are the dimensions of <strong>the data itself</strong> as it leaves each stage, speed slots by time instants at the start, and they tell the story on their own. A recording starts as <strong>320 &times; 12,000 numbers</strong> and ends as <strong>one number between 0 and 1</strong>. The network itself is already in the picture: the third box of the lower row is where each finished window enters the CNN and leaves as two scores, so everything before that box exists to manufacture its input.<br><br>The lower row is labelled <strong>per fold</strong>. A <strong>fold</strong> is one round of the validation loop: one subject is set aside as the test, every step of the lower row runs using only the other 57 subjects, and then the round repeats with a different subject set aside, 58 rounds in all. So yes, the data a fold trains on is exactly the recordings that do <em>not</em> belong to its held-out subject. The two rows split the work accordingly: everything identical for every fold is computed once and cached (upper row), and everything that must not see the held-out subject is re-done inside each fold (lower row). One scope note: this figure is the <strong>deep-learning path only</strong>. The classical baseline never sees windows at all; it has its own five-stage flow, drawn in section 2.")}

{defn2("The whole reduction, in numbers", '''
<p>The stages below are explained one at a time in the rest of this section.
Together they answer how a recording becomes a single score.</p>
<div class="tblwrap"><table>
<thead><tr><th>Stage</th><th>Shape</th><th>What one number in it is</th></tr></thead><tbody>
<tr><td>One recording</td><td class="num">320 &times; 12,000</td>
<td>320 speed slots against 12,000 instants. Each value is <strong>how much
reflected energy came back at that speed at that instant</strong>, so one column
is a snapshot of the whole body at one moment: rows near zero are the slow parts
(the trunk), rows far from zero are the fast parts (a swinging foot).</td></tr>
<tr><td>One window</td><td class="num">320 &times; 4,800</td>
<td>The same, cut down to 3.0 seconds. Recordings differ in length, so they yield
between <strong>2 and 11</strong> windows each, 4.8 on average, which is how 348
recordings give 1,673 windows rather than exactly 348 &times; 5. That some people
yield more windows than others is itself a problem, handled in stage 4.</td></tr>
<tr><td>After resizing</td><td class="num">224 &times; 224</td>
<td>The same picture, redrawn on a smaller grid.</td></tr>
<tr><td>After stacking</td><td class="num">2 &times; 224 &times; 224</td>
<td>The two resized window pictures, foot channel and torso channel, held
together in one array.</td></tr>
<tr><td>Network output</td><td class="num">2 numbers</td>
<td>What the network produces for one window: a control score and a PD
score.</td></tr>
<tr><td>Per window</td><td class="num">1 number</td>
<td>The window&rsquo;s two scores converted into a single probability, between 0
and 1, that this window comes from a PD walk.</td></tr>
<tr><td><strong>Per subject</strong></td><td class="num"><strong>1 number</strong></td>
<td>The <strong>average of all the window probabilities belonging to one
person</strong>: their final PD score, and the answer the thesis reports.</td></tr>
</tbody></table></div>
''')}

{step(1, "Cut each recording into fixed-length windows")}
{intro("The first stage takes one variable-length recording and produces several fixed-length pieces. The figure shows the cuts drawn onto a real recording.")}

{fig("pipe_windowing.png",
  "Above: one real recording with every cut point drawn as a white line, and each resulting window labelled. Below: the same windows as bars, so the overlap between them is visible.",
  "<strong>What the figure shows.</strong> A 7.5 second recording produces four windows. The lower panel makes the key detail visible: <strong>the windows overlap</strong>. Window 1 does not begin where window 0 ends. It begins 1.5 seconds after window 0 <em>began</em>, so the two share 1.5 seconds of content.<br><br><strong>What it means.</strong> Every input to the network is now exactly 3.0 seconds, whatever the length of the walk it came from. Recording length can no longer reach the network through the size of its input. It can still reach it through the <em>number</em> of windows, which is dealt with in stage 4.")}

{defn2("Why windows of 3.0 seconds, starting every 1.5 seconds", '''
<p><strong>Why 3.0 seconds long.</strong> One stride, left foot down to left foot
down again, takes about <strong>1.0 to 1.2 seconds</strong>. One of the three
clinically motivated measurements asks how much a person&rsquo;s strides differ
from each other, and to compare strides at least two whole ones must be visible
in the same window. Three seconds is roughly the shortest window that always
holds <strong>two to three complete strides</strong>; shorter would break that
measurement, and longer would cut the already small number of windows further.</p>
<p><strong>What the 1.5 second hop means.</strong> Nothing more than where the
next window begins: a new window starts every 1.5 seconds, and each one lasts
3.0 seconds.</p>
<div class="math">window 0 covers 0.0 to 3.0 s
window 1 covers 1.5 to 4.5 s
window 2 covers 3.0 to 6.0 s
<span class="mnote">each window starts 1.5 s after the previous one and lasts 3.0 s,
so neighbours share half their content</span></div>
<p><strong>Why they overlap on purpose.</strong> Two reasons. Starting a new
window only every 3 seconds would produce half as many windows from an already
small dataset. And any stride that happened to sit across a cut would be split
in two, seen whole by no window; with the overlap, every moment of the walk
appears complete inside at least one window.</p>
<p>A finer setting, 2.0 second windows starting every 0.5 seconds, would give far
more windows, but neighbours would then share <strong>75 %</strong> of their
content, so the extra windows are near-copies rather than new information.</p>
''')}

{step(2, "Compress the brightness range")}
{intro("The second stage changes the numbers rather than the shape. It is one line of arithmetic, but skipping it would make the foot signal nearly invisible to the network.")}

{defn2("Why the values need compressing", '''
<p>The trunk is a large, flat reflector and a foot is a small one, so the trunk
sends back <strong>on the order of a hundred times more energy</strong>. Written
on a straight scale the picture is then almost useless: the trunk band occupies
the top of the range and everything a foot does is squeezed into the bottom few
percent, where a network cannot separate it from noise.</p>
<div class="math">compressed = log(1 + value)
<span class="mnote">adding 1 first is what keeps zero mapped to zero, so silence stays silence</span></div>
<p>A logarithm turns <em>ratios</em> into <em>distances</em>. Before it, the gap
between 1 and 10 looks tiny next to the gap between 1,000 and 10,000. After it,
both are the same size, because both are a factor of ten. Large values are pulled
in hard, small ones are barely moved, and the foot band becomes as legible as the
trunk band.</p>
''')}

{defn2("Where the scaling is finished, and why not here", '''
<p>Networks train badly on inputs that are not centred near zero, so the numbers
also have to be <strong>standardised</strong>: subtract an average, divide by a
spread.</p>
<div class="math">standardised = (value &minus; mean) / spread
<span class="mnote">mean = the average brightness over all the training windows;
spread = the standard deviation, how far a typical value sits from that average</span></div>
<p>The question is <em>whose</em> average. A <strong>fold</strong> is one
repetition of the validation loop: one subject held out, the rest used to train.
There are 58 folds because there are 58 subjects, and the word has nothing to do
with the diagnosis label.</p>
<p>If the average were computed over the whole dataset, the held-out subject would
have helped set the scale against which its own input is later measured. That is a
small leak, but the entire argument of this thesis is about not having any, so the
cached files hold the compressed values <strong>only</strong>, and the
standardisation is applied when the data is loaded, using
<strong>the training subjects of that fold</strong>.</p>
<p>The alternative is to standardise each window against its own average, which is
what many published pipelines do by default. It is not used here, because
<strong>standardising a window against itself deletes its absolute
brightness</strong>, and that one deletion would do two different things at once.
It could delete a <em>signal</em>: brightness is how strongly the moving body
reflects, which is part of what <strong>bradykinesia</strong> changes. And it would
also delete a known <em>nuisance</em>: the two processing runs differ in overall
brightness too. Since we cannot know in advance whether brightness is carrying
disease, artifact, or both, neither option is safe to just assert. So the default
pipeline <strong>keeps</strong> brightness, everything is later re-run once more
with only this one choice changed, and section 5 compares the two, using the
within-batch score to tell which of the two things the brightness was actually
carrying.</p>
''')}

{step(3, "Redraw on a common grid and stack the two radars")}
{intro("The third stage makes every window the same size in pixels and joins the two radars into one array. This is the last stage that touches the picture; what comes out is exactly what the network sees.")}

{fig("pipe_model_input.png",
  "One cached window. Channel 0 is the foot-aimed node and channel 1 is the torso-aimed node, both recorded at the same moment during the same walk by the same subject.",
  "<strong>What the figure shows.</strong> Two pictures of the same three seconds. The brightness at any point is the energy the Fourier transform found at that speed and instant, after the compression of stage 2, so a bright pixel means a lot of the body was moving at that speed then. The two look different because the nodes are aimed at different heights: the foot channel is spiky and reaches high speeds, the torso channel is a smooth band near zero.<br><br><strong>What it means.</strong> The network is given both at once rather than one at a time, because the source publication {p} establishes that the relationship <em>between</em> trunk and foot motion is what breaks down in Parkinson's. Concretely: in a healthy walk the trunk is not a passive block. It makes a small speed-up at <strong>every heel strike</strong>, so the smooth torso band pulses in time with the foot spikes. Parkinson's stiffens the trunk and shortens the steps, and those pulses weaken and drift out of step with the feet. That pattern exists only <em>between</em> the two pictures; a model shown one channel at a time has nothing to represent it with.".format(p=PAPER))}

{defn2("How the resizing is done, and what happens to the windows afterwards", '''
<p><strong>Each window is resized on its own</strong>, from 320 &times; 4,800 to
224 &times; 224, by <strong>bilinear interpolation</strong>, the same operation
that shrinks a photograph: every pixel of the new grid is a weighted average of
the input pixels around its position. 224 is not tuned; it is the input size the
pretrained network expects. The two resized pictures are then
<strong>stacked</strong> into one 2 &times; 224 &times; 224 array, the way a
colour photo stacks red, green and blue.</p>
<p><strong>Each window becomes its own file on disk</strong>, and a
<strong>catalogue of the windows</strong> is written beside them: one row per
window, recording which subject it came from, their diagnosis, which recording
it was cut from and where within it, and where the file sits. Training never
touches the raw recordings again; it works from this catalogue, so picking out
any group of subjects, say the 49 that train in one fold, is just filtering the
catalogue by subject. (There are still 58 subjects in total; <strong>49</strong>
is how many <em>train</em> inside one validation fold, where the other 9 are
held aside: 8 to decide when to stop training, 1 as the test subject. That
split is the subject of section 4.)</p>
''')}

{key("<strong>The dataset did grow, but not in the way that would help.</strong> 348 recordings became {nw} windows, so the network sees roughly five times more examples. Those examples are <strong>not independent</strong>: five windows from one walk overlap each other and come from one person, so they carry far less than five recordings' worth of information. This is why the number of <em>subjects</em>, still 58, remains the real constraint, why folds are split by subject and never by window, and why uncertainty is estimated by resampling subjects rather than windows.".format(nw=NW))}

{step(4, "Stop long recordings from carrying more weight")}
{intro("Windowing fixed the size of each input but not the number of them. This stage measures that problem and corrects it.")}

{fig("pipe_window_count.png",
  "Each dot is one subject. On the left, the share of training influence they would carry as recorded. On the right, the same after the correction.",
  "<strong>What the figure shows.</strong> On the left the dots spread from about 0.5 to 2.0, so some subjects would contribute four times as much to training as others, purely because their recordings are longer. PD subjects average <strong>30.9 windows against 27.3</strong>, and window count on its own separates the two groups at <strong>AUC 0.62</strong>. On the right every dot sits on 1.0.<br><br><strong>What it means.</strong> Without a correction a network could reach 0.62 by responding to how many windows a person contributed, which is a restatement of how long they took. The correction is one multiplication: each window's contribution to training is scaled by <strong>1 divided by the number of windows its subject has</strong>. Someone with 40 windows counts 1/40 per window, someone with 20 counts 1/20, so every subject adds up to <strong>exactly the same total influence</strong>, which is why every dot on the right lands on 1.0. The confound is removed at the point where it would have acted.")}

{step(5, "Record which direction the subject was walking")}
{intro("The last stage adds no processing. It attaches one extra number to every window so that a specific experiment becomes possible later. Why that experiment is worth setting up comes first; the figure then shows what the label looks like and what the experiment would cost.")}

{defn2("Why the turn is worth isolating", '''
<p>The source publication {p} does not compute its gait parameters on whole
recordings. It first separates out the walking passes and
<strong>discards the turning, standing and sitting stages</strong>, then measures
gait on what is left. We currently keep everything.</p>
<p>That difference matters here for a reason specific to this dataset. The
exploratory analysis found that PD subjects take about <strong>10 % longer</strong>
in the variant with the chair, and that those extra seconds are concentrated in
<strong>standing up, hesitating and turning</strong> rather than in steady walking.
So the confound and the discarded stages are <em>the same moments</em>.</p>
<p>Restricting the analysis to windows well inside a walking pass therefore attacks
the confound at its source rather than correcting for it afterwards. It is not free:
it discards about a third of the data, and with 58 subjects that is a real cost. The
direction is recorded for every window so the trade can be measured rather than
assumed.</p>
'''.format(p=PAPER))}

{fig("pipe_turn.png",
  "Left: how much of each window's energy sits at positive Doppler, meaning motion towards the node. Right: how many windows survive if windows near the midpoint are discarded.",
  "<strong>What the figure shows.</strong> The left panel has three groups. Windows near <strong>0</strong> lie inside the outward pass, where all motion is away from the node. Windows near <strong>1</strong> lie inside the return pass. Windows near <strong>0.5</strong> contain the moment the subject turned around, so they hold energy in both directions at once. The right panel prices the exclusion: removing everything within 0.1 of the midpoint keeps <strong>65 %</strong> of the windows.<br><br><strong>What it means.</strong> The experiment described above is now cheap to run: discard the mixed-direction windows and everything near them, re-run the pipeline, and see whether the score survives on steady walking alone. The right panel says the price of that experiment in advance.")}

{key("<strong>The frozen configuration.</strong> 3.0 s windows, 1.5 s hop, contrast-enhanced representation, compression by log(1+x) with standardisation deferred into each fold, 224 &times; 224, two channels, inverse-count weighting, every window kept. It produced <strong>{nw} windows from 348 recordings</strong> with no failures. Every alternative named above is tested later by an <strong>ablation</strong>: removing or changing one single component of the pipeline and re-running everything, to measure how that specific part affects the overall performance. The choices are measured, not asserted.".format(nw=NW))}
"""))

# ═══════════════════════════ 2 ═══════════════════════════
rows = []
for mdl in ["logreg", "svm_rbf", "random_forest"]:
    cells = []
    for fs in ["all_features", "duration_invariant", "shape_clean", "confounded_only", "duration_only"]:
        v = R.get(f"{mdl}|{fs}", {}).get("auc")
        cls = ' class="hi"' if (v is not None and v >= 0.66) else ""
        cells.append(f'<td{cls}>{v:.3f}</td>' if v is not None else "<td>n/a</td>")
    rows.append(f'<tr><td><strong>{mdl}</strong></td>' + "".join(cells) + "</tr>")
GRID = "".join(rows)

S.append(("The classical baseline, and why it exists", f"""
{key("<strong>The deep models are the goal of this thesis; the classifiers in this section are the yardstick.</strong> Before training any network, we measure how far three ordinary statistical classifiers get using the 25 handcrafted measurements. Whatever they score becomes the bar. A network that cannot clearly beat that bar has added nothing, however sophisticated it is. The grid of numbers this section produces is reported in section 5, where every deep result is read against it.")}

<h3>How one classical prediction is made</h3>
{intro("Before any numbers, here is the procedure. It runs once per subject, so the whole thing happens 58 times to produce one score.")}

{fig("pipe_classical_flow.png",
  "The five stages of one fold of the classical baseline. The loop runs once per subject.",
  "<strong>Stage 1, hold out one subject.</strong> All of that person's recordings are set aside. Nothing about them touches the rest of the stage.<br><br><strong>Stage 2, fill gaps.</strong> This stage is a guard rather than a repair. If any measurement were undefined for some recording, say a frequency band with no energy in it, which would make an average of that band meaningless, it would be replaced by the median of that measurement across the training subjects, so no recording is discarded and no value is borrowed from the held-out person. On this dataset the guard never fires: all 348 recordings produced all 25 measurements. It is stated because a protocol has to say in advance what happens when something is missing, not because something was.<br><br><strong>Stage 3, put everything on one scale.</strong> The measurements have wildly different units: a spectral centroid is a frequency in the hundreds, a ratio is a fraction below 1. Each is rescaled to average 0 and spread 1 using the training subjects only, so no measurement dominates merely because its numbers are bigger.<br><br><strong>Stage 4, fit the classifier</strong> on the 57 remaining subjects. The cohort is 57 % control and 43 % PD, so a model that always answers control would already look 57 % right; to remove that temptation, <strong>an error on a PD example is made proportionally more expensive</strong>, each class's mistakes scaled by the inverse of its share, so the smaller PD group carries the same total influence on the fit as the larger control group.<br><br><strong>Stage 5, score and average.</strong> The classifier never answers &ldquo;PD&rdquo; or &ldquo;control&rdquo; outright. It answers with a <strong>probability between 0 and 1</strong>, one per recording, and the held-out subject's six probabilities are averaged into <strong>one number for that person</strong>. Only after all 58 subjects have their number are the 58 probabilities compared against the true diagnoses, and that comparison is the AUC.")}

{defn("Why the average, and why subjects rather than recordings", "Averaging gives the answer a clinician would actually want, one per person. It also makes the score <strong>insensitive to how many recordings a subject happens to have</strong>, which is the same confound the window weighting handles on the deep side. Scoring recordings instead would let a subject with more recordings count more.")}

{defn("The same protocol as the deep models, with one piece missing", "The classical models are validated leave-one-subject-out exactly like the networks, but they skip the inner 8-subject validation group. That group exists for one job only: deciding, epoch by epoch, when to stop training a network. Logistic regression and the other two fit in a single pass with fixed settings, nothing unfolds over time and nothing is tuned along the way, so there is nothing for a validation group to decide. Hence <strong>57 train / 1 test</strong> here, against <strong>49 train / 8 validation / 1 test</strong> on the deep side.")}

<h3>Three classifiers, chosen to differ in what they can express</h3>
{intro("Three models are compared, not to find a winner, but because they can represent different kinds of pattern, so their agreement and disagreement is itself informative. If only logistic regression succeeds, the signal is a simple trend: more of some measurement means more likely PD. If only the forest succeeds, the signal is conditional, something like &ldquo;this measurement matters only when that one is high&rdquo;, which a straight line cannot express. And if all three fail together, the reasonable reading is that no signal of any of these shapes is present at this sample size, rather than that one algorithm was unlucky.")}

<div class="tblwrap"><table>
<thead><tr><th>Model</th><th>Why it suits this problem</th></tr></thead><tbody>
<tr><td><strong>Logistic regression</strong></td>
<td>Gives every measurement a weight, adds up the evidence, and turns the total into a probability. <strong>The most restrictive of the three, therefore the hardest to fool</strong>: with 58 subjects a flexible model can fit noise, this one mostly cannot. Its weights are also readable, so a result comes with an explanation.</td></tr>
<tr><td><strong>Support vector machine (RBF kernel)</strong></td>
<td>Draws a <strong>curved</strong> boundary between the groups, placed as far as possible from the nearest subjects on either side. Catches signals that only show up in combinations of measurements, and keeping that wide margin is a built-in caution that helps at this sample size.</td></tr>
<tr><td><strong>Random forest</strong></td>
<td>Grows many decision trees on random parts of the data and averages their votes. Catches <strong>thresholds and conditional patterns</strong>, a signal that lives in a cut-off rather than a trend, and it ranks the measurements by usefulness for free, which feeds the interpretation.</td></tr>
</tbody></table></div>

<h3>Five sets of measurements, and what each one is for</h3>
{intro("The same three models are also run on five different sets of the measurements, giving a grid of 15 numbers. The five sets are not attempts at a better score, and they are not tuned over. They differ in exactly one respect: how much of the recording-length confound each is allowed to contain, from all of it down to nothing but it. Reading the grid across is therefore a controlled experiment. If the score holds up as the length-carrying measurements are removed, the models were reading gait. If it falls step by step as length is removed, they were reading the recording length.")}

<div class="tblwrap"><table>
<thead><tr><th>Set</th><th>Size</th><th>What is in it</th><th>The question it answers</th></tr></thead><tbody>
<tr><td><code>all_features</code></td><td class="num">{n_all}</td>
<td>Every measurement kept for analysis.</td>
<td>How well can we do with no restraint? This is the number most published work would report.</td></tr>
<tr><td><code>duration_invariant</code></td><td class="num">{n_inv}</td>
<td>Averages, spectral shapes and ratios. Anything that is a total was removed,
because a total keeps growing the longer the walk lasts, while an average or a
ratio does not.</td>
<td>How well can we do once recording length is removed <strong>by how the
measurements are defined</strong>?</td></tr>
<tr><td><code>shape_clean</code></td><td class="num">{n_cln}</td>
<td>The subset of the above that, when checked directly against each
recording&rsquo;s actual length, does not follow it. The check exists because
definitions were not enough: a longer recording holds more standing and
turning, and that extra slow content drags some averages with it.</td>
<td>How well can we do once length is removed <strong>in practice, not just on
paper</strong>?</td></tr>
<tr><td><code>confounded_only</code></td><td class="num">{n_cnf}</td>
<td>Only the totals, which grow with the length of the recording.</td>
<td>How much does the confound alone buy?</td></tr>
<tr><td><code>duration_only</code></td><td class="num">1</td>
<td>The length of the recording, and nothing else.</td>
<td><strong>The floor.</strong> Any result not clearly above this has demonstrated nothing about gait.</td></tr>
</tbody></table></div>

<p>The score every configuration is judged by, the <strong>AUC</strong>, and the
reason it is the right one, are defined in <strong>section 4</strong>, together
with the leave-one-subject-out protocol that produces it.</p>

{later("This section set up the yardstick; it deliberately shows no numbers. The complete 15-cell grid it produces opens <strong>section 5</strong>, where it is the first thing every deep result is read against.")}
"""))

# ═══════════════════════════ 3 ═══════════════════════════
S.append(("The deep models", f"""
<p>With the yardstick defined and the input pipeline built, what remains is
choosing the networks themselves. Two are used, and the choice of both is
dominated by one number: <strong>58 subjects</strong>.</p>

{fig("pipe_models.png",
  "Trainable parameters for each option on a logarithmic scale, against the number of training windows available.",
  "<strong>What the figure shows.</strong> The vertical line is {nw} training windows. A model to the right of it has <strong>more free parameters than we have examples</strong>, which means it can in principle memorise every training window exactly and learn nothing general.<br><br><strong>What it means.</strong> The two multi-million-parameter options sit far beyond the line and are run only to demonstrate that overfitting, never as headline configurations. Of the two used, the probe sits below the line outright, and SmallCNN sits close enough to it that its capacity is held in check by dropout, augmentation and weighting rather than assumed safe.".format(nw=NW))}

<h3>Model A: SmallCNN, trained from scratch</h3>
{intro("The first network is built for this problem and learns everything from the radar data. It is deliberately small.")}

{fig("pipe_smallcnn.png",
  "Every layer of SmallCNN with the shape of the tensor leaving it. Box height is proportional to the size of that tensor.",
  "<strong>Reading it left to right.</strong> The picture gets smaller while the number of feature maps grows, which is the standard shape of a convolutional network: trade spatial detail for a richer description of what is present.")}

<div class="tblwrap"><table>
<thead><tr><th>Layer</th><th>What it does</th><th>Why it is there</th></tr></thead><tbody>
<tr><td><strong>Convolution 3&times;3</strong></td>
<td>Slides a small 3&times;3 filter across the picture and records how strongly each position matches it. Sixteen filters in the first block, then 32, then 64.</td>
<td>A filter learns a local pattern: an edge, a bright ridge, a repeating band. Sliding it means the pattern is recognised <strong>wherever</strong> it occurs, which suits gait because a stride can begin anywhere in a window.</td></tr>
<tr><td><strong>Batch normalisation</strong></td>
<td>Rescales the values leaving a layer back to a stable range.</td>
<td>Keeps training numerically well behaved and allows a larger learning rate, so fewer epochs are needed.</td></tr>
<tr><td><strong>ReLU</strong></td>
<td>Replaces every negative value with zero.</td>
<td>Without a non-linear step, stacked convolutions collapse into a single one and depth buys nothing.</td></tr>
<tr><td><strong>Max pooling</strong></td>
<td>Halves the width and height by keeping the strongest value in each 2&times;2 square.</td>
<td>Discards precise position while keeping presence, and cuts the computation of every later layer fourfold.</td></tr>
<tr><td><strong>Global average pooling</strong></td>
<td>Replaces each 56&times;56 feature map by its single average value, leaving 64 numbers.</td>
<td><strong>The most consequential choice in the design.</strong> Flattening that map into a dense layer instead would need roughly <strong>200,000 weights in that one layer</strong>, eight times the whole network. Averaging removes them, and forces each filter to mean something regardless of where in the window it fired.</td></tr>
<tr><td><strong>Dropout 0.3</strong></td>
<td>During training, randomly ignores 30 % of those 64 numbers each step.</td>
<td>Stops the network leaning on any single feature, which is the cheapest defence against memorising 58 people.</td></tr>
<tr><td><strong>Linear</strong></td>
<td>Turns the 64 numbers into 2 scores.</td>
<td>The two scores become a probability of PD.</td></tr>
</tbody></table></div>

<h3>Model B: ResNet-18, pretrained, mostly frozen</h3>
{intro("The second network is not built for this problem at all. It was trained on a million everyday photographs, and the question is whether what it learned there transfers to spectrograms. Using it here takes exactly two adaptations, and both are drawn on the figure.")}

{fig("pipe_resnet.png",
  "Every stage of ResNet-18 as used here, with the shape of the tensor leaving it. Dashed grey stages are frozen at the values learned on ImageNet; only the final linear layer trains on radar data.",
  "<strong>Reading it left to right.</strong> The same overall shape as SmallCNN, the picture shrinking while the description grows richer, but 18 layers deep and already trained. The two adaptations are visible directly. <strong>Adaptation 1</strong> is at the entrance: the first layer expected colour photographs, so its red, green and blue filters are averaged into one, and the average serves both radar channels. <strong>Adaptation 2</strong> is the freezing, marked by the long bracket: 11.17 million weights are locked at their ImageNet values and act as a fixed description generator, so the whole training problem reduces to the last box, a single linear layer with 1,026 weights, fewer than the classical baseline fits.")}

{defn2("Adaptation 1: three colour channels into two radar channels", '''
<p>The pretrained first layer expects red, green and blue. We have foot and torso.
Discarding that layer and starting it over would throw away the edge and texture
detectors that make pretraining worth doing, so the three colour filters are
<strong>averaged into one</strong> and the average is copied into both of our
channels.</p>
<div class="math">new filter = mean( red filter, green filter, blue filter )
<span class="mnote">a spectrogram has no colour, so averaging is the right reduction rather than a compromise</span></div>
''')}

{defn2("Adaptation 2: how much of the network to retrain", '''
<p>This is the decision that matters, and the parameter counts settle it:</p>
<div class="tblwrap"><table>
<thead><tr><th>Setting</th><th>Trainable</th><th>Verdict</th></tr></thead><tbody>
<tr><td><strong>Final layer only</strong></td><td class="num">1,026</td>
<td><strong>The default.</strong> The frozen network becomes a fixed description
generator and a single linear layer is fitted on top. Fewer free parameters than
the handcrafted baseline uses.</td></tr>
<tr><td>Final block and final layer</td><td class="num">8,394,754</td>
<td>Run only as a capacity check: roughly 5,000 free parameters per training
window, so overfitting is expected and the run is there to show it.</td></tr>
<tr><td>Everything</td><td class="num">11,174,402</td>
<td>Not run. Shown for scale.</td></tr>
</tbody></table></div>
<p>Retraining only the final layer is called <strong>linear probing</strong>. It
asks a precise question: is the disease already separable in the description
ImageNet learned, without letting the network reshape that description around 58
people?</p>
''')}

{key("The two networks <strong>bracket the problem</strong> rather than compete. SmallCNN can learn radar-specific structure but has only {nw} correlated windows to learn it from. ResNet-18 brings structure learned from a million images but is not allowed to adapt it. If both land in the same place, that place is a property of the data rather than of either model. If they diverge, the direction says which of the two limits is binding.".format(nw=NW))}

{defn2("What the source publication contributes to the design " + PAPER, '''
<p>The paper that built and validated this radar system contains no classifier,
but two of its measurement findings shape our experiments. Neither changes the
architecture; both decide <em>what is tested</em>:</p>
<div class="tblwrap"><table>
<thead><tr><th>What the paper establishes</th><th>What this design does with it</th></tr></thead><tbody>
<tr><td>The <strong>foot-aimed nodes keep measuring reliably</strong> on motorically
impaired subjects, while the torso-aimed nodes lose reliability on exactly those
subjects.</td>
<td>This predicts that the foot channel carries more of the usable signal. The
foot-only run in section 5 is therefore a <strong>test of a stated
hypothesis</strong>, made before seeing any result, not a blind sweep over channel
combinations.</td></tr>
<tr><td>In a healthy walk the trunk <strong>speeds up slightly at every heel
strike</strong>. In Parkinson&rsquo;s, that trunk-to-step coupling weakens: the
trunk stiffens and stops answering the feet.</td>
<td>The marker lives <em>between</em> the channels, not inside either one. That is
why stage 3 of section 1 hands the network <strong>both channels together</strong>:
a single-channel model has nothing to represent the coupling with.</td></tr>
</tbody></table></div>
''')}
"""))

# ═══════════════════════════ 4 ═══════════════════════════
S.append(("The validation protocol", f"""
{key("<strong>The protocol is dictated by the question, so it comes before any result.</strong> The thesis asks: could radar screening flag Parkinson&rsquo;s in <strong>a person the system has never seen before</strong>? That is how such a tool would actually be used, on patients it was never trained on, so the test must always be a subject the model has never met, which is exactly what leave-one-subject-out enforces, 58 times over. Everything else in this section follows from that one sentence. The choice was driven by two findings that point the same way: our own exploratory measurement that <strong>a walk identifies the walker</strong>, so any split that mixes a person&rsquo;s recordings rewards recognising the person; and the literature&rsquo;s own cautionary tale, a published network whose high score turned out to come from <strong>recognising the recording setup</strong> rather than the disease. It is the single decision this thesis most depends on.")}

<h3>Why not validate the way the literature does</h3>
{intro("Most published work on radar micro-Doppler splits recordings at random into a training set and a test set. That is the obvious thing to do, it is easier, and it produces much higher numbers. It is not used here, and the reason is not caution.")}

<p>Three findings, taken together, make a random split measure the wrong thing on
this dataset:</p>

<ol>
<li><strong>A walk identifies the walker.</strong> The exploratory analysis found
that a subject&rsquo;s own recordings are about <strong>2.5 times more alike</strong>
than anyone else&rsquo;s, for all 58 subjects without exception. Two published
works recognise <em>individuals</em> from radar micro-Doppler at over 93 % and
96 %. Identity is the strongest thing in this signal.</li>
<li><strong>A random split therefore leaks it.</strong> With six recordings per
person, a random split puts some of a subject&rsquo;s recordings in training and
the rest in test. A model can then score well by recognising the person and
recalling the label it already saw, without learning anything about
Parkinson&rsquo;s. The score is real; the conclusion drawn from it is false.</li>
<li><strong>The field shows this is not hypothetical.</strong> Of the sixteen works
reviewed, only <strong>two</strong> use subject-independent validation. The
cautionary tale is {HAYASHI}: their spectrogram network reached <strong>97.8 %
accuracy</strong> telling young from elderly walkers, but the two groups had been
recorded at <em>different sites</em>, and every room leaves its own pattern of
<strong>background noise</strong>, the ambient reflections and sensor hiss that
fill the spectrogram behind the walker. The network had learned to recognise the
recording setup, not the gait, and the authors themselves flagged the number as
unreliable.</li>
</ol>

<p>These three findings were also put to a direct test. In a side experiment we
trained the same network used in this document under the literature&rsquo;s own
random split, and the signs of overfitting were unmistakable: the score jumped
to the literature&rsquo;s range, and the moment the same trained model was
scored on subjects excluded from training entirely, it fell to chance,
recognising <em>who</em> it had seen rather than <em>what</em> they have. That
experiment is written up in its own document and is taken up in depth in the
discussion chapter of the report.</p>

{key("The deciding argument is simpler than any of those, though. <strong>A validation protocol should match how the tool would actually be used.</strong> A clinical screening tool is applied to <strong>a person it has never seen</strong>. A random split answers a different question: given more recordings of somebody already known to the model, can it label them? That question is never asked in practice, so an answer to it, however high, does not measure what the thesis claims to measure.")}

<p>The cost is accepted openly. Leave-one-subject-out on this data will produce a
number far below the accuracies quoted in the literature. That gap is
<strong>mostly the protocol, not the method</strong>, and saying so is part of the
contribution.</p>

<h3>How one fold is built</h3>
{intro("The argument above says what must be true: every subject is scored by a model that never saw them. The machinery that makes it true is the <strong>fold</strong>, the same word introduced in section 1: one round of the loop, in which one subject is set aside as the test, the remaining 57 are split into the two working groups shown below, one network is trained from scratch, and one score is stored. The figure shows a single fold; the loop runs 58 of them, once per subject.")}
{fig("pipe_split.png",
  "One fold of the protocol, with segment widths drawn to scale. This is repeated 58 times, once with each subject held out.",
  "<strong>What the figure shows.</strong> Three groups of subjects with three different jobs, and the boundaries between them are never crossed. <strong>Train</strong>, 49 subjects, fits the weights. <strong>Inner validation</strong>, 8 subjects, decides which training epoch to keep. <strong>Test</strong>, 1 subject, is scored once at the end.<br><br><strong>What it means.</strong> The middle group is the part most published work omits, and it is what allows the held-out subject to stay untouched. Without it, the only way to decide when to stop training is to watch the test subject, which is leakage. The fold's standardisation statistics come from the training group alone for the same reason. The figure is a snapshot of one fold; the full procedure, and how the 58 repetitions combine into one number, is written out step by step just below.")}

<div class="tblwrap"><table>
<thead><tr><th>Group</th><th>Size</th><th>What it decides</th><th>What would go wrong without it</th></tr></thead><tbody>
<tr><td><strong>Train</strong></td><td class="num">49</td><td>The weights of the network.</td><td>Nothing to learn from.</td></tr>
<tr><td><strong>Inner validation</strong></td><td class="num">8</td><td>Which epoch to keep, and any configuration choice.</td><td>Either no stopping rule at all, or one measured on the test subject, which is exactly the leakage this protocol exists to prevent.</td></tr>
<tr><td><strong>Test</strong></td><td class="num">1</td><td><strong>Nothing.</strong> It is scored and reported.</td><td>Every number in the thesis would be optimistic by an unknown amount.</td></tr>
</tbody></table></div>

{defn("Why the inner validation group needs 8 subjects", "It must contain <strong>both classes</strong>, or the quantity used to compare epochs cannot be computed at all. Eight subjects, chosen so that four are control and four are PD, gives a comparison that is coarse but well defined. Fewer would make the stopping decision very noisy. More would take subjects away from training, which is the scarcer resource at this size.")}

{defn2("The full loop, step by step", '''
<p>The figure shows one fold as a snapshot. Here is the same thing as a
procedure, because the repetition is where the guarantee comes from:</p>
<ol>
<li><strong>Choose the test subject.</strong> Subject 1 of 58 is set aside,
untouched.</li>
<li><strong>Split the rest.</strong> Of the remaining 57, eight (four control,
four PD) become the inner validation group; the other 49 are the training
group.</li>
<li><strong>Compute the scale.</strong> The standardisation mean and spread from
stage 2 of section 1 are computed on the 49 training subjects&rsquo; windows
only.</li>
<li><strong>Train a fresh network</strong> on the 49 subjects&rsquo; windows.
After every epoch it is paused and scored on the 8 validation subjects, one
averaged probability per subject.</li>
<li><strong>Keep the best epoch.</strong> When the validation score stops
improving, training stops, and the weights from the best epoch are restored.</li>
<li><strong>Score the test subject, once.</strong> Their windows get
probabilities, the probabilities are averaged into one number, that number is
stored, and the network is <strong>thrown away</strong>.</li>
<li><strong>Start over from a blank slate</strong> with subject 2 held out, then
subject 3, and so on. 58 separate trainings, nothing carried from one fold to
the next.</li>
</ol>
<p>After the last fold there are 58 stored numbers, one per subject, and each was
produced by a model that had never seen that person. Those 58 numbers against the
58 true diagnoses give the single AUC that is reported.</p>
''')}

<h3>Why the reported number is an AUC</h3>
{defn2("Area under the ROC curve", '''
<p>Every subject leaves the loop above with a probability of being PD. To turn
those into decisions a threshold is needed, and any single threshold is a choice
that could be argued with. <strong>AUC avoids the choice entirely</strong>. It has
a direct reading:</p>
<div class="math">AUC = the chance that a randomly chosen PD subject
gets a higher score than a randomly chosen control subject
<span class="mnote">0.5 means the ordering is no better than a coin flip; 1.0 means every PD subject outranks every control</span></div>
<p>Three properties make it the right headline here. It needs
<strong>no threshold</strong>. It is unaffected by the <strong>57 to 43
imbalance</strong>, unlike plain accuracy, which a model can reach 57 % on by
always answering control. And it measures <strong>ranking</strong>, which is what a
screening tool is for: flagging who should be looked at more closely.</p>
<p>Balanced accuracy, sensitivity and specificity are reported beside it so the
practical behaviour is visible too.</p>
''')}

{warn("The literature this thesis compares itself against overwhelmingly reports <strong>accuracy</strong>, and usually on a random split. That is not the same measurement under the same conditions, so a 95 % accuracy there and a far lower AUC here are different quantities, and the higher number is not automatically the better work. Any comparison table must say so or it will read as though this work simply performed worse.")}

{key("<strong>The protocol is the contribution.</strong> Only 2 of the 16 reviewed works validate this way. The same paper whose network hit <strong>97.8 % accuracy</strong> on background noise, " + HAYASHI + ", also separates young from elderly walkers at <strong>94.9 %</strong>, so age alone can masquerade as disease. Against that background, a modest number obtained under this protocol is worth more than a high number obtained without it, and demonstrating that difference is a result in itself. One safeguard completes it: every choice this document describes was <strong>frozen and committed to version control before any deep model was trained</strong>, so none of it could be quietly tuned to the results.")}
"""))

# ═══════════════════════════ 5 ═══════════════════════════
S.append(("What the training found", f"""
<p>Everything above described decisions. This section reports what happened when
they were executed, starting with the classical yardstick that section 2 set up,
then <strong>eleven deep configurations across four model families</strong>,
every one under the same leave-one-subject-out protocol, all results read
against the same scoreboard.</p>

{defn("The scoreboard, restated once", "The corpus was processed in two runs whose composition differs sharply (29 % PD in one, 63 % in the other), so anything separating the runs also separates the groups. The <strong>pooled</strong> score over all 58 subjects can therefore look respectable while measuring the artifact. The number that can only come from gait is the AUC <strong>within a single batch</strong>, and the two floors to clear are <strong>0.610</strong> (recording length alone) and <strong>0.664</strong> (the batch label alone).")}

<h3>First, the yardstick: what the classical baseline found</h3>
{intro("The grid below is the complete classical result promised in section 2: three models across five measurement sets, each cell scored by leave-one-subject-out over all 58 subjects.")}

<div class="tblwrap"><table>
<thead><tr><th>Model</th><th>all_features</th><th>duration_invariant</th><th>shape_clean</th><th>confounded_only</th><th>duration_only</th></tr></thead>
<tbody>{GRID}</tbody></table></div>

{key("Read the logistic regression row <strong>right to left</strong>, because that is the order in which it becomes uncomfortable. Recording length alone scores <strong>{d:.3f}</strong>. The measurements that merely encode length score <strong>{c:.3f}</strong>, the best cell in the table. The measurements that describe the shape of the walk, with length removed, score <strong>{s:.3f}</strong>. The further the measurements get from elapsed time, <strong>the worse the classifier does</strong>.".format(d=dur['auc'], c=best['auc'], s=clean['auc']))}

<p>Two numbers beside the grid matter more than the grid itself:</p>
<ul>
<li>The confidence interval on the best cell runs from <strong>{lo:.3f} to
{hi:.3f}</strong>. It comfortably contains 0.5, so with 58 subjects even the
strongest classical result is <strong>not distinguishable from chance</strong>.</li>
<li>Shuffling the diagnosis labels at random and refitting gives
<strong>p = 0.092</strong> for the clean measurement set. Above the conventional
threshold, so the honest statement is that this configuration <strong>has not been
shown</strong> to beat chance.</li>
</ul>

<p>This is the bar the deep models had to clear. What follows is what happened
when they tried.</p>

<h3>The campaign at a glance</h3>
{fig("res_campaign.png",
  "Every configuration on one axis. The blue dot is the pooled subject-level AUC over all 58 subjects; the grey triangle and square are the same model scored within batch A and batch B alone. Vertical lines mark chance and the two confound floors.",
  "<strong>What the figure shows.</strong> Two patterns, and they are the whole story. The blue dots cluster in a narrow band <strong>between the two floors</strong>, from 0.52 to 0.67. And the grey markers sit far to their left: <strong>every configuration falls to chance or below once the comparison stays inside one batch</strong>.<br><br><strong>What it means.</strong> The pooled scores are not measuring gait plus noise. They are largely measuring the batch composition and the residue of recording length, and when those are held fixed there is nothing left. No configuration, from 1,026 to 35,586 trainable weights, from scratch-trained to ImageNet-transferred to the literature&rsquo;s own envelope recipe, escapes this.")}

<h3>The three model families, first pass</h3>
<div class="tblwrap"><table>
<thead><tr><th>Model</th><th>Pooled AUC</th><th>95&nbsp;% CI</th><th>Within A / B</th><th>Score tracks length</th></tr></thead><tbody>
<tr><td><strong>SmallCNN</strong> (23,682 weights)</td><td class="num">{_c('smallcnn','pooled')}</td><td class="num">[{CAMP['smallcnn']['ci'][0]:.2f}, {CAMP['smallcnn']['ci'][1]:.2f}]</td><td class="num">{_c('smallcnn','batch_A',2)} / {_c('smallcnn','batch_B',2)}</td><td class="num">+0.15</td></tr>
<tr><td><strong>ResNet-18 probe</strong> (1,026)</td><td class="num">{_c('resnet18_fc','pooled')}</td><td class="num">[{CAMP['resnet18_fc']['ci'][0]:.2f}, {CAMP['resnet18_fc']['ci'][1]:.2f}]</td><td class="num">{_c('resnet18_fc','batch_A',2)} / {_c('resnet18_fc','batch_B',2)}</td><td class="num">+0.16</td></tr>
<tr><td><strong>Envelope-LSTM</strong> (35,586)</td><td class="num">{_c('envlstm','pooled')}</td><td class="num">[{CAMP['envlstm']['ci'][0]:.2f}, {CAMP['envlstm']['ci'][1]:.2f}]</td><td class="num">{_c('envlstm','batch_A',2)} / {_c('envlstm','batch_B',2)}</td><td class="num">&minus;0.05</td></tr>
</tbody></table></div>

{defn2("The envelope model deserves one paragraph of its own", (
"<p>It was added because it is the approach the literature itself trusts most: the "
"one published result whose own authors preferred it over their higher-scoring "
"spectrogram network, which had been caught reading a site artifact. Instead of "
"images it sees <strong>three velocity curves</strong> per window, the peak foot "
"speed, the average foot speed and the average trunk speed, so the per-recording "
"image texture, and any artifact living in it, is stripped away before the model "
"ever looks.</p>"
"<p>It behaved exactly as that design predicts. Its score is the only one with "
"<strong>no correlation with recording length at all</strong>, and once the "
"texture was gone its pooled score fell to <strong>0.56</strong>: the honest "
"number, unaided by any nuisance channel. In 28 of its 58 folds no epoch beyond "
"the first improved on the first epoch's result, the optimiser agreeing with "
"the exploratory analysis that the two groups&rsquo; velocity curves overlap "
"almost completely.</p>"))}

<h3>An adversarial review, and the corrected protocol</h3>
<p>Before trusting the numbers above, the entire training pipeline was put
through an <strong>adversarial review</strong>: ten independent reviewers, five
reading the code through different lenses and five re-checking every claim
against the source. Twenty findings survived verification. The five that
mattered were fixed, the fixes were committed <strong>before</strong> any model
was re-run, and none of them was informed by a test result:</p>

<div class="tblwrap"><table>
<thead><tr><th>Finding</th><th>Why it corrupted the result</th><th>Fix</th></tr></thead><tbody>
<tr><td><strong>The &ldquo;frozen&rdquo; backbone was not frozen</strong></td>
<td>Freezing weights does not freeze batch normalisation: the pretrained network&rsquo;s internal statistics kept drifting during training, so the probe learned against features that moved under it and was evaluated on different ones.</td>
<td>The frozen blocks are now pinned so their statistics cannot change.</td></tr>
<tr><td><strong>Epoch selection re-imported the duration confound</strong></td>
<td>The epoch to keep was chosen by a score that weights each subject by their window count, and PD subjects contribute ~14&nbsp;% more windows. The confound removed from training re-entered through model selection.</td>
<td>Epochs are now compared exactly the way subjects are scored: one averaged probability per validation subject.</td></tr>
<tr><td><strong>Resizing aliased the foot band</strong></td>
<td>The time axis is downsampled 21-fold; without an anti-aliasing filter the spiky foot band turned into noise that differed between two windows of the same walk.</td>
<td>The resize now filters before sampling; the cache was rebuilt.</td></tr>
<tr><td><strong>The augmentation flipped the wrong axis</strong></td>
<td>Reversing time plays a stride backwards, a movement no walker produces. And the added noise was too small to have any effect, so the wrong flip was effectively the only augmentation.</td>
<td>The flip is now on the Doppler axis, which turns walking away into walking towards, a movement every subject really performs. Noise raised to a meaningful level; random time and Doppler masking added.</td></tr>
<tr><td><strong>Diluted turn labels</strong></td>
<td>The per-recording background floor dragged every window&rsquo;s direction score toward the midpoint, so the turn-exclusion experiment silently discarded four whole subjects.</td>
<td>The background is now subtracted before the direction is measured.</td></tr>
</tbody></table></div>

{fig("res_v1v2.png",
  "Left: how strongly each model's subject scores track recording length, before and after the corrections. Right: the one apparently strong result, foot-only in batch B, re-tested under the corrected protocol.",
  "<strong>Left panel.</strong> The corrections cut the length-tracking of every model by two thirds or more, without changing any conclusion: this is what removing a bias looks like when there is no signal underneath it.<br><br><strong>Right panel.</strong> The single best number in the campaign, foot-only at 0.70 inside batch B, was the one candidate for a real gait signal. Re-run under the corrected protocol, <strong>the two batches swap places</strong>: B falls from 0.70 to 0.42 while A rises from 0.42 to 0.51. A genuine effect does not change which half of the data it lives in when selection noise is removed. It was noise.")}

<h3>The corrected runs</h3>
<div class="tblwrap"><table>
<thead><tr><th>Model, corrected protocol</th><th>Pooled AUC</th><th>Within A / B</th><th>Score tracks length</th></tr></thead><tbody>
<tr><td><strong>SmallCNN</strong></td><td class="num">{_c('smallcnn_v2','pooled')}</td><td class="num">{_c('smallcnn_v2','batch_A',2)} / {_c('smallcnn_v2','batch_B',2)}</td><td class="num">+0.05</td></tr>
<tr><td><strong>ResNet-18 probe</strong></td><td class="num">{_c('resnet18_fc_v2','pooled')}</td><td class="num">{_c('resnet18_fc_v2','batch_A',2)} / {_c('resnet18_fc_v2','batch_B',2)}</td><td class="num">+0.08</td></tr>
<tr><td><strong>Foot channel only</strong></td><td class="num">{_c('abl_foot_v2','pooled')}</td><td class="num">{_c('abl_foot_v2','batch_A',2)} / {_c('abl_foot_v2','batch_B',2)}</td><td class="num">+0.04</td></tr>
<tr><td><strong>Turn excluded</strong> (n=54)</td><td class="num">{_c('abl_noturn_v2','pooled')}</td><td class="num">{_c('abl_noturn_v2','batch_A',2)} / {_c('abl_noturn_v2','batch_B',2)}</td><td class="num">&minus;0.10</td></tr>
</tbody></table></div>

<p>One number in this table is quietly the most eloquent of the campaign. Under
the corrected stopping rule, the ResNet probe&rsquo;s best epoch averaged
<strong>0.5</strong>: with a stationary feature map and an honest comparison, the
probe converges after roughly one pass and never improves again. The training
loop itself is reporting that there is nothing further to extract.</p>

<h3>What each ablation taught</h3>
<div class="tblwrap"><table>
<thead><tr><th>Ablation</th><th>Pooled</th><th>Within A / B</th><th>The lesson</th></tr></thead><tbody>
<tr><td><strong>Per-window z-score</strong></td><td class="num">{_c('abl_zscore','pooled')}</td><td class="num">{_c('abl_zscore','batch_A',2)} / {_c('abl_zscore','batch_B',2)}</td>
<td>Deleting per-recording brightness removes most of the length residue. Nothing usable is uncovered beneath it.</td></tr>
<tr><td><strong>Raw |STFT|</strong></td><td class="num">{_c('abl_stft','pooled')}</td><td class="num">{_c('abl_stft','batch_A',2)} / {_c('abl_stft','batch_B',2)}</td>
<td>Bypassing the contrast enhancement <em>restores</em> absolute energy, and the length-tracking doubles to +0.28. The pooled rise is the nuisance channel, not gait.</td></tr>
<tr><td><strong>Turn excluded</strong> (n=54)</td><td class="num">{_c('abl_noturn','pooled')}</td><td class="num">{_c('abl_noturn','batch_A',2)} / {_c('abl_noturn','batch_B',2)}</td>
<td>Restricting to steady walking collapses even the pooled score to chance. The models were reading the turn-and-stand content, which is where the group time difference lives.</td></tr>
<tr><td><strong>Foot channel only</strong></td><td class="num">{_c('abl_foot','pooled')}</td><td class="num">{_c('abl_foot','batch_A',2)} / {_c('abl_foot','batch_B',2)}</td>
<td>The best pooled number of the campaign, and the source of the batch-B candidate that the corrected re-run dissolved.</td></tr>
</tbody></table></div>

{warn("The corrected turn labels surfaced one discovery that is about the <strong>data</strong> rather than the models. Once the background floor is subtracted, <strong>24 of the 58 subjects have no window at all with a clean single-direction foot signature</strong>, in an all-or-nothing pattern per subject. The likely cause is the per-recording combination of the two foot radars, in which one node&rsquo;s Doppler axis is flipped before merging, so the sign convention of the combined channel may differ from recording to recording. This is now a standing question for the data owners, and it gates the one untried signal source the review identified: left/right asymmetry, which would need the per-node arrays.")}

{key("A statement the report must carry: <strong>eleven deep configurations were explored in total</strong>, and every one is reported here, including the failures. With 58 subjects, quoting only the best of eleven runs would manufacture exactly the optimism this thesis criticises in the literature. The best pooled number (0.674) and the best within-batch number (0.704) are both shown <em>with</em> the re-runs that dissolved them.")}
"""))

S.append(("What it means, and what comes next", f"""
{key("<strong>The verdict.</strong> Under leak-free, subject-independent validation, no model family and no configuration demonstrates a gait signal above the acquisition artifact on this dataset: not the 25 handcrafted measurements, not a network trained from scratch, not transferred ImageNet features, not the literature&rsquo;s envelope recipe. The pooled scores that look respectable are accounted for by two nuisance channels, the processing-run composition and the recording length, and every intervention that removes those channels removes the score with them.")}

<p>This outcome was pre-registered as a possibility before the first model ran,
and it is not a failure of the project. It is, to our knowledge, the first
careful demonstration that <strong>radar-based Parkinson&rsquo;s classification
performance can be an acquisition artifact</strong>, on the same validated
hardware the field would build on, shown across four independent method
families. Published work in this niche reports 90 to 98 percent under
validation that the review chapter shows to be leaky; the one signal here that
survives every control is the clinically real observation that
<strong>PD subjects take longer</strong>, and it is carried by elapsed time, not
by the spectral shape of the walk.</p>

<h3>What could still change the picture</h3>
<div class="tblwrap"><table>
<thead><tr><th>Missing piece</th><th>What it would enable</th></tr></thead><tbody>
<tr><td><strong>Subject ages</strong></td><td>The dominant untestable confound. With ages, an age-matched analysis could say whether even the duration signal is disease or ageing.</td></tr>
<tr><td><strong>Per-node radar arrays</strong></td><td>Left/right asymmetry, an established PD marker that the combined channels destroy, and the answer to the direction-convention question above.</td></tr>
<tr><td><strong>A batch-balanced cohort</strong></td><td>The definitive test: with diagnosis decoupled from processing run, the within-batch and pooled scores would finally have to agree.</td></tr>
</tbody></table></div>

<h3>What happens next</h3>
<div class="tblwrap"><table>
<thead><tr><th>Step</th><th>What it produces</th></tr></thead><tbody>
<tr><td><strong>Attention maps</strong> from the saved checkpoints</td><td>The picture of <em>what</em> the models attended to. If it is the turn segments and background rather than the gait bands, the artifact story becomes visible rather than statistical.</td></tr>
<tr><td><strong>The formal confound battery</strong></td><td>Every headline model re-scored duration-matched and on the chair-free variant only, completing the table the results chapter is built around.</td></tr>
<tr><td><strong>Writing</strong></td><td>These findings become the modelling and results sections of the report, under the same style rules as everything above.</td></tr>
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
<title>The Modelling Setup &middot; Radar micro-Doppler Parkinson's</title>
<style>{CSS}</style></head><body>
<div id="bar"></div>
<div class="wrap">
<header class="hero">
  <p class="kicker">Master's thesis &middot; Methods, before any model is trained</p>
  <h1>The Modelling Setup</h1>
  <p class="sub">How a recording becomes a tensor, which classifiers set the yardstick,
  which networks were trained and why exactly those, the validation protocol that makes
  any of it worth reporting, and what the full training campaign found.</p>
  <div class="statlead">Where the campaign ended:</div>
  <div class="stats">
    <div class="stat"><b>{NW}</b><span>windows cached<i>from 348 recordings</i></span></div>
    <div class="stat"><b>11</b><span>configurations run<i>4 model families, all leak-free</i></span></div>
    <div class="stat"><b>0.674</b><span>best pooled score<i>did not survive re-testing</i></span></div>
    <div class="stat"><b>0.664</b><span>the floor to beat<i>acquisition batch alone</i></span></div>
    <div class="stat"><b>0</b><span>configurations with within-batch signal<i>the finding itself</i></span></div>
  </div>
</header>
<nav class="toc"><h4>Contents</h4><ol>{toc}</ol></nav>
<main>{body}</main>
</div>
<button id="top" title="Back to top">&uarr;</button>
<script>{JS}</script>
</body></html>"""

out = ROOT / "MODELLING_SETUP.html"
out.write_text(html)
ndef = html.count('class="defn"')
print(f"wrote {out} ({len(html)/1e6:.2f} MB)")
print(f"sections={len(S)} figures={html.count('data:image/jpeg')} definitions={ndef}")
