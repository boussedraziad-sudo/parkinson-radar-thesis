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
  "The whole path from a stored recording to one score per person. The upper row runs once and writes a cache to disk; the lower row runs inside every fold of the validation loop.",
  "Read the shapes along the bottom of each box, because they are the actual array dimensions and they tell the story on their own. A recording starts as <strong>320 &times; 12,000 numbers</strong> and ends as <strong>one number between 0 and 1</strong>. The split between the two rows is deliberate: everything identical for every fold is computed once, and everything that must not see the held-out subject is deferred into the fold.")}

{defn2("The whole reduction, in numbers", '''
<p>The stages below are explained one at a time in the rest of this section.
Together they answer how a recording becomes a single score.</p>
<div class="tblwrap"><table>
<thead><tr><th>Stage</th><th>Shape</th><th>What one number in it is</th></tr></thead><tbody>
<tr><td>One recording</td><td class="num">320 &times; 12,000</td>
<td>320 speed slots against 12,000 instants. Each value is <strong>how much
reflected energy came back at that speed at that instant</strong>.</td></tr>
<tr><td>One window</td><td class="num">320 &times; 4,800</td>
<td>The same, cut down to 3.0 seconds. About 5 windows come from one recording.</td></tr>
<tr><td>After resizing</td><td class="num">224 &times; 224</td>
<td>The same picture, redrawn on a smaller grid.</td></tr>
<tr><td>After stacking</td><td class="num">2 &times; 224 &times; 224</td>
<td>Two of those pictures held together, one per radar.</td></tr>
<tr><td>Network output</td><td class="num">2 numbers</td>
<td>A score for control and a score for PD, for <em>that window</em>.</td></tr>
<tr><td>Per window</td><td class="num">1 number</td>
<td>Those two turned into a probability of PD, between 0 and 1.</td></tr>
<tr><td><strong>Per subject</strong></td><td class="num"><strong>1 number</strong></td>
<td>The <strong>average</strong> of that person&rsquo;s window probabilities. This
is the answer the thesis reports.</td></tr>
</tbody></table></div>
''')}

{step(1, "Cut each recording into fixed-length windows")}
{intro("The first stage takes one variable-length recording and produces several fixed-length pieces. The figure shows the cuts drawn onto a real recording.")}

{fig("pipe_windowing.png",
  "Above: one real recording with every cut point drawn as a white line, and each resulting window labelled. Below: the same windows as bars, so the overlap between them is visible.",
  "<strong>What the figure shows.</strong> A 7.5 second recording produces four windows. The lower panel makes the key detail visible: <strong>the windows overlap</strong>. Window 1 does not begin where window 0 ends. It begins 1.5 seconds after window 0 <em>began</em>, so the two share 1.5 seconds of content.<br><br><strong>What it means.</strong> Every input to the network is now exactly 3.0 seconds, whatever the length of the walk it came from. Recording length can no longer reach the network through the size of its input. It can still reach it through the <em>number</em> of windows, which is dealt with in stage 4.")}

{defn2("Why 3.0 seconds, and what the 1.5 second hop actually means", '''
<p><strong>The window length.</strong> One stride, meaning left foot down to left
foot down again, takes roughly <strong>1.0 to 1.2 seconds</strong>. A 3 second
window therefore holds <strong>two to three complete strides</strong>. That
matters because one of the three clinically motivated measurements is
<strong>stride-to-stride regularity</strong>, and regularity is a comparison
between strides: with only one stride in view there is nothing to compare it
against, so the measurement does not exist. Three seconds is close to the
shortest window in which it does.</p>
<p><strong>The hop is not a delay.</strong> It is the distance between the
<em>starting points</em> of consecutive windows, not a gap between them:</p>
<div class="math">window 0 covers 0.0 to 3.0 s
window 1 covers 1.5 to 4.5 s
window 2 covers 3.0 to 6.0 s
<span class="mnote">a hop of 1.5 s with a length of 3.0 s means neighbours share half their content</span></div>
<p>The overlap is deliberate. Sliding by a full 3 seconds would give roughly half
as many windows from an already small dataset, and any stride pattern straddling
a boundary would be split in two and seen by neither window. Overlapping means
every moment of the walk appears whole inside at least one window.</p>
<p>The alternative of 2.0 second windows hopping 0.5 seconds produces far more
windows, but neighbours would then share <strong>75 %</strong> of their content,
so the extra windows are near-copies rather than new information. That setting is
kept as an ablation.</p>
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
what many published pipelines do by default. It is not used here, and the reason is
worth stating because it is a decision rather than an oversight:
<strong>standardising a window against itself deletes its absolute
brightness</strong>, and absolute brightness is how strongly the body reflects,
which is exactly what <strong>bradykinesia</strong> changes. It would remove the
signal being looked for. It also happens to suppress the difference between the two
processing runs. Those two effects pull in opposite directions, so it is reported as
an ablation and the within-batch analysis is what separates the two causes.</p>
''')}

{step(3, "Redraw on a common grid and stack the two radars")}
{intro("The third stage makes every window the same size in pixels and joins the two radars into one array. This is the last stage that touches the picture; what comes out is exactly what the network sees.")}

{fig("pipe_model_input.png",
  "One cached window. Channel 0 is the foot-aimed node and channel 1 is the torso-aimed node, both recorded at the same moment during the same walk by the same subject.",
  "<strong>What the figure shows.</strong> Two pictures of the same three seconds. The brightness at any point is the energy the Fourier transform found at that speed and instant, after the compression of stage 2, so a bright pixel means a lot of the body was moving at that speed then. The two look different because the nodes are aimed at different heights: the foot channel is spiky and reaches high speeds, the torso channel is a smooth band near zero.<br><br><strong>What it means.</strong> The network is given both at once rather than one at a time, because the source publication {p} establishes that the relationship <em>between</em> trunk and foot motion is what breaks down in Parkinson's. A model shown one channel cannot represent that relationship at all.".format(p=PAPER))}

{defn2("How the resizing is done, and what happens to the windows afterwards", '''
<p><strong>Each window is resized on its own</strong>, from 320 &times; 4,800 to
224 &times; 224, by <strong>bilinear interpolation</strong>: each pixel of the new
grid takes a weighted average of the four input pixels surrounding the position it
corresponds to, with nearer pixels weighted more. It is the same operation as
resizing a photograph. The target of 224 is not tuned; it is the input size the
pretrained network expects.</p>
<p>The two resized pictures are then <strong>stacked</strong> into one array of
shape <strong>2 &times; 224 &times; 224</strong>, in the same way a colour photo
holds red, green and blue as three stacked layers. Here the two layers are the two
radars.</p>
<p><strong>Each window becomes its own file</strong>, and a single table records
one row per window: which subject it came from, their diagnosis, which test
variant and repeat, its position within the recording, and where the file sits.
That table is the only thing training reads, which is what makes it possible to say
&ldquo;train on these 49 people&rdquo; in one line.</p>
''')}

{key("<strong>The dataset did grow, but not in the way that would help.</strong> 348 recordings became {nw} windows, so the network sees roughly five times more examples. Those examples are <strong>not independent</strong>: five windows from one walk overlap each other and come from one person, so they carry far less than five recordings' worth of information. This is why the number of <em>subjects</em>, still 58, remains the real constraint, why folds are split by subject and never by window, and why uncertainty is estimated by resampling subjects rather than windows.".format(nw=NW))}

{defn2("Does redrawing on a smaller grid lose anything", '''
<p>It does not, and the check is a comparison between how finely the grid is
sampled and how finely the measurement can actually resolve.</p>
<div class="tblwrap"><table>
<thead><tr><th>Axis</th><th>After resizing</th><th>What the measurement can truly resolve</th><th>Result</th></tr></thead><tbody>
<tr><td><strong>Time</strong></td><td>3.0 s over 224 columns = <strong>13.4 ms</strong> per column</td>
<td>The analysis window that produced the data was <strong>50 ms</strong> wide</td>
<td>About four columns still describe one independent measurement</td></tr>
<tr><td><strong>Speed</strong></td><td>1600 Hz over 224 rows = <strong>7.1 Hz</strong> per row</td>
<td>The same 50 ms window separates components <strong>20 Hz</strong> apart</td>
<td>About three rows still describe one independent measurement</td></tr>
</tbody></table></div>
<p>Both axes remain <strong>oversampled</strong> after resizing. What is removed is
duplication that was already there, not detail. The intuition is a photograph
saved at a resolution far beyond what the lens can resolve: shrinking it discards
pixels, not information.</p>
''')}

{step(4, "Stop long recordings from carrying more weight")}
{intro("Windowing fixed the size of each input but not the number of them. This stage measures that problem and corrects it.")}

{fig("pipe_window_count.png",
  "Each dot is one subject. On the left, the share of training influence they would carry as recorded. On the right, the same after the correction.",
  "<strong>What the figure shows.</strong> On the left the dots spread from about 0.5 to 2.0, so some subjects would contribute four times as much to training as others, purely because their recordings are longer. PD subjects average <strong>30.9 windows against 27.3</strong>, and window count on its own separates the two groups at <strong>AUC 0.62</strong>. On the right every dot sits on 1.0.<br><br><strong>What it means.</strong> Without the correction a network could reach 0.62 by responding to how many windows a person contributed, which is a restatement of how long they took. Each window is therefore weighted by the inverse of its subject's window count, so all 58 subjects carry <strong>exactly the same total weight</strong>. The confound is removed at the point where it would have acted.")}

{step(5, "Record which direction the subject was walking")}
{intro("The last stage adds no processing. It attaches one extra number to every window so that a specific experiment becomes possible later.")}

{fig("pipe_turn.png",
  "Left: how much of each window's energy sits at positive Doppler, meaning motion towards the node. Right: how many windows survive if windows near the midpoint are discarded.",
  "<strong>What the figure shows.</strong> The left panel has three groups. Windows near <strong>0</strong> lie inside the outward pass, where all motion is away from the node. Windows near <strong>1</strong> lie inside the return pass. Windows near <strong>0.5</strong> contain the moment the subject turned around, so they hold energy in both directions at once. The right panel prices the exclusion: removing everything within 0.1 of the midpoint keeps <strong>65 %</strong> of the windows.<br><br><strong>What it means.</strong> This makes a targeted experiment possible, and the reason it is targeted is the important part.")}

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

{key("<strong>The frozen configuration.</strong> 3.0 s windows, 1.5 s hop, contrast-enhanced representation, compression by log(1+x) with standardisation deferred into each fold, 224 &times; 224, two channels, inverse-count weighting, every window kept. It produced <strong>{nw} windows from 348 recordings</strong> with no failures. Each alternative named above becomes an ablation, so the choices are tested rather than asserted.".format(nw=NW))}
"""))

# ═══════════════════════════ 2 ═══════════════════════════
rows = []
for mdl in ["logreg", "svm_rbf", "random_forest"]:
    cells = []
    for fs in ["all_features", "duration_invariant", "shape_clean", "confounded_only", "duration_only"]:
        v = R.get(f"{mdl}|{fs}", {}).get("auc")
        cls = ""
        if v is not None:
            if v >= 0.66: cls = ' class="hi"'
            elif v < 0.50: cls = ' class="lo"'
        cells.append(f'<td{cls}>{v:.3f}</td>' if v is not None else "<td>n/a</td>")
    rows.append(f'<tr><td><strong>{mdl}</strong></td>' + "".join(cells) + "</tr>")
GRID = "".join(rows)

S.append(("The classical baseline, and why it exists", f"""
{key("<strong>The goal of this thesis is the deep model.</strong> The classical classifiers in this section are not competitors, they are the <strong>yardstick</strong>. They answer a question that has to be settled first: how far can ordinary statistics get using the 25 handcrafted measurements? Whatever number they reach is the number a network has to beat before anything can be claimed for it. Every result below is repeated in the results chapter beside the deep models.")}

<h3>How one classical prediction is made</h3>
{intro("Before any numbers, here is the procedure. It runs once per subject, so the whole thing happens 58 times to produce one score.")}

{fig("pipe_classical_flow.png",
  "The five stages of one fold of the classical baseline. The loop runs once per subject.",
  "<strong>Stage 1, hold out one subject.</strong> All of that person's recordings are set aside. Nothing about them touches the rest of the stage.<br><br><strong>Stage 2, fill gaps.</strong> A few measurements are occasionally missing. They are replaced by the median of the same measurement across the training subjects, so no recording has to be discarded and no value is invented from the held-out person.<br><br><strong>Stage 3, put everything on one scale.</strong> The measurements have wildly different units: a spectral centroid is a frequency in the hundreds, a ratio is a fraction below 1. Each is rescaled to average 0 and spread 1 using the training subjects only, so no measurement dominates merely because its numbers are bigger.<br><br><strong>Stage 4, fit the classifier</strong> on the 57 remaining subjects, with the two classes weighted to offset the 57 to 43 imbalance so the model is not rewarded for always answering control.<br><br><strong>Stage 5, score and average.</strong> The held-out subject's six recordings each get a probability, and those are averaged into <strong>one number for that person</strong>.")}

{defn("Why the average, and why subjects rather than recordings", "Averaging gives the answer a clinician would actually want, one per person. It also makes the score <strong>insensitive to how many recordings a subject happens to have</strong>, which is the same confound the window weighting handles on the deep side. Scoring recordings instead would let a subject with more recordings count more.")}

<h3>Three classifiers, chosen to differ in what they can express</h3>
{intro("Three models are compared, not to find a winner, but because they can represent different kinds of pattern. If a signal exists but only one of them finds it, that itself says what shape the signal has.")}

<div class="tblwrap"><table>
<thead><tr><th>Model</th><th>How it works</th><th>What it can and cannot represent</th><th>Why it suits this problem</th></tr></thead><tbody>
<tr><td><strong>Logistic regression</strong></td>
<td>Gives every measurement a weight, adds them up, and squashes the total into a probability between 0 and 1.</td>
<td>Only <strong>additive</strong> effects. It can say &ldquo;a lower centroid pushes towards PD&rdquo;, but it cannot say &ldquo;a low centroid matters only when rhythmicity is also high&rdquo;.</td>
<td><strong>The most restrictive, therefore the hardest to fool.</strong> With 58 subjects a flexible model can fit noise; this one mostly cannot. Its weights are also readable, so a result comes with an explanation.</td></tr>
<tr><td><strong>Support vector machine, RBF kernel</strong></td>
<td>Draws a boundary between the two groups, allowed to curve, and places it as far as possible from the nearest subjects on either side.</td>
<td>Smooth <strong>curved</strong> boundaries, so combinations that only matter jointly.</td>
<td>Catches a signal that is real but not additive. Maximising the margin is a built-in caution, which helps at this sample size.</td></tr>
<tr><td><strong>Random forest</strong></td>
<td>Grows many decision trees, each on a random part of the data, and averages their votes.</td>
<td><strong>Thresholds and interactions</strong>, with no assumption that the relationship is smooth.</td>
<td>Catches a signal that lives in a cut-off rather than a trend. It also ranks the measurements by usefulness for free, which feeds the interpretability discussion.</td></tr>
</tbody></table></div>

<h3>Five sets of measurements, and what each one is for</h3>
{intro("The same three models are run five times, each on a different set of measurements. The sets are not variations to be tuned over. Each is a specific question, and reading them side by side is what makes the result interpretable.")}

<div class="tblwrap"><table>
<thead><tr><th>Set</th><th>Size</th><th>What is in it</th><th>The question it answers</th></tr></thead><tbody>
<tr><td><code>all_features</code></td><td class="num">{n_all}</td>
<td>Every measurement kept for analysis.</td>
<td>How well can we do with no restraint? This is the number most published work would report.</td></tr>
<tr><td><code>duration_invariant</code></td><td class="num">{n_inv}</td>
<td>Averages, spectral shapes and ratios. Anything that is a total was removed.</td>
<td>How well can we do once recording length is removed <strong>by construction</strong>?</td></tr>
<tr><td><code>shape_clean</code></td><td class="num">{n_cln}</td>
<td>The subset of the above that also passes a direct test against measured duration.</td>
<td>How well can we do once length is removed <strong>by measurement</strong>? This exists because construction turned out not to be enough.</td></tr>
<tr><td><code>confounded_only</code></td><td class="num">{n_cnf}</td>
<td>Only the totals, which grow with the length of the recording.</td>
<td>How much does the confound alone buy?</td></tr>
<tr><td><code>duration_only</code></td><td class="num">1</td>
<td>The length of the recording, and nothing else.</td>
<td><strong>The floor.</strong> Any result not clearly above this has demonstrated nothing about gait.</td></tr>
</tbody></table></div>

{defn2("Seven measurements are computed but not used, and it is worth saying which", '''
<p>32 measurements come out of the extraction step and <strong>25</strong> go into
the analysis. The seven left out are not failures, they are duplicates, and
excluding them keeps the number of statistical tests honest:</p>
<div class="tblwrap"><table>
<thead><tr><th>Left out</th><th>Why</th></tr></thead><tbody>
<tr><td>The four per-channel band averages, meaning the torso-band and foot-band
averages taken separately within each of the two channels</td>
<td>They are the two halves of a ratio that is already in the set. Keeping the
ratio and both of its parts counts the same information three times.</td></tr>
<tr><td>The two alternative foot-to-torso ratios normalised by band width</td>
<td>A second way of writing a ratio that is already present. The two versions
differ by a constant factor, so they carry the same ordering between subjects.</td></tr>
<tr><td>The cross-channel average ratio</td>
<td>Duplicates the total-energy ratio already in the set.</td></tr>
</tbody></table></div>
<p>All seven are retained in the stored table, so the decision can be reversed and
checked rather than taken on trust.</p>
''')}

<h3>Why the reported number is an AUC</h3>
{defn2("Area under the ROC curve", '''
<p>Every subject leaves the classifier with a probability of being PD. To turn
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

{warn("The literature this thesis compares itself against overwhelmingly reports <strong>accuracy</strong>, and usually on a random split. That is not the same measurement under the same conditions, so an accuracy of 95 % there is not a better result than an AUC of 0.62 here. Any comparison table must say so or it will read as though this work simply performed worse.")}

<h3>What the baseline found</h3>
{intro("The grid below is the complete classical result: three models across five measurement sets, each scored by leave-one-subject-out over all 58 subjects.")}

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

{warn("Two cells sit <em>below</em> 0.5, at 0.287 and 0.450. Both are the single-measurement <code>duration_only</code> column fitted by the SVM and the forest. Given one input and 57 training subjects those two models fit the noise in the training folds and invert on the held-out one. It is a useful demonstration that a model can score <strong>worse than guessing</strong>, and it is why the floor is quoted from logistic regression, the only one of the three that handles a single input sensibly.")}

{later("These are the classical numbers. The deep models are what the preprocessing was built for and their results are being computed now. When they arrive they are reported against this grid and against both floors, <code>duration_only</code> at 0.610 and acquisition batch at 0.664.")}
"""))

# ═══════════════════════════ 3 ═══════════════════════════
S.append(("The deep models", f"""
<p>Two networks are used, and the choice of both is dominated by one number:
<strong>58 subjects</strong>.</p>

{fig("pipe_models.png",
  "Trainable parameters for each option on a logarithmic scale, against the number of training windows available.",
  "<strong>What the figure shows.</strong> The vertical line is {nw} training windows. A model to the right of it has <strong>more free parameters than we have examples</strong>, which means it can in principle memorise every training window exactly and learn nothing general.<br><br><strong>What it means.</strong> Two of the four options sit there, and both are reported as ablations that demonstrate the overfitting rather than as headline configurations. The two used sit to the left.".format(nw=NW))}

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
{intro("The second network is not built for this problem at all. It was trained on a million everyday photographs, and the question is whether what it learned there transfers to spectrograms.")}

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
<td>Ablation. Roughly 5,000 free parameters per training window, so overfitting is
expected and the run is there to show it.</td></tr>
<tr><td>Everything</td><td class="num">11,174,402</td>
<td>Not run. Shown for scale.</td></tr>
</tbody></table></div>
<p>Retraining only the final layer is called <strong>linear probing</strong>. It
asks a precise question: is the disease already separable in the description
ImageNet learned, without letting the network reshape that description around 58
people?</p>
''')}

{key("The two networks <strong>bracket the problem</strong> rather than compete. SmallCNN can learn radar-specific structure but has only {nw} correlated windows to learn it from. ResNet-18 brings structure learned from a million images but is not allowed to adapt it. If both land in the same place, that place is a property of the data rather than of either model. If they diverge, the direction says which of the two limits is binding.".format(nw=NW))}

{defn("What the source publication contributes to the design " + PAPER, "Two things, and both shape the experiments rather than the architecture. It finds that <strong>foot nodes stay reliable for motorically impaired subjects while torso nodes lose reliability</strong>, which predicts that the foot channel carries more of the signal and turns the channel ablation into a test of a stated hypothesis rather than a sweep. And it establishes that the trunk normally tracks each heel strike while <strong>that coupling breaks down in Parkinson's</strong>, which is the reason both channels are given to the network together.")}
"""))

# ═══════════════════════════ 4 ═══════════════════════════
S.append(("The validation protocol", f"""
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
reviewed, only <strong>two</strong> use subject-independent validation. One
published network reached <strong>97.8 %</strong> and was later shown to be keying
on <em>background noise differences</em> between recording sites rather than on
gait.</li>
</ol>

{key("The deciding argument is simpler than any of those, though. <strong>A validation protocol should match how the tool would actually be used.</strong> A clinical screening tool is applied to <strong>a person it has never seen</strong>. A random split answers a different question: given more recordings of somebody already known to the model, can it label them? That question is never asked in practice, so an answer to it, however high, does not measure what the thesis claims to measure.")}

<p>The cost is accepted openly. Leave-one-subject-out on this data will produce a
number far below the accuracies quoted in the literature. That gap is
<strong>mostly the protocol, not the method</strong>, and saying so is part of the
contribution.</p>

<h3>How one fold is built</h3>
{fig("pipe_split.png",
  "One fold of the protocol, with segment widths drawn to scale. This is repeated 58 times, once with each subject held out.",
  "<strong>What the figure shows.</strong> Three groups of subjects with three different jobs, and the boundaries between them are never crossed. <strong>Train</strong>, 49 subjects, fits the weights. <strong>Inner validation</strong>, 8 subjects, decides which training epoch to keep. <strong>Test</strong>, 1 subject, is scored once at the end.<br><br><strong>What it means.</strong> The middle group is the part most published work omits, and it is what allows the held-out subject to stay untouched. Without it, the only way to decide when to stop training is to watch the test subject, which is leakage. The fold's standardisation statistics come from the training group alone for the same reason.")}

<div class="tblwrap"><table>
<thead><tr><th>Group</th><th>Size</th><th>What it decides</th><th>What would go wrong without it</th></tr></thead><tbody>
<tr><td><strong>Train</strong></td><td class="num">49</td><td>The weights of the network.</td><td>Nothing to learn from.</td></tr>
<tr><td><strong>Inner validation</strong></td><td class="num">8</td><td>Which epoch to keep, and any configuration choice.</td><td>Either no stopping rule at all, or one measured on the test subject, which is exactly the leakage this protocol exists to prevent.</td></tr>
<tr><td><strong>Test</strong></td><td class="num">1</td><td><strong>Nothing.</strong> It is scored and reported.</td><td>Every number in the thesis would be optimistic by an unknown amount.</td></tr>
</tbody></table></div>

{defn("Why the inner validation group needs 8 subjects", "It must contain <strong>both classes</strong>, or the quantity used to compare epochs cannot be computed at all. Eight subjects, chosen so that four are control and four are PD, gives a comparison that is coarse but well defined. Fewer would make the stopping decision very noisy. More would take subjects away from training, which is the scarcer resource at this size.")}

<h3>From windows to one number per person</h3>
<div class="math">subject score = the average of that subject&rsquo;s window probabilities
<span class="mnote">the average rather than the maximum, so a subject with more windows does not thereby get a more confident score</span></div>
<p>The 58 subject scores then give one AUC for the whole run. The decision
threshold used for the supporting metrics is set to the <strong>prevalence of the
positive class</strong> rather than to 0.5, because a model trained on an
imbalanced corpus is not calibrated and an uncalibrated half-way cut
systematically under-predicts the smaller group. The AUC itself does not depend on
this choice.</p>

<h3>Pre-registration</h3>
<p>Every choice described in this document was made <strong>before any deep model
was trained</strong>, and the complete configuration was committed to version
control as a single frozen object at <code>b0e7221</code>. The reporting runs use
it unchanged. The number of configurations explored is stated in the report,
because with 58 subjects the optimism introduced by repeated tuning is real and
cannot be engineered away.</p>

{key("<strong>The protocol is the contribution.</strong> Only 2 of the 16 reviewed works validate this way; a published network reached 97.8 % by keying on background noise; and radar separates young from elderly walkers at 94.9 %, so age alone can masquerade as disease. Against that background, a modest number obtained under this protocol is worth more than a high number obtained without it, and demonstrating that difference is a result in itself.")}
"""))

# ═══════════════════════════ 5 ═══════════════════════════
S.append(("What happens next", f"""
<p>The pipeline is built and the reporting runs are under way. A full 58-fold run
costs <strong>16 minutes</strong> on the GPU against 88 on the processor, which is
what makes the full experiment programme affordable.</p>

<div class="tblwrap"><table>
<thead><tr><th>Step</th><th>What it produces</th><th>Cost</th><th>State</th></tr></thead><tbody>
<tr><td>Freeze and commit the configuration</td><td>The pre-registration hash quoted in the report.</td><td class="num">done</td><td><code>b0e7221</code></td></tr>
<tr><td>SmallCNN, full leave-one-subject-out</td><td>The first honest deep-model number, against both floors.</td><td class="num">16 min</td><td>running</td></tr>
<tr><td>ResNet-18, full run</td><td>Whether pretrained descriptions transfer to this problem.</td><td class="num">~20 min</td><td>queued</td></tr>
<tr><td>Six ablations</td><td>Channel, representation, window length, augmentation, normalisation, turn exclusion.</td><td class="num">~2 h</td><td>next</td></tr>
<tr><td>Grad-CAM</td><td>Whether the network attends to the foot and torso bands or to artifacts.</td><td class="num">~30 min</td><td>next</td></tr>
<tr><td>Confound battery</td><td>Every model re-scored within batch, duration-matched, test2 only, window-count controlled.</td><td class="num">~2 h</td><td>next</td></tr>
</tbody></table></div>

{warn("One expectation is worth setting before the numbers arrive, because it changes how they should be read rather than whether they are worth having. The classical baseline reaches <strong>0.62 on gait-shape measurements against 0.61 for recording length alone</strong>, and the acquisition batch label by itself reaches <strong>0.664</strong>. A deep model has to clear both before any claim about gait can be made. If it does not, that is a publishable result rather than a failure: it would be the first careful demonstration that reported radar performance on this task can be an acquisition artifact.")}
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
  <p class="sub">How a recording becomes a tensor, which classifiers set the yardstick and
  what they already say, which networks are used and why exactly those, and the
  validation protocol that makes any of it worth reporting.</p>
  <div class="statlead">Where the pipeline stands:</div>
  <div class="stats">
    <div class="stat"><b>{NW}</b><span>windows cached<i>from 348 recordings</i></span></div>
    <div class="stat"><b>2</b><span>architectures<i>23,682 and 1,026 trainable weights</i></span></div>
    <div class="stat"><b>58</b><span>folds per run<i>one per subject, 16 min</i></span></div>
    <div class="stat"><b>0.62</b><span>the classical yardstick<i>gait shape, subject-independent</i></span></div>
    <div class="stat"><b>0.664</b><span>the floor to beat<i>acquisition batch alone</i></span></div>
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
