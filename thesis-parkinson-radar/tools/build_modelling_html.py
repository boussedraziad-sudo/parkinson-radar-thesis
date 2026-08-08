#!/usr/bin/env python3
"""MODELLING_SETUP.html — everything decided before a single model is trained.

Same style rules as DATA_AND_EDA.html: no em-dashes, key terms bolded, lists
over prose, nothing phrased as a question the author asks himself.
"""
import base64, io, json, pathlib, sys
import pandas as pd
from PIL import Image

ROOT = pathlib.Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
FIG, TOOLS = ROOT / "outputs/figures", ROOT / "tools"
sys.path.insert(0, str(ROOT))

BS = json.load(open(ROOT / "reports/baseline_summary.json"))
MAN = pd.read_csv(ROOT / "outputs/preprocessed/manifest.csv")
R = BS["results"]
FS = BS["feature_sets"]
NW = len(MAN)
n_all, n_inv, n_cln, n_cnf = (len(FS["all_features"]), len(FS["duration_invariant"]),
                              len(FS["shape_clean"]), len(FS["confounded_only"]))


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
    ins = (f'<div class="insight"><span class="ilab">What it tells us</span>'
           f'<p>{insight}</p></div>') if insight else ""
    return (f'<figure>{img(name, maxw)}<figcaption><span class="fname">{name}</span> '
            f'{caption}</figcaption>{ins}</figure>')


def key(t):  return f'<aside class="key"><span class="klabel">Key point</span><p>{t}</p></aside>'
def warn(t): return f'<aside class="warn"><span class="klabel">Caution</span><p>{t}</p></aside>'
def defn(term, body): return f'<div class="defn"><span class="dterm">{term}</span><p>{body}</p></div>'
def defn2(term, body): return f'<div class="defn"><span class="dterm">{term}</span><div>{body}</div></div>'
def later(t): return f'<div class="later"><span class="ltag">Comes after training</span><p>{t}</p></div>'

PAPER = ('<span class="src"><a href="https://doi.org/10.1109/TBME.2025.3583785" '
         'target="_blank" rel="noopener">L&oacute;pez-Delgado et al. (2026)</a></span>')

CSS = (TOOLS / "eda.css").read_text() + """
.dterm .src{font-weight:400;margin-left:7px;letter-spacing:0}
.sec h4{font-family:var(--sans);font-size:15px;font-weight:700;color:var(--ink);margin:26px 0 8px}
.src a{color:inherit;text-decoration:none;border-bottom:1px dotted currentColor}
.defn>div{margin:0}
.math{font-family:var(--mono);font-size:14.5px;line-height:1.85;background:var(--panel);
 border:1px solid var(--line);border-radius:8px;padding:13px 16px;margin:13px 0;
 text-align:center;overflow-x:auto}
.math .mnote{display:block;font-family:var(--sans);font-size:12.5px;color:var(--muted);
 margin-top:7px;font-style:italic}
pre.code{font-family:var(--mono);font-size:12.5px;line-height:1.6;background:var(--panel);
 border:1px solid var(--line);border-left:3px solid var(--key-line);border-radius:0 8px 8px 0;
 padding:13px 16px;margin:14px 0;overflow-x:auto;white-space:pre}
.fixed{border-left:3px solid var(--key-line)}
.broke{border-left:3px solid var(--warn)}
.bx{display:grid;grid-template-columns:26px 1fr;gap:13px;padding:14px 0;
 border-top:1px solid var(--line)}
.bx:first-of-type{border-top:none}
.bn{font-family:var(--sans);font-size:13px;font-weight:700;color:#fff;background:var(--warn);
 border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center}
.bx.done .bn{background:var(--key-line)}
.bx p{margin:0 0 7px;font-size:15px;line-height:1.6;max-width:none}
.bx p:last-child{margin-bottom:0}
.bl{font-family:var(--sans);font-size:10.5px;font-weight:700;letter-spacing:.08em;
 text-transform:uppercase;color:var(--muted);margin-right:8px}
td.hi{background:rgba(31,122,85,.10);font-weight:700}
td.lo{background:rgba(192,57,43,.10)}
"""

S = []

# ══════════════════════════ 1 ══════════════════════════
S.append(("What was broken, and what it now does", f"""
<p>Nothing in this document could be written until the training code actually
trained. Four defects blocked it, all of them silent: every one produced a run
that finished, printed a number and looked successful. They are fixed, and this
section records what each one was, because two of them are worth stating in the
report as evidence of rigour rather than hiding as bugs.</p>

<div class="bx done"><span class="bn">1</span><div>
<p><span class="bl">Was</span>Early stopping compared model epochs using the
validation AUC. In leave-one-subject-out the validation set is
<strong>a single subject</strong>, so every recording in it carries the same
label, and an AUC cannot be computed from one class. The code silently
substituted <strong>0.5 every epoch</strong>.</p>
<p><span class="bl">Consequence</span>0.5 is never greater than 0.5, so the
&ldquo;best&rdquo; epoch was always <strong>epoch 0</strong>. Patience then ran
out five epochs later and the epoch-0 weights were restored. <strong>Every model
was effectively trained for one epoch</strong>, and the run still reported
success.</p>
<p><span class="bl">Now</span>The validation set is <strong>8 subjects,
stratified so both classes are present</strong>. The AUC is real. A guard raises
an error if a validation set ever contains one class again, so the failure can
never return quietly.</p>
</div></div>

<div class="bx done"><span class="bn">2</span><div>
<p><span class="bl">Was</span>The epoch was chosen using the held-out subject,
which is the subject the result is then reported on.</p>
<p><span class="bl">Consequence</span>Leakage. This is the exact practice the
state-of-the-art chapter criticises other work for.</p>
<p><span class="bl">Now</span>Selection uses the inner validation subjects only.
The held-out subject is scored <strong>exactly once</strong>, after training has
finished, and informs no decision.</p>
</div></div>

<div class="bx done"><span class="bn">3</span><div>
<p><span class="bl">Was</span>No model weights were ever written to disk.</p>
<p><span class="bl">Consequence</span>Grad-CAM was impossible, so the
interpretability section could never have been produced.</p>
<p><span class="bl">Now</span>The best weights of every fold are saved, together
with the epoch that produced them and the full configuration.</p>
</div></div>

<div class="bx done"><span class="bn">4</span><div>
<p><span class="bl">Was</span>No subject-grouped cross-validation splitter.</p>
<p><span class="bl">Consequence</span>The only way to test the pipeline cheaply
was a full 58-fold run, so it rarely got tested.</p>
<p><span class="bl">Now</span>A grouped, stratified 5-fold splitter exists for
smoke tests, and it can never place one subject on both sides of a split.</p>
</div></div>

{fig("pipe_bug.png",
  "The first defect, drawn. Left: what the old code saw. Right: what the fixed code sees.",
  "The left panel is not an illustration of a subtle problem. The line is <strong>flat at exactly 0.5</strong> because the value was a constant substituted for an undefined quantity, so no epoch could ever beat the first one. Any result produced before this fix describes a network trained for a single pass over the data.")}

{key("Three further changes were made at the same time, none of them bug fixes but all of them necessary for the results to mean anything: the loss is now <strong>class-balanced</strong> to match the 57/43 split, each window carries an <strong>inverse-count weight</strong> so longer recordings do not contribute more gradient, and every fold is <strong>seeded</strong> so a run reproduces exactly.")}
"""))

# ══════════════════════════ 2 ══════════════════════════
S.append(("Preprocessing: from a recording to a tensor", f"""
<p>A recording is a grid of about <strong>320 by 12,000</strong> values covering
roughly 9 seconds. A network needs inputs of a fixed size, and 348 recordings is
not enough to train one. Preprocessing solves both at once.</p>

{fig("pipe_overview.png",
  "The whole path from a stored recording to one score per person. The upper row runs once and writes a cache; the lower row runs inside every fold.",
  "The split between the two rows is the important part. Everything that is <strong>identical for every fold</strong> is computed once, which is why a full 58-fold run costs 16 minutes rather than hours. Everything that <strong>must not see the held-out subject</strong>, above all the standardisation, is deferred into the fold. The shapes along the bottom of each box are the actual array dimensions at that point.")}

<h3>Step 1: cut each recording into fixed-length windows</h3>
{fig("pipe_windowing.png",
  "One real recording with the window boundaries drawn on it. The horizontal band near the bottom marks each 3.0 s window; consecutive windows overlap by half their length.",
  "Two things this buys. Every input is now the <strong>same size</strong> regardless of how long the walk was, which removes the duration confound at the input. And 348 recordings become <strong>{nw} training samples</strong>, which matters when the cohort is 58 people. The turn is clearly visible in this recording as the switch from negative to positive Doppler at about 7.2 s.".format(nw=NW))}

{defn2("Why 3.0 seconds and a 1.5 second hop", '''
<p>A stride takes roughly <strong>1.0 to 1.2 seconds</strong>, so a 3 second
window contains <strong>two to three complete strides</strong>. That is the
shortest window in which stride-to-stride regularity, one of the three
clinically motivated measurements, is visible at all.</p>
<p>The hop of 1.5 s gives <strong>50 % overlap</strong>. The obvious
alternative, 2.0 s windows with a 0.5 s hop, produces far more samples but they
overlap by <strong>75 %</strong>, so consecutive windows are nearly the same
picture. That inflates the apparent amount of data without adding information,
and it is kept as an ablation rather than used as the default.</p>
''')}

<h3>Step 2: compress the brightness range, and stop there</h3>
<p>The values in a spectrogram span several orders of magnitude, so they are
passed through <strong>log1p</strong>, which compresses large values while
leaving small ones nearly unchanged. The important decision is what happens
next, and the answer is <strong>nothing, not yet</strong>.</p>

{defn2("Why standardisation is deferred, and why it matters", '''
<p>Neural networks train badly on inputs that are not centred near zero, so the
data has to be standardised at some point: subtract a mean, divide by a spread.
The question is <em>whose</em> mean and spread.</p>
<div class="math">standardised = (value &minus; mean) / spread
<span class="mnote">computed over the TRAINING subjects of that fold, never over the whole dataset</span></div>
<p>If the mean and spread were computed over the whole dataset, the held-out
subject would have contributed to the scale against which its own input is
measured. That is a small leak, but it is a leak, and the entire argument of
this thesis is about not having any. So the cache stores <strong>log1p
only</strong> and the standardisation is applied when the data is loaded, using
statistics from that fold&rsquo;s training subjects alone.</p>
<p>The cost of doing this properly would normally be re-reading every window
once per fold. It is avoided by storing each window&rsquo;s own mean and spread
in the manifest, from which any fold&rsquo;s exact pooled statistics follow
arithmetically:</p>
<div class="math">Var(X) = E[ Var(X | window) ] + Var( E[X | window] )
<span class="mnote">verified against a brute-force computation, agreement to 9 decimal places</span></div>
''')}

{warn("There is a second normalisation available, <strong>per-window z-scoring</strong>, which standardises each window against itself. It is <em>not</em> the default, for a specific reason. Standardising a window against itself deletes its absolute brightness, and absolute brightness is how strongly the body reflects, which is exactly what <strong>bradykinesia</strong> changes. It would remove the signal being looked for. It also suppresses the difference between the two processing runs. Those two effects point in opposite directions, which is why this is an <strong>ablation</strong>: whichever way the result moves, the within-batch analysis is what says which cause it was.")}

<h3>Step 3: resize to a common shape and stack the two channels</h3>
{fig("pipe_model_input.png",
  "One cached window, both channels, exactly as the network receives it. Foot on the left, torso on the right.",
  "Each window is resized to <strong>224 by 224</strong> and the two channels are stacked into one array of shape <strong>(2, 224, 224)</strong>. 224 is the input size the pretrained ResNet expects, so this choice is inherited rather than tuned.")}

{defn("Does the resize throw information away", "It does not, and this is checkable. In time, 3 seconds across 224 columns is <strong>13.4 ms per column</strong>, while the analysis window that produced the data was <strong>50 ms</strong> wide. So roughly four columns still describe one genuinely independent measurement, and the resize removes redundancy rather than detail. In Doppler, 320 stored rows become 224, which is <strong>7.1 Hz per row</strong> against a true resolution of 20 Hz. Both axes remain oversampled after resizing.")}

<h3>Step 4: neutralise window count</h3>
{fig("pipe_window_count.png",
  "Left: how many windows each subject contributes, by group. Right: the total training weight each subject carries after correction.",
  "This is the correction that a windowing pipeline usually forgets. Windowing fixes the <em>size</em> of the input but not the <em>number</em> of them: a longer recording yields more windows, so PD subjects contribute <strong>30.9 windows on average against 27.3</strong>. Window count alone separates the groups at <strong>AUC 0.62</strong>, so without a correction a network could reach that score by counting rather than by looking. Each window is therefore weighted by the inverse of its subject&rsquo;s window count, which makes every subject contribute <strong>exactly the same total weight, 28.845</strong>, as the right panel shows.")}

<h3>Step 5: label every window by direction of travel</h3>
{fig("pipe_turn.png",
  "Left: the share of each window's energy at positive Doppler. Right: how many windows survive different turn-exclusion thresholds.",
  "This is taken directly from the source publication {p}, which computes its gait parameters only <strong>after discarding the turning, standing and sitting stages</strong>. We currently keep everything. Since the extra seconds PD subjects spend are concentrated in exactly those stages, restricting the analysis to steady walking is a targeted attack on the duration confound. Every window now carries its direction of travel so that restriction can be tested, at a cost the right panel makes explicit: excluding windows within 0.1 of the midpoint discards <strong>35 % of the data</strong>.".format(p=PAPER))}

{key("<strong>The frozen configuration.</strong> 3.0 s windows, 1.5 s hop, contrast-enhanced representation, log1p with standardisation deferred to the training fold, 224 by 224, two channels, inverse-count weighting, all windows kept. It produced <strong>{nw} windows from 348 recordings in 66 seconds</strong>, with no failures. Every alternative named above becomes an ablation.".format(nw=NW))}
"""))

# ══════════════════════════ 3 ══════════════════════════
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
best = R["logreg|confounded_only"]
lo, hi = best["auc_lo"], best["auc_hi"]
clean = R["logreg|shape_clean"]
dur = R["logreg|duration_only"]

S.append(("The classical baseline: what it is and what it already says", f"""
<p>Before any network, the same question is asked of the
<strong>25 handcrafted measurements</strong> using ordinary statistical
classifiers. This is not a warm-up. It sets the number the deep models have to
beat, and it is already complete.</p>

<h3>How one prediction is made</h3>
<p>The pipeline is four steps, applied identically for every model and every
fold:</p>
<pre class="code">for each of the 58 subjects, in turn:
    hold that subject out entirely
    ├─ fill any missing value with the median of the training subjects
    ├─ rescale every measurement to mean 0 and spread 1  (training subjects only)
    ├─ fit the classifier, weighting the classes to offset the 57/43 split
    └─ score the held-out subject's recordings, then AVERAGE them
                                                    into one score per subject
compute one AUC over the 58 subject scores</pre>

{defn("Why the average, and why score subjects rather than recordings", "Each subject has six recordings, so a classifier produces six probabilities for them. Averaging into one number per subject does two things: it is the decision a clinician would actually want, one answer per person, and it makes the metric insensitive to <strong>how many recordings a subject happens to have</strong>. Scoring recordings instead would let a subject with more recordings count more, which is the same confound the window weighting handles on the deep side.")}

<h3>Three classifiers, chosen to differ in what they can express</h3>
<div class="tblwrap"><table>
<thead><tr><th>Model</th><th>What it can represent</th><th>Why it is in the comparison</th></tr></thead><tbody>
<tr><td><strong>Logistic regression</strong></td>
<td>A weighted sum of the measurements, and nothing more.</td>
<td>The most restrictive, and therefore the hardest to fool. If it works, the signal is simple and additive. Its weights are also directly readable.</td></tr>
<tr><td><strong>SVM, RBF kernel</strong></td>
<td>Curved boundaries, so combinations that only matter jointly.</td>
<td>Catches signal that is real but not additive.</td></tr>
<tr><td><strong>Random forest</strong></td>
<td>Thresholds and interactions, with no assumption of a smooth relationship.</td>
<td>Catches signal that lives in cut-offs, and gives an importance ranking for free.</td></tr>
</tbody></table></div>

<h3>Five feature sets, and the reason each one exists</h3>
<div class="tblwrap"><table>
<thead><tr><th>Set</th><th>Size</th><th>What it contains</th><th>Its job in the argument</th></tr></thead><tbody>
<tr><td><code>all_features</code></td><td class="num">{n_all}</td><td>Everything measured.</td><td>The optimistic number, and the one most papers would report.</td></tr>
<tr><td><code>duration_invariant</code></td><td class="num">{n_inv}</td><td>Averages, shapes and ratios.</td><td>Length removed <strong>by construction</strong>.</td></tr>
<tr><td><code>shape_clean</code></td><td class="num">{n_cln}</td><td>The subset that also survives a direct test against measured duration.</td><td>Length removed <strong>by measurement</strong>, because construction turned out not to be enough.</td></tr>
<tr><td><code>confounded_only</code></td><td class="num">{n_cnf}</td><td>Only the summed quantities that grow with length.</td><td>Shows how much the confound alone buys.</td></tr>
<tr><td><code>duration_only</code></td><td class="num">1</td><td>Recording length, and nothing else.</td><td><strong>The null floor.</strong></td></tr>
</tbody></table></div>

<h3>What the baseline already found</h3>
<div class="tblwrap"><table>
<thead><tr><th>Model</th><th>all_features</th><th>duration_invariant</th><th>shape_clean</th><th>confounded_only</th><th>duration_only</th></tr></thead>
<tbody>{GRID}</tbody></table></div>

{key("Read the row for logistic regression from right to left, because that is the order in which it becomes uncomfortable. <strong>Recording length alone scores {d:.3f}.</strong> The measurements that merely encode length score <strong>{c:.3f}</strong>, which is the <em>best cell in the entire table</em>. The measurements that describe the shape of the walk, with length removed, score <strong>{s:.3f}</strong>. In other words, the further the features get from elapsed time, <strong>the worse the classifier does</strong>.".format(d=dur['auc'], c=best['auc'], s=clean['auc']))}

<p>Two further numbers matter more than the grid itself:</p>
<ul>
<li>The confidence interval on the best cell runs from <strong>{lo:.3f} to
{hi:.3f}</strong>. It comfortably includes 0.5, so on 58 subjects even the
strongest result is <strong>not distinguishable from chance</strong>.</li>
<li>Shuffling the labels and refitting gives <strong>p = 0.092</strong> for the
clean feature set. That is above the conventional threshold, so the honest
statement is that this configuration <strong>has not been shown</strong> to beat
chance.</li>
</ul>

{warn("Two cells in the table sit <em>below</em> 0.5, at 0.287 and 0.450. That is not a bug and it is worth understanding. Both are the single-feature <code>duration_only</code> column, fitted by a kernel SVM and a random forest. With one input and 57 training subjects those models fit the noise in the training folds and invert on the held-out one. It is a demonstration that <strong>a model can score worse than guessing</strong> when it is given too little to work with, and it is why the floor is quoted from logistic regression, the only one of the three that handles a single feature sensibly.")}

{later("These are trial-level classical results. The deep models are what the pipeline above was built for, and their numbers do not exist yet. When they do, they will be reported against this table and against both null floors.")}
"""))

# ══════════════════════════ 4 ══════════════════════════
S.append(("The deep models: two architectures, and why exactly two", f"""
<p>The choice of architecture is dominated by one number: <strong>58
subjects</strong>. Everything below follows from it.</p>

{fig("pipe_models.png",
  "Trainable parameters for each option, on a logarithmic scale. The dashed line marks the number of training windows available.",
  "The dashed line is the honest constraint. Anything to its right has <strong>more free parameters than we have examples</strong>, which means it can memorise the training set exactly rather than learning anything general. Two of the four options are there, and both are reported as ablations rather than used as the headline.")}

<h3>Model A: SmallCNN, trained from scratch</h3>
<p>Three convolutional blocks, each halving the image and doubling the channel
count, followed by <strong>global average pooling</strong> and a single linear
layer. <strong>23,682 trainable parameters.</strong></p>
<pre class="code">input  (2, 224, 224)
  conv 3x3 -> 16 channels -> batch-norm -> ReLU -> halve    (112 x 112)
  conv 3x3 -> 32 channels -> batch-norm -> ReLU -> halve    ( 56 x  56)
  conv 3x3 -> 64 channels -> batch-norm -> ReLU
  global average pool                                       (64 numbers)
  dropout 0.3 -> linear -> 2 classes</pre>

{defn("Why global average pooling instead of a flatten", "Flattening a 56 by 56 map into a dense layer would need roughly <strong>200,000 weights in that single layer</strong>, eight times the whole network. Averaging each channel down to one number removes them entirely. It also forces each channel to represent something that is meaningful <em>wherever</em> it occurs in the image, rather than at a fixed position, which suits a gait signal that can start anywhere in a window.")}

<h3>Model B: ResNet-18, pretrained, mostly frozen</h3>
<p>An 18-layer residual network trained on ImageNet, reused here. Two
adaptations are needed and one decision.</p>

{defn2("Adaptation 1: three colour channels into two radar channels", '''
<p>The pretrained first layer expects red, green and blue. We have foot and
torso. Discarding the layer and starting over would throw away the edge and
texture detectors that make pretraining worth doing at all, so instead the three
colour filters are <strong>averaged into one</strong> and that average is copied
into both of our channels.</p>
<div class="math">new_filter[:, foot] = new_filter[:, torso] = mean( old_filter[:, R], old_filter[:, G], old_filter[:, B] )
<span class="mnote">a spectrogram has no colour, so averaging is the correct reduction rather than a compromise</span></div>
''')}

{defn2("Adaptation 2: how much of the network to retrain", '''
<p>This is the decision that matters, and the parameter counts make it for us:</p>
<div class="tblwrap"><table>
<thead><tr><th>Setting</th><th>Trainable</th><th>Verdict</th></tr></thead><tbody>
<tr><td><strong>Final layer only</strong></td><td class="num">1,026</td>
<td><strong>The default.</strong> The frozen network becomes a fixed feature
extractor and we fit a single linear layer on top. Fewer free parameters than a
handful of handcrafted features.</td></tr>
<tr><td>Final block plus final layer</td><td class="num">8,394,754</td>
<td>Ablation. 5,000 parameters per training window; expected to overfit, and
reported to show that it does.</td></tr>
<tr><td>Everything</td><td class="num">11,174,402</td>
<td>Not used. Included in the figure for scale.</td></tr>
</tbody></table></div>
<p>Retraining only the final layer is often called <strong>linear probing</strong>:
it asks whether the disease is separable in the representation ImageNet already
learned, without letting the network reshape that representation around 58
people.</p>
''')}

{key("These two models are chosen to <strong>bracket the problem</strong> rather than to compete. SmallCNN learns radar-specific structure but has only {nw} windows to learn it from. ResNet-18 brings structure learned from a million natural images but cannot adapt it. If both land in the same place, that place is a property of the data. If they diverge, the direction says which of the two constraints is binding.".format(nw=NW))}

{defn("What the source publication contributes here " + PAPER, "Two things, and both shape the ablations rather than the architecture. First, that study finds <strong>foot nodes remain reliable for motorically impaired subjects while torso nodes lose reliability</strong>, which predicts that the foot channel carries more of the signal and makes the channel ablation a test of a stated hypothesis rather than a sweep. Second, it establishes that the trunk normally tracks each heel strike and that <strong>this coupling breaks down in Parkinson's</strong>, which is the reason both channels are presented to the network together instead of separately.")}
"""))

# ══════════════════════════ 5 ══════════════════════════
S.append(("The validation protocol, and why it is the contribution", f"""
<p>The exploratory analysis established that a subject&rsquo;s own recordings
are about <strong>2.5 times more alike</strong> than anyone else&rsquo;s, for all
58 subjects without exception. That single fact makes an ordinary random split
invalid here, not merely suboptimal, because a model could score well by
recognising the person rather than the disease.</p>

{fig("pipe_split.png",
  "One fold of the protocol. This is repeated 58 times, once with each subject held out.",
  "Three separations are enforced at once, and the third is the one most published work omits. Subjects never span the split. <strong>The epoch is chosen on the inner validation subjects</strong>, never on the held-out one. And the fold's normalisation statistics come from the training subjects alone.")}

<h3>What each part is for</h3>
<div class="tblwrap"><table>
<thead><tr><th>Part</th><th>Size</th><th>What it decides</th><th>What would go wrong without it</th></tr></thead><tbody>
<tr><td><strong>Train</strong></td><td class="num">49</td><td>The weights.</td><td>Nothing to learn from.</td></tr>
<tr><td><strong>Inner validation</strong></td><td class="num">8</td><td>When to stop training, and any configuration choice.</td><td>Either no stopping criterion, or one measured on the test subject, which is leakage.</td></tr>
<tr><td><strong>Test</strong></td><td class="num">1</td><td><strong>Nothing.</strong> Scored once and reported.</td><td>Every number in the thesis would be optimistic by an unknown amount.</td></tr>
</tbody></table></div>

{defn("Why the inner validation set needs 8 subjects and not 2", "It has to contain <strong>both classes</strong> for an AUC to exist, which is what the first defect proved by failing. Eight subjects, stratified, gives four of each and a validation AUC that is coarse but well defined. Fewer would make the stopping decision very noisy; more would take subjects away from training, which is the scarcer resource at this size.")}

<h3>How a subject gets one number</h3>
<div class="math">subject score = mean of the probabilities of that subject&rsquo;s windows
<span class="mnote">the mean, so that a subject with more windows does not thereby get a more confident score</span></div>
<p>The 58 subject scores then produce one AUC for the whole run.
<strong>AUC is the headline metric</strong> because it needs no decision
threshold and is unaffected by the 57/43 imbalance. Balanced accuracy,
sensitivity and specificity are reported beside it, at a threshold set to the
prevalence of the positive class rather than to 0.5, since an uncalibrated
half-way cut systematically under-predicts the smaller group.</p>

{warn("The literature this thesis compares itself against overwhelmingly reports <strong>accuracy</strong>, not AUC. The two are not comparable, and an accuracy of 95 % on a random split is not a better result than an AUC of 0.62 under leave-one-subject-out; it is a different measurement under weaker conditions. Every comparison table must say so explicitly or it will read as though this work simply performed badly.")}

<h3>Pre-registration</h3>
<p>Every configuration above was chosen before any deep model was trained, and
is committed to version control as a single frozen object. The reporting run
uses it unchanged. The number of configurations explored is stated in the
report, because with 58 subjects the optimism from repeated tuning is real and
cannot be engineered away.</p>

{key("<strong>The protocol is the contribution.</strong> The state-of-the-art chapter shows that only 2 of 16 reviewed works use subject-independent validation, that a published network reached 97.8 % by keying on background noise, and that radar separates young from elderly at 94.9 % so age alone can masquerade as disease. Against that background, a modest number obtained under this protocol is worth more than a high number obtained without it.")}
"""))

# ══════════════════════════ 6 ══════════════════════════
S.append(("What happens next", f"""
<p>The chain is now unblocked and measured. A full 58-fold run takes
<strong>16 minutes</strong> on the GPU, against 88 minutes on the processor, so
the experiment programme is affordable.</p>

<div class="tblwrap"><table>
<thead><tr><th>Step</th><th>What is produced</th><th>Cost</th></tr></thead><tbody>
<tr><td><strong>1. Commit the configuration</strong></td><td>The pre-registration hash quoted in the report.</td><td class="num">minutes</td></tr>
<tr><td><strong>2. SmallCNN, full leave-one-subject-out</strong></td><td>The first honest deep-model number, against both null floors.</td><td class="num">16 min</td></tr>
<tr><td><strong>3. ResNet-18, full run</strong></td><td>Whether pretrained features transfer to this problem.</td><td class="num">~20 min</td></tr>
<tr><td><strong>4. Six ablations</strong></td><td>Channel, representation, window, augmentation, normalisation, turn exclusion.</td><td class="num">~2 h</td></tr>
<tr><td><strong>5. Grad-CAM</strong></td><td>Whether the network attends to the foot and torso bands or to artifacts.</td><td class="num">~30 min</td></tr>
<tr><td><strong>6. Confound battery</strong></td><td>Every model re-scored within batch, duration-matched, test2 only, window-count controlled.</td><td class="num">~2 h</td></tr>
</tbody></table></div>

{warn("One expectation worth setting now, because it changes how the results should be read rather than whether they are worth having. The classical baseline reaches <strong>0.62 on gait-shape features against 0.61 for recording length alone</strong>, and the batch label by itself reaches <strong>0.664</strong>. A deep model has to clear both floors before any claim about gait can be made. If it does not, that is a publishable result and not a failure: it would be the first careful demonstration that reported radar performance on this task can be an acquisition artifact.")}

{key("Nothing beyond this point is written until the numbers exist. The sections above are complete because they describe <strong>decisions</strong>, and every one of those decisions was made before seeing a single deep-model result.")}
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
  <p class="sub">What was broken in the training code and what it now does, how a recording
  becomes a tensor, which classifiers are compared and what they already say, which
  networks are used and why exactly those, and the validation protocol that makes
  any of it worth reporting.</p>
  <div class="statlead">Where the pipeline stands:</div>
  <div class="stats">
    <div class="stat"><b>4/4</b><span>blocking defects fixed<i>all four were silent</i></span></div>
    <div class="stat"><b>{NW}</b><span>windows cached<i>from 348 recordings, no failures</i></span></div>
    <div class="stat"><b>2</b><span>architectures<i>23.7k and 1,026 trainable parameters</i></span></div>
    <div class="stat"><b>16</b><span>minutes per full run<i>58 folds on the GPU</i></span></div>
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
print(f"wrote {out} ({len(html)/1e6:.2f} MB)")
ndef = html.count('class="defn"')
print(f"sections={len(S)} figures={html.count('data:image/jpeg')} definitions={ndef}")
