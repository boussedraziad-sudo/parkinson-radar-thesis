#!/usr/bin/env python3
"""DISCUSSION.html: the discussion chapter, drafted for review.

Same style rules as the other documents: no em-dashes, key terms bolded,
intro -> figure -> reading, numbers pulled from stored results where they
exist as files.
"""
import base64, io, json, pathlib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = pathlib.Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
FIG = ROOT / "outputs/figures"
CSFIG = ROOT / "classical_split/outputs/figures"
AXFIG = ROOT / "experiments/alexnet/outputs/figures"
CAMP = json.load(open(ROOT / "outputs/runs/campaign_summary.json"))
BG = json.load(open(ROOT / "experiments/bg_suppressed/outputs/bg_suppressed_summary.json"))
CS = json.load(open(ROOT / "classical_split/outputs/results.json"))

# --- AlexNet experiment: stored results ---
AXD = ROOT / "experiments/alexnet/outputs"
AX_EXTRA = json.load(open(AXD / "alexnet_loso_extra.json"))
AX_SUMM = json.load(open(AXD / "alexnet_loso_summary.json"))["summary"]
AX_RAND = json.load(open(AXD / "alexnet_random.json"))
_rs = pd.read_csv(ROOT / "outputs/runs/resnet18_fc_v2_folds.csv")
RS_CORR_NWIN = float(np.corrcoef(_rs["subject_prob"], _rs["n_test_windows"])[0, 1])

# --- Grad-CAM per-window summary: band statistics computed from the stored CSV ---
GC = pd.read_csv(ROOT / "outputs/metrics/gradcam_summary.csv")

def gc_stats(model):
    g = GC[GC["model"] == model]
    dens_cols = ["att_mean_torso", "att_mean_foot", "att_mean_bg"]
    dens = {b: float(g[f"att_mean_{b}"].mean()) for b in ("torso", "foot", "bg")}
    called = {c: float(g[(g["prob_pd"] >= 0.5) == bool(c)][dens_cols].mean().mean())
              for c in (1, 0)}
    bylab = {l: {"bg_mass": float(g.loc[g["label"] == l, "att_mass_bg"].mean()),
                 "com": float(g.loc[g["label"] == l, "att_com_abs_hz"].mean())}
             for l in (1, 0)}
    return {"dens": dens, "called": called, "bylab": bylab}

GC_SC = gc_stats("smallcnn_v2")
GC_RS = gc_stats("resnet18_fc_v2")


def cs_mean(exp, tag, grp, field):
    vals = [r[tag][grp][field] for r in CS if r["experiment"] == exp]
    return float(np.mean(vals))


def img(name, base=FIG, maxw=1500, q=84):
    p = base / name
    if not p.exists():
        return f'<p style="color:#b00">[missing figure: {name}]</p>'
    im = Image.open(p).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" alt="{name}">'


def fig(name, caption, insight=None, base=FIG, maxw=1500):
    ins = (f'<div class="insight"><span class="ilab">Reading the figure</span>'
           f'<p>{insight}</p></div>') if insight else ""
    return (f'<figure>{img(name, base, maxw)}<figcaption><span class="fname">{name}</span> '
            f'{caption}</figcaption>{ins}</figure>')


def key(t):  return f'<aside class="key"><span class="klabel">Key point</span><p>{t}</p></aside>'
def warn(t): return f'<aside class="warn"><span class="klabel">Caution</span><p>{t}</p></aside>'
def defn(term, body): return f'<div class="defn"><span class="dterm">{term}</span><p>{body}</p></div>'
def intro(t): return f'<div class="intro"><p>{t}</p></div>'

CSS = (ROOT / "tools/eda.css").read_text() + """
.sec h4{font-family:var(--sans);font-size:15px;font-weight:700;color:var(--ink);margin:26px 0 8px}
.intro{border-left:3px solid var(--accent);padding:2px 0 2px 16px;margin:22px 0 6px}
.intro p{margin:0;font-size:15.5px;line-height:1.62;max-width:none;color:var(--ink)}
"""

S = []

# ═══ 1 ═══
S.append(("What this chapter argues", f"""
<p>The results chapter measured what this dataset supports under
subject-independent validation: subject-level scores that sit in the same
band as the score recording duration provides on its own. Two questions
follow from that measurement, and this chapter answers both with experiments
rather than argument. The first is why published radar gait studies report
so much more. <strong>Section 2</strong> reproduces the literature&rsquo;s
numbers on this very dataset by changing nothing but the validation design,
first with the campaign&rsquo;s own network and then with the field&rsquo;s
favourite architecture, and shows that those numbers collapse the moment a
test subject is genuinely new. The second is what the scores obtained here
are actually made of. <strong>Section 3</strong> reads the models&rsquo; own
attention maps, <strong>Section 4</strong> deletes what they attend to and
confirms that no hidden gait signature was underneath, <strong>Section
5</strong> states the findings the data does establish, and <strong>Sections
6 and 7</strong> close with the limitations and the concrete path to
stronger results.</p>

{key("<strong>The framing the chapter defends.</strong> The gap between this thesis's numbers and the literature's is not a performance gap. It is a <strong>measurement gap</strong>: the two are answering different questions. The literature's question, can a model label recordings of people it has already seen, is answered here too, and with literature-grade numbers. The clinical question, can a model flag a person it has never seen, is the one this thesis insists on, and its honest answer on this dataset is: not yet, and not demonstrably from gait shape.")}
"""))

# ═══ 2 ═══
S.append(("The random-split demonstration: the gap is the protocol", f"""
{intro("The claim that the literature's numbers are inflated by validation design would normally rest on reading papers. Here it was tested: the same network and data as the main campaign, trained the field's way, with a random hold-out split, early stopping on window-level scores and no window weighting, so that the whole published recipe is reproduced rather than argued about.")}

<div class="tblwrap"><table>
<thead><tr><th>Validation design</th><th>Window AUC</th><th>What it measures</th></tr></thead><tbody>
<tr><td><strong>Window-level random split</strong> (frame-level)</td>
<td class="num">{cs_mean('window_split','test','window','auc'):.2f}</td>
<td>Nearly identical windows of the same walk on both sides of the split.</td></tr>
<tr><td><strong>Recording-level random split</strong> (the literature&rsquo;s hold-out)</td>
<td class="num">{cs_mean('recording_split','test','window','auc'):.2f}</td>
<td>Every test recording comes from a person the model trained on.</td></tr>
<tr><td><strong>Same trained model, subjects excluded from training</strong></td>
<td class="num">{cs_mean('collapse_test','never_seen','window','auc'):.2f}</td>
<td>The person is unknown; only the disease could help.</td></tr>
<tr><td><strong>Leave-one-subject-out</strong> (the thesis protocol)</td>
<td class="num">{CAMP['resnet18_fc_v2']['pooled']:.2f}</td>
<td>Every subject scored by a model that never saw them (subject-level AUC).</td></tr>
</tbody></table></div>

{defn("How to read the table", "All four rows use the <strong>same architecture</strong>, the ResNet-18 transfer model of the main campaign; the first three rows are trained identically and differ only in what the split keeps apart. Row by row. The <strong>window split</strong> shuffles the three-second windows themselves (3 repeats), so nearly identical overlapping frames of the same walk sit on both sides of the split. The <strong>recording split</strong> shuffles whole walks (5 repeats), so each test walk is new but the walker's other walks are in training, which is the design most published studies use. In the <strong>third row</strong>, eight subjects (five control, three PD) are set aside before anything is trained; the model trains recording-split style on the remaining fifty subjects, and the finished model then classifies every window of the eight strangers' recordings; the score is over those windows (3 repeats, a fresh eight each time). In the <strong>last row</strong>, one subject is set aside, the model trains on the other 57, and then scores every window of all of that person's recordings (typically ten walks); the average of those window probabilities becomes that one person's score. Repeating this 58 times, once per subject, gives 58 scores, and the AUC over them is the number shown. One unit caveat: the first three rows are window-level AUC averaged over the repeats, the level at which the literature scores, while the last row is subject-level, one score per person. The trend is the point: the more the split leaks, the higher the number, and stripping the leakage returns the same model, on the same data, to the modest band the thesis reports.")}

{fig("cs_collapse.png",
  "The collapse test, the third row of the table: three ResNet-18 transfer models, one line each. The left end of each line is that model's window AUC on test recordings of subjects it saw in training; the right end is the same model on the eight subjects excluded from training.",
  "<strong>What the figure shows.</strong> Each line is one trained ResNet-18 transfer model, not one subject: three repeats were run, each with a different set of eight subjects excluded. On the left, every model scores high on test recordings of people whose other walks it trained on. On the right, the very same models are asked about the eight people they have never seen, and every line falls; two of the three land at chance.<br><br><strong>What it means.</strong> A model that had learned the disease would hold its score on strangers, because Parkinson&rsquo;s looks like Parkinson&rsquo;s in people it never met. A model that learned <em>people</em> scores high on the familiar and collapses on the unknown, and that is what happens. The inflated numbers are real numbers; they are simply answers to the recognition question, not the screening question.",
  base=CSFIG)}

{defn("Why this opens the discussion", "Every later interpretation depends on this point. Once it is established that the protocol alone manufactures a 0.2 to 0.3 AUC difference on this very dataset, comparisons with published accuracies stop being embarrassing and start being diagnostic: the field&rsquo;s numbers tell us more about its validation habits than about radar&rsquo;s clinical value.")}

<h4>The same demonstration with the field&rsquo;s own network</h4>

{intro("The demonstration above uses the thesis's own network, and a sceptic could object that the collapse says something about that particular model rather than about the protocol. So the experiment was repeated with <strong>AlexNet</strong>, the architecture behind the field's headline spectrogram result, adapted exactly as the thesis's ResNet transfer model: ImageNet-pretrained, first convolution averaged to the two radar channels, classifier retrained (54.5 million trainable parameters) at the source literature's learning rate. Same window cache, same augmentation; the random-split runs also adopt the literature's training conventions, early stopping on window-level scores and no window weighting.")}

<div class="tblwrap"><table>
<thead><tr><th>Validation design</th><th>Train / validation / test</th><th>Subject AUC</th><th>95&nbsp;% CI</th><th>Test subjects seen in training</th></tr></thead><tbody>
<tr><td><strong>Random split</strong>, seed 1</td>
<td>70 / 15 / 15&nbsp;% of recordings</td>
<td class="num">{AX_RAND[0]['subject']['auc']:.3f}</td><td class="num">&ndash;</td><td>100&nbsp;%</td></tr>
<tr><td><strong>Random split</strong>, seed 2</td>
<td>70 / 15 / 15&nbsp;% of recordings</td>
<td class="num">{AX_RAND[1]['subject']['auc']:.3f}</td><td class="num">&ndash;</td><td>100&nbsp;%</td></tr>
<tr><td><strong>Random split</strong>, seed 3</td>
<td>70 / 15 / 15&nbsp;% of recordings</td>
<td class="num">{AX_RAND[2]['subject']['auc']:.3f}</td><td class="num">&ndash;</td><td>100&nbsp;%</td></tr>
<tr><td><strong>Leave-one-subject-out</strong></td>
<td>49 / 8 / 1 subjects, repeated 58 times</td>
<td class="num">{AX_EXTRA['pooled']:.3f}</td>
<td class="num">[{AX_EXTRA['ci'][0]:.3f}, {AX_EXTRA['ci'][1]:.3f}]</td><td>0&nbsp;%</td></tr>
</tbody></table></div>

<p>A <strong>seed</strong> is one repeat of the split: the same pool of
recordings shuffled with a different random starting point, so the three
rows are three independent draws of the literature&rsquo;s protocol. In the
random split the units dealt into the 70/15/15 piles are whole recordings;
in leave-one-subject-out the units are people, with 49 subjects training the
model, 8 steering early stopping, and 1 being scored, rotated until each of
the 58 has been the test subject once.</p>

{fig("alex_ladder.png",
  "AlexNet's subject-level AUC under the literature's random split (red dots, one per seed) and under leave-one-subject-out (blue dot with its 95 % confidence interval), against the chance and duration-alone lines.",
  f"<strong>What the figure shows.</strong> Each red dot is one repeat of the literature's protocol: the same recordings shuffled 70/15/15 with a different seed, scored per subject on the held-out share ({AX_RAND[0]['subject']['auc']:.2f}, {AX_RAND[1]['subject']['auc']:.2f} and {AX_RAND[2]['subject']['auc']:.2f}, each dot labelled with its score). The blue dot is the same architecture under leave-one-subject-out, {AX_EXTRA['pooled']:.2f}, with whiskers marking the 95&nbsp;% confidence interval [{AX_EXTRA['ci'][0]:.2f}, {AX_EXTRA['ci'][1]:.2f}]. The grey line is chance (0.50); the violet line is the duration floor (0.610), the score recording duration alone provides.<br><br><strong>What it means.</strong> Three readings. First, the field's own architecture reaches {AX_RAND[0]['subject']['auc']:.2f} and {AX_RAND[1]['subject']['auc']:.2f} the moment every test subject is someone it trained on, and drops the moment they are not: the split, not the architecture, decides whether the number is literature-grade or honest. Second, the inflated number is itself unstable: reshuffling the same recordings moves it from 0.89 down to 0.74, a 0.16 swing between seeds. Third, the honest interval reaches down past the duration line, so even the best compliant score cannot be distinguished from what elapsed time alone provides.",
  base=AXFIG)}

<p>An AUC summarises how well the scores <em>rank</em> people; a screening
decision is a yes or no. The confusion matrices below show the same
network&rsquo;s actual decisions twice, first on <strong>subjects it saw in
training</strong> and then on <strong>subjects it has never seen</strong>,
so the two protocols can be compared decision by decision.</p>

{fig("alex_confusion_pair.png",
  "AlexNet's decisions on subjects seen in training (left: random split, window level, pooled over the three seeds) and on never-seen subjects (right: leave-one-subject-out, one decision per person, threshold at the 43 % PD prevalence).",
  "<strong>How each side is counted.</strong> The two panels count different things. On the left, each cell counts <strong>three-second windows</strong>: every test window is classified on its own at the default 0.5 cut-off, pooled over the three seeds' test sets. On the right, each cell counts <strong>people</strong>: all windows of a person's recordings are scored, their average becomes that person's single probability, and the person is called PD when it exceeds the prevalence threshold of 0.43. Neither panel is an AUC; these are the actual yes/no decisions.<br><br><strong>What the figure shows.</strong> On seen subjects (left) the model is balanced and looks useful: about 70&nbsp;% of each class is called correctly. On never-seen subjects (right) the balance breaks: it catches 21 of the 25 PD patients, but only by flagging 18 of the 33 healthy controls along with them, a specificity of 45&nbsp;%, worse than a coin flip.<br><br><strong>What it means.</strong> The left panel is what the literature's validation rewards, and it is real; it is also the answer to the recognition question, because every one of those windows belongs to a person whose other walks were in training. The right panel is the clinical question: faced with a stranger, the network cannot separate healthy from ill, only lean toward calling recordings PD, so of the 39 people it would send onward for review, 18 are healthy. The same architecture is clinic-ready on people it knows and close to a biased guess on the population a screening tool actually exists for.",
  base=AXFIG)}

<p>Two details make the honest AlexNet score itself instructive. Its
subject scores carry <strong>more duration residue than any of the
campaign&rsquo;s headline models</strong> (correlation
+{AX_EXTRA['corr_nwin']:.2f} with window count, against
+{RS_CORR_NWIN:.2f} for the ResNet transfer model): with 54.5
million parameters free, it drinks more deeply from the same confound, it
does not find a new signal. And its best validation epoch averages
{AX_SUMM['mean_best_epoch']:.1f}, meaning the score is essentially in place
after a single pass over the data, the learning profile of an easy shortcut
rather than of a subtle clinical pattern.</p>
"""))

# ═══ 3 ═══
S.append(("What the models actually read", f"""
{intro("The results chapter reported the scores; the attention analysis explains them. Grad-CAM maps were computed for every window of each of the <strong>58 held-out subjects</strong>, for <strong>both image models</strong>: the compact CNN trained from scratch (SmallCNN) and the pretrained deep network (the ResNet-18 transfer model). One gallery per model follows, so the finding can be checked on each.")}

<h4>The compact CNN trained from scratch (SmallCNN)</h4>

{fig("gradcam_gallery.png",
  "Where the SmallCNN's PD output looks, for confident correct decisions (top) and confident mistakes (bottom): spectrogram in grey, attention in colour.",
  "<strong>What the figure shows.</strong> In every window called PD, right or wrong, the attention floods the empty background; the gait envelope itself is the one region left unhighlighted. In every window called control the map stays dark.<br><br><strong>What it means.</strong> The decision variable is a property of the recording, its background and enhancement texture, not of the walk. The mistakes obey the same rule, which is why they are systematic rather than random: noisy-backed controls get flagged, clean-backed patients get missed.")}

<h4>The pretrained deep network (ResNet-18 transfer model)</h4>

{fig("gradcam_gallery_resnet.png",
  "The same gallery for the ResNet-18 transfer model: confident correct decisions on top, confident mistakes below.",
  f"<strong>What the figure shows.</strong> The deep network obeys the same rule, expressed more diffusely: windows it calls PD are flooded with attention across the whole image, silent rows included, while windows it calls control are left almost unattended. Averaged over every window of all 58 held-out subjects, its attention density on called-PD windows is {GC_RS['called'][1]:.2f} against {GC_RS['called'][0]:.2f} on called-control windows; for the compact CNN the same pair is {GC_SC['called'][1]:.2f} against {GC_SC['called'][0]:.2f}.<br><br><strong>What it means.</strong> Two different architectures, one trained from scratch and one built on frozen ImageNet features, converge on the same behaviour: respond to the overall character of the recording, not to any localised gait structure. The finding is a property of what the images offer, not a quirk of one network.")}

<p>Three quantitative facts, computed over every window of all 58 held-out
subjects, complete the picture. For the compact CNN, attention density is highest in
the silent rows above 500&nbsp;Hz ({GC_SC['dens']['bg']:.2f} per pixel
against {GC_SC['dens']['torso']:.2f} in the torso band), a region containing
no gait energy, and its attention sits visibly higher in frequency on PD
windows than on control windows (centre of mass
{GC_SC['bylab'][1]['com']:.0f} against {GC_SC['bylab'][0]['com']:.0f}&nbsp;Hz).
For the compact CNN, turn-straddling windows also draw <strong>four to six
times</strong> the attention of steady-walking windows, matching the ablation
in which excluding the turn collapsed its score. The transfer model differs
in one telling way: because its backbone is frozen, its maps average out
<strong>identical between the true classes</strong> (background attention
share {GC_RS['bylab'][1]['bg_mass']:.3f} on PD windows against
{GC_RS['bylab'][0]['bg_mass']:.3f} on control windows), so a fixed ImageNet
description of these images contains nothing class-specific to point at, and
its thin trained head can only read overall texture.</p>

{key("Taken together with the exploratory analysis, the mechanism of every score in this thesis is now explicit: <strong>elapsed time</strong> (carried by turn-and-stand content and window counts) plus <strong>per-recording character</strong> (carried by background and enhancement texture). Both are properties a screening tool must not rely on; neither is the spectral shape of a Parkinsonian walk.")}
"""))

# ═══ 4 ═══
S.append(("The intervention test: removing what they read", f"""
{intro("An interpretability finding invites an intervention: if the models read the background, delete the background and see what remains. The follow-up experiment, pre-registered before any window was rebuilt, subtracted each recording's background profile and cropped the silent rows beyond 500 Hz, then re-ran both image models under the unchanged protocol.")}

<div class="tblwrap"><table>
<thead><tr><th>Model</th><th>Standard input</th><th>Background-suppressed</th><th>95&nbsp;% CI (suppressed)</th></tr></thead><tbody>
<tr><td><strong>SmallCNN</strong></td><td class="num">{CAMP['smallcnn_v2']['pooled']:.3f}</td><td class="num">{BG['smallcnn_bg']['pooled']:.3f}</td><td class="num">[{BG['smallcnn_bg']['ci'][0]:.3f}, {BG['smallcnn_bg']['ci'][1]:.3f}]</td></tr>
<tr><td><strong>ResNet-18 probe</strong></td><td class="num">{CAMP['resnet18_fc_v2']['pooled']:.3f}</td><td class="num">{BG['resnet18_fc_bg']['pooled']:.3f}</td><td class="num">[{BG['resnet18_fc_bg']['ci'][0]:.3f}, {BG['resnet18_fc_bg']['ci'][1]:.3f}]</td></tr>
</tbody></table></div>

<p>Two conclusions, both informative:</p>
<ul>
<li><strong>No hidden gait signal surfaced.</strong> If the texture had been
masking a real gait signature, removing it should have lifted the scores
clearly above the duration floor. Every confidence interval still contains
0.610.</li>
<li><strong>The nuisance is distributed, not localised.</strong> The scores
did not collapse either: enough per-recording character survives inside the
gait band, plausibly in the per-recording contrast enhancement, for the
models to keep finding it. The recording-level nuisance is a property of how
each file was processed, not one removable region of the image.</li>
</ul>

"""))

# ═══ 5 ═══
S.append(("What was found, stated positively", f"""
{intro("Not everything in this thesis is a negative result. The findings below are established by the data and hold under the strict protocol; each row gives the finding, the evidence in brief, and what it tells us.")}

<div class="tblwrap"><table>
<thead><tr><th>Finding</th><th>Evidence</th><th>What it tells us</th></tr></thead><tbody>
<tr><td><strong>PD subjects take measurably longer</strong>, and the extra time concentrates in standing up and turning.</td>
<td>The duration gap between the groups is clear in the recording lengths, is largest in the walk variant that includes the chair phase, and shrinks when the turn is removed.</td>
<td>Radar captures the slowness of movement that clinicians already test for; the timing of a walk is a real, usable signal.</td></tr>
<tr><td><strong>Radar timing is a credible screening quantity.</strong></td>
<td>Recording duration alone separates the groups above chance under honest, subject-independent validation.</td>
<td>An automated timed-up-and-go measurement is this dataset&rsquo;s trustworthy first-line signal, and it needs no deep network at all.</td></tr>
<tr><td><strong>A walk identifies the walker.</strong></td>
<td>Recordings of the same person are far more alike than recordings of different people, for every one of the 58 subjects.</td>
<td>Subject-independent validation is mandatory, and radar gait could serve as a biometric even where diagnosis fails.</td></tr>
<tr><td><strong>The field&rsquo;s validation gap is real and measurable.</strong></td>
<td>Relaxing the protocol on this one dataset reproduces literature-grade numbers that collapse on unseen subjects.</td>
<td>Published accuracies reflect validation design as much as clinical signal, so honest evaluation has to be demanded before numbers are compared.</td></tr>
</tbody></table></div>

<p>Against the literature, these results are not an outlier; they are what the
few honest data points predict. Of the sixteen works reviewed in the state of
the art, only two report a genuinely subject-independent protocol, and one of
them, the lightweight radar framework of Samimi Fard et al., saw its scores
fall the moment it moved from a pooled split to leave-one-person-out
evaluation. The study closest to ours in modality, the radar gait work of
Hayashi and Saho et al., preferred its modest envelope model over a
spectrogram network whose higher score its own authors distrusted, because
that network had keyed on background differences between recording sites.
This thesis extends that honest line to Parkinson&rsquo;s screening and
explains, mechanistically, where the distrusted scores come from.</p>
"""))

# ═══ 6 ═══
S.append(("Limitations, and what would change the picture", f"""
<div class="tblwrap"><table>
<thead><tr><th>Limitation</th><th>Consequence, and what would resolve it</th></tr></thead><tbody>
<tr><td><strong>No age metadata.</strong></td>
<td>The dominant untestable confound: PD subjects are typically older, radar separates young from elderly walkers at 94.9&nbsp;%, and even the surviving duration signal could partly be ageing. Subject ages would allow an age-matched analysis.</td></tr>
<tr><td><strong>Combined foot channels hide left/right asymmetry.</strong></td>
<td>Asymmetry is an established PD marker, and it is structurally invisible after the two foot radars are merged into one channel. The original per-sensor recordings would make it measurable.</td></tr>
<tr><td><strong>58 subjects.</strong></td>
<td>With 58 people, every score carries a wide uncertainty range, so a small real effect can neither be proven nor ruled out at this size. Only a much larger group, in the hundreds, can settle small differences.</td></tr>
<tr><td><strong>No clinical information beyond the label.</strong></td>
<td>Each subject is only marked PD or control. Without disease stage, symptom severity or medication state, it is impossible to test whether the signal grows with severity, or how treatment changes it, both of which matter for a screening tool.</td></tr>
</tbody></table></div>
"""))

# ═══ 7 ═══
S.append(("Future work: the path to stronger results", f"""
{intro("The limitations above are not dead ends. Ordered from immediately actionable to long-term, this is how stronger and trustworthy results can be built on this line of work.")}

<div class="tblwrap"><table>
<thead><tr><th>Step</th><th>What it would deliver</th></tr></thead><tbody>
<tr><td><strong>1. Model the timing signature directly.</strong></td>
<td>Extract turn duration, sit-to-stand time and cadence from the spectrogram
as explicit quantities, instead of hoping a network discovers them in pixels.
Timing is the one signal this work showed to be real: patients take longer,
and the extra time concentrates in standing up and turning. A model built
directly on these timing quantities would be an automated version of the
timed-up-and-go test that clinicians already use, which is also what the
source radar system was validated to measure.</td></tr>
<tr><td><strong>2. Obtain subject ages.</strong></td>
<td>An age-matched analysis would settle whether the surviving signal is
disease or ageing, and could also unmask real signal that age variation
currently hides.</td></tr>
<tr><td><strong>3. Record a larger, session-balanced cohort.</strong></td>
<td>With 58 subjects every score carries a wide uncertainty range, so even a
genuinely better model could not prove itself here. Demonstrating a spectral
gait signature clearly above the duration floor needs a cohort in the low
hundreds, recorded in sessions balanced across the two groups.</td></tr>
<tr><td><strong>4. Keep the evaluation standard.</strong></td>
<td>Whatever the model, evaluate it on people it has never seen and report it
next to the duration floor and a confidence interval. The protocol built for
this thesis is pre-registered, ablation-tested and reusable as the design
document for every step above.</td></tr>
</tbody></table></div>

{key("<strong>The closing argument.</strong> This thesis set out to build a radar Parkinson&rsquo;s classifier and found something more useful: a demonstration, on validated clinical hardware, of exactly how such classifiers come to look better than they are, and a protocol that prevents it. Every step above inherits that protocol, which is what makes the path forward credible: whatever signal richer data holds, the standard defended here will measure it honestly, on people the model has never seen.")}
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
<title>The Discussion &middot; Radar micro-Doppler Parkinson's</title>
<style>{CSS}</style></head><body>
<div id="bar"></div>
<div class="wrap">
<header class="hero">
  <p class="kicker">Master's thesis &middot; Discussion chapter, drafted for review</p>
  <h1>The Discussion</h1>
  <p class="sub">What the results mean: the protocol experiment that reproduces
  the literature's numbers, the attention analysis that explains them, the
  intervention that tests them, the positive findings, the limitations, and
  the path to stronger results.</p>
  <div class="statlead">The argument in numbers:</div>
  <div class="stats">
    <div class="stat"><b>{cs_mean('recording_split','test','window','auc'):.2f}</b><span>with the literature's split<i>same model, same data</i></span></div>
    <div class="stat"><b>{cs_mean('collapse_test','never_seen','window','auc'):.2f}</b><span>on never-seen subjects<i>the collapse test</i></span></div>
    <div class="stat"><b>4-6&times;</b><span>attention on turn windows<i>vs steady walking</i></span></div>
    <div class="stat"><b>0.610</b><span>duration alone<i>the honest, useful signal</i></span></div>
  </div>
</header>
<nav class="toc"><h4>Contents</h4><ol>{toc}</ol></nav>
<main>{body}</main>
</div>
<button id="top" title="Back to top">&uarr;</button>
<script>{JS}</script>
</body></html>"""

out = ROOT / "DISCUSSION.html"
out.write_text(html)
print(f"wrote {out} ({len(html)/1e6:.2f} MB)")
print(f"sections={len(S)} figures={html.count('data:image/jpeg')}")
print("missing:", html.count("missing figure"), "emdash:", html.count(chr(8212)))
