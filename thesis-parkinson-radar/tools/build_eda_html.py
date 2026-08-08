#!/usr/bin/env python3
"""DATA_AND_EDA.html v3 — data insights only. ML content deferred to a later document.

Style rules enforced here (they carry through to the report):
  * no em-dashes in prose
  * key terms in <strong>
  * lists instead of long prose blocks
  * NO source-file names in the prose (figure labels are for our own navigation only)
  * nothing phrased as a question the author is asking himself
"""
import base64, io, json, pathlib
from PIL import Image

ROOT = pathlib.Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
FIG = ROOT / "outputs/figures"
TOOLS = ROOT / "tools"
N = json.load(open(TOOLS / "eda_numbers.json"))
FP = json.load(open(ROOT / "outputs/metrics/subject_fingerprint.json"))
EX = FP["examples"]
def exrow(k):
    e = EX[k]
    g = "Parkinson's" if str(e["group"]).lower().startswith("pd") else "control"
    return (f'<tr><td><strong>{e["id"]}</strong><br><span class="sm">{g}</span></td>'
            f'<td class="num">{e["within"]}</td><td class="num">{e["between"]}</td>'
            f'<td class="num"><strong>{e["times"]}&times;</strong></td>'
            f'<td>{e["caption"]}</td></tr>')
d1, d2 = N["dur_test1"], N["dur_test2"]
NT, NS, NF = N["n_trials"], N["n_subj"], N["n_feat"]
ns, nt, m = N["n_sig"], N["n_tested"], N["max_delta"]

def img(name, maxw=1500, q=82):
    p = FIG / name
    if not p.exists(): return f'<p style="color:#b00">[missing figure: {name}]</p>'
    im = Image.open(p).convert("RGB")
    if im.width > maxw: im = im.resize((maxw, round(im.height*maxw/im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" alt="{name}">'

def fig(name, caption, insight=None, maxw=1500):
    ins = f'<div class="insight"><span class="ilab">What it tells us</span><p>{insight}</p></div>' if insight else ""
    return f'<figure>{img(name,maxw)}<figcaption><span class="fname">{name}</span> {caption}</figcaption>{ins}</figure>'

def key(t):  return f'<aside class="key"><span class="klabel">Key point</span><p>{t}</p></aside>'
def warn(t): return f'<aside class="warn"><span class="klabel">Caution</span><p>{t}</p></aside>'
def defn(term, body): return f'<div class="defn"><span class="dterm">{term}</span><p>{body}</p></div>'
def defn2(term, body): return f'<div class="defn"><span class="dterm">{term}</span><div>{body}</div></div>'
def later(t): return f'<div class="later"><span class="ltag">Covered in the modelling chapter</span><p>{t}</p></div>'
def src(t):  return f'<span class="src">{t}</span>'   # inline source pointer, becomes a citation in the report

# ── source registry ─────────────────────────────────────────────────────────
# Every work cited anywhere in this document. Each becomes a bibliography entry
# in the written report; the short label is what appears inline.
REFS = {
 "paper": ("I. E. L&oacute;pez-Delgado, V. Navarro-L&oacute;pez, F. Grandas-P&eacute;rez, "
   "J. I. Godino-Llorente and J. Grajal, &ldquo;Radar Network for Gait Monitoring: "
   "Technology and Validation&rdquo;, <em>IEEE Transactions on Biomedical Engineering</em>, "
   "vol. 73, no. 1, pp. 393&ndash;403, 2026.",
   "https://doi.org/10.1109/TBME.2025.3583785"),
 "postuma": ("R. B. Postuma <em>et al.</em>, &ldquo;MDS Clinical Diagnostic Criteria for "
   "Parkinson&rsquo;s Disease&rdquo;, <em>Movement Disorders</em>, vol. 30, no. 12, "
   "pp. 1591&ndash;1601, 2015.",
   "https://doi.org/10.1002/mds.26424"),
 "mirelman": ("A. Mirelman <em>et al.</em>, &ldquo;Gait Impairments in Parkinson&rsquo;s "
   "Disease&rdquo;, <em>The Lancet Neurology</em>, vol. 18, no. 7, pp. 697&ndash;708, 2019.",
   "https://doi.org/10.1016/S1474-4422(19)30044-4"),
 "hausdorff": ("J. M. Hausdorff, &ldquo;Gait Dynamics in Parkinson&rsquo;s Disease: Common and "
   "Distinct Behavior Among Stride Length, Gait Variability, and Fractal-Like Scaling&rdquo;, "
   "<em>Chaos</em>, vol. 19, no. 2, art. 026113, 2009.",
   "https://doi.org/10.1063/1.3147408"),
 "mannwhitney": ("H. B. Mann and D. R. Whitney, &ldquo;On a Test of Whether One of Two Random "
   "Variables Is Stochastically Larger Than the Other&rdquo;, <em>The Annals of Mathematical "
   "Statistics</em>, vol. 18, no. 1, pp. 50&ndash;60, 1947.",
   "https://doi.org/10.1214/aoms/1177730491"),
 "cliff": ("N. Cliff, &ldquo;Dominance Statistics: Ordinal Analyses to Answer Ordinal "
   "Questions&rdquo;, <em>Psychological Bulletin</em>, vol. 114, no. 3, pp. 494&ndash;509, 1993.",
   "https://doi.org/10.1037/0033-2909.114.3.494"),
 "romano": ("J. Romano, J. D. Kromrey, J. Coraggio and J. Skowronek, &ldquo;Appropriate "
   "Statistics for Ordinal Level Data&rdquo;, Annual Meeting of the Florida Association of "
   "Institutional Research, 2006. Source of the negligible / small / medium / large thresholds "
   "used for the effect size.",
   "https://www.researchgate.net/publication/237544991"),
 "bh": ("Y. Benjamini and Y. Hochberg, &ldquo;Controlling the False Discovery Rate: A Practical "
   "and Powerful Approach to Multiple Testing&rdquo;, <em>Journal of the Royal Statistical "
   "Society: Series B</em>, vol. 57, no. 1, pp. 289&ndash;300, 1995.",
   "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x"),
}
LABEL = {"paper": "L&oacute;pez-Delgado et al. (2026)", "postuma": "Postuma et al. (2015)",
         "mirelman": "Mirelman et al. (2019)", "hausdorff": "Hausdorff (2009)",
         "mannwhitney": "Mann &amp; Whitney (1947)", "cliff": "Cliff (1993)",
         "romano": "Romano et al. (2006)", "bh": "Benjamini &amp; Hochberg (1995)"}

def ref(key, locator=""):
    """Inline citation. Becomes a \\cite with a page locator in the report."""
    txt = LABEL[key] + (", " + locator if locator else "")
    return (f'<span class="src"><a href="{REFS[key][1]}" target="_blank" '
            f'rel="noopener">{txt}</a></span>')

CSS = (TOOLS / "eda.css").read_text() + """
.dterm .src{font-weight:400;margin-left:7px;letter-spacing:0}
.sec h4{font-family:var(--sans);font-size:15px;font-weight:700;color:var(--ink);
 margin:26px 0 8px}
.nw{white-space:nowrap;font-family:var(--mono);font-size:.92em}
.wex .math{margin:9px 0 3px}
.wex .wex{margin:0}
.src a{color:inherit;text-decoration:none;border-bottom:1px dotted currentColor}
.src a:hover{color:var(--link);border-bottom-color:var(--link)}
.defn>div{margin:0}
.math{font-family:var(--mono);font-size:15px;line-height:1.9;background:var(--panel);
 border:1px solid var(--line);border-radius:8px;padding:13px 16px;margin:13px 0;
 text-align:center;overflow-x:auto}
.math .mnote{display:block;font-family:var(--sans);font-size:12.5px;color:var(--muted);
 margin-top:7px;font-style:italic}
.wex{border:1px dashed var(--line);border-radius:9px;padding:12px 16px;margin:14px 0}
.wlab{font-family:var(--sans);font-size:10.5px;font-weight:700;letter-spacing:.09em;
 text-transform:uppercase;color:var(--muted);display:block;margin-bottom:7px}
.wex p{margin:7px 0;font-size:14.5px;line-height:1.6;max-width:none}
table.pairs{min-width:0;width:auto;margin:10px auto;font-size:13px}
table.pairs td,table.pairs th{padding:6px 12px;text-align:center;white-space:nowrap}
table.pairs th{font-size:11px}
.hi{color:var(--key-line);font-weight:700}.lo{color:var(--warn);font-weight:700}
.chg{display:grid;grid-template-columns:34px 1fr;gap:14px;padding:15px 0;
 border-top:1px solid var(--line)}
.chg:first-of-type{border-top:none}
.cn{font-family:var(--sans);font-size:15px;font-weight:700;color:var(--muted);
 border:1.5px solid var(--line);border-radius:50%;width:32px;height:32px;
 display:flex;align-items:center;justify-content:center}
.chg p{margin:0 0 7px;font-size:15.5px;line-height:1.6;max-width:none}
.chg p:last-child{margin-bottom:0}
.cl{font-family:var(--sans);font-size:10.5px;font-weight:700;letter-spacing:.08em;
 text-transform:uppercase;color:var(--muted);margin-right:8px}
"""
S = []

# ══════════════════ 1 ══════════════════
S.append(("Where the data comes from", f"""
<p>Every measurement in this thesis comes from one acquisition campaign at the
<strong>Universidad Politécnica de Madrid</strong>, published as
<a href="https://doi.org/10.1109/TBME.2025.3583785" target="_blank" rel="noopener">López-Delgado
et al., &ldquo;Radar Network for Gait Monitoring: Technology and Validation&rdquo;,
<em>IEEE Transactions on Biomedical Engineering</em>, vol. 73, no. 1, pp.
393&ndash;403, 2026</a>. That study validates the measurement system against an
optical motion-capture reference. It stops at measuring gait parameters and
contains <strong>no disease classifier</strong>, which is the gap this thesis
addresses.</p>

<h3>How a radar measurement happens</h3>
<p>A radar is not a camera. It works in four steps, repeated 1600 times a second:</p>
<ol>
<li>The radar <strong>transmits a chirp</strong>, a brief burst whose frequency
sweeps upward. This sweeping is what the term FMCW (frequency-modulated
continuous-wave) refers to.</li>
<li>The chirp <strong>reflects off the body</strong> and returns a fraction of a
microsecond later.</li>
<li>The radar <strong>compares</strong> the returned chirp against the one it is
transmitting now. The difference between them encodes how far away the reflector
is and how fast it is moving.</li>
<li>Repeating this produces a continuous stream of measurements. Applying a
<strong>Fourier transform</strong> to short slices of that stream converts it into
the images we work with.</li>
</ol>

{fig("setup_and_signal_chain.png",
  "Above: the acquisition geometry, drawn from the published parameters. Below: the five steps that turn a transmitted chirp into the two pictures we analyse.",
  "The geometry explains <em>why</em> the data has <strong>two channels</strong>. Foot-aimed nodes sit at 0.15 m and the torso node at 1.0 m, so each captures a different body part travelling at a different speed. The chain below explains <em>where this thesis begins</em>: the first four steps were already performed by the team that built the system, and what reaches us is the output of the fifth.")}

{fig("paper_fig4_setup.png",
  "The acquisition environment as published: above, the corridor layout with the five phases of the test numbered in order; below, a photograph of the same room. " + ref("paper", "Fig. 4, p. 396"),
  "The two panels share one colour key, printed inside each. <strong>Blue is a torso node</strong> at 1 m, <strong>green is a foot node</strong> at 0.15 m, <strong>dark grey is the processing unit</strong>, and in the photograph <strong>pink marks a VICON motion-capture camera</strong>. There is one blue and one green node at <em>each</em> end of the walkway, so a subject is measured while walking away and again while walking back. The schematic also numbers the test itself: <strong>1 stand up, 2 walk, 3 turn around, 4 walk, 5 sit down</strong>, with the chair drawn at the right-hand end. The VICON cameras are the optical reference against which the radar was validated, and they are the reason this dataset can be trusted as a measurement.", 1200)}

{fig("paper_fig1_node.png",
  "One 24 GHz radar node, and the kind of spectrogram it produces. " + ref("paper", "Fig. 1(a), p. 395"),
  "Each node is a single small board carrying two antenna patches, one transmitting and one receiving, with the radio circuits on one PCB and the control circuits on another. The image on the monitor is a spectrogram of a walk, the same kind of image analysed throughout this document, which shows how directly the hardware output maps to what we model.", 1100)}

<h3>The parameters and what each one controls</h3>
<p>Every value in the table was <strong>set by the team that built the
system</strong> rather than chosen for this thesis, so they are inherited
constraints. They are reproduced here unchanged from the source publication
{ref("paper", "Section III, p. 396, and Table III")}.</p>

<div class="tblwrap"><table>
<thead><tr><th>Parameter</th><th>Value</th><th>What it controls</th></tr></thead><tbody>
<tr><td>Carrier frequency</td><td class="num">23 GHz</td>
<td>Fixes the <strong>wavelength</strong> at 1.303 cm, which is the conversion factor between frequency shift and speed.</td></tr>
<tr><td>Chirp bandwidth</td><td class="num">1.4 GHz</td>
<td><strong>Range resolution</strong>: how finely two reflectors at different distances can be told apart.</td></tr>
<tr><td>Chirp duration</td><td class="num">625 µs</td>
<td>One measurement every 625 µs is <strong>1600 per second</strong>. That caps the measurable frequency shift at half the rate, giving the <strong>±800 Hz</strong> span of our data.</td></tr>
<tr><td>Clutter filter</td><td class="num">10 Hz high-pass</td>
<td>Discards reflections from <strong>stationary objects</strong> such as walls and furniture.</td></tr>
<tr><td>Analysis window</td><td class="num">50 ms</td>
<td>The <strong>time against frequency trade-off</strong>. A 50 ms window can separate frequencies <strong>20 Hz</strong> apart.</td></tr>
<tr><td>Values stored per column</td><td class="num">320 over ±800 Hz</td>
<td>One value every 5 Hz.</td></tr>
<tr><td>Foot nodes / torso nodes</td><td class="num">2 / 1</td>
<td>The validated network layout, which is why we receive <strong>two channels</strong>.</td></tr>
<tr><td>Corridor / standoff</td><td class="num">3 m / 1 m</td>
<td>A short walk, so recordings are brief: median <strong>{N['dur']['50%']} s</strong>.</td></tr>
<tr><td>Beamwidth</td><td class="num">40°</td>
<td>The field of view, so it sets when a subject enters and leaves the measurement.</td></tr>
</tbody></table></div>

{defn("Doppler bin", "Each column of a spectrogram is a list of numbers, one per frequency slot. Each slot is a <strong>bin</strong>. Our files hold 320 bins spanning &minus;800 Hz to +800 Hz, so bins sit 5 Hz apart. The 50 ms analysis window can genuinely separate frequencies 20 Hz apart, so roughly <strong>four neighbouring bins describe one real measurement</strong>, in the same way enlarging a photograph adds pixels but no new detail. Nothing is wrong with this; it simply means the Doppler axis is smoother than it is precise.")}

<h3>The protocol: Timed-Up-and-Go</h3>
<p>Each subject performed the <strong>TUG</strong> test, a standard clinical
mobility assessment: stand up from a chair, walk 3 m, turn, walk back, sit down.
Two variants were recorded, three times each.</p>

{fig("explain_tug.png",
  "The phases of each variant, with the measured median duration for each group.",
  "This single protocol difference is the most consequential fact in the dataset. <strong>test1 includes the chair</strong>, and rising from a chair is one of the hardest movements for someone with Parkinson's, so the extra seconds accumulate there: PD recordings run <strong>+{p1} % longer</strong>. <strong>test2 removes the chair</strong> and the gap falls to <strong>+{p2} %</strong>, small enough to be ordinary variation between people. The comparison was made with a <strong>rank-based test</strong>, which simply asks how often a PD recording is longer than a control recording chosen at random. The resulting probability value is small for test1 and large for test2, meaning the test1 gap is unlikely to be coincidence while the test2 gap is entirely consistent with coincidence.".format(p1=d1['pct'], p2=d2['pct']))}

<h3>The cohort</h3>
<div class="tblwrap"><table>
<thead><tr><th>Group</th><th>Subjects</th><th>test1 recordings</th><th>test2 recordings</th></tr></thead><tbody>
<tr><td>Healthy control</td><td class="num">{N['n_ctrl']}</td><td class="num">99</td><td class="num">99</td></tr>
<tr><td>Parkinson's disease</td><td class="num">{N['n_pd']}</td><td class="num">76</td><td class="num">74</td></tr>
<tr><td><strong>Total</strong></td><td class="num"><strong>{NS}</strong></td><td class="num" colspan="2"><strong>{NT} recordings</strong></td></tr>
</tbody></table></div>
<p>The intended design is 2 variants × 3 repeats = 6 recordings per subject,
achieved by 56 of the 58. One subject has a seventh recording and one is missing
one; both are examined in section 4.</p>

{key("<strong>58 subjects is the defining constraint of this project.</strong> It shapes every later decision: small models rather than large ones, reusing networks pretrained elsewhere rather than training from scratch, and validation that holds out one person at a time rather than a fixed test set.")}
"""))

# ══════════════════ 2 ══════════════════
S.append(("From a radio wave to a picture", f"""
<p>The images we analyse are built from radio measurements. Understanding that
conversion is what makes everything afterwards interpretable.</p>

<h3>The Doppler effect</h3>
<p>The familiar version is an ambulance siren: the pitch rises as it approaches and
drops as it recedes, because the motion compresses or stretches the sound waves.
Radar applies the same physics to radio waves. A moving body part shifts the
frequency of the wave that bounces off it, and the size of that shift tells us
<strong>how fast that part was moving</strong>.</p>

{fig("explain_doppler.png",
  "A: motion squeezes the returning wave. B: converting a frequency shift into a speed. C: known walking speeds place the two analysis bands.",
  "A worked example: a shift of 153 Hz corresponds to <strong>1.0 m/s</strong>, which is ordinary walking pace. This conversion is exact and fixed by the radar's wavelength, so the vertical axis of every spectrogram can be read directly as a speed axis.")}

<h3>The vertical axis is a speed axis. Speed of what, precisely</h3>
<p>At any instant the radar receives reflections from <strong>every moving part of
the body at once</strong>: both feet, both shins, the thighs, the trunk, the arms.
Each part moves at its own speed, so each produces its own frequency shift, and the
Fourier transform separates them.</p>
<p>A single vertical column of a spectrogram is therefore a
<strong>distribution of speeds present at that moment</strong>. Brightness at
height <em>f</em> means "this much of the reflected energy came from parts of the
body moving at the speed corresponding to <em>f</em>". It is not the walking speed
of the person; it is <strong>how the body's motion is spread across speeds</strong>.</p>

{key("This is why a spectrogram of walking has rich internal structure rather than being one line. The torso contributes a steady band near 150 Hz throughout the recording, while the feet sweep from 0 Hz up towards 500 Hz and back with every step.")}

<h3>Where the two band boundaries come from</h3>
<p>The boundaries are inherited from the source publication, which places them
using <strong>established walking speeds</strong> rather than fitting them to any
particular dataset {ref("paper", "Section III, pp. 396&ndash;398")}:</p>
<ul>
<li>A trunk walking at roughly 1 m/s produces about <strong>153 Hz</strong>.
Allowing headroom for faster walkers gives a <strong>torso band below 200 Hz</strong>.</li>
<li>A foot at peak swing reaches roughly 3.3 m/s, which is about
<strong>506 Hz</strong>, giving a <strong>foot band from 200 to 500 Hz</strong>.</li>
</ul>
<p>Our own recordings broadly support this split. Averaged over a sample of
trials, the torso channel places <strong>52 % of its energy below 200 Hz</strong>,
while the foot channel is shifted upward as expected.</p>

{warn("The 500 Hz upper edge is a <strong>convention, not a physical limit</strong>. Measured across a sample of our recordings, roughly <strong>21 % of the foot channel&rsquo;s energy sits above it</strong>. In other words, about a fifth of the fast motion the radar actually captured falls outside the window we chose to call the foot band, so any measurement of that band <strong>under-reports how much fast movement occurred</strong>. That matters in one concrete way: if the disease reduces the very fastest foot motion, the effect would be partly invisible to a band that stops at 500 Hz. Widening the upper edge is therefore worth testing, and it is the reason the boundary is presented here as a choice rather than as a fact about the body.")}

<h3>What is inside one recording</h3>
{fig("explain_matfile.png",
  "The contents of one recording file and how the arrays relate to one another.",
  "Two practical consequences. The arrays arrive already oriented as speed against time, so no rearranging is needed. And only the enhanced spectrograms and the two axes are read in normal operation, which is why loading one recording takes about <strong>0.12 s</strong> and the full 106 GB is never read.")}

{defn("Contrast enhancement", "The spectrograms in the dataset have had their brightness rescaled before being saved. The reason is simple: the trunk reflects far more energy than a foot does, so on a raw image the trunk would appear bright and the feet would be almost invisible. The rescaling brings both into view at once, much like the HDR setting on a phone camera brightening shadows so you can see detail in them. Only these rescaled versions are used in the main analysis.")}

"""))

# ══════════════════ 3 ══════════════════
S.append(("Reading a spectrogram", f"""
<p>Before any statistics, it is worth being able to look at one recording and
describe what physically happened. Three things are directly readable.</p>

{fig("guide_data_explained.png",
  "One control subject and one PD subject, each seen by both radars. The two recordings are matched for duration (8.0 s against 8.1 s) so that length is not the visible difference. White dashed lines mark the torso band at ±200 Hz, orange marks the foot band from 200 to 500 Hz.", None, 1600)}

{key("An important detail: the <strong>two left panels are the same recording of the same control subject</strong>, captured simultaneously by the foot-aimed and torso-aimed radars. The same applies to the two right panels, which show one PD subject. These are not different people and not different moments, but <strong>two simultaneous views of a single walk</strong>.")}

<h3>1. The turn is visible, and it divides every recording in two</h3>
<p>In all four panels the energy sits <strong>below</strong> zero for the first half
and <strong>above</strong> zero for the second. This is where the sign of the axis
matters:</p>
<ul>
<li><strong>Negative</strong> means moving <strong>away</strong> from the radar.</li>
<li><strong>Positive</strong> means moving <strong>towards</strong> it.</li>
</ul>
<p>The sign change therefore marks the exact instant the subject
<strong>turned around</strong>: about 6.2 s for the control and 7.6 s for the PD
subject. Every recording contains two walking passes separated by one turn. This
matters clinically, because turning is the slowest and least stable part of the
walk and difficulty turning is characteristic of Parkinson's.</p>

<h3>2. The repeating arcs are individual steps</h3>
<p>In the foot panels the bright region is not smooth; it forms repeating arches.
Each arch is one <strong>foot swing</strong>: the foot accelerates from rest, peaks
mid-swing, then slows to a stop at heel strike. The
<strong>valleys between arches are the moments the foot is flat on the
ground</strong>, moving at zero speed. Steps can be counted directly off the image,
and the spacing between arches is the step rhythm.</p>

{fig("annotated_ce_foot.png", "A single foot-channel spectrogram with its physical structure annotated.", None, 1400)}

<h3>3. The two channels capture different physics</h3>
{fig("guide_foot_vs_torso.png",
  "The same walk seen by the foot-aimed and torso-aimed nodes.",
  "The foot channel is <strong>spiky and periodic</strong>, dominated by fast 200 to 500 Hz swings. The torso channel is <strong>one smooth band</strong> near 0 to 200 Hz. They are not two views of the same thing.")}

<h3>Why the pairing of the two channels carries information</h3>
<p>In healthy walking the two channels are <strong>locked to each other in
time</strong>. The clearest way to see it is to follow the single most important
instant in a stride, the <strong>heel strike</strong>, the moment one foot lands.
The source publication marks that instant in two independent ways, one in each
channel {ref("paper", "Section III-B, p. 397")}:</p>
<div class="tblwrap"><table>
<thead><tr><th>Moment in the stride</th><th>Foot channel shows</th><th>Torso channel shows</th></tr></thead><tbody>
<tr><td>Mid-swing</td><td>a bright peak high on the axis, near 500 Hz</td><td>the trunk gliding forward at its steady pace</td></tr>
<tr><td><strong>Heel strike</strong></td><td><strong>a local minimum</strong>: a foot on the ground is momentarily still</td><td><strong>a local peak</strong>: the trunk is at its fastest as the weight transfers</td></tr>
<tr><td>Push-off</td><td>climbs again as the other foot swings</td><td>eases back down from that peak</td></tr>
</tbody></table></div>
<p>In healthy walkers those two markers <strong>land at the same instant</strong>,
stride after stride, so either channel alone identifies the heel strike. The same
publication then reports the finding that matters here: that agreement
<strong>is lost in participants with Parkinson's disease</strong>, so the trunk no
longer tracks each heel strike reliably, and the recommendation is to read the
event from the feet instead {ref("paper", "Section III-B and Fig. 9, p. 398")}.</p>
{key("The information is therefore not in either channel alone, but in <strong>whether the two rise and fall together</strong>. A measurement taken from the foot channel by itself, or the torso channel by itself, cannot express that alignment. Any method intended to capture it has to see both channels at the same time.")}

{fig("control_vs_pd_ce_foot.png", "Control and PD foot channels side by side.", None, 1400)}

{warn("In the comparison above the PD example has a visibly noisier background, and it is tempting to read that as a sign of the disease. It may instead reflect <strong>which processing run the file came from</strong>. The dataset was not produced in a single pass: it carries the traces of two separate runs with slightly different noise characteristics, and that difference is taken up in the modelling chapter. This is exactly the kind of obvious-looking visual difference that turns out to be a recording artifact, which is why no conclusion here rests on comparing a few images by eye.")}
"""))

# ══════════════════ 4 ══════════════════
S.append(("Quality control: how every recording was checked", f"""
<p>All {NT} recordings were checked automatically before any analysis, so that a
damaged file could not quietly become a finding. The checks run in two stages,
cheapest first.</p>

<h3>Stage 1: read only the file index and the two small axes</h3>
<p>This stage reads the file's table of contents <strong>without loading the large
arrays</strong>, so it costs milliseconds per recording.</p>
<div class="tblwrap"><table>
<thead><tr><th>Check</th><th>What it would catch</th></tr></thead><tbody>
<tr><td>All six expected variables present</td><td>A truncated or partly written file</td></tr>
<tr><td>All four spectrogram arrays share one shape</td><td>A file assembled from mismatched pieces</td></tr>
<tr><td>Speed axis spans the expected range, with the expected spacing</td><td>A recording processed with different settings</td></tr>
<tr><td>Duration between 5 s and 180 s</td><td>An aborted recording, or one left running</td></tr>
</tbody></table></div>

<h3>Stage 2: load the two enhanced spectrograms</h3>
<div class="tblwrap"><table>
<thead><tr><th>Check</th><th>What it would catch</th></tr></thead><tbody>
<tr><td>Every value is a finite number</td><td>Numerical failure somewhere upstream</td></tr>
<tr><td>The image has real contrast, not one flat value</td><td>A recording where the sensor captured nothing</td></tr>
</tbody></table></div>
<p>Each recording produces one row of pass or fail flags in a quality-control
table, so any exclusion would be traceable rather than silent.</p>

{key("<strong>All {n} recordings pass every check.</strong> There are no invalid values, no blank spectrograms, and the speed axis is consistent across the dataset to within a tiny fraction of one bin. Nothing was excluded. The dataset is unusually clean, which matters because it means the modest results reported later <strong>cannot be attributed to damaged inputs</strong>.".format(n=NT))}

<h3>The two irregular subjects, examined rather than assumed</h3>
{fig("11_fisp022_trial4.png",
  "Similarity between every pair of recordings for the subject with a seventh recording.",
  "The concern was that the extra file might be an accidental copy of another. It is not. No pair reaches the similarity threshold that would indicate duplication, and the extra recording resembles the others no more closely than they resemble each other. <strong>It is a genuine repeat attempt</strong>, so it was kept.")}
{fig("11_fisp048_test2.png",
  "The available recordings for the subject missing one repeat.",
  "The concern was a partially written file. The five remaining recordings are internally consistent with no sign of truncation, so <strong>the missing recording was a clean deletion</strong> and the subject was kept with five.")}
"""))

# ══════════════════ 5 ══════════════════
S.append(("Turning each recording into numbers", f"""
<p>A spectrogram is a large grid of roughly 320 × 1000 values. Statistical
comparison between groups needs a <strong>short, fixed list of numbers per
recording</strong> instead, so each recording is summarised.</p>

<p>For one recording the procedure is:</p>
<ol>
<li>Take the <strong>foot</strong> spectrogram and compute a set of descriptive
numbers over that entire grid: its average brightness, its peak, its spread, and
so on.</li>
<li>Repeat the identical calculation on the <strong>torso</strong> spectrogram.</li>
<li>Add two ratios that compare the channels against each other.</li>
</ol>
<p>The result is <strong>one row of {NF} numbers per recording</strong>, giving a
table of <strong>{NT} rows</strong>. Each number describes the whole of one
recording: the average brightness of a recording is the average over all of its
pixels, not an average across recordings or across subjects.</p>

{later("How these numbers are then used, which models consume them and how those models are validated, belongs to the modelling work and is presented in the modelling chapter. This section covers only <strong>what is measured and what each measurement means</strong>.")}

<div class="tblwrap"><table>
<thead><tr><th>Measurement</th><th>What it captures physically</th><th>Grows with recording length?</th></tr></thead><tbody>
<tr><td>Average brightness</td><td>Overall reflected energy density</td><td><span class="pill ok">no</span></td></tr>
<tr><td>Peak brightness</td><td>The single strongest moment of motion</td><td><span class="pill ok">no</span></td></tr>
<tr><td>Spread of brightness</td><td>How much the signal varies</td><td><span class="pill ok">no</span></td></tr>
<tr><td><strong>Spectral centroid</strong></td><td><strong>The body's average speed</strong>, the balance point of the energy on the speed axis</td><td><span class="pill ok">no</span></td></tr>
<tr><td><strong>Spectral bandwidth</strong></td><td><strong>The range of speeds present</strong>: is the motion uniform or varied</td><td><span class="pill ok">no</span></td></tr>
<tr><td>Spectral entropy</td><td>How evenly the energy is spread across speeds</td><td><span class="pill ok">no</span></td></tr>
<tr><td>Band averages</td><td>Mean energy inside the torso band or the foot band</td><td><span class="pill ok">no</span></td></tr>
<tr><td><strong>Foot-to-torso ratio</strong></td><td><strong>What share of the motion is fast (feet) rather than slow (trunk)</strong></td><td><span class="pill ok">no</span></td></tr>
<tr><td><strong>Rhythmicity</strong></td><td><strong>How regular the stride timing is</strong>, measured relative to the average</td><td><span class="pill ok">no</span></td></tr>
<tr><td>Band and total energy sums</td><td>Energy added up over the whole recording</td><td><span class="pill bad">YES</span></td></tr>
<tr><td>Absolute envelope variability</td><td>Variation in energy over time, unscaled</td><td><span class="pill bad">YES</span></td></tr>
</tbody></table></div>

{defn("Why the last two rows are a problem", "An <strong>average</strong> does not change much when a recording is longer, because averaging over 500 or 1000 columns gives a comparable answer. A <strong>sum</strong> does: double the recording and the sum roughly doubles. Since PD subjects walk for longer, any summed measurement partly encodes <em>how long the walk lasted</em> rather than <em>how the person walked</em>. Those measurements are retained deliberately so the effect can be demonstrated and quantified, and they are kept separate from the measurements used to describe gait.")}

<h3>Three measurements chosen for clinical reasons</h3>
<p>Parkinson's disease produces specific, named motor symptoms. Three of the
measurements above were selected because each should respond to one of them, and
the expected direction of the response can be stated in advance.</p>
<p>The symptoms themselves are <strong>not our own observations</strong>. Each is
taken from the clinical literature and carries a source in the table below:
<strong>bradykinesia</strong> is the one motor sign that the international diagnostic
criteria require to be present before a diagnosis can be made at all
{ref("postuma")}, while <strong>shorter steps with reduced foot
clearance</strong> and <strong>increased stride-to-stride variability</strong> are
the two gait changes most consistently reported in reviews of the condition
{ref("mirelman")} {ref("hausdorff")}. What this thesis adds is only the third
column: how each symptom should show up in a radar measurement.</p>
<div class="tblwrap"><table>
<thead><tr><th>Symptom, and where it is established</th><th>Effect on the walk</th><th>Expected effect on the measurement</th></tr></thead><tbody>
<tr><td><strong>Bradykinesia</strong><br><span class="sm">slowness of movement</span><br>{ref("postuma")}</td>
<td>Every body part moves more slowly, so all frequency shifts are smaller.</td>
<td>The <strong>spectral centroid falls</strong>, because the energy sits lower on the speed axis.</td></tr>
<tr><td><strong>Reduced foot clearance</strong><br><span class="sm">shuffling, shorter steps</span><br>{ref("mirelman")}</td>
<td>The foot travels less far and less quickly, so less energy reaches the fast band.</td>
<td>The <strong>foot-to-torso ratio falls</strong>, because the fast share of the motion shrinks.</td></tr>
<tr><td><strong>Gait variability</strong><br><span class="sm">irregular stride to stride</span><br>{ref("hausdorff")}</td>
<td>Steps become less evenly timed, so energy over time is less regular.</td>
<td><strong>Rhythmicity rises</strong>, because variation relative to the average increases.</td></tr>
</tbody></table></div>

{key("Committing to those directions beforehand turns each measurement into a <strong>testable prediction</strong> rather than an observation to be explained afterwards. If the centroid rises when the disease predicts a fall, the difference cannot be the symptom it was chosen to detect, and something else is driving it. Predicting the direction in advance is what makes that distinction possible.")}
"""))

# ══════════════════ 6 ══════════════════
S.append(("What the exploratory analysis found", f"""
<h3>How the two groups are compared, and how the numbers are worked out</h3>
<p>Every comparison in this section is done on <strong>one measurement at a
time</strong>. Take one measurement, say the spectral centroid. Read its value off
every recording, so there is <strong>one number per recording</strong>, and label
each number control or PD. Everything below operates on those two lists of
numbers.</p>
<p>All three quantities come from the <strong>same single act of counting</strong>:
compare every PD number against every control number, one pair at a time, and
count how often the PD one is larger. A worked example with seven numbers runs
through all three, so the arithmetic can be followed by hand.</p>

{defn2("The example we will use throughout", '''
<p>Suppose one measurement gives these values:</p>
<div class="math">PD = { 9, 12, 15 }&nbsp;&nbsp;&nbsp;&nbsp;Control = { 7, 8, 11, 14 }
<span class="mnote">3 PD numbers and 4 control numbers, so 3 &times; 4 = 12 pairs to check</span></div>
<p>Every pair is checked once. A tick means the PD number was the larger one:</p>
<div class="tblwrap"><table class="pairs">
<thead><tr><th></th><th>vs 7</th><th>vs 8</th><th>vs 11</th><th>vs 14</th><th>PD wins</th></tr></thead>
<tbody>
<tr><td><strong>PD 9</strong></td><td class="hi">&#10003;</td><td class="hi">&#10003;</td><td class="lo">&#10007;</td><td class="lo">&#10007;</td><td><strong>2</strong></td></tr>
<tr><td><strong>PD 12</strong></td><td class="hi">&#10003;</td><td class="hi">&#10003;</td><td class="hi">&#10003;</td><td class="lo">&#10007;</td><td><strong>3</strong></td></tr>
<tr><td><strong>PD 15</strong></td><td class="hi">&#10003;</td><td class="hi">&#10003;</td><td class="hi">&#10003;</td><td class="hi">&#10003;</td><td><strong>4</strong></td></tr>
</tbody></table></div>
<p><strong>PD was larger in 9 pairs out of 12</strong> and smaller in the other 3.
That single count is all three numbers below.</p>
''')}

{defn2("1. The comparison itself, called the Mann-Whitney U test " + ref("mannwhitney"), '''
<p><em>U</em> is simply that count of wins:</p>
<div class="math">U = number of pairs where the PD value is larger&nbsp; = &nbsp;<strong>9</strong>
<span class="mnote">ties, which are rare here, count as half a win each</span></div>
<p>The useful part is knowing what <em>U</em> would be if the two groups were
really the same. Then PD would win about half the pairs:</p>
<div class="math">U expected if no difference &nbsp;=&nbsp; (n<sub>PD</sub> &times; n<sub>ctrl</sub>) / 2 &nbsp;=&nbsp; 12 / 2 &nbsp;=&nbsp; <strong>6</strong></div>
<p>We got 9 where 6 was expected, so PD tends to sit higher. Counting wins rather
than comparing averages matters here because our measurements are lopsided, with a
few very large values, and one extreme recording can drag an average around while
it can only ever win its own pairs.</p>
''')}

{defn2("2. How big the gap is, called Cliff's delta " + ref("cliff"), '''
<p>The same count, rescaled so it does not depend on how many recordings there are:</p>
<div class="math">&delta; = (wins &minus; losses) / total pairs = (9 &minus; 3) / 12 = <strong>0.50</strong>
<span class="mnote">equivalently &delta; = 2U / (n<sub>PD</sub> &times; n<sub>ctrl</sub>) &minus; 1</span></div>
<p><strong>What the number means.</strong> &delta; runs from &minus;1 to +1.
Zero means the two groups are interchangeable, so picking one of each at random is
a coin flip. A value of 0.50 means PD wins 75 % of pairs and loses 25 %. Getting
all the way to 1 would need <em>every</em> PD value to beat <em>every</em> control
value, with no overlap at all.</p>
<p>The size labels printed on the plots use the standard cut-offs of <strong>0.147,
0.33 and 0.474</strong> for small, medium and large ''' + ref("romano") + '''. So
the biggest value anywhere in our data, ''' + str(m) + ''', sits just under the
line for large, and a typical value of about 0.28 counts as <strong>small to
moderate</strong>: a real tendency, but with the two groups heavily overlapping.</p>
''')}

{defn2("3. How surprising the gap is, the probability value (p)", '''
<p>The count of wins would still wander a little above and below 6 purely by luck,
even with two identical groups. The probability value asks how often luck alone
would produce a count <strong>at least as lopsided as ours</strong>.</p>
<div class="math">p = chance of a count this far from the expected 6, if the groups were identical</div>
<p><strong>What the number means.</strong> Small means hard to explain by luck.
Our test1 duration comparison gives <strong>p = 0.0016</strong>, so a gap that
large would turn up by coincidence about <strong>once in 600 attempts</strong>.
Our test2 comparison gives <strong>p = 0.14</strong>, roughly one attempt in seven,
which is ordinary enough that luck remains a perfectly good explanation.</p>
<p><strong>What it does not mean.</strong> It says nothing about size. With enough
recordings, even a difference far too small to matter will produce a tiny
<em>p</em>. That is exactly why &delta; is always reported next to it.</p>
''')}

{defn2("4. Correcting for testing 25 things at once (BH-FDR) " + ref("bh"), '''
<p>Running the comparison on 25 measurements creates a problem. Even with nothing
real going on, about one test in twenty looks convincing by luck, so
<strong>25 tests buy you roughly one false alarm for free</strong>. The
Benjamini-Hochberg correction fixes this by sorting the 25 probability values
smallest to largest and inflating each one according to where it lands in that
order:</p>
<div class="math">p<sub>fdr</sub> = p &times; 25 / (its rank in the sorted list)
<span class="mnote">then adjusted so the corrected values never decrease down the list</span></div>
<div class="wex"><span class="wlab">Worked through</span>
<p>Take a measurement whose raw <em>p</em> is <strong>0.030</strong>, and follow
what happens to it in two different situations.</p>
<p><strong>If it is the 2nd smallest of the 25:</strong> 0.030 &times; 25 / 2 =
<strong>0.375</strong>. It fails. Only two of the 25 tests looked convincing, which
is roughly what luck alone hands out, so neither can be trusted.</p>
<p><strong>If it is the 24th smallest:</strong> 0.030 &times; 25 / 24 =
<strong>0.031</strong>. It passes. Twenty-four tests looked convincing, far more
than luck produces, so they are unlikely to be false alarms.</p>
<p>Same raw number, opposite verdict. The correction judges each result
<strong>in the company of the other 24</strong>, which is exactly the behaviour
needed when 25 things are tested at once.</p></div>
<p>The stricter alternative, which simply multiplies every value by 25, was
rejected because it assumes the 25 measurements are unrelated. Ours are heavily
related to one another, so that method would be so cautious that real differences
would be thrown away.</p>
''')}

{defn("Terms you will see written on the plots", "The figures in this section were produced with the standard names for these methods printed on their axes, so they are worth matching to the descriptions above. <strong>MWU</strong> is the Mann-Whitney U test. <strong>p</strong> or <strong>p_fdr</strong> is the probability value, before and after the correction. <strong>delta</strong> is Cliff&rsquo;s delta. <strong>BH-FDR</strong> is the correction itself. And <strong>trial level</strong> means each recording was counted as one observation, which is discussed at the end of this section.")}

{key("A large dataset can make a <strong>tiny</strong> difference look statistically convincing. That is why every comparison in this section reports the effect size alongside the probability value. The probability tells us the difference is probably real; the effect size tells us whether it is big enough to matter.")}

<h3>What the exploratory analysis changed</h3>
<p>Three things were planned one way at the start and had to be done differently
once the data had been looked at.</p>

<div class="chg"><span class="cn">1</span><div>
<p><span class="cl">Planned</span>Measure how accurately a <strong>model</strong>
can separate Parkinson's from healthy control using these recordings.</p>
<p><span class="cl">Found</span>PD subjects <strong>take longer</strong> to
complete the test.</p>
<p><span class="cl">Revised</span>A model responding only to elapsed time would
already score well, having learned nothing about gait. The objective became
measuring <strong>how much separation survives once duration is removed</strong>.</p>
</div></div>

<div class="chg"><span class="cn">2</span><div>
<p><span class="cl">Planned</span>Split the recordings at random into a set to
train on and a set to test on, as is standard.</p>
<p><span class="cl">Found</span>Two recordings of the <strong>same subject</strong>
are far more alike than recordings of two different subjects.</p>
<p><span class="cl">Revised</span>A model could score well simply by
<strong>recognising the individual</strong> and recalling the label attached to
them. Splitting is therefore done <strong>by subject</strong>, never by recording.</p>
</div></div>

<div class="chg"><span class="cn">3</span><div>
<p><span class="cl">Planned</span>Decide by inspection which measurements grow
purely because a recording is longer, and exclude them.</p>
<p><span class="cl">Found</span>Those judgements were <strong>wrong in both
directions</strong>. Several measurements assumed to be safe track recording length
closely.</p>
<p><span class="cl">Revised</span>Membership of the length-independent set is now
<strong>decided by testing each measurement against the data</strong> rather than
by assumption.</p>
</div></div>

<p>The third one has a reason worth knowing. A longer recording is
<strong>not the same walk played slowly</strong>. It has more standing still, more
hesitating and more turning in it, and less plain walking. The mixture inside it is
different. So even an average shifts when a recording is longer, and averages were
supposed to be the safe kind.</p>

{key("This is why the exploratory analysis is presented as a result rather than as preparatory work. It did not simply describe the data. It <strong>changed what question could honestly be asked, how any answer must be checked, and which measurements are trustworthy</strong>. That set of decisions is the methodological contribution of this thesis.")}

<h3>Cohort composition</h3>
{fig("eda_class_balance.png", "Subjects per group, and how many recordings each subject contributed.",
  "The imbalance is mild: {c} control against {p} PD, roughly 57 % to 43 %. It is small enough not to require rebalancing the data, but large enough that plain accuracy would be misleading, since always guessing the larger group would score 57 % while being useless.".format(c=N['n_ctrl'], p=N['n_pd']))}

<h3>Finding 1: PD subjects take longer, and the chair is why</h3>
<p>To recall what is being compared: <strong>test1 is the variant that includes
standing up from a chair</strong>, and <strong>test2 is the same walk without the
chair</strong>.</p>
{fig("eda_duration_confound.png", "Distribution of recording durations for each group, split by variant. Dashed lines mark each group's median.",
  "The difference is <strong>concentrated in the chair variant</strong>. In test1 the control median is {c1} s against {p1} s for PD, a gap of <strong>+{pc1} %</strong>, unlikely to be coincidence and of small-to-moderate size (effect size {dd1}). In test2 the gap falls to <strong>+{pc2} %</strong> with an effect size of {dd2}, which is negligible and entirely consistent with ordinary variation. Removing the chair removes most of the difference.".format(
    c1=d1['ctrl_med'], p1=d1['pd_med'], pc1=d1['pct'], dd1=d1['delta'], pc2=d2['pct'], dd2=d2['delta']))}
{fig("11_duration_by_test.png", "The same comparison repeated independently, using a separate analysis path.",
  "This is a <strong>replication check</strong> rather than a repeat of the same calculation. It was produced by a different analysis written independently, and it reaches the same conclusion: the group difference is concentrated in the chair variant and largely absent without it. Two independent routes agreeing is what allows this to be stated as established.")}

{key("This finding is why <strong>test2 is treated as the cleaner comparison</strong> throughout the rest of the work. In test2 the two groups take almost the same time, so any difference found there cannot be explained by one group simply spending longer in front of the radar.")}

<h3>Finding 2: almost everything looks convincing, and that is the problem</h3>
<p>Comparing the two groups across all 25 measurements gives a result that looks
excellent at first glance: <strong>{ns} of the {nt} show a difference unlikely to be
coincidence</strong>. It is in fact a warning sign, and the reason is easiest to see
through an analogy.</p>

<p>Imagine measuring 25 different things about two groups of cars, and finding that
<strong>24 of them differ</strong>: engine noise, fuel use, tyre wear, brake
temperature, and so on. There are two possible readings. Either the two groups of
cars differ in 24 independent ways, or <strong>one group was simply driven
faster</strong>, and every one of those 24 measurements is partly reporting speed.
The second explanation is far simpler, and it predicts exactly the pattern
observed.</p>

<p>The same logic applies here. It is not plausible that a walk differs between the
groups in 24 independent ways. It is very plausible that <strong>one shared factor
differs</strong> and shows up in nearly every measurement. Finding 1 already
identified that factor: PD recordings last longer.</p>

{key("The lesson is that <strong>a large number of significant results is not strong evidence, it is a hint that the measurements are not independent</strong>. Had only two or three measurements differed, that would have been more convincing, because it would suggest something specific rather than something pervasive.")}

{fig("eda_effect_sizes.png", "One horizontal bar per measurement, 25 in all. The length of a bar is Cliff's delta for that measurement: how far apart the two groups sit. Bars are sorted longest first, and the dotted vertical guides mark the negligible, small and medium boundaries.",
  "<strong>What the colours mean.</strong> A bar is <span style='color:#c0392b;font-weight:700'>red</span> if that measurement was flagged in section 5 as one that grows simply because a recording is longer, which is the summed quantities. It is <span style='color:#2e86c1;font-weight:700'>blue</span> otherwise. The names down the side are the measurements defined in section 5.<br><br><strong>What we see.</strong> Two things. First, ranked by size, the <strong>five longest bars are all red</strong>, and the first blue one comes sixth at 0.41. Second, <strong>not one bar reaches the line marked large</strong>; the longest of them all is {m}, and it stops just short.<br><br><strong>What that means for us.</strong> The biggest differences between the two groups are coming from measurements that partly report <em>how long the walk took</em>, not <em>how the person walked</em>. Once those are set aside, what remains is modest. So there is no single number in this list that separates the groups on its own, and any method built on them has to combine several of them to get anywhere.".format(m=m))}

{fig("eda_top_feature_distributions.png", "One panel per measurement, showing the strongest of the measurements that do not grow with recording length. Inside each panel there are two shapes, one per group. The shape shows how the recordings are spread across values: it is wide where many recordings share a value and narrow where few do.",
  "<strong>What we see.</strong> In every panel the two shapes <strong>sit almost on top of one another</strong>. Their centres are slightly offset, which is the difference the statistics picked up, but the bodies of the two shapes cover nearly the same range of values.<br><br><strong>What that means for us.</strong> This is the thing a probability value cannot tell you. A small probability says the two groups differ <em>on average</em>. It does not say whether <strong>one unknown person</strong> could be placed in the right group, and that is the only question a diagnostic tool actually has to answer. These pictures answer it, and the answer is that for most individuals these measurements would barely help: the value you would read off a PD subject is a value plenty of control subjects also produce.<br><br>This is worth stating plainly because it sets expectations for everything that follows. The modelling results reported later are modest, and <strong>this overlap is why</strong>. It was visible in the data before any model was built, so it is a property of the problem rather than a failure of a particular method.")}

<h3>Finding 3: a walk identifies the walker</h3>

<p>This one has nothing to do with metres. Everybody walks the same 3 m corridor.
The word <strong>distance</strong> is used here to mean something else:
<strong>how unalike two recordings are</strong>.</p>

<p>Working it out is short. Every recording is already a list of measurements from
section 5, so two recordings can be compared by seeing how far apart their two lists
are, taking all the measurements together at once. Identical recordings give zero,
and the more the measurements disagree the larger the number. Only the measurements
that do not grow with recording length go in, {FP['n_features']} of them, so what is being compared
is the <strong>shape of a walk</strong> rather than how long it took.</p>

<p>That gives each person two numbers, and those two numbers are what the figure
plots:</p>
<ul>
<li><strong>Within</strong>: how far apart <em>that person&rsquo;s own</em>
recordings sit from each other.</li>
<li><strong>Between</strong>: how far they sit from <em>other people&rsquo;s</em>
recordings.</li>
</ul>


{fig("11_within_vs_between.png", "One dot per subject, {ns} in all. How far across a dot sits is the distance among <em>that person's own</em> recordings. How far up it sits is the distance to ten <em>other people's</em> recordings picked at random. The dashed line marks where those two would be equal, which is where every dot would land if a walk carried nothing personal in it. Four dots are ringed and named to show the range: the most and least distinctive subjects, a typical one, and the one whose own recordings vary the most.".format(ns=FP['n_subjects']),
  "<strong>What we see.</strong> Nothing lands near the line. <strong>Every one of the {ns} dots sits above it</strong>, with no exceptions. A person's own recordings are a typical distance of <strong>{w}</strong> apart, while other people's sit at <strong>{b}</strong>. So a person's own walks are about <strong>{r} times more alike</strong> than anyone else's.<br><br><strong>What that means for us.</strong> Each person leaves a recognisable mark on their own recordings, the way handwriting is recognisable across different pages. That is a hazard rather than a useful finding. Suppose one recording of a PD subject were used to build a method and a second recording of the same subject were used to test it. The method would not need to learn anything about Parkinson's at all. It could recognise <em>that individual</em> and report back the label it had already been shown. The score would be genuine and the conclusion drawn from it would be wrong.<br><br><strong>What follows.</strong> Two rules, applied everywhere in this thesis. Every subject is held out <strong>completely</strong>, never one recording at a time. And when uncertainty is estimated, <strong>whole subjects are resampled</strong> rather than individual recordings, because six recordings of one person are not six independent observations, and treating them as six would make the results look more certain than they are.".format(
    ns=FP['n_subjects'], w=FP['within_median'], b=FP['between_median'], r=FP['times_more_alike']))}

"""))

toc ="".join(f'<li><a href="#s{i}"><span class="tn">{i}</span>{t}</a></li>' for i,(t,_) in enumerate(S,1))
body = "".join(f'<section id="s{i}" class="sec"><h2><span class="secnum">{i}</span>{t}</h2>{c}</section>' for i,(t,c) in enumerate(S,1))

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
<title>The Data &amp; EDA · Radar micro-Doppler Parkinson's</title>
<style>{CSS}</style></head><body>
<div id="bar"></div>
<div class="wrap">
<header class="hero">
  <p class="kicker">Master's thesis · Data &amp; exploratory analysis</p>
  <h1>The Data, End to End</h1>
  <p class="sub">Where these recordings come from, what is physically inside them, how each
  recording is turned into measurements, and what the exploratory analysis established.
  Modelling and results are covered separately.</p>
  <div class="statlead">The numbers that describe the dataset:</div>
  <div class="stats">
    <div class="stat"><b>{NS}</b><span>people recorded<i>{N['n_ctrl']} healthy, {N['n_pd']} with Parkinson's</i></span></div>
    <div class="stat"><b>{NT}</b><span>walk recordings<i>6 per person, 106 GB in total</i></span></div>
    <div class="stat"><b>2</b><span>simultaneous views<i>one aimed at the feet, one at the trunk</i></span></div>
    <div class="stat"><b>{NF}</b><span>measurements per recording<i>summarising each spectrogram</i></span></div>
    <div class="stat"><b>3</b><span>findings that shaped the work<i>see section 6</i></span></div>
  </div>
</header>
<nav class="toc"><h4>Contents</h4><ol>{toc}</ol></nav>
<main>{body}</main>
</div>
<button id="top" title="Back to top">↑</button>
<script>{JS}</script>
</body></html>"""

out = ROOT / "DATA_AND_EDA.html"
out.write_text(html)
nd = html.count('class="defn"'); nl = html.count('class="later"')
print(f"wrote {out} ({len(html)/1e6:.2f} MB)")
print(f"sections={len(S)} figures={html.count('data:image/jpeg')} definitions={nd} deferrals={nl}")
