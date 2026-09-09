#!/usr/bin/env python3
"""REMAINING_SECTIONS.html: drafts of every report section not yet written,
in the order they appear in the report.

Order in the report: Resumen/Summary (frontmatter) -> Acknowledgements ->
Chapter 1 (Introduction, Objectives) -> Results section 4.7 -> Annex A ->
Annex B. Acronyms are already complete. Each section names the LaTeX file it
will be transcribed into once reviewed. Citation keys for the transcription
are shown inline as [key] tags.

Same style rules as the other documents: no em-dashes, key terms bolded,
numbers only where they are already established results.
"""
import pathlib

ROOT = pathlib.Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")

def key(t):  return f'<aside class="key"><span class="klabel">Key point</span><p>{t}</p></aside>'
def warn(t): return f'<aside class="warn"><span class="klabel">Decision needed</span><p>{t}</p></aside>'
def defn(term, body): return f'<div class="defn"><span class="dterm">{term}</span><p>{body}</p></div>'
def intro(t): return f'<div class="intro"><p>{t}</p></div>'
def slot(f): return f'<p class="slot">Goes into: <code>{f}</code></p>'
def cite(k): return f'<code class="ck">[{k}]</code>'

# ---- Annex B budget: computed here so the arithmetic cannot drift ----
HOURS, RATE = 360, 15.0
LABOUR = HOURS * RATE
PC = 3200.0 * 7 / 60          # MacBook Pro 2021 (M1 Max, 32 GB), 7 months of a 5-year amortisation
CD = LABOUR + PC              # direct costs
GG = 0.15 * CD                # general expenses, 15% over CD
BI = 0.06 * (CD + GG)         # industrial benefit, 6% over CD + CI
FUNG = 100.0 + 300.0          # printing + binding
SUBTOT = CD + GG + BI + FUNG
IVA = 0.21 * SUBTOT
TOTAL = SUBTOT + IVA

def eur(x):
    s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s}&nbsp;&euro;"

CSS = (ROOT / "tools/eda.css").read_text() + """
.sec h4{font-family:var(--sans);font-size:15px;font-weight:700;color:var(--ink);margin:26px 0 8px}
.intro{border-left:3px solid var(--accent);padding:2px 0 2px 16px;margin:22px 0 6px}
.intro p{margin:0;font-size:15.5px;line-height:1.62;max-width:none;color:var(--ink)}
.slot{font-family:var(--sans);font-size:12.5px;color:var(--ink-3);margin:2px 0 14px}
.slot code{background:var(--surface);padding:2px 7px;border-radius:5px}
code.ck{background:var(--surface);padding:1px 5px;border-radius:4px;font-size:12px;color:var(--accent)}
.draft{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px 22px;margin:14px 0}
.draft p{max-width:none}
.wc{font-family:var(--sans);font-size:12px;color:var(--ink-3);margin-top:8px}
"""

S = []

# ═══ 1 ═══ Resumen (ES)
S.append(("Resumen y palabras clave (frontmatter, Spanish)", f"""
{slot("frontmatter/abstract.tex &middot; first block &middot; limit 500 words")}
{intro("The template requires the Spanish Resumen and the English Summary, both at most 500 words, structured context, problem, data, method, result, conclusion. The keywords are already in place in the file. Have a native speaker read the Spanish before submission.")}

<div class="draft">
<p>La enfermedad de Parkinson es el segundo trastorno neurodegenerativo
m&aacute;s frecuente, y la alteraci&oacute;n de la marcha figura entre sus
signos motores m&aacute;s tempranos e incapacitantes. El radar de onda
continua modulada en frecuencia (FMCW) permite observar la marcha sin
contacto, sin c&aacute;maras y sin sensores corporales: cada recorrido se
convierte en un <strong>espectrograma micro-Doppler</strong>, una imagen
tiempo-frecuencia que separa las velocidades del torso y de las
extremidades. Ello lo hace atractivo para la evaluaci&oacute;n objetiva del
movimiento en la cl&iacute;nica y en el hogar; sin embargo, ning&uacute;n
trabajo publicado hab&iacute;a evaluado un clasificador de Parkinson frente
a controles sobre estos espectrogramas con <strong>validaci&oacute;n
independiente del sujeto</strong>, aquella en la que cada persona puntuada
es alguien que el modelo nunca vio durante el entrenamiento, que es la
condici&oacute;n real de una herramienta de cribado.</p>

<p>Este trabajo construye esa evaluaci&oacute;n sobre una red de radar
cl&iacute;nicamente validada y una cohorte de 58 sujetos (33 controles
sanos y 25 pacientes). Se desarrolla un proceso completo y prerregistrado:
ventanas de duraci&oacute;n fija sobre los espectrogramas, modelos
cl&aacute;sicos basados en caracter&iacute;sticas y cuatro familias de
aprendizaje profundo, incluida la arquitectura de referencia de la
literatura, todo ello bajo validaci&oacute;n cruzada dejando-un-sujeto-fuera,
con predictores de referencia deliberadamente simples junto a cada modelo e
incertidumbre estimada a nivel de sujeto.</p>

<p>Bajo este protocolo, ning&uacute;n clasificador demuestra conocer la
enfermedad m&aacute;s all&aacute; de lo que revela la duraci&oacute;n de la
grabaci&oacute;n, pues los pacientes sencillamente tardan m&aacute;s. El
an&aacute;lisis de atenci&oacute;n y un experimento de intervenci&oacute;n
atribuyen la puntuaci&oacute;n restante a propiedades de cada
grabaci&oacute;n y no a la forma espectral de la marcha; y reproducir sobre
estos mismos datos el reparto aleatorio habitual en la literatura muestra
cu&aacute;nto infla ese dise&ntilde;o el rendimiento: modelos aparentemente
listos para la cl&iacute;nica se desploman ante desconocidos, porque
reconocen a la persona y no a la enfermedad. Lo que sobrevive es
cl&iacute;nicamente coherente: el radar mide con fiabilidad los tiempos de
la marcha, en especial al levantarse y al girar, lo que equivale a una
prueba <em>Timed-Up-and-Go</em> automatizada.</p>

<p>La contribuci&oacute;n es doble: la primera evaluaci&oacute;n
independiente del sujeto de este problema, y una demostraci&oacute;n
mecanicista de c&oacute;mo los clasificadores de marcha por radar llegan a
parecer mejores de lo que son, junto con el protocolo de evaluaci&oacute;n
que lo impide.</p>
<p class="wc">Draft length: about 330 words (limit 500).</p>
</div>
"""))

# ═══ 2 ═══ Summary (EN)
S.append(("Summary and keywords (frontmatter, English)", f"""
{slot("frontmatter/abstract.tex &middot; second block &middot; limit 500 words")}

<div class="draft">
<p>Parkinson's disease is the second most common neurodegenerative
disorder, and gait impairment is among its earliest and most disabling
motor signs. Frequency-Modulated Continuous Wave (FMCW) radar can observe
gait without contact, cameras or wearable sensors: a walk becomes a
<strong>micro-Doppler spectrogram</strong>, a time-frequency image that
resolves the velocities of the torso and the limbs. This makes radar
attractive for objective motor assessment in clinics and homes, yet no
published work had evaluated a Parkinson's-versus-control classifier on
these spectrograms under <strong>subject-independent validation</strong>,
in which every scored person is one the model never saw during training,
which is the condition a screening tool faces in practice.</p>

<p>This thesis builds that evaluation on a clinically validated radar
network and a cohort of 58 subjects (33 healthy controls, 25 patients). A
complete, pre-registered pipeline is developed: fixed-length windowing of
the spectrograms, classical feature-based baselines and four deep learning
families, including the field's reference architecture, all evaluated under
leave-one-subject-out cross-validation, with deliberately simple reference
predictors reported beside every model and uncertainty estimated at the
subject level.</p>

<p>Under this protocol, no classifier demonstrates more knowledge of the
disease than a predictor based on recording duration alone, since patients
simply take longer. Attention analysis and an intervention experiment trace
the remaining score to properties of each recording rather than to the
spectral shape of the walk, and reproducing the literature's usual random
hold-out on this same dataset shows how strongly that design inflates
performance: models that look clinic-ready on familiar subjects collapse on
strangers, recognising the person rather than the disease. What survives is
clinically coherent: radar reliably measures the timing of the walk,
especially rising from the chair and turning, which amounts to an automated
<em>Timed-Up-and-Go</em> test.</p>

<p>The contribution is twofold: the first subject-independent evaluation of
this problem, and a mechanistic demonstration of how radar gait classifiers
come to look better than they are, together with the evaluation protocol
that prevents it.</p>
<p class="wc">Draft length: about 300 words (limit 500).</p>
</div>
"""))

# ═══ 3 ═══ Acknowledgements
S.append(("Acknowledgements (frontmatter)", f"""
{slot("frontmatter/acknowledgements.tex")}
{intro("Personal by nature, so this is only a scaffold with the names the file's own notes list. Rewrite it in your own voice.")}

<div class="draft">
<p>I would like to thank my supervisor, Prof.&nbsp;Juan Ignacio
Godino-Llorente, for his guidance and for holding this work to the standard
it defends. I am grateful to Ignacio L&oacute;pez-Delgado and the GAPS team
for building the radar network and for entrusting me with the dataset, and
to every participant who walked the corridor that made this study
possible.</p>
<p>To my parents, for the confidence they placed in me long before any
result justified it, and for making it possible for me to study far from
home. To my family, for their patience with the distance and with every
call this thesis made me miss. And to my friends, in Madrid and back home,
who kept me company through the long weeks of writing and listened to far
more about radar spectrograms than they ever signed up for. This work is
theirs as much as mine.</p>
</div>
"""))

# ═══ 4 ═══ Chapter 1: Introduction
S.append(("Chapter 1 &middot; Introduction", f"""
{slot("chapters/ch1_introduction/introduction.tex")}
{intro("Four sections, as the file's outline plans: Motivation, Problem Statement, Contributions, and Structure of this Document. Citation keys for the LaTeX transcription are marked inline.")}

<h4>1.1 Motivation</h4>
<div class="draft">
<p>Parkinson's disease is the second most common neurodegenerative disorder,
and its diagnosis remains clinical: a neurologist examines the patient's
movement against consensus criteria {cite("postuma2015")}. Among the motor
signs, <strong>gait</strong> holds a special place. It deteriorates early,
it disables profoundly, and it is measurable: patients walk more slowly,
with shorter steps, reduced arm swing and, characteristically, greater
stride-to-stride variability and marked difficulty with turns and with
rising from a chair {cite("mirelman2019")}{cite("hausdorff2009")}. A
technology that measured gait objectively, repeatedly and outside the
clinic could support earlier referral, longitudinal tracking and therapy
adjustment.</p>

<p>The instruments available today each concede something. Optical motion
capture is the laboratory gold standard but is expensive, confined to
instrumented rooms and requires markers. Wearable sensors travel with the
patient but must be worn, charged and accepted, a real burden for elderly
users. Cameras are cheap but raise immediate privacy objections in the
home. <strong>Radar</strong> concedes none of these: it is contactless,
works through clothing and in darkness, records no identifiable image, and
its micro-Doppler return resolves the velocities of the torso and the limbs
individually {cite("seifert2020")}{cite("gurbuz2024")}. A walk in front of a
radar becomes a time-frequency image, a <strong>micro-Doppler
spectrogram</strong>, in which the torso traces a slow band and each foot a
fast periodic signature.</p>

<p>This thesis works on data from a clinically validated multi-node FMCW
radar network built precisely for gait monitoring, whose parameter
extraction has been validated in both healthy and Parkinson's populations
{cite("lopezdelgado2026")}. The cohort contains 58 subjects, 33 healthy
controls and 25 Parkinson's patients, each walking a short corridor in a
protocol modelled on the clinical Timed-Up-and-Go test. The question this
thesis asks of that data is the natural next one: can the disease itself be
read from the spectrograms?</p>
</div>

<h4>1.2 Problem Statement</h4>
<div class="draft">
<p>The radar gait literature has answered a different question. Its
validation studies establish which biomechanical parameters radar measures
reliably {cite("seifert2020")}{cite("lopezdelgado2026")}, and its
classification studies discriminate coarse groups, almost always young
versus elderly walkers, reporting accuracies above 90&nbsp;%
{cite("hayashi2021")}{cite("hoshiga2021")}. No published work classifies
Parkinson's disease against controls from micro-Doppler spectrograms under
<strong>subject-independent validation</strong>, in which every scored
person is one the model never saw during training.</p>

<p>That validation detail is not a technicality; it decides what the number
means. Micro-Doppler encodes individual identity strongly enough that
published systems recognise <em>people</em> from their walk
{cite("papanastasiou2020")}{cite("ni2020")}. Any evaluation that lets
recordings of the same person appear on both sides of the split therefore
lets a model score highly by recognising the walker rather than the
disease. The field's own record contains the warning: a spectrogram network
reaching 97.8&nbsp;% was shown by its own authors to have keyed on
background differences between recording sites rather than on gait
{cite("hayashi2021")}. Confounds compound the risk: Parkinson's cohorts are
older than their controls, patients take longer to complete the test, and
every one of these regularities is available to a classifier as a
shortcut.</p>

<p>The problem this thesis addresses is therefore twofold: to build and
evaluate the first subject-independent Parkinson's-versus-control
classifier on radar micro-Doppler, and to do so under a protocol in which
every score is defensible, with the confounds measured, floored and
controlled rather than ignored.</p>
</div>

<h4>1.3 Contributions</h4>
<div class="draft">
<p>The thesis makes five contributions:</p>
<ul>
<li><strong>The first subject-independent evaluation</strong> of
Parkinson's-versus-control classification from radar micro-Doppler gait
signatures, spanning classical models and four deep model families,
including the architecture behind the field's headline result
{cite("hayashi2021")}.</li>
<li><strong>A pre-registered confound-control methodology:</strong>
fixed-length windows, nested leave-one-subject-out validation, explicit
null floors beside every model, and subject-level bootstrap confidence
intervals, frozen before any result was known.</li>
<li><strong>An honest quantification of the achievable signal:</strong>
under that protocol, no classifier is distinguishable from recording
duration alone, and the finding is stated as a property of the data's
signal content with its uncertainty, not as a modelling failure.</li>
<li><strong>A mechanistic explanation of the scores:</strong> attention
analysis of both image models, an intervention experiment that deletes what
they attend to, and a protocol experiment that reproduces the literature's
numbers on this very dataset and shows them collapsing on unseen
subjects.</li>
<li><strong>A constructive clinical finding and a reusable protocol:</strong>
radar reliably measures the timing signature of the disease, an automated
Timed-Up-and-Go, and the evaluation protocol stands as the design document
for the richer datasets the field needs next.</li>
</ul>
</div>

<h4>1.4 Structure of this Document</h4>
<div class="draft">
<p><strong>Chapter&nbsp;2</strong> reviews the state of the art: radar
micro-Doppler as a gait sensing modality, the classification studies
nearest to this task, and the validation pitfalls that recur throughout the
field. <strong>Chapter&nbsp;3</strong> specifies the materials and methods:
the dataset, the preprocessing and windowing, the classical and deep model
families, the leave-one-subject-out protocol, and the confound-control and
evaluation machinery, all fixed before training. <strong>Chapter&nbsp;4</strong>
reports the results: the exploratory characterisation that constrained the
design, the null floors, the classical and deep campaigns, the exploratory
variants and the decision-level performance. <strong>Chapter&nbsp;5</strong>
discusses what the results mean, demonstrates experimentally where the gap
to the literature comes from, states the positive findings and the
limitations, concludes, and lays out the path to stronger results. Two
annexes cover the ethical, social, economic and environmental aspects and
the project budget.</p>
</div>
"""))

# ═══ 5 ═══ Chapter 1: Objectives
S.append(("Chapter 1 &middot; Objectives", f"""
{slot("chapters/ch1_introduction/objectives.tex")}
{intro("Five objectives, kept exactly as the file's outline plans them so the conclusions of Chapter 5 can be mapped back one to one.")}

<div class="draft">
<p>The general objective of this thesis is to determine whether Parkinson's
disease can be classified from FMCW radar micro-Doppler gait spectrograms
under an evaluation a clinician could trust. It is pursued through five
specific objectives:</p>
<ul>
<li><strong>O1.</strong> Characterise the dataset and identify the
confounds capable of producing spurious classification performance, before
any model is trained.</li>
<li><strong>O2.</strong> Establish a confound-controlled classical baseline
under subject-independent (leave-one-subject-out) validation, including
deliberately simple reference predictors, such as one that knows only how
long each recording lasts, that set the score any real model must clearly
beat.</li>
<li><strong>O3.</strong> Develop and evaluate deep models on windowed
micro-Doppler spectrograms, a compact convolutional network, an
ImageNet-pretrained probe and a sequence model on velocity envelopes, under
the identical protocol.</li>
<li><strong>O4.</strong> Verify with interpretability methods (Grad-CAM)
what the trained networks actually attend to, and test the finding with an
intervention experiment rather than leaving it as a picture.</li>
<li><strong>O5.</strong> Report every result against the explicit null
floors with honest uncertainty, and reconcile the outcome with the
published literature by reproducing its validation design on this same
dataset.</li>
</ul>
<p>Chapter&nbsp;5 returns to these objectives one by one; the note in
<code>conclusions.tex</code> already marks where the mapping attaches.</p>
</div>
"""))

# ═══ 6 ═══ Annex A
S.append(("Annex A &middot; Ethical, economic, social and environmental aspects", f"""
{slot("chapters/annexes/annexA_ethics.tex &middot; sections A1&ndash;A4")}
{intro("The annex is mandatory (EUR-ACE/ABET) and its A1&ndash;A4 structure is fixed by the template, whose headings are in Spanish. The draft below is in English to match the body; if the school expects the annex itself in Spanish, it translates directly. It is deliberately synthetic, as the template requests.")}

<h4>A1. Introducci&oacute;n</h4>
<div class="draft">
<p>This project develops and evaluates a machine-learning system that
screens for Parkinson's disease from contactless radar recordings of gait.
It touches three sensitive domains at once: <strong>health data</strong>
from a vulnerable population, <strong>clinical claims</strong> produced by
machine-learning models, and a sensing technology whose reach into private
spaces is precisely its selling point. The relevant dimensions are
therefore ethical (validity of clinical claims, privacy), social (access to
earlier screening for an ageing population), economic (low-cost monitoring
against expensive laboratory instrumentation) and, to a lesser degree,
environmental (computational footprint).</p>
</div>

<h4>A2. Descripci&oacute;n de impactos relevantes</h4>
<div class="draft">
<p>Three impacts dominate, with their stakeholders:</p>
<ul>
<li><strong>The validity of clinical machine-learning claims.</strong>
Stakeholders: patients, clinicians, the research field. An overstated
screening accuracy harms twice: false alarms send healthy people to
neurology consultations, and false trust delays real diagnoses. This thesis
found that the field's evaluation habits inflate performance substantially,
and made honest evaluation its central contribution.</li>
<li><strong>Privacy of gait data.</strong> Stakeholders: monitored persons
and their households. Radar records no image, which is its privacy
advantage; yet this work also measured that a person's walk identifies them
(all 58 of 58 subjects distinguishable), so gait recordings are biometric
data under GDPR and must be treated as such: anonymised identifiers,
restricted access, purpose limitation. The dataset used here is fully
anonymised and was collected under the clinical validation study of the
source system.</li>
<li><strong>Access and cost of motor assessment.</strong> Stakeholders:
health systems, patients in areas without movement-disorder specialists. A
radar Timed-Up-and-Go, the one measurement this thesis validates as
trustworthy, is cheap, contactless and automatable, and could extend
objective motor assessment beyond specialist clinics.</li>
</ul>
</div>

<h4>A3. An&aacute;lisis detallado de uno de los impactos</h4>
<div class="draft">
<p>The impact analysed in depth is the first: <strong>what happens when
clinical machine-learning performance is overstated</strong>, because this
thesis measured the mechanism directly. On this very dataset, the
evaluation design used by most published work produces subject-level AUCs
up to 0.89; scoring the same models on people excluded from training
returns them to a band indistinguishable from a predictor that knows only
how long the recording lasted. A deployment decision taken on the first
number would field a screening tool whose real behaviour on new patients is
close to a biased guess: in the decision analysis, catching 21 of 25
patients cost falsely flagging 18 of 33 healthy controls. The ethical
posture adopted is therefore procedural rather than declarative:
pre-registration of the analysis plan, subject-independent validation,
explicit null floors beside every reported score, confidence intervals at
the subject level, and publication of the negative result. This is the
engineering-responsibility standard the EUR-ACE criteria ask the graduate
to demonstrate, applied to the claim itself rather than to the artefact.</p>
</div>

<h4>A4. Conclusiones</h4>
<div class="draft">
<p>The project's sustainability profile is favourable: it consumes only
modest computation (a single personal computer; no data-centre training),
reuses an existing clinical dataset rather than collecting new one, and its
product is methodological, a protocol that prevents overstated clinical
claims. The sustainability criteria added genuine value: treating the
validity of the claim as the primary ethical object is what turned an
apparently negative result into the thesis's contribution, and the
privacy analysis (gait as a biometric) is itself one of the work's
findings.</p>
</div>
"""))

# ═══ 8 ═══ Annex B
S.append(("Annex B &middot; Economic budget", f"""
{slot("chapters/annexes/annexB_budget.tex &middot; replace the template's example table")}
{intro("Where these numbers come from: the <strong>rates and percentages are the template's own</strong> (15 &euro;/h labour, 15 % general expenses, 6 % industrial benefit, 21 % VAT, and the 100/300 &euro; printing and binding lines all appear in the school's example table); only the <strong>quantities are estimates to confirm</strong>: 360 hours (12-ECTS TFM at 30 h/ECTS, confirmed), the computer is the author's MacBook Pro 2021 (M1 Max, 32 GB) valued at its 3.200 &euro; purchase price, and 7 months of use matches the project's actual span. The radar network and the dataset were provided by the research group at no direct cost, which the annex states, and all software used is open source.")}

{defn("How to read the budget", "The school requires the report to price the project as if a company had been hired to do it, line by line. <strong>Labour</strong> values the student's own working hours at a reference hourly rate. <strong>Material resources</strong> charge only the slice of the equipment's life the project consumed: the computer cost 3.200&nbsp;&euro; and lasts about 5 years (60 months) in accounting terms, the project used it for 7 of those months, so it charges 3.200&nbsp;&times;&nbsp;7/60&nbsp;=&nbsp;373,33&nbsp;&euro;. <strong>General expenses</strong> is a flat 15&nbsp;% surcharge standing in for everything not itemised (electricity, internet, workspace). <strong>Industrial benefit</strong> is the standard 6&nbsp;% profit margin a contractor would add on top. <strong>Consumables</strong> are the printing and binding of the report itself. VAT is then applied to the sum, exactly as on any invoice, giving the total a client would pay.")}

<div class="tblwrap"><table>
<thead><tr><th>Concept</th><th>Detail</th><th>Amount</th></tr></thead><tbody>
<tr><td><strong>Labour (direct cost)</strong></td>
<td>{HOURS} hours (12 ECTS &times; 30 h) at {RATE:.0f}&nbsp;&euro;/h</td>
<td class="num">{eur(LABOUR)}</td></tr>
<tr><td><strong>Material resources (direct cost)</strong></td>
<td>Personal computer: MacBook Pro 2021 (M1 Max, 32&nbsp;GB), purchase
3.200,00&nbsp;&euro;, used 7 months, amortised over 5 years. Radar network and dataset provided by the research
group (no direct cost); software entirely open source (0&nbsp;&euro;).</td>
<td class="num">{eur(PC)}</td></tr>
<tr><td><strong>General expenses (indirect)</strong></td>
<td>15&nbsp;% over direct costs</td>
<td class="num">{eur(GG)}</td></tr>
<tr><td><strong>Industrial benefit</strong></td>
<td>6&nbsp;% over direct + indirect costs</td>
<td class="num">{eur(BI)}</td></tr>
<tr><td><strong>Consumables</strong></td>
<td>Printing (100,00&nbsp;&euro;) and binding (300,00&nbsp;&euro;)</td>
<td class="num">{eur(FUNG)}</td></tr>
<tr><td><strong>Subtotal</strong></td><td></td>
<td class="num"><strong>{eur(SUBTOT)}</strong></td></tr>
<tr><td><strong>VAT (21&nbsp;%)</strong></td><td></td>
<td class="num">{eur(IVA)}</td></tr>
<tr><td><strong>TOTAL</strong></td><td></td>
<td class="num"><strong>{eur(TOTAL)}</strong></td></tr>
</tbody></table></div>

{defn("Transcription note", "The LaTeX annex keeps the template's own table layout (the school's expected format); only the example numbers are replaced by the ones above. The 'Otro equipamiento' row becomes the radar-network line with an explicit zero direct cost and a footnote that the hardware belongs to the research group's clinical validation study.")}
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
<title>The Remaining Sections &middot; Radar micro-Doppler Parkinson's</title>
<style>{CSS}</style></head><body>
<div id="bar"></div>
<div class="wrap">
<header class="hero">
  <p class="kicker">Master's thesis &middot; every unwritten section, drafted for review</p>
  <h1>The Remaining Sections</h1>
  <p class="sub">In the order they appear in the report: the Spanish Resumen and
  English Summary, the acknowledgements scaffold, Chapter 1 (Introduction and
  Objectives), and Annexes A and B. The acronym list is already complete, and
  the pending Results &sect;4.7 has been dropped from the report along with
  the methods promise that announced it. Each section names the LaTeX file it
  will be transcribed into once you approve it.</p>
  <div class="statlead">What this closes:</div>
  <div class="stats">
    <div class="stat"><b>2</b><span>abstracts drafted<i>Resumen + Summary</i></span></div>
    <div class="stat"><b>Ch. 1</b><span>fully drafted<i>4 + 1 sections</i></span></div>
    <div class="stat"><b>A + B</b><span>annexes drafted<i>ethics + budget</i></span></div>
    <div class="stat"><b>0</b><span>pending sections left<i>&sect;4.7 dropped</i></span></div>
  </div>
</header>
<nav class="toc"><h4>Contents</h4><ol>{toc}</ol></nav>
<main>{body}</main>
</div>
<button id="top" title="Back to top">&uarr;</button>
<script>{JS}</script>
</body></html>"""

out = ROOT / "REMAINING_SECTIONS.html"
out.write_text(html)
print(f"wrote {out} ({len(html)/1e3:.0f} KB)")
print(f"sections={len(S)} emdash={html.count(chr(8212))}")
