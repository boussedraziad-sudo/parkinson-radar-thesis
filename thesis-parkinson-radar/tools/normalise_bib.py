#!/usr/bin/env python3
"""
Normalise the thesis bibliography.

Rules applied:
  * volume / number / pages moved out of the journal string into real fields
  * redundant PMC ids and repository UUIDs dropped (the url already resolves there)
  * "accepted to <journal>" moved from the journal string into note
  * entry types corrected (a conference paper is @inproceedings, not @article;
    a journal article is @article, not @misc)
  * NOTHING is invented: only data already present in the file is relocated.
"""
import re, pathlib

# key -> (entrytype, {field: value})   author/title/year/url/doi are carried over
SPEC = {
 "hayashi2021":  ("article", {"journal": "Sensors"}),
 "alanazi2022":  ("article", {"journal": "Sensors", "volume": "22", "number": "15", "pages": "5470"}),
 "ha2023":       ("article", {"journal": "Sensors", "volume": "23", "number": "21", "pages": "8743"}),
 "gurbuz2024":   ("article", {"journal": "IEEE Open Journal of Engineering in Medicine and Biology",
                              "volume": "5", "pages": "735--749"}),
 "cai2023":      ("article", {"journal": "IEEE Sensors Journal", "volume": "23", "number": "10",
                              "pages": "10998--11006"}),
 "papanastasiou2020": ("inproceedings",
                       {"booktitle": "2020 17th European Radar Conference (EuRAD)",
                        "publisher": "IEEE",
                        "note": "Open-access version: MSc thesis, Delft University of Technology"}),
 "seifert2020":  ("article", {"journal": "arXiv preprint arXiv:2005.05280 [eess.SP]"}),
 "fard2026":     ("article", {"journal": "Scientific Reports", "volume": "16", "pages": "9650"}),
 "seyfioglu2018":("article", {"journal": "arXiv preprint arXiv:1811.08361",
                              "note": "Accepted to IEEE Transactions on Aerospace and Electronic Systems"}),
 "ni2020":       ("article", {"journal": "IET Radar, Sonar \\& Navigation", "volume": "14",
                              "number": "10", "pages": "1640--1646"}),
 "neurosurgery2025": ("article", {"journal": "Diagnostics", "volume": "15", "number": "16",
                                  "pages": "2046"}),
 "hoshiga2021":  ("article", {"journal": "Measurement: Sensors", "volume": "18", "pages": "100103"}),
 "park2016":     ("article", {"journal": "Sensors", "volume": "16", "number": "12", "pages": "1990"}),
 "erol2020":     ("article", {"journal": "arXiv preprint arXiv:2001.08582",
                              "note": "Accepted to IEEE Transactions on Aerospace and Electronic Systems"}),
 "nguyen2024":   ("article", {"journal": "PLOS ONE", "volume": "19", "number": "8",
                              "pages": "e0308045"}),
 "lopezdelgado2026": ("article", {"journal": "IEEE Transactions on Biomedical Engineering",
                                  "volume": "73", "number": "1", "pages": "393--403"}),
}
ORDER = ["author", "title", "journal", "booktitle", "publisher", "volume", "number",
         "pages", "year", "doi", "url", "note"]

def parse(text):
    out = {}
    for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', text, re.S):
        typ, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*\{(.*?)\}(?=\s*,\s*\n|\s*\n?$)', body, re.S):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split())
        out[key] = (typ, fields)
    return out

def emit(key, typ, f):
    lines = [f"@{typ}{{{key},"]
    present = [k for k in ORDER if f.get(k)]
    for i, k in enumerate(present):
        comma = "," if i < len(present) - 1 else ""
        lines.append(f"  {k} = {{{f[k]}}}{comma}")
    lines.append("}")
    return "\n".join(lines)

for p in ["/Users/ziad.boussedra/Desktop/Master Thesis/Ziad_Boussedra_TFM/biblio_thesis.bib",
          "/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar/references.bib"]:
    path = pathlib.Path(p)
    entries = parse(path.read_text())
    blocks, changed, retyped = [], 0, []
    for key, (typ, f) in entries.items():
        if key not in SPEC:
            blocks.append(emit(key, typ, f)); continue
        newtyp, over = SPEC[key]
        keep = {k: f.get(k) for k in ("author", "title", "year", "doi", "url") if f.get(k)}
        keep.update(over)
        if newtyp != typ:
            retyped.append(f"{key}: @{typ} -> @{newtyp}")
        if keep != f or newtyp != typ:
            changed += 1
        blocks.append(emit(key, newtyp, keep))
    header = ("% Bibliography — Master's thesis, Ziad Boussedra\n"
              "% Radar micro-Doppler Parkinson's classification. 16 verified sources.\n"
              "% Normalised 2026-07-29: volume/number/pages in real fields; PMC ids and\n"
              "% repository UUIDs removed (urls resolve there); entry types corrected.\n\n")
    path.write_text(header + "\n\n".join(blocks) + "\n")
    print(f"{path.name}: {len(entries)} entries, {changed} normalised")
    for r in retyped:
        print(f"    type fix -> {r}")
