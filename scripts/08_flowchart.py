"""Draw the cohort flow chart (CONSORT style) from the counts in results/flowchart/flow_counts.json.

Steps: all ICU stays -> adults (>= 18 y) -> SOFA computable within W h -> lactate within W h -> base excess within W h
-> not died within the first W h,
for W = 6 and W = 24 (one figure each, plus a side-by-side figure). Coordinates are in inches.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'flowchart'
F = json.load(open(OUT / 'flow_counts.json'))
plt.rcParams['font.family'] = 'DejaVu Sans'

FS, LH, PAD = 9.0, 0.21, 0.09          # font size (pt), line height / padding (inches)
WIDTH, HEIGHT = 10.0, 10.0
LX, LW, RX, RW = 0.3, 4.3, 5.2, 4.5    # main column / exclusion column (x, width)
GAP = 0.8                             # vertical space between main boxes (holds the arrow)


def box(ax, x, y_top, w, lines, fs=FS):
    h = 2 * PAD + LH * len(lines)
    ax.add_patch(Rectangle((x, y_top - h), w, h, fill=False, lw=1.0, ec='black'))
    for i, line in enumerate(lines):
        ax.text(x + 0.12, y_top - PAD - i * LH - 0.02, line, fontsize=fs, va='top', ha='left',
                fontweight='bold' if i == 0 else 'normal')
    return y_top - h


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=11, lw=1.0, color='black',
                                 shrinkA=0, shrinkB=0))


def draw(ax, W, title=None):
    d = F[f'W{W}']; m = d['excl_sofa_missing_organ']
    main = [
        [f"{F['all']['stays']:,} ICU stays in MIMIC-III v1.4", f"({F['all']['patients']:,} patients)"],
        [f"{F['adult']['stays']:,} adult ICU stays", f"({F['adult']['patients']:,} patients)"],
        [f"{d['sofa']['stays']:,} stays with SOFA computable in the first {W} h", f"({d['sofa']['patients']:,} patients)"],
        [f"{d['lactate']['stays']:,} stays with lactate in the first {W} h", f"({d['lactate']['patients']:,} patients)"],
        [f"{d['be']['stays']:,} stays with base excess in the first {W} h", f"({d['be']['patients']:,} patients)"],
        [f"{d['final']['stays']:,} stays included  ({d['final']['patients']:,} patients)",
         f"{d['final_died_28d']:,} died within 28 days of ICU admission",
         f"{d['final_alive_28d']:,} alive at 28 days"],
    ]
    excl = [
        [f"{F['excl_age'] + F['excl_time']:,} excluded",
         f"{F['excl_age']:,} age < 18 y at ICU admission",
         f"   ({F['excl_age_nicu']:,} NICU, {F['excl_age'] - F['excl_age_nicu']:,} paediatric in adult units)",
         f"{F['excl_time']:,} ICU discharge time not after admission"],
        [f"{d['excl_sofa']:,} excluded: SOFA not computable in {W} h",
         "missing organ system (not mutually exclusive):",
         f"   liver (bilirubin) {m['liver']:,}",
         f"   respiration (PaO2 and FiO2) {m['respiration']:,}",
         f"   coagulation {m['coagulation']:,}   CNS (GCS) {m['CNS']:,}",
         f"   cardiovascular {m['cardiovascular']:,}   renal {m['renal']:,}"],
        [f"{d['excl_lactate']:,} excluded", f"no lactate measured in the first {W} h"],
        [f"{d['excl_be']:,} excluded", f"no base excess measured in the first {W} h",
         "(base excess is reported with every PaO2, already required for SOFA)"],
        [f"{d['excl_died_in_window']:,} excluded", f"died within the first {W} h of ICU admission",
         "(outcome already observed inside the observation window)"],
    ]
    y = HEIGHT - 0.35
    for i, lines in enumerate(main):
        y_bottom = box(ax, LX, y, LW, lines)
        if i < len(excl):
            y_next_top = y_bottom - GAP
            xm = LX + LW / 2
            arrow(ax, xm, y_bottom, xm, y_next_top)
            ymid = y_bottom - GAP / 2
            h = 2 * PAD + LH * len(excl[i])
            box(ax, RX, ymid + h / 2, RW, excl[i], fs=FS - 0.6)
            arrow(ax, xm, ymid, RX, ymid)
            y = y_next_top
    ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.axis('off')
    if title:
        ax.text(LX, HEIGHT - 0.08, title, fontsize=11, fontweight='bold', va='top')


note = ("Labs are counted from 6 h before ICU admission (admission panel); SOFA 'computable' = all six organ systems measured "
        "(PaO2 & FiO2, platelets, bilirubin, MAP or vasopressor, GCS, creatinine or urine output). "
        "28-day mortality = death from any cause within 28 days of ICU admission (PATIENTS.DOD, includes post-discharge deaths).")
for W in [6, 24]:
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    draw(ax, W)
    fig.text(0.03, 0.02, f"Fig. Cohort selection, {W}-hour observation window. " + note, fontsize=7.5, wrap=True)
    plt.subplots_adjust(0, 0, 1, 1); plt.savefig(OUT / f'flowchart_{W}h.png', dpi=200); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(2 * WIDTH, HEIGHT))
for ax, W in zip(axes, [6, 24]):
    draw(ax, W, title=f'{W}-hour observation window')
plt.subplots_adjust(0, 0, 1, 1, wspace=0); plt.savefig(OUT / 'flowchart_6h_24h.png', dpi=170); plt.close()
print('saved', OUT)
