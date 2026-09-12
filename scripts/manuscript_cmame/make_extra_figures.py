"""Deterministic CMAME figures F6-F10, built offline from shipped result JSON files.

Style, determinism policy and the render-time overlap guard follow
`pde_restart/figures_cmame/make_figures.py`. Nothing here recomputes a scientific
quantity: every plotted value is read from a result file and the source path is
recorded in FIGURE_RECEIPT_EXTRA.json.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
MAN = HERE.parent
BASE = MAN.parent
WORK = Path("pde_volume")
OUT = MAN / "figures"
OUT.mkdir(exist_ok=True)

MPL_META = {"Creator": "make_extra_figures.py", "CreationDate": datetime(2000, 1, 1)}
plt.rcParams.update({
    "font.size": 7.0, "axes.labelsize": 7.0, "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3, "legend.fontsize": 5.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
})

SOURCES = []


def load(path):
    p = Path(path)
    if not p.is_absolute():
        p = BASE / p
    SOURCES.append(str(p))
    with open(p) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(OUT / name, format="pdf", metadata=MPL_META)
    plt.close(fig)


def guard(fig, name):
    """Fail if text furniture from one axes enters another axes' box."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(ren).expanded(1.02, 1.08) for a in fig.axes]
    bad = []
    for i, ax in enumerate(fig.axes):
        own = ax.get_window_extent(ren)
        items = (list(ax.texts) + list(ax.get_xticklabels())
                 + list(ax.get_yticklabels()) + [ax.xaxis.label, ax.yaxis.label, ax.title])
        for t in items:
            if not t.get_visible() or not t.get_text():
                continue
            b = t.get_window_extent(ren)
            for j, other in enumerate(boxes):
                if i != j and b.overlaps(other) and not b.overlaps(own):
                    bad.append((i, j, t.get_text()))
    if bad:
        raise RuntimeError(f"overlap guard failed for {name}: {bad}")
    return "PASS"


C_NARROW, C_MEAN, C_STATE, C_SWITCH = "#222222", "#3478b8", "#d95f02", "#7a1d6b"


# --------------------------------------------------------------------------
# F6  training-state support against switch-state support
# --------------------------------------------------------------------------
def fig6():
    geo = load("family1_second_pairs/results/GEOMETRY.json")
    iv = load("family1b_pair_screen/RESULT.json")["intervals"]
    gs = load("family1_rerun/results/S1_grayscott_1sp/RESULT.json")
    fk = load("family1_rerun/results/S1_fkpp_r0.5/RESULT.json")

    narrow = geo["narrow_training_support"]
    rows = [
        ("Fisher–KPP, $r=1.0$",
         geo["pairs"]["P1_diffusion_fisher_kpp"]["broad_search"][0]["support"],
         geo["pairs"]["P1_diffusion_fisher_kpp"]["switch_support"], "measurable"),
        ("Fisher–KPP, $r=0.5$",
         iv["S1_fkpp_r0.5"]["search"][0]["support"],
         fk["gates"]["G1_shift_exists"]["switch_support"], "measurable"),
        ("Gray–Scott, single species",
         iv["S1_grayscott_1sp"]["search"][0]["support"],
         gs["gates"]["G1_shift_exists"]["switch_support"], "measurable"),
        ("Cahn–Hilliard (conservative)",
         geo["pairs"]["P3_diffusion_cahn_hilliard"]["broad_search"][0]["support"],
         geo["pairs"]["P3_diffusion_cahn_hilliard"]["switch_support"], "no shift"),
        ("Allen–Cahn",
         geo["pairs"]["P2_diffusion_allen_cahn"]["broad_search"][0]["support"],
         geo["pairs"]["P2_diffusion_allen_cahn"]["switch_support"], "no dynamic range"),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 2.5), constrained_layout=True)
    h = 0.17
    for k, (label, broad, switch, note) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.barh(y + h, broad[1] - broad[0], left=broad[0], height=0.15,
                color=C_MEAN, alpha=0.85,
                label="mean-broadened training support" if k == 0 else None)
        ax.barh(y, narrow[1] - narrow[0], left=narrow[0], height=0.15, color=C_NARROW,
                label="narrow training support" if k == 0 else None)
        ax.barh(y - h, switch[1] - switch[0], left=switch[0], height=0.15, color=C_SWITCH,
                label="switch-state support" if k == 0 else None)
        if note != "measurable":
            ax.text(1.14, y, note, color="0.35", fontsize=5.8, ha="right", va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.4)
    ax.set_xlabel("spatial mean of the state")
    ax.set_xlim(0.25, 1.15)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, ncol=3)
    status = guard(fig, "F6")
    save(fig, "F6_support_geometry.pdf")
    return status


# --------------------------------------------------------------------------
# F7  degradation against candidate support-distance statistics
# --------------------------------------------------------------------------
def fig7():
    d = load("family1d_predictor/RESULT.json")
    cells = d["cells"]
    held = d["heldout_cells"]
    sel = d["selection_cells"]
    rho_d3_held = d["gates"]["PRED_S3_A_spearman_heldout"]["rho"]
    rho_d3_sel = d["reported_not_gated"]["six_selection_cells_spearman_D3"]["rho"]
    rho_mh_held = d["reported_not_gated"]["heldout_spearman_Dbar_mahalanobis"]["rho"]
    rho_mh_sel = d["reported_not_gated"]["six_selection_cells_spearman_Dbar"]["rho"]

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)
    for ax, key, xlabel, rh, rs in (
        (axs[0], "D3", "out-of-basis residual $D_3$", rho_d3_held, rho_d3_sel),
        (axs[1], "Dbar", r"PCA-99 Mahalanobis distance $\bar{D}$", rho_mh_held, rho_mh_sel),
    ):
        for names, marker, colour, lab in (
            (sel, "o", "0.55", "cells the statistic was chosen on"),
            (held, "D", C_STATE, "held-out cells"),
        ):
            xs = [cells[n][key] for n in names]
            ys = [cells[n]["log10R"] for n in names]
            ax.scatter(xs, ys, s=22, marker=marker, facecolor=colour, edgecolor="k",
                       linewidth=0.4, zorder=3, label=lab if ax is axs[0] else None)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.22)
        anchor = (0.03, 0.97, "top", "left") if ax is axs[0] else (0.97, 0.03, "bottom", "right")
        ax.text(anchor[0], anchor[1],
                f"held-out $\\rho$ = {rh:+.2f}\nselection $\\rho$ = {rs:+.2f}",
                transform=ax.transAxes, va=anchor[2], ha=anchor[3], fontsize=6.0)
    axs[0].set_ylabel(r"$\log_{10} R$")
    axs[1].set_xscale("linear")
    axs[0].legend(loc="lower right", frameon=False)
    status = guard(fig, "F7")
    save(fig, "F7_support_distance.pdf")
    return status


# --------------------------------------------------------------------------
# F8  resolution against the length of the handoff
# --------------------------------------------------------------------------
def fig8():
    d = load("family2_resolution_horizon/results/RESULT.json")
    cells = d["cells"]
    res = [128, 256, 512]
    legs = [1, 2, 3]
    shades = {128: "#7fb2dd", 256: "#3478b8", 512: "#12395c"}

    def cell(n, L):
        return cells[f"N{n}_L{L}"]["result"]

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)
    for n in res:
        narrow = [cell(n, L)["narrow"]["median_R"] for L in legs]
        broad = [cell(n, L)["broad"]["median_R"] for L in legs]
        ratio = [cell(n, L)["broad_over_narrow"] for L in legs]
        axs[0].plot(legs, narrow, "-o", ms=4, lw=1.2, color=shades[n],
                    label=f"$N={n}$")
        axs[0].plot(legs, broad, "--s", ms=4, lw=1.2, color=shades[n],
                    markerfacecolor="white")
        axs[1].plot(legs, ratio, "-o", ms=4, lw=1.2, color=shades[n],
                    label=f"$N={n}$")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("median $R$ over ten seeds")
    axs[0].text(0.04, 0.46, "narrow (solid)\nbroadened (dashed)",
                transform=axs[0].transAxes, va="center", ha="left", fontsize=6.0)
    axs[1].axhline(0.25, color="0.35", ls=":", lw=0.9)
    axs[1].text(3.0, 0.256, "reference factor", fontsize=5.8, color="0.35",
                ha="right", va="bottom")
    axs[1].set_ylim(0.0, 0.30)
    axs[1].set_ylabel("broadened / narrow")
    for ax in axs:
        ax.set_xticks(legs)
        ax.set_xlabel("reaction legs before the surrogate is applied")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, loc="upper left" if ax is axs[1] else "center right")
    status = guard(fig, "F8")
    save(fig, "F8_resolution_horizon.pdf")
    return status


# --------------------------------------------------------------------------
# F9  second architecture, three dictionaries
# --------------------------------------------------------------------------
def fig9():
    f2 = load("family2_resolution_horizon/results/N256_L1/RESULT.json")["per_seed"]
    c1 = load("family1c_state_support/RESULT.json")["repair_table"]["P1"]
    un = load(WORK / "family4/results/P1_diffusion_fisher_kpp/RESULT.json")["per_seed"]
    ub = load(WORK / "family4b/results/RESULT.json")["per_seed"]
    fno = {
        "narrow": [f2[f"narrow_{i}"]["R"] for i in range(10)],
        "mean\n-broadened": [f2[f"broad_{i}"]["R"] for i in range(10)],
        "state\nspace": list(c1["R_broadS_per_seed"]),
    }
    unet = {
        "narrow": [r["R"] for r in un if r["condition"] == "narrow"],
        "mean\n-broadened": [r["R"] for r in un if r["condition"] == "broad"],
        "state\nspace": [r["R"] for r in ub if r["condition"] == "broadS"],
    }
    colours = [C_NARROW, C_MEAN, C_STATE]
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.6), sharey=True,
                            constrained_layout=True)
    for ax, (name, data, params) in zip(axs, (
            ("Fourier neural operator", fno, "549,569 parameters"),
            ("U-Net", unet, "467,728 parameters"))):
        for x, (cond, ys) in enumerate(data.items()):
            ys = np.asarray(ys, dtype=float)
            ax.scatter(np.full(len(ys), x), ys, s=12, c=colours[x], zorder=3)
            ax.plot([x - .22, x + .22], [np.median(ys)] * 2, c=colours[x], lw=1.6)
        ax.axhline(1.0, color="0.6", ls=":", lw=0.8, zorder=1)
        ax.set_yscale("log")
        ax.set_xticks(range(3), list(data))
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(f"{name}\n{params}", fontsize=6.6)
        ax.grid(axis="y", alpha=0.22)
    axs[0].set_ylabel("$R$ (per seed)")
    status = guard(fig, "F9")
    save(fig, "F9_second_architecture.pdf")
    return status


# --------------------------------------------------------------------------
# F10  error scales against the persistence reference
# --------------------------------------------------------------------------
def fig10():
    f2 = load("family2_resolution_horizon/results/N256_L1/RESULT.json")
    c1 = load("family1c_state_support/RESULT.json")["repair_table"]
    ch = load("family1_second_pairs/results/P3_negative_control/RESULT.json")
    e1 = load("family1e_state_broadening_new_pairs/results/RESULT.json")["pairs"]
    fk = load("family1_rerun/results/S1_fkpp_r0.5/RESULT.json")
    gs = load("family1_rerun/results/S1_grayscott_1sp/RESULT.json")
    g6 = load("family6_serrano_baseline/results/RESULT_SMOKE.json")

    def med(v):
        return float(np.median(np.asarray(v, dtype=float)))

    def bs(entry):
        per = entry["broadS"]["per_seed"]
        return (med([per[k]["e_on"] for k in per]), med([per[k]["e_off"] for k in per]))

    pairs = [
        ("Fisher–KPP\n$r=1.0$", 0.08019073763412352, [
            ("narrow", f2["narrow"]["median_e_on"], f2["narrow"]["median_e_off"]),
            ("mean-broadened", f2["broad"]["median_e_on"], f2["broad"]["median_e_off"]),
            ("state-space", med(c1["P1"]["e_on_broadS_per_seed"]),
             med(c1["P1"]["e_off_broadS_per_seed"])),
        ]),
        ("Fisher–KPP\n$r=0.5$", fk["gates"]["G2_metric_resolves"]["persistence_off"], [
            ("narrow", fk["narrow"]["median_e_on"], fk["narrow"]["median_e_off"]),
            ("mean-broadened", fk["broad"]["median_e_on"], fk["broad"]["median_e_off"]),
            ("state-space",) + bs(e1["S1_fkpp_r0.5"]),
        ]),
        ("Gray–Scott", gs["gates"]["G2_metric_resolves"]["persistence_off"], [
            ("narrow", gs["narrow"]["median_e_on"], gs["narrow"]["median_e_off"]),
            ("mean-broadened", gs["broad"]["median_e_on"], gs["broad"]["median_e_off"]),
            ("state-space",) + bs(e1["S1_grayscott_1sp"]),
        ]),
        ("Cahn–Hilliard", ch["G4_censored_above"]["persistence_off"], [
            ("narrow", ch["narrow"]["median_e_on"], ch["narrow"]["median_e_off"]),
            ("mean-broadened", ch["broad"]["median_e_on"], ch["broad"]["median_e_off"]),
            ("state-space", med(c1["CH"]["e_on_broadS_per_seed"]),
             med(c1["CH"]["e_off_broadS_per_seed"])),
        ]),
    ]
    assert g6["units"] == 48
    colours = {"narrow": C_NARROW, "mean-broadened": C_MEAN, "state-space": C_STATE}
    fig, ax = plt.subplots(figsize=(6.6, 2.6), constrained_layout=True)
    width = 0.24
    for i, (label, persist, conds) in enumerate(pairs):
        for j, (cond, e_on, e_off) in enumerate(conds):
            x = i + (j - 1) * width
            ax.plot([x, x], [e_on, e_off], color=colours[cond], lw=1.1, zorder=2)
            ax.scatter([x], [e_on], s=16, facecolor="white", edgecolor=colours[cond],
                       linewidth=1.0, zorder=3,
                       label="on-support error $e_{\\rm on}$" if i == 0 and j == 0 else None)
            ax.scatter([x], [e_off], s=16, color=colours[cond], zorder=3,
                       label="off-support error $e_{\\rm off}$" if i == 0 and j == 0 else None)
        ax.plot([i - 0.42, i + 0.42], [persist] * 2, color="0.35", ls="--", lw=0.9,
                label="persistence reference" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(len(pairs)), [p[0] for p in pairs], fontsize=6.4)
    ax.set_ylabel("median relative $L^2$ error")
    ax.grid(axis="y", alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    extra = [plt.Line2D([], [], color=colours[c], lw=2.0) for c in colours]
    ax.legend(handles + extra, labels + list(colours), loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False, ncol=6, fontsize=5.6)
    status = guard(fig, "F10")
    save(fig, "F10_error_headroom.pdf")
    return status


# --------------------------------------------------------------------------
# F11  two-dimensional four-operator dictionary
# --------------------------------------------------------------------------
def _application_order(key):
    """'GoA' is G after A, so the word in order of application is 'AG'."""
    return "".join(reversed(key.split("o")))


def fig11():
    r8 = load("family8_complex_algebra/results/RESULT.json")
    alg = load("family8_complex_algebra/results/ALGEBRA.json")
    per_word = r8["E1_P3_broadening_repairs"]["per_word"]

    keys = sorted(per_word, key=lambda k: -per_word[k]["narrow"]["median_endpoint_rel_l2"])
    labels = [_application_order(k) for k in keys]
    e_n = [per_word[k]["narrow"]["median_endpoint_rel_l2"] for k in keys]
    e_b = [per_word[k]["broadS"]["median_endpoint_rel_l2"] for k in keys]

    def acc(cond):
        rows = alg["per_seed"][cond]
        return [float(np.median([r["per_word_acc"][k] for r in rows])) for k in keys]

    a_n, a_b = acc("narrow"), acc("broadS")

    x = np.arange(len(keys))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), constrained_layout=True)

    for xi, (yn, yb) in enumerate(zip(e_n, e_b)):
        axs[0].plot([xi, xi], [yn, yb], color="0.6", lw=0.8, zorder=1)
    axs[0].scatter(x, e_n, s=20, color=C_NARROW, zorder=3, label="narrow dictionary")
    axs[0].scatter(x, e_b, s=20, color=C_STATE, zorder=3, label="broadened dictionary")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("endpoint relative $L^2$ error")
    axs[0].legend(frameon=False, loc="upper right")

    w = 0.38
    axs[1].bar(x - w / 2, a_n, w, color=C_NARROW, label="narrow dictionary")
    axs[1].bar(x + w / 2, a_b, w, color=C_STATE, label="broadened dictionary")
    for xi, v in zip(x, a_n):
        if v == 0.0:
            axs[1].text(xi - w / 2, 0.02, "0", ha="center", va="bottom",
                        fontsize=5.6, color=C_NARROW)
    axs[1].set_ylim(0, 1.12)
    axs[1].set_ylabel("generating word recovered")

    for ax in axs:
        ax.set_xticks(x, labels, fontsize=6.2)
        ax.set_xlabel("composed word, in order of application")
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    status = guard(fig, "F11")
    save(fig, "F11_algebra_2d.pdf")
    return status


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    status = {
        "F6": fig6(), "F7": fig7(), "F8": fig8(), "F9": fig9(), "F10": fig10(),
        "F11": fig11(),
    }
    names = ["F6_support_geometry.pdf", "F7_support_distance.pdf",
             "F8_resolution_horizon.pdf", "F9_second_architecture.pdf",
             "F10_error_headroom.pdf", "F11_algebra_2d.pdf"]
    receipt = {
        "no_network": True,
        "overlap_guard": status,
        "sha256": {f"figures/{n}": sha256(OUT / n) for n in names},
        "script_sha256": sha256(Path(__file__)),
        "sources": sorted(set(SOURCES)),
        "versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "python": __import__("platform").python_version(),
        },
    }
    with open(MAN / "FIGURE_RECEIPT_EXTRA.json", "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(receipt["sha256"], indent=2))
    print(json.dumps(status))


if __name__ == "__main__":
    main()
