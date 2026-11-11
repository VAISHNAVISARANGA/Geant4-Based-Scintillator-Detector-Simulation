#!/usr/bin/env python3
"""
analyze.py - consolidated analysis and report generator for ScintillatorSimulator.

Reads   results/summary.json, results/events.csv (and, when present, results/photon_data.csv)
Writes  results/plots/*.png, results/analysis_metrics.json, results/report.html

Usage:  python3 analyze.py [--results results] [--tol-keV 1.0] [--no-archive]

Sections (matching the report):
  1 Gamma interaction performance     4 Photodetection performance
  2 Scintillation performance         5 Detector response
  3 Optical transport performance     6 Final material evaluation (no score is produced)

Every number is labelled as one of:  INPUT (from config.yaml, with its provenance status),
SIMULATED (counted in the Monte Carlo), CALCULATED (analytic / Geant4 cross-section table),
DERIVED (computed here from simulated numbers).  Nothing is fabricated: comparisons that need
several simulations (energy / thickness / linearity) appear only when such data really exist.
Dependencies: numpy, pandas, matplotlib (scipy is used for fitting if available).
"""
import argparse, base64, datetime, glob, html, io, json, math, os, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

FWHM_K = 2.0 * math.sqrt(2.0 * math.log(2.0))   # 2.3548
MIN_PEAK_EVENTS_REPORT = 30      # below this, resolution is NOT reported
MIN_PEAK_EVENTS_GOOD = 100       # below this, a low-statistics warning is attached

# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
METRICS = {}      # section -> list of dict(label, value, unit, kind, note)
PLOTS = {}        # section -> list of dict(file, title, caption)
NOTES = {}        # section -> list of text notes / warnings
OUT = {"results": "results", "plots": "results/plots"}


def add(sec, label, value, unit="", kind="SIMULATED", note="", err=None):
    METRICS.setdefault(sec, []).append(dict(label=label, value=value, unit=unit, kind=kind, note=note, err=err))


def note(sec, text):
    NOTES.setdefault(sec, []).append(text)


def fmt(v, err=None, digits=4):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        s = f"{int(v):,}"
    elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    else:
        v = float(v)
        s = f"{v:.{digits}g}" if (abs(v) >= 1e-3 or v == 0) else f"{v:.3e}"
        if abs(v) >= 1e5:
            s = f"{v:.4g}"
    if err is not None and not (isinstance(err, float) and (math.isnan(err) or math.isinf(err))):
        s += f" ± {float(err):.2g}"
    return s


def binom_err(k, n):
    if n <= 0:
        return float("nan")
    p = k / n
    return math.sqrt(max(p * (1 - p), 0) / n)


def savefig(fig, name, sec, title, caption):
    path = os.path.join(OUT["plots"], name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    PLOTS.setdefault(sec, []).append(dict(file=path, title=title, caption=caption))


def edges(h):
    return np.linspace(h["lo"], h["hi"], h["bins"] + 1)


def gauss_lin(x, a, mu, sg, b0, b1):
    return a * np.exp(-0.5 * ((x - mu) / sg) ** 2) + b0 + b1 * (x - mu)


# ----------------------------------------------------------------------------------------------
# photopeak finding / fitting (re-used for archived runs)
# ----------------------------------------------------------------------------------------------
def photopeak(signal, truth_full_mask, n_emitted, n_entered, label=""):
    """Locate and fit the full-energy peak of a signal spectrum (signal units = photoelectrons or fired cells).

    The peak is *seeded* from the truth-tagged full-energy events (location and rough width only);
    the Gaussian(+linear background) fit is then performed on ALL events in the window, so the
    continuum and tails are part of the measurement, as in a real spectrum.
    """
    res = dict(ok=False, reason="", n_truth=int(truth_full_mask.sum()))
    sig_t = signal[truth_full_mask & (signal > 0)]
    if len(sig_t) < MIN_PEAK_EVENTS_REPORT:
        res["reason"] = (f"only {len(sig_t)} full-energy (truth-tagged) events with signal; "
                         f"at least {MIN_PEAK_EVENTS_REPORT} are needed -> increase simulation.events")
        return res
    med = float(np.median(sig_t))
    mad = 1.4826 * float(np.median(np.abs(sig_t - med)))
    s0 = max(mad, 0.02 * med, 1.0)
    # truth-tagged cross-check (no background, exact selection)
    res["truth_mean"] = float(sig_t.mean()); res["truth_std"] = float(sig_t.std(ddof=1))
    res["truth_resolution_pct"] = FWHM_K * res["truth_std"] / res["truth_mean"] * 100.0
    d0 = sig_t - sig_t.mean()
    res["truth_skew"] = float((d0 ** 3).mean() / (d0.std() ** 3 + 1e-30))
    lo, hi = med - 4.0 * s0, med + 4.0 * s0
    w = signal[(signal >= lo) & (signal <= hi)]
    bw = max(s0 / 4.0, 1.0)
    nb = max(int(round((hi - lo) / bw)), 8)
    cnt, ed = np.histogram(w, bins=nb, range=(lo, hi))
    ctr = 0.5 * (ed[1:] + ed[:-1])
    mu = sg = a = b0 = b1 = err_mu = err_sg = None
    if HAVE_SCIPY:
        try:
            p0 = [max(cnt.max(), 1), med, s0, max(cnt[:2].mean(), 0.0), 0.0]
            popt, pcov = curve_fit(gauss_lin, ctr, cnt, p0=p0, sigma=np.sqrt(np.maximum(cnt, 1.0)), absolute_sigma=True, maxfev=20000)
            a, mu, sg, b0, b1 = popt; sg = abs(sg)
            perr = np.sqrt(np.diag(pcov)); err_mu, err_sg = float(perr[1]), float(perr[2])
        except Exception as ex:  # fall through to moments
            res["fit_warning"] = f"curve_fit failed ({ex}); truncated-moment estimate used"
    if mu is None:    # truncated moments, iterated (no scipy)
        mu, sg = med, s0
        for _ in range(20):
            sel = w[np.abs(w - mu) < 1.5 * sg]
            if len(sel) < 5:
                break
            mu, sg = float(sel.mean()), float(sel.std()) / 0.7 ** 1.0   # ~0.7 = truncation factor at +-1.5 sigma
        a = float(cnt.max()); b0 = b1 = 0.0
        res["fit_warning"] = res.get("fit_warning", "scipy not available; moment-based estimate (no background model, no uncertainty)")
    # ---- fit-quality gate: reject physically implausible or inconsistent fits ----------------------
    gate = []
    if not (lo <= mu <= hi): gate.append("fitted position outside the fit window")
    if sg <= 0 or sg / max(mu, 1e-9) > 0.6 or sg / max(mu, 1e-9) < 0.003: gate.append("fitted width implausible")
    if err_mu is not None and err_mu / max(mu, 1e-9) > 0.1: gate.append("position uncertainty > 10 %")
    if abs(mu - res["truth_mean"]) > res["truth_std"]: gate.append("fitted position disagrees with the full-energy events by more than 1 sigma")
    res["method"] = "Gaussian + linear background fit to the full spectrum"
    if gate:
        res["fit_gate_failed"] = "; ".join(gate)
        res["method"] = "truth-tagged full-energy events (Gaussian fit failed the quality gate: " + res["fit_gate_failed"] + ")"
        mu, sg, err_mu, err_sg = res["truth_mean"], res["truth_std"], res["truth_std"] / math.sqrt(len(sig_t)), None
        a = b0 = b1 = None
    fwhm = FWHM_K * sg
    res.update(ok=True, mu=float(mu), sigma=float(sg), fwhm=float(fwhm), mu_err=err_mu, sigma_err=err_sg,
               resolution_pct=float(fwhm / mu * 100.0), fit_window=(float(lo), float(hi)),
               fit_hist=(ctr.tolist(), cnt.tolist()), fit_params=None if a is None else [float(a), float(mu), float(sg), float(b0), float(b1)])
    if err_mu is not None and err_sg is not None and sg > 0:
        res["resolution_err_pct"] = float(res["resolution_pct"] * math.sqrt((err_sg / sg) ** 2 + (err_mu / mu) ** 2))
    # efficiency: events within mu +- FWHM minus fitted linear background
    lo2, hi2 = mu - fwhm, mu + fwhm
    raw = int(((signal >= lo2) & (signal <= hi2)).sum())
    bkg = 0.0
    if b0 is not None and a is not None and HAVE_SCIPY and err_mu is not None:
        xs = np.linspace(lo2, hi2, 400)
        bkg_density = np.clip(b0 + b1 * (xs - mu), 0, None)          # counts per fit-bin
        _trap = getattr(np, 'trapezoid', None) or np.trapz   # NumPy 2 renamed trapz
        bkg = float(_trap(bkg_density, xs) / bw)
    net = max(raw - bkg, 0.0)
    res.update(window=(float(lo2), float(hi2)), raw_counts=raw, bkg_counts=bkg, net_counts=net,
               eff_abs=net / n_emitted if n_emitted else float("nan"),
               eff_abs_err=math.sqrt(max(raw, 1)) / n_emitted if n_emitted else float("nan"),
               eff_int=net / n_entered if n_entered else float("nan"))
    res["warning_low_stats"] = bool(len(sig_t) < MIN_PEAK_EVENTS_GOOD)
    return res


# ----------------------------------------------------------------------------------------------
# archive handling: other runs of the SAME material/geometry/detector at different gamma energies
# ----------------------------------------------------------------------------------------------
def canonical_config(S):
    c = json.loads(json.dumps(S["config"]))
    c.get("source", {}).pop("energy", None)
    c.get("simulation", {}).pop("events", None); c.get("simulation", {}).pop("seed", None)
    c.pop("output", None); c.pop("visualization", None)
    return json.dumps(c, sort_keys=True)


def load_archive(S, results_dir):
    runs = []
    key = canonical_config(S)
    for d in sorted(glob.glob(os.path.join(results_dir, "archive", "*"))):
        sj, ej = os.path.join(d, "summary.json"), os.path.join(d, "events.csv")
        if not (os.path.exists(sj) and os.path.exists(ej)):
            continue
        try:
            S2 = json.load(open(sj))
            same_run = (S2["source"]["energy_keV"] == S["source"]["energy_keV"] and S2["run"]["seed"] == S["run"]["seed"] and S2["run"]["events"] == S["run"]["events"])
            if same_run:
                continue       # the current run itself (archived copy) - do not count it twice
            if canonical_config(S2) == key and S2["run"]["mode_effective"] == S["run"]["mode_effective"]:
                runs.append((os.path.basename(d), S2, pd.read_csv(ej)))
        except Exception:
            continue
    return runs


# ----------------------------------------------------------------------------------------------
# SECTION 1 - gamma interaction performance
# ----------------------------------------------------------------------------------------------
def section1(S, ev, tol, archive):
    sec = "s1"
    Eg = S["source"]["energy_keV"]; N = len(ev)
    entered = S["totals"]["events_primary_entered_crystal"]
    inter = int(ev.interacted.sum()); dep = ev.energy_deposited_keV > 0
    nd = int(dep.sum())
    full = ev.energy_deposited_keV >= Eg - tol
    nfull = int((full & dep).sum()); npart = int((dep & ~full).sum())
    add(sec, "Gamma events simulated", N, kind="SIMULATED")
    add(sec, "Events whose primary entered the crystal", entered, kind="SIMULATED")
    add(sec, "Events with a gamma interaction in the crystal", inter, kind="SIMULATED", note="any process other than transport (photoelectric, Compton, Rayleigh, pair)")
    add(sec, "Interaction fraction (per emitted gamma)", inter / N, kind="SIMULATED", err=binom_err(inter, N))
    if entered:
        add(sec, "Interaction fraction (per gamma entering the crystal)", inter / entered, kind="DERIVED", err=binom_err(inter, entered))
    add(sec, "Fraction of events depositing energy", nd / N, kind="SIMULATED", err=binom_err(nd, N),
        note="Rayleigh-only events interact without depositing energy")
    add(sec, "Mean deposited energy, all events", float(ev.energy_deposited_keV.mean()), "keV", "DERIVED", err=float(ev.energy_deposited_keV.std(ddof=1) / math.sqrt(N)) if N > 1 else None)
    if nd:
        add(sec, "Mean deposited energy, events depositing energy", float(ev.energy_deposited_keV[dep].mean()), "keV", "DERIVED")
    add(sec, f"Full-energy deposition fraction (E_dep ≥ E_γ − {tol:g} keV, per emitted gamma)", nfull / N, kind="SIMULATED", err=binom_err(nfull, N),
        note="Truth quantity from the simulation (not a detector observable)")
    add(sec, "Partial-energy deposition fraction (0 < E_dep < E_γ − tol, per emitted gamma)", npart / N, kind="SIMULATED", err=binom_err(npart, N))
    if nd:
        add(sec, "Full-energy fraction among events depositing energy", nfull / nd, kind="DERIVED", err=binom_err(nfull, nd))
    esc = ev.energy_escaped_net_keV[dep]
    if len(esc):
        add(sec, "Mean energy escaping the crystal, events that deposited energy", float(esc.mean()), "keV", "SIMULATED",
            note="net kinetic energy of particles leaving the crystal (tracked, not inferred); gammas that cross without interacting are excluded")
        add(sec, "Fraction of depositing events with escaping energy > tol (partial absorption)", float((esc > tol).mean()), kind="SIMULATED")
    add(sec, "Max |energy closure| (entering − deposited − escaped)", float(ev.energy_closure_keV.abs().max()), "keV", "SIMULATED",
        note="bookkeeping self-check; should be ~0 (positron escape before annihilation would show here)")
    fp = S["totals"]["first_interaction_process_counts"]
    add(sec, "First interaction process counts", ", ".join(f"{k}: {v}" for k, v in sorted(fp.items()) if k != "none"), kind="SIMULATED",
        note="phot = photoelectric, compt = Compton, Rayl = Rayleigh, conv = pair production")
    g = S["gamma_calculations"]
    add(sec, "Total attenuation length at E_γ", g["total_attenuation_length_mm"], "mm", "CALCULATED", note="Geant4 cross-section tables (G4EmCalculator)")
    for k, lbl in [("mu_photoelectric_per_mm", "photoelectric"), ("mu_compton_per_mm", "Compton"), ("mu_rayleigh_per_mm", "Rayleigh"), ("mu_pair_per_mm", "pair production")]:
        add(sec, f"Linear attenuation coefficient, {lbl}", g[k], "1/mm", "CALCULATED")
    if g.get("expected_interaction_probability_pencil_beam") is not None:
        ex = g["expected_interaction_probability_pencil_beam"]; sim = inter / N
        z = (sim - ex) / max(binom_err(inter, N), 1e-12)
        add(sec, f"Expected interaction probability, pencil beam through {g['pencil_beam_chord_length_mm']:.4g} mm", ex, kind="CALCULATED",
            note=f"1 − exp(−L/λ). Simulation differs by {z:+.1f}σ")
    # --- plots ---
    # 1. deposited energy
    fig, ax = plt.subplots(figsize=(7, 4.2))
    e = ev.energy_deposited_keV[dep]
    if len(e):
        bw = max(Eg / 250.0, 0.5)
        ax.hist(e, bins=np.arange(0, Eg + 4 * bw, bw), color="#3b7ddd", histtype="stepfilled", alpha=0.8)
        ax.set_yscale("log")
    ax.axvline(Eg, color="k", ls="--", lw=0.8, label=f"E_γ = {Eg:g} keV")
    ce = Eg - Eg / (1 + 2 * Eg / 510.999)
    ax.axvline(ce, color="r", ls=":", lw=0.9, label=f"Compton edge (calc.) {ce:.0f} keV")
    ax.set_xlabel("Deposited energy in crystal [keV]"); ax.set_ylabel("Events per bin")
    ax.set_title("Deposited-energy spectrum (simulation truth, no detector response)")
    ax.legend(); savefig(fig, "s1_deposited_energy.png", sec, "Deposited-energy histogram",
                         "True energy deposited in the crystal per event (events with E_dep > 0). This is Geant4 truth; the detector-response spectrum is in Section 5.")
    # 2. spatial maps
    H = S["histograms"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, (xl, yl) in zip(axs, ["edep_map_xy_keV", "edep_map_xz_keV", "edep_map_yz_keV"], [("x", "y"), ("x", "z"), ("y", "z")]):
        h = H[key]; c = np.array(h["counts"]).T
        im = ax.imshow(c, origin="lower", extent=[h["xlo"], h["xhi"], h["ylo"], h["yhi"]], aspect="equal", cmap="viridis")
        ax.set_xlabel(f"{xl} [mm]"); ax.set_ylabel(f"{yl} [mm]"); ax.set_title(f"Energy deposit, {xl}{yl} projection")
        fig.colorbar(im, ax=ax, label="keV (summed)", fraction=0.046)
    fig.tight_layout(); savefig(fig, "s1_edep_spatial.png", sec, "Energy-deposition spatial distribution",
                                "Total deposited energy summed over all events, projected on the three crystal planes (50×50 bins).")
    # 3. interaction probability vs energy (only with several real runs)
    runs = [(S, ev)] + [(S2, e2) for _, S2, e2 in archive]
    en = sorted({round(r[0]["source"]["energy_keV"], 6) for r in runs})
    if len(en) >= 2:
        pts = {}
        for S2, e2 in runs:
            pts.setdefault(round(S2["source"]["energy_keV"], 6), []).append((int(e2.interacted.sum()), len(e2), S2["gamma_calculations"].get("expected_interaction_probability_pencil_beam")))
        xs = sorted(pts); k = np.array([sum(p[0] for p in pts[x]) for x in xs]); n = np.array([sum(p[1] for p in pts[x]) for x in xs])
        ex = [pts[x][0][2] for x in xs]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.errorbar(xs, k / n, yerr=[binom_err(a, b) for a, b in zip(k, n)], fmt="o", label="simulated (Monte Carlo)")
        if all(v is not None for v in ex):
            ax.plot(xs, ex, "s--", mfc="none", label="calculated (Geant4 cross sections)")
        ax.set_xlabel("Gamma energy [keV]"); ax.set_ylabel("Interaction probability"); ax.legend(); ax.set_title("Interaction probability vs gamma energy")
        savefig(fig, "s1_interaction_vs_energy.png", sec, "Interaction probability vs gamma energy", f"Built from {len(xs)} real simulation runs ({', '.join(f'{x:g}' for x in xs)} keV) of identical configuration.")
    else:
        note(sec, "Interaction probability vs gamma energy: NOT shown — only one gamma energy has been simulated. Run ./run_all.sh --archive <tag> at other energies.")
    # 4. thickness: calculation only
    if g.get("pencil_beam_chord_length_mm") is not None and g["total_attenuation_length_mm"] > 0:
        L = g["pencil_beam_chord_length_mm"]; lam = g["total_attenuation_length_mm"]
        t = np.linspace(0, max(3 * L, 4 * lam * 0.7), 200)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(t, 1 - np.exp(-t / lam), label="calculated: 1 − exp(−t/λ), λ from Geant4 tables")
        ax.errorbar([L], [inter / N], yerr=[binom_err(inter, N)], fmt="o", color="C3", label="this simulation (pencil beam)")
        ax.set_xlabel("Path length / thickness along beam [mm]"); ax.set_ylabel("Interaction probability"); ax.legend(); ax.set_title("Thickness dependence (calculation, not a sweep)")
        savefig(fig, "s1_thickness_calc.png", sec, "Thickness-dependent interaction probability (calculated)",
                "A Beer–Lambert CALCULATION using the Geant4 attenuation length; no thickness-sweep simulation was performed (by design). The red point is this run.")
    else:
        note(sec, "Thickness-dependent performance: NOT shown (needs a pencil-beam source for a valid path-length calculation, or thickness-dependent simulations).")
    return dict(Eg=Eg, N=N, entered=entered, full=full & dep, dep=dep)


# ----------------------------------------------------------------------------------------------
# SECTION 2 - scintillation performance
# ----------------------------------------------------------------------------------------------
def section2(S, ev, ctx):
    sec = "s2"
    si = S["scintillation_inputs"]; Eg = ctx["Eg"]; dep = ctx["dep"]; full = ctx["full"]
    ly = si["light_yield_photons_per_MeV"]
    add(sec, "Configured light yield", ly, "photons/MeV", f"INPUT ({si['light_yield_status']})", note="Material property supplied by the user")
    add(sec, "Scintillation yield fluctuation scale", si["resolution_scale"], "", "INPUT", note="1 = Poisson")
    add(sec, "Emission peak wavelength", si["emission_peak_nm"], "nm", "INPUT")
    add(sec, "Emission-spectrum FWHM", si["emission_fwhm_nm"] if si["emission_fwhm_nm"] is not None else "not supplied", "nm" if si["emission_fwhm_nm"] is not None else "", "INPUT",
        note="spectral width of the emission band — NOT the gamma photopeak FWHM")
    add(sec, "Spectrum model used", si["spectrum_mode"], "", "INPUT")
    add(sec, "Rise time", si["rise_time_ns"], "ns", "INPUT")
    for i, c in enumerate(si["decay_components"], 1):
        add(sec, f"Decay component {i}", f"{c['decay_time_ns']:g} ns, fraction {c['fraction']:g}", "", "INPUT")
    ng = ev.n_scint_photons.values; ed = ev.energy_deposited_keV.values
    nd = dep.values
    add(sec, "Generated scintillation photons, total", int(ng.sum()), kind="SIMULATED")
    add(sec, "Mean generated photons per event (all events)", float(ng.mean()), kind="SIMULATED", err=float(ng.std(ddof=1) / math.sqrt(len(ng))) if len(ng) > 1 else None)
    if nd.any():
        add(sec, "Mean generated photons per depositing event", float(ng[nd].mean()), kind="DERIVED")
        eff_ly = ng.sum() / (ed.sum() / 1000.0)
        add(sec, "Generated photons per MeV actually deposited", float(eff_ly), "photons/MeV", "DERIVED",
            note=f"ratio to configured yield = {eff_ly / ly:.4f}. Configured yield (input) and generated photons per event (output) are different quantities.")
    fm = (full & dep).values
    if fm.sum() >= 5:
        m, v = ng[fm].mean(), ng[fm].var(ddof=1)
        add(sec, f"Full-energy events: mean generated photons", float(m), kind="DERIVED")
        add(sec, "Full-energy events: variance / mean of generated photons", float(v / m), kind="SIMULATED",
            note=f"expected ≈ resolution_scale² = {si['resolution_scale'] ** 2:g} (Poisson/Gaussian yield statistics; small extra variance from E_dep spread within tolerance)")
    H = S["histograms"]
    # plot 1 emission spectrum
    sp = si["emission_spectrum_used"]; wl = np.array(sp["wavelength_nm"]); I = np.array(sp["relative_intensity_per_wavelength"])
    hw = H["generated_wavelength_nm"]; ew = edges(hw); cw = 0.5 * (ew[1:] + ew[:-1]); cc = np.array(hw["counts"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if cc.sum() > 0:
        ax.bar(cw, cc / cc.max(), width=np.diff(ew), color="#9ecae1", label="SIMULATED: generated photons (histogram)")
    ax.plot(wl, I, "k-", lw=1.6, label=f"INPUT: emission spectrum ({si['spectrum_mode']})")
    ax.axvline(si["emission_peak_nm"], color="r", ls=":", label=f"INPUT peak {si['emission_peak_nm']:g} nm")
    ax.set_xlabel("Wavelength [nm]"); ax.set_ylabel("Relative intensity (dN/dλ, peak = 1)"); ax.legend(fontsize=8); ax.set_title("Scintillation emission spectrum")
    savefig(fig, "s2_emission_spectrum.png", sec, "Emission spectrum vs wavelength", "Black: input spectrum. Blue bars: wavelengths of the photons actually generated (checks the sampling).")
    # plot 2 generated vs deposited
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    idx = np.where(nd)[0]
    if len(idx) > 6000:
        idx = np.random.default_rng(1).choice(idx, 6000, replace=False)
    ax.scatter(ed[idx], ng[idx], s=5, alpha=0.5, label="SIMULATED events")
    xx = np.linspace(0, max(ed.max(), 1), 50); ax.plot(xx, ly * xx / 1000.0, "r-", lw=1, label="INPUT: light yield × E_dep")
    ax.set_xlabel("Deposited energy [keV]"); ax.set_ylabel("Generated scintillation photons"); ax.legend(); ax.set_title("Generated photons vs deposited energy")
    savefig(fig, "s2_generated_vs_edep.png", sec, "Generated photons versus deposited energy", "Each point is one event. The red line is the configured yield; scatter around it comes from the stochastic yield model.")
    # plot 3 emission time
    ht = H["emission_time_ns"]; et = edges(ht); ct = np.array(ht["counts"]); cen = 0.5 * (et[1:] + et[:-1])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.step(cen, ct, where="mid", color="#3b7ddd", label="SIMULATED emission times")
    comps = si["decay_components"]; tr = si["rise_time_ns"]; tot = ct.sum()
    if tot > 0:
        pdf = np.zeros_like(cen)
        for c in comps:
            td = c["decay_time_ns"]
            pdf += c["fraction"] * ((np.exp(-cen / td) - np.exp(-cen / tr)) / (td - tr) if tr > 0 and abs(td - tr) > 1e-12 else np.exp(-cen / td) / td)
        ax.plot(cen, pdf * tot * (et[1] - et[0]) / max(ht["entries"] - ht["overflow"] - ht["underflow"], 1) * (tot / max(tot, 1)), "r--", label="INPUT-based expectation")
    ax.set_yscale("log"); ax.set_ylim(bottom=max(ct[ct > 0].min() * 0.5, 0.5) if (ct > 0).any() else 0.5)
    ax.set_xlabel("Emission time after first energy deposit [ns]"); ax.set_ylabel("Photons per bin"); ax.legend(); ax.set_title("Photon emission-time distribution")
    savefig(fig, "s2_emission_time.png", sec, "Photon emission time distribution",
            f"Dashed: expected shape from the configured rise time ({tr:g} ns) and decay component(s); overflow beyond the plotted range: {int(ht['overflow'])} photons.")
    # plot 4 generated count distribution
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if nd.any():
        bins = np.linspace(0, ng.max() * 1.02 + 1, 80)
        ax.hist(ng[nd], bins=bins, color="#9ecae1", label="all depositing events")
        if fm.sum() >= 5:
            ax.hist(ng[fm], bins=bins, color="#08519c", label="full-energy events (truth)")
    ax.set_xlabel("Generated photons per event"); ax.set_ylabel("Events"); ax.legend(); ax.set_title("Generated-photon-count distribution")
    savefig(fig, "s2_generated_count.png", sec, "Generated photon count distribution", "Spread of the full-energy peak reflects the yield statistics only (before optical transport and photodetection).")
    # photon-log reconciliation
    note(sec, "Light yield is a configured INPUT. The generated photon number per event is an OUTPUT of the stochastic model and equals yield × deposited energy only on average.")


# ----------------------------------------------------------------------------------------------
# SECTION 3 - optical transport
# ----------------------------------------------------------------------------------------------
def section3(S, ev, ctx, results_dir):
    sec = "s3"
    o = S["totals"]["optical"]; T = o["terminal"]
    gen_s = o["generated_scintillation_photons"]; gen_all = o["total_optical_tracks"]
    add(sec, "Generated scintillation photons", gen_s, kind="SIMULATED")
    if o["generated_wls_photons"]:
        add(sec, "Photons generated by wavelength shifting (secondary tracks)", o["generated_wls_photons"], kind="SIMULATED")
    add(sec, "Photons transported to at least one surface", o["photons_reaching_a_boundary"], kind="SIMULATED", note=f"{o['photons_reaching_a_boundary'] / gen_all:.4f} of all optical tracks")
    def fr(k): return T[k] / gen_all
    add(sec, "Bulk-absorbed photons", T["bulk_absorbed"], kind="SIMULATED", note=f"fraction {fr('bulk_absorbed'):.4f}")
    add(sec, "Surface-absorbed photons", T["surface_absorbed"], kind="SIMULATED", note=f"fraction {fr('surface_absorbed'):.4f} (reflector loss = 1 − reflectivity)")
    add(sec, "Escaped photons", T["escaped"], kind="SIMULATED", note=f"fraction {fr('escaped'):.4f}")
    if T["wls_converted"]:
        add(sec, "Photons absorbed by wavelength shifter", T["wls_converted"], kind="SIMULATED")
    add(sec, "Photons reaching (entering) the sensor", o["photons_incident_on_sensor"], kind="SIMULATED", note=f"fraction {o['photons_incident_on_sensor'] / gen_all:.4f}")
    add(sec, "Unclassified terminal fates", T["unclassified"], kind="SIMULATED", note="must be 0; non-zero indicates a configuration or physics problem")
    add(sec, "Sum of terminal classes − optical tracks", int(sum(T.values()) - gen_all), kind="SIMULATED", note="bookkeeping check; must be 0 (each photon has exactly one fate)")
    oce = o["photons_incident_on_sensor"] / gen_s if gen_s else float("nan")
    add(sec, "Optical collection efficiency (photons reaching sensor / generated)", oce, kind="DERIVED", err=binom_err(o["photons_incident_on_sensor"], gen_s) if oce <= 1 else None,
        note="before photodetection efficiency; excludes Fresnel-reflected photons that did not enter the sensor window")
    add(sec, "Rayleigh scatters", o["rayleigh_scatters"], kind="SIMULATED"); add(sec, "Mie scatters", o["mie_scatters"], kind="SIMULATED")
    add(sec, "Boundary reflections (all types)", o["boundary_reflections"], kind="SIMULATED")
    # per-event optical collection variation
    m = ev.n_scint_photons > 0
    ocev = (ev.n_sensor_incident[m] / ev.n_scint_photons[m]).values
    fm = (ctx["full"] & ctx["dep"])[m].values
    if fm.sum() >= 5:
        add(sec, "Per-event collection efficiency, full-energy events: mean", float(ocev[fm].mean()), kind="DERIVED")
        add(sec, "Per-event collection efficiency, full-energy events: relative std", float(ocev[fm].std(ddof=1) / ocev[fm].mean()), kind="DERIVED",
            note="position-dependent light collection — one source of energy-resolution broadening")
    # plot 1 bars
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labs = ["generated\n(all tracks)", "bulk\nabsorbed", "surface\nabsorbed", "escaped", "wls\nconverted", "reached\nsensor"]
    vals = [gen_all, T["bulk_absorbed"], T["surface_absorbed"], T["escaped"], T["wls_converted"], o["photons_incident_on_sensor"]]
    b = ax.bar(labs, vals, color=["#444", "#d95f02", "#e6ab02", "#7570b3", "#999", "#1b9e77"])
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:,}\n{v / gen_all:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.2); ax.set_ylabel("Optical photons"); ax.set_title("Fate of optical photons (SIMULATED)")
    savefig(fig, "s3_photon_fates.png", sec, "Generated vs absorbed vs escaped vs sensor-reaching photons", "Terminal classes are mutually exclusive; bulk + surface + escaped + wls + sensor-reaching (detected or not) add up to the generated tracks.")
    # plot 2 arrival time
    h = S["histograms"]["sensor_arrival_time_ns"]; e = edges(h); c = np.array(h["counts"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.step(0.5 * (e[1:] + e[:-1]), c, where="mid", color="#1b9e77"); ax.set_yscale("log")
    if (c > 0).any(): ax.set_ylim(bottom=0.5)
    ax.set_xlabel("Arrival time at sensor after first energy deposit [ns]"); ax.set_ylabel("Photons per bin"); ax.set_title("Photon arrival-time distribution")
    savefig(fig, "s3_arrival_time.png", sec, "Photon arrival-time distribution", f"Photons entering the sensor window (before PDE). Overflow beyond plot range: {int(h['overflow'])}. Includes emission delay + optical transit.")
    # plot 3 face-wise losses + collection vs depth
    fl = o["face_loss"]; faces = fl["faces"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(faces)); w = 0.38
    axs[0].bar(x - w / 2, fl["surface_absorbed"], w, label="surface absorbed", color="#e6ab02"); axs[0].bar(x + w / 2, fl["escaped"], w, label="escaped", color="#7570b3")
    axs[0].set_xticks(x); axs[0].set_xticklabels(faces, rotation=20); axs[0].set_ylabel("Photons"); axs[0].set_title(f"Face-wise optical loss (sensor on {S['geometry']['sensor_face']})"); axs[0].legend()
    face = S["geometry"]["sensor_face"]; ax_i = "xyz".index(face[1]); sgn = 1 if face[0] == "+" else -1
    coord = ev[["edep_centroid_x_mm", "edep_centroid_y_mm", "edep_centroid_z_mm"]].values[:, ax_i]
    half = S["geometry"]["dimensions_mm"][ax_i] / 2
    depth = half - sgn * coord      # distance of the energy-deposit centroid from the sensor-side face
    ok = ((ev.n_scint_photons > 0).values) & np.isfinite(depth)
    if ok.sum() > 10:
        d = depth[ok]; q = (ev.n_sensor_incident.values[ok] / ev.n_scint_photons.values[ok])
        bins = np.linspace(0, 2 * half, 9); ix = np.digitize(d, bins) - 1
        mx, my, me = [], [], []
        for k in range(8):
            s = q[ix == k]
            if len(s) >= 3: mx.append(0.5 * (bins[k] + bins[k + 1])); my.append(s.mean()); me.append(s.std(ddof=1) / math.sqrt(len(s)))
        axs[1].errorbar(mx, my, yerr=me, fmt="o-"); axs[1].set_xlabel("Distance of energy deposit from sensor-side face [mm]"); axs[1].set_ylabel("Collection efficiency"); axs[1].set_title("Light collection vs interaction depth")
    else:
        axs[1].text(0.5, 0.5, "too few events", ha="center"); axs[1].axis("off")
    savefig(fig, "s3_face_losses.png", sec, "Face-wise optical losses and collection vs depth",
            "Left: where photons were lost (the face nearest to the loss point; 'coupling_edge' = lateral edge of the optical coupling layer). Right: position dependence of light collection.")
    # photon-log reconciliation
    pf = os.path.join(results_dir, "photon_data.csv")
    if S["photon_log"]["mode"] == "all" and not S["photon_log"]["truncated"] and os.path.exists(pf):
        ph = pd.read_csv(pf, usecols=["terminal_class"])
        vc = ph.terminal_class.value_counts().to_dict()
        ok = len(ph) == gen_all and all(vc.get(k, 0) == v for k, v in T.items() if v)
        add(sec, "photon_data.csv reconciles with summary.json (every class count equal)", "yes" if ok else "NO — inconsistent", kind="DERIVED")
    else:
        note(sec, f"photon_data.csv reconciliation skipped (photon_log = {S['photon_log']['mode']}, truncated = {S['photon_log']['truncated']}).")
    return oce


# ----------------------------------------------------------------------------------------------
# SECTION 4 - photodetection
# ----------------------------------------------------------------------------------------------
def section4(S, ev, ctx, results_dir):
    sec = "s4"
    d = S["detector_inputs"]; o = S["totals"]["optical"]; sipm = d["type"] == "SiPM"
    pde = d["pde"]
    add(sec, "Sensor type", d["type"], kind="INPUT")
    add(sec, "Configured PDE", pde["constant"] if pde["constant"] is not None else "wavelength table (%d points)" % len(pde["table_values"]), "", f"INPUT ({pde['status']})")
    inc, det = o["photons_incident_on_sensor"], o["photons_detected"]
    add(sec, "Photons entering the sensor window", inc, kind="SIMULATED")
    add(sec, "Photons detected (photoelectrons)", det, kind="SIMULATED")
    if inc:
        add(sec, "Detection efficiency (detected / entering)", det / inc, kind="DERIVED", err=binom_err(det, inc), note="should reproduce the PDE averaged over the arriving spectrum")
    pe = ev.n_detected_photoelectrons.values; dep = ctx["dep"].values; fm = (ctx["full"] & ctx["dep"]).values
    add(sec, "Mean photoelectrons per event (all events)", float(pe.mean()), kind="DERIVED", err=float(pe.std(ddof=1) / math.sqrt(len(pe))) if len(pe) > 1 else None)
    if dep.any(): add(sec, "Mean photoelectrons per depositing event", float(pe[dep].mean()), kind="DERIVED")
    if fm.sum() >= 3:
        add(sec, "Mean photoelectrons, full-energy events (truth-tagged)", float(pe[fm].mean()), kind="DERIVED", err=float(pe[fm].std(ddof=1) / math.sqrt(fm.sum())))
        add(sec, "Photoelectron yield at E_γ (full-energy events)", float(pe[fm].mean() / ctx["Eg"]), "PE/keV", "DERIVED")
    if S["photon_log"]["mode"] != "none" and os.path.exists(os.path.join(results_dir, "photon_data.csv")) and pde["constant"] is None:
        ph = pd.read_csv(os.path.join(results_dir, "photon_data.csv"), usecols=["terminal_class", "wavelength_nm"])
        w = ph.wavelength_nm[ph.terminal_class.isin(["sensor_detected", "sensor_not_detected"])]
        if len(w):
            add(sec, "PDE averaged over photons that reached the sensor", float(np.interp(w, d["pde"]["table_wavelength_nm"], d["pde"]["table_values"]).mean()), kind="CALCULATED", note="compare with the detection efficiency above")
    if sipm:
        add(sec, "SiPM microcell saturation modelled", "yes" if d.get("model_saturation") else "no", kind="INPUT")
        if d.get("model_saturation"):
            add(sec, "Microcell count", d["microcell_count"], kind="INPUT"); add(sec, "Recovery time", d["recovery_time_ns"] if d.get("model_recovery") else "not modelled", "ns" if d.get("model_recovery") else "", "INPUT")
            lost = o["photons_lost_to_saturation"]
            add(sec, "Photoelectrons lost to saturation/dead cells", lost, kind="SIMULATED", note=f"{lost / det:.3%} of detected photons" if det else "")
        add(sec, "Microcells fired (SiPM signal)", o["microcells_fired"], kind="SIMULATED")
        add(sec, "Mean avalanche charge per event", float(ev.charge_electrons.mean()), "electrons", "DERIVED", note="fired cells × configured gain (no gain fluctuation modelled)")
    else:
        add(sec, "Mean charge per event", float(ev.charge_electrons.mean()) if ev.charge_electrons.notna().any() else "not modelled (PMT gain not supplied)", "electrons" if ev.charge_electrons.notna().any() else "", "DERIVED")
    tf = ev.t_first_detection_ns.dropna()
    if len(tf):
        add(sec, "Mean time of first detected photon", float(tf.mean()), "ns", "SIMULATED", note="relative to first energy deposit; no single-photon timing jitter modelled")
        add(sec, "Median time of first detected photon", float(tf.median()), "ns", "SIMULATED")
    hd = S["histograms"]["detection_time_ns"]
    # plots
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    m = ev.n_sensor_incident > 0
    ax.scatter(ev.n_sensor_incident[m], ev.n_detected_photoelectrons[m], s=5, alpha=0.5, label="SIMULATED events")
    xx = np.array([0, max(ev.n_sensor_incident.max(), 1)]); ax.plot(xx, xx * (det / inc if inc else 0), "r-", lw=1, label=f"mean ratio {det / inc:.3f}" if inc else "")
    ax.set_xlabel("Photons entering the sensor per event"); ax.set_ylabel("Photons detected per event"); ax.legend(); ax.set_title("Sensor-incident vs detected photons")
    savefig(fig, "s4_incident_vs_detected.png", sec, "Sensor-incident versus detected photons", "Binomial thinning by the PDE: each photon entering the sensor window is detected with probability PDE(λ).")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if dep.any(): ax.hist(pe[dep], bins=80, color="#9ecae1", label="depositing events"); 
    if fm.sum() >= 3: ax.hist(pe[fm], bins=40, color="#08519c", label="full-energy (truth)")
    ax.set_xlabel("Photoelectrons per event"); ax.set_ylabel("Events"); ax.legend(); ax.set_title("Photoelectron distribution")
    savefig(fig, "s4_pe_histogram.png", sec, "Photoelectron histogram", "Photoelectron count per event (before SiPM saturation).")
    e = edges(hd); c = np.array(hd["counts"])
    fig, ax = plt.subplots(figsize=(7, 4.2)); ax.step(0.5 * (e[1:] + e[:-1]), c, where="mid", color="#d95f02"); ax.set_yscale("log")
    if (c > 0).any(): ax.set_ylim(bottom=0.5)
    ax.set_xlabel("Detection time after first energy deposit [ns]"); ax.set_ylabel("Detected photons per bin"); ax.set_title("Detection-time distribution")
    savefig(fig, "s4_detection_time.png", sec, "Detection-time distribution", f"Detected photons only. Overflow beyond range: {int(hd['overflow'])}. Single-photon time jitter is not modelled.")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    idx = np.where(dep)[0]
    if len(idx) > 6000: idx = np.random.default_rng(2).choice(idx, 6000, replace=False)
    ax.scatter(ev.energy_deposited_keV.values[idx], pe[idx], s=5, alpha=0.5, label="photoelectrons")
    if sipm and d.get("model_saturation"): ax.scatter(ev.energy_deposited_keV.values[idx], ev.n_cells_fired_or_signal.values[idx], s=5, alpha=0.5, label="fired microcells (after saturation)")
    ax.set_xlabel("Deposited energy [keV]"); ax.set_ylabel("Signal"); ax.legend(); ax.set_title("Photoelectrons vs deposited energy")
    savefig(fig, "s4_pe_vs_edep.png", sec, "Photoelectrons versus deposited energy", "Each point is one event.")


# ----------------------------------------------------------------------------------------------
# SECTION 5 - detector response
# ----------------------------------------------------------------------------------------------
def section5(S, ev, ctx, archive):
    sec = "s5"
    Eg = ctx["Eg"]; N = ctx["N"]
    sig = ev.n_cells_fired_or_signal.values.astype(float)
    unit = "fired microcells" if S["detector_inputs"]["type"] == "SiPM" and S["detector_inputs"].get("model_saturation") else "photoelectrons"
    fm = (ctx["full"] & ctx["dep"]).values
    r = photopeak(sig, fm, N, ctx["entered"])
    result = dict(fit=r, unit=unit)
    if not r["ok"]:
        add(sec, "Photopeak analysis", "NOT DETERMINED", kind="DERIVED", note=r["reason"])
        note(sec, "Energy resolution is NOT reported: " + r["reason"])
        return result
    k = Eg / r["mu"]
    add(sec, "Peak determination method", r["method"], kind="DERIVED")
    if r.get("fit_gate_failed"):
        note(sec, "The Gaussian fit to the full spectrum was REJECTED by the quality gate (" + r["fit_gate_failed"] + "). Peak position and width below come from truth-tagged full-energy events instead; this is optimistic because real spectra contain background and unresolved tails. Increase simulation.events for a reliable fit.")
    add(sec, f"Photopeak position (signal)", r["mu"], unit, "DERIVED", err=r["mu_err"], note="Gaussian + linear-background fit to the signal spectrum")
    add(sec, "Energy calibration (single-point at E_γ)", k, "keV per " + ("cell" if unit.startswith("fired") else "PE"), "DERIVED", note="E_reco = signal × this factor; the photopeak is placed at E_γ by construction, so only the width is informative")
    add(sec, "Photopeak position (reconstructed)", Eg, "keV", "DERIVED", note="equals the calibration energy by construction")
    add(sec, "Photopeak σ (signal)", r["sigma"], unit, "DERIVED", err=r["sigma_err"])
    add(sec, "Gamma photopeak FWHM", r["fwhm"] * k, "keV", "DERIVED", err=(r["sigma_err"] * FWHM_K * k) if r["sigma_err"] else None, note="gamma-line width in the reconstructed energy spectrum (≠ emission-spectrum FWHM in nm)")
    add(sec, "Energy resolution = FWHM / E × 100", r["resolution_pct"], "%", "DERIVED", err=r.get("resolution_err_pct"), note="from the fit to the full spectrum")
    add(sec, "Energy resolution from truth-tagged full-energy events (cross-check)", r["truth_resolution_pct"], "%", "DERIVED", note=f"2.355·std/mean of events selected with simulation truth; no background; peak skewness {r['truth_skew']:+.2f}")
    npe = np.mean(ev.n_detected_photoelectrons.values[fm]) if fm.any() else float("nan")
    if npe > 0:
        add(sec, "Photoelectron-statistics-only expectation, 2.355/√N_pe", FWHM_K / math.sqrt(npe) * 100, "%", "CALCULATED", note="lower bound from Poisson photon counting alone (N_pe = mean photoelectrons in full-energy events); additional broadening shows up in the simulated value")
    # decomposition of the resolution (calculated from simulated per-event numbers)
    m = (ev.n_scint_photons > 0).values & fm
    if m.sum() >= 10 and npe > 0:
        oce = ev.n_sensor_incident.values[m] / ev.n_scint_photons.values[m]
        sc = float(oce.std(ddof=1) / oce.mean())
        tot = math.sqrt(1.0 / npe + sc ** 2)
        add(sec, "Relative spread of optical collection (full-energy events)", sc, kind="DERIVED", note="position-dependent light collection (includes ~binomial noise of the photon count)")
        add(sec, "Quadrature estimate: 2.355·√(1/N_pe + σ_collection²)", FWHM_K * tot * 100, "%", "CALCULATED",
            note="should be close to the truth-tagged resolution above if these two effects dominate")
        if sc > 2.0 / math.sqrt(npe):
            note(sec, f"The energy resolution is dominated by position-dependent optical collection (spread {sc:.1%}), not by photoelectron counting (≈{1 / math.sqrt(npe):.1%}). Changing reflector, surface finish, sensor size/position or absorption length will change it strongly.")
    add(sec, "Full-energy events used for seeding", r["n_truth"], kind="SIMULATED", note="statistical quality: " + ("LOW (<100) — treat the resolution with caution" if r["warning_low_stats"] else "adequate (≥100)"))
    add(sec, "Photopeak counts in μ ± FWHM (raw)", r["raw_counts"], kind="SIMULATED"); add(sec, "Estimated background in window", r["bkg_counts"], kind="DERIVED")
    add(sec, "Photopeak efficiency, absolute (net counts / emitted gammas)", r["eff_abs"], kind="DERIVED", err=r["eff_abs_err"])
    add(sec, "Photopeak efficiency, intrinsic (net counts / gammas entering crystal)", r["eff_int"], kind="DERIVED")
    note(sec, "Photopeak efficiency uses the window μ ± FWHM minus a fitted linear background; the truth full-energy fraction in Section 3 is an upper bound. They differ when the peak is broad or non-Gaussian (depth-dependent light collection produces a low-side tail).")
    if abs(r.get("truth_skew", 0)) > 0.5:
        note(sec, f"The full-energy peak is clearly asymmetric (skewness {r['truth_skew']:+.2f}), so a Gaussian FWHM is only approximate. Compare the fitted value with the truth-tagged value and inspect the spectrum plot.")
    if r["warning_low_stats"]: note(sec, f"LOW STATISTICS: only {r['n_truth']} full-energy events; resolution uncertainty is large. Increase simulation.events.")
    if r.get("fit_warning"): note(sec, r["fit_warning"])
    # reconstructed spectrum plot
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    pos = sig > 0
    Er = sig[pos] * k
    bw = max(Eg / 100.0, 0.5)
    ax.hist(Er, bins=np.arange(0, max(Eg * 1.15, Er.max() * 1.03) + bw, bw), color="#9ecae1", histtype="stepfilled", label="reconstructed (detector response)")
    xs = np.linspace(r["fit_window"][0], r["fit_window"][1], 300)
    # fit was on signal-histogram with bin width bw_sig; convert counts to the plot's bin width
    ctr, cnt = r["fit_hist"]; bws = ctr[1] - ctr[0]
    if r["fit_params"] is not None:
        ax.plot(xs * k, gauss_lin(xs, *r["fit_params"]) * (bw / (bws * k)), "r-", lw=1.6, label=f"Gaussian fit: FWHM {r['fwhm'] * k:.1f} keV ({r['resolution_pct']:.2f} %)")
    else:
        ax.axvspan((r["mu"] - r["fwhm"] / 2) * k, (r["mu"] + r["fwhm"] / 2) * k, color="r", alpha=0.15, label=f"truth-tagged peak: FWHM {r['fwhm'] * k:.1f} keV ({r['resolution_pct']:.2f} %); fit rejected")
    ax.set_xlabel("Reconstructed energy [keV]"); ax.set_ylabel("Events per bin"); ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_title("Reconstructed energy spectrum (single-point calibration)")
    ax.set_ylim(bottom=0.5)
    savefig(fig, "s5_reconstructed_spectrum.png", sec, "Reconstructed energy spectrum",
            f"Signal = {unit}, scaled so the fitted photopeak sits at E_γ. Includes: yield statistics, optical collection variation, PDE statistics" + (", SiPM saturation" if unit.startswith("fired") else "") + ". Excludes electronics noise, nonproportionality, crosstalk/afterpulsing.")
    # linearity / multi-energy
    runs = [(S, ev)] + [(S2, e2) for _, S2, e2 in archive]
    pts = {}
    for S2, e2 in runs:
        E2 = S2["source"]["energy_keV"]
        s2 = e2.n_cells_fired_or_signal.values.astype(float)
        full2 = (e2.energy_deposited_keV >= E2 - 1.0) & (e2.energy_deposited_keV > 0)
        p = photopeak(s2, full2.values, len(e2), S2["totals"]["events_primary_entered_crystal"])
        if p["ok"]:
            pts[round(E2, 6)] = p
    if len(pts) >= 2:
        E = np.array(sorted(pts)); mu = np.array([pts[e]["truth_mean"] for e in E]); er = np.array([pts[e]["truth_std"] / math.sqrt(pts[e]["n_truth"]) for e in E]); rs = np.array([pts[e]["resolution_pct"] for e in E])
        rs_ok = np.array([pts[e].get("fit_gate_failed") is None for e in E])
        b, a = np.polyfit(E, mu, 1); resid = (mu - (a + b * E)) / mu * 100
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].errorbar(E, mu, yerr=er, fmt="o"); xx = np.linspace(0, E.max() * 1.05, 50); axs[0].plot(xx, a + b * xx, "r-", label=f"linear fit: {b:.4g} /keV")
        axs[0].set_xlabel("Gamma energy [keV]"); axs[0].set_ylabel(f"Photopeak position [{unit}]"); axs[0].legend(); axs[0].set_title("Signal linearity")
        axs[1].axhline(0, color="k", lw=0.5); axs[1].plot(E, resid, "o-"); axs[1].set_xlabel("Gamma energy [keV]"); axs[1].set_ylabel("Deviation from linear fit [%]"); axs[1].set_title("Non-linearity of the simulated chain")
        axs[2].plot(E, rs, "-", color="0.6"); axs[2].plot(E[rs_ok], rs[rs_ok], "o", color="C0", label="Gaussian fit"); axs[2].plot(E[~rs_ok], rs[~rs_ok], "s", mfc="none", color="C3", label="truth-tagged (fit rejected)"); axs[2].legend(fontsize=7); axs[2].set_xlabel("Gamma energy [keV]"); axs[2].set_ylabel("Energy resolution [%]"); axs[2].set_title("Resolution vs energy")
        fig.tight_layout()
        savefig(fig, "s5_linearity.png", sec, "Signal linearity and resolution vs energy",
                f"From {len(E)} simulation runs with identical configuration at {', '.join(f'{e:g}' for e in E)} keV; position = mean signal of truth-tagged full-energy events. Scintillator nonproportionality is NOT modelled, so deviations come only from the optical/photodetection chain: depth-dependent light collection (the interaction depth changes with gamma energy for a given beam geometry) and SiPM saturation.")
        add(sec, "Signal linearity: max |deviation| from linear fit (with intercept)", float(np.abs(resid).max()), "%", "DERIVED", note=f"{len(E)} energies; mean signal of truth-tagged full-energy events")
        add(sec, "Signal per keV at each energy", ", ".join(f"{e:g} keV: {m / e:.3f}" for e, m in zip(E, mu)), "signal/keV", "DERIVED", note="constant if the chain were perfectly proportional")
        result["linearity"] = dict(E=E.tolist(), mu=mu.tolist(), res=rs.tolist(), fit_ok=rs_ok.tolist())
    else:
        note(sec, "Signal linearity: NOT evaluated — it needs simulations at several gamma energies (run ./run_all.sh --archive <tag> for each energy).")
    return result


# ----------------------------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------------------------
CSS = """
body{font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222;line-height:1.45}
h1{border-bottom:3px solid #3b7ddd;padding-bottom:.2em}h2{margin-top:2em;border-bottom:1px solid #ccc;padding-bottom:.2em}
table{border-collapse:collapse;width:100%;margin:.8em 0;font-size:.92em}th,td{border:1px solid #ddd;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f0f4fa}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.78em;font-weight:600;color:#fff;white-space:nowrap}
.INPUT{background:#6a51a3}.SIMULATED{background:#2b8cbe}.CALCULATED{background:#238b45}.DERIVED{background:#636363}
.experimental{background:#006d2c}.literature{background:#2171b5}.calculated{background:#238b45}.estimated{background:#d94801}.unknown{background:#cb181d}
.warn{background:#fff3cd;border-left:5px solid #e0a800;padding:.5em .9em;margin:.7em 0}.err{background:#f8d7da;border-left:5px solid #c0392b;padding:.5em .9em;margin:.7em 0}
.fig{margin:1em 0}.fig img{max-width:100%;border:1px solid #ddd}.cap{font-size:.88em;color:#444}.small{font-size:.85em;color:#555}
code{background:#f4f4f4;padding:1px 4px}
"""


def badge(kind):
    base = kind.split(" ")[0]
    extra = kind[len(base):].strip()
    s = f'<span class="badge {base}">{html.escape(base)}</span>'
    if extra:
        st = extra.strip("()")
        s += f' <span class="badge {st}">{html.escape(st)}</span>'
    return s


def metrics_table(sec):
    rows = METRICS.get(sec, [])
    if not rows:
        return ""
    out = ['<table><tr><th>Quantity</th><th>Value</th><th>Unit</th><th>Type</th><th>Note</th></tr>']
    for r in rows:
        out.append(f"<tr><td>{html.escape(r['label'])}</td><td class='num'>{html.escape(fmt(r['value'], r['err']))}</td><td>{html.escape(r['unit'])}</td><td>{badge(r['kind'])}</td><td class='small'>{html.escape(r['note'])}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def plots_html(sec):
    out = []
    for p in PLOTS.get(sec, []):
        b = base64.b64encode(open(p["file"], "rb").read()).decode()
        out.append(f'<div class="fig"><b>{html.escape(p["title"])}</b><br><img src="data:image/png;base64,{b}" alt="{html.escape(p["title"])}"><div class="cap">{html.escape(p["caption"])}</div></div>')
    return "\n".join(out)


def notes_html(sec, cls="warn"):
    return "".join(f'<div class="{cls}">{html.escape(t)}</div>' for t in NOTES.get(sec, []))


def not_run_html(S, stage):
    st = S["stages"][stage]
    miss = "".join(f"<li>{html.escape(m)}</li>" for m in st["missing_inputs"])
    return f'<div class="err"><b>Stage "{stage}" was NOT run.</b> {html.escape(st["reason"])}<br>Missing / unknown inputs:<ul>{miss or "<li>(none listed; stage not requested)</li>"}</ul>No performance claim is made for this stage.</div>'


def provenance_table(S, only_known=False):
    out = ['<table><tr><th>Input</th><th>Value</th><th>Unit</th><th>Status</th><th>Source</th><th>Conditions</th></tr>']
    for p in S["provenance"]:
        if only_known and not p["known"]:
            continue
        out.append(f"<tr><td><code>{html.escape(p['path'])}</code></td><td>{html.escape(p['value'])}</td><td>{html.escape(p['unit'])}</td><td>{badge(p['status'] or 'unknown')}</td><td class='small'>{html.escape(p['source'])}</td><td class='small'>{html.escape(p['conditions'])}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def build_report(S, ctx5, archive):
    mat = S["material"]; cfg = S["config"]
    comp_rows = "".join(f"<tr><td>{e['symbol']}</td><td>{e['Z']}</td><td class='num'>{e['atoms_per_formula_unit']:g}</td><td class='num'>{e['molar_mass_g_mol']:.4f}</td><td class='num'>{e['mass_fraction']:.5f}</td></tr>" for e in mat["composition"])
    g = S["geometry"]; src = S["source"]; run = S["run"]
    full = run["mode_effective"] == "full"
    estimated = [p for p in S["provenance"] if p["status"] == "estimated" and p["known"]]
    unknown = [p for p in S["provenance"] if not p["known"]]
    literature = [p for p in S["provenance"] if p["status"] == "literature" and p["known"]]
    h = []
    h.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Scintillator simulation report - {html.escape(mat['name'])}</title><style>{CSS}</style></head><body>")
    h.append(f"<h1>Scintillator Simulation Report: {html.escape(mat['name'])}</h1>")
    h.append(f"<p class='small'>Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by analyze.py from a simulation run on {html.escape(run['timestamp_utc'])} · {html.escape(S['software']['name'])} v{html.escape(S['software']['version'])} · Geant4 {html.escape(S['software']['geant4_version'])}</p>")
    h.append("<p>Legend: " + " ".join(badge(k) for k in ["INPUT", "SIMULATED", "CALCULATED", "DERIVED"]) + " &nbsp; Input status: " + " ".join(badge(k) for k in ["experimental", "literature", "calculated", "estimated", "unknown"]) + "</p>")
    h.append(f"<div class='warn'><b>Scope.</b> Geant4 cannot discover scintillation properties from composition. Everything in the scintillation, optical and detector sections is conditional on the INPUT values and their status. Mode used: <b>{html.escape(run['mode_effective'])}</b>"
             + (f" ({len(estimated)} inputs are <b>estimated</b>)." if full else ". No scintillation, optical-transport or photodetection performance is claimed in this mode.") + "</div>")
    # 1 material
    h.append("<h2>1. Material information</h2>")
    h.append(f"<p><b>{html.escape(mat['name'])}</b>, {html.escape(mat['physical_state'])}, density <b>{mat['density_g_cm3']:g} g/cm³</b>, temperature <b>{mat['temperature_K']:g} K</b> "
             f"<span class='small'>(temperature is recorded only; Geant4 gamma physics is temperature independent)</span>. Radiation length {mat['radiation_length_mm']:.4g} mm, electron density {mat['electron_density_per_cm3']:.4g} cm⁻³ (calculated by Geant4).</p>")
    h.append(f"<table><tr><th>Element</th><th>Z</th><th>Atoms / formula unit</th><th>Molar mass [g/mol]</th><th>Mass fraction</th></tr>{comp_rows}</table>")
    h.append("<p><b>Input provenance</b> (every scientific input as supplied):</p>" + provenance_table(S))
    # 2 configuration
    h.append("<h2>2. Simulation configuration</h2>")
    rows = [("Mode requested → used", f"{run['mode_requested']} → {run['mode_effective']}"), ("Events / seed", f"{run['events']} / {run['seed']}"), ("Physics list", run["physics_list"]),
            ("Production cut", f"{run['production_cut_mm']:g} mm"), ("Crystal", f"{g['crystal_shape']} {g['dimensions_mm'][0]:g} × {g['dimensions_mm'][1]:g} × {g['dimensions_mm'][2]:g} mm"),
            ("Surrounding medium", g["surrounding_material"]), ("Wrapper / finish", f"{g['reflector']} / {g['surface_finish']}"),
            ("Source", f"{src['particle']} {src['type']}, {src['energy_keV']:g} keV, position {src['position_mm']} mm, direction {[round(x, 4) for x in src['direction']]}"),
            ("Wall time", f"{run['wall_time_s']:.1f} s (single thread)")]
    if full:
        rows += [("Sensor", f"{S['detector_inputs']['type']} on face {g['sensor_face']}, {g['sensor_active_area_mm'][0]:g} × {g['sensor_active_area_mm'][1]:g} mm², coupling {g['coupling']}")]
    if run["cli_overrides"]:
        rows.append(("Command-line overrides", json.dumps(run["cli_overrides"])))
    h.append("<table>" + "".join(f"<tr><th style='width:22%'>{html.escape(a)}</th><td>{html.escape(b)}</td></tr>" for a, b in rows) + "</table>")
    pm = S["physics_models"]
    h.append("<p><b>Physics implemented:</b></p><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in pm["implemented"]) + "</ul>")
    h.append("<p>The complete configuration (including the original <code>config.yaml</code> text) is stored in <code>summary.json</code>, so the run is reproducible with the same seed.</p>")
    # 3 gamma
    h.append("<h2>3. Gamma interaction performance</h2>" + notes_html("s1") + metrics_table("s1") + plots_html("s1"))
    # 4-7
    for num, title, sec, stage in [(4, "Scintillation performance", "s2", "scintillation"), (5, "Optical transport", "s3", "optical_transport"), (6, "Photodetection", "s4", "photodetection")]:
        h.append(f"<h2>{num}. {title}</h2>")
        h.append(notes_html(sec) + metrics_table(sec) + plots_html(sec) if full else not_run_html(S, stage))
    h.append("<h2>7. Detector response</h2>")
    if full:
        h.append(notes_html("s5") + metrics_table("s5") + plots_html("s5"))
        impl, omit = [], []
        impl = ["Scintillation yield statistics (" + ("Poisson" if S["scintillation_inputs"]["resolution_scale"] == 1 else f"scale {S['scintillation_inputs']['resolution_scale']:g}") + ")", "Optical collection variation (full Monte Carlo transport)", "Photodetection statistics (binomial PDE)"]
        if S["detector_inputs"]["type"] == "SiPM" and S["detector_inputs"].get("model_saturation"): impl.append("SiPM microcell saturation" + (" with dead-time recovery" if S["detector_inputs"].get("model_recovery") else ""))
        omit = ["Scintillator nonproportionality / Birks quenching", "Electronics noise, baseline, shaping", "SiPM crosstalk, afterpulsing, dark counts; single-photon time jitter", "Light-yield dependence on temperature"]
        h.append("<p><b>Effects included in the energy-resolution estimate:</b> " + "; ".join(impl) + ".</p><p><b>Effects omitted (the simulated resolution is therefore optimistic):</b> " + "; ".join(omit) + ".</p>")
        h.append("<p class='small'>Energy resolution (%) = (gamma photopeak FWHM [keV] / photopeak energy [keV]) × 100. This is unrelated to the emission-spectrum FWHM in nm.</p>")
    else:
        h.append(not_run_html(S, "photodetection"))
    # 8 limitations
    h.append("<h2>8. Scientific limitations</h2><ul>")
    h.append("<li>Geant4 does not predict light yield, emission spectrum, decay times, absorption length or refractive index from composition. These are inputs.</li>")
    h.append("<li>Not modelled in this run:</li><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in pm["not_modelled"]) + "</ul>")
    h.append("<li>The crystal is a single idealised box; defects, inhomogeneity, surface damage and non-ideal wrapping are represented only through the supplied scalar/tabulated parameters.</li>")
    h.append("<li>Statistical uncertainties shown are Monte Carlo only; they do not include uncertainty in the inputs.</li></ul>")
    # 9 uncertainties
    h.append("<h2>9. Estimates and modeling assumptions</h2>")
    h.append(f"<p>Inputs by status: " + ", ".join(f"{k}: {sum(1 for p in S['provenance'] if p['status'] == k and p['known'])}" for k in ["experimental", "literature", "calculated", "estimated"]) + f", unknown / not provided: {len(unknown)}.</p>")
    if estimated:
        h.append("<p><b>Estimated inputs</b> (results depend directly on these):</p><ul>" + "".join(f"<li><code>{html.escape(p['path'])}</code> = {html.escape(p['value'])} {html.escape(p['unit'])} — <i>{html.escape(p['source'])}</i></li>" for p in estimated) + "</ul>")
    if S["assumptions_applied"]:
        h.append("<p><b>Assumptions applied by the program</b> because an optional input was omitted:</p><ul>" + "".join(f"<li>{html.escape(a)}</li>" for a in S["assumptions_applied"]) + "</ul>")
    if unknown:
        h.append("<p><b>Unknown / not provided inputs:</b> " + ", ".join(f"<code>{html.escape(p['path'])}</code>" for p in unknown) + "</p>")
    h.append("<p>Sensitivity to any input is studied by editing <code>config.yaml</code> and running again; no automatic parameter sweeps are performed.</p>")
    # 10 final
    h.append("<h2>10. Final material evaluation</h2>")
    h.append("<p class='small'>This is a factual summary of what was simulated. No overall good/bad score is assigned.</p>")
    h.append(final_evaluation_html(S, ctx5, full, estimated, literature))
    h.append("</body></html>")
    return "\n".join(h)


def val(sec, label_start):
    for r in METRICS.get(sec, []):
        if r["label"].startswith(label_start):
            return r
    return None


def final_evaluation_html(S, ctx5, full, estimated, literature):
    out = []
    mat = S["material"]
    formula = ", ".join("%s%g" % (e["symbol"], e["atoms_per_formula_unit"]) for e in mat["composition"])
    out.append(f"<p><b>Material:</b> {html.escape(mat['name'])} ({formula}), ρ = {mat['density_g_cm3']:g} g/cm³, T = {mat['temperature_K']:g} K.</p>")
    r = val("s1", "Interaction fraction (per emitted"); f = val("s1", "Full-energy deposition fraction"); a = val("s1", "Total attenuation length")
    out.append(f"<p><b>Gamma interaction ({S['source']['energy_keV']:g} keV):</b> interaction fraction {fmt(r['value'], r['err'])}, full-energy deposition fraction {fmt(f['value'], f['err'])} per emitted gamma; attenuation length {fmt(a['value'])} mm (calculated). "
               "These depend only on composition, density, geometry and source — they are the most trustworthy part of this report when those inputs are measured.</p>")
    if not full:
        out.append("<p><b>Scintillation, optical transport and photodetection:</b> not simulated (missing inputs listed in the stage report above). No detector-performance statement can be made for this material from this run.</p>")
    else:
        ly = val("s2", "Configured light yield"); g = val("s2", "Generated photons per MeV"); oce = val("s3", "Optical collection efficiency"); pe = val("s4", "Mean photoelectrons, full-energy")
        out.append(f"<p><b>Scintillation:</b> configured {fmt(ly['value'])} photons/MeV ({ly['kind']}); the simulation generated {fmt(g['value']) if g else 'n/a'} photons per MeV deposited.</p>")
        out.append(f"<p><b>Optical transport:</b> collection efficiency {fmt(oce['value'], oce['err'])} (fraction of generated photons entering the sensor).</p>")
        if pe: out.append(f"<p><b>Photodetection:</b> {fmt(pe['value'], pe['err'])} photoelectrons for full-energy events.</p>")
        fit = ctx5.get("fit", {})
        if fit.get("ok"):
            e = fit.get("resolution_err_pct")
            out.append(f"<p><b>Detector response:</b> simulated energy resolution {fit['resolution_pct']:.2f}{f' ± {e:.2f}' if e else ''} % at {S['source']['energy_keV']:g} keV (Gaussian fit; "
                       f"{fit['n_truth']} full-energy events{', LOW STATISTICS' if fit['warning_low_stats'] else ''}). Because nonproportionality, electronics noise and other effects are omitted, this is a <b>lower bound</b> for what the same inputs would give in a real detector, not a prediction.</p>")
        else:
            out.append(f"<p><b>Detector response:</b> energy resolution not determined — {html.escape(fit.get('reason', 'n/a'))}</p>")
    out.append("<p><b>Properties requiring experimental verification before conclusions are drawn:</b></p><ul>")
    for p in estimated + literature:
        out.append(f"<li><code>{html.escape(p['path'])}</code> ({p['status']}): {html.escape(p['value'])} {html.escape(p['unit'])}</li>")
    out.append("</ul>")
    out.append("<p><b>Limitations</b> are listed in Section 8 and <b>estimates and modeling assumptions</b> in Section 9.</p>")
    return "".join(out)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results"); ap.add_argument("--tol-keV", type=float, default=1.0, help="tolerance defining a full-energy deposit (truth)")
    ap.add_argument("--no-archive", action="store_true", help="ignore results/archive/* runs")
    a = ap.parse_args()
    R = a.results; OUT["results"] = R; OUT["plots"] = os.path.join(R, "plots"); os.makedirs(OUT["plots"], exist_ok=True)
    for f in glob.glob(os.path.join(OUT["plots"], "*.png")):
        os.remove(f)
    try:
        S = json.load(open(os.path.join(R, "summary.json"))); ev = pd.read_csv(os.path.join(R, "events.csv"))
    except Exception as ex:
        sys.exit(f"cannot read results in '{R}': {ex}\nRun the simulator first (./run_all.sh).")
    if S.get("schema_version") != 1:
        print("warning: unexpected summary.json schema_version", file=sys.stderr)
    full = S["run"]["mode_effective"] == "full"
    archive = [] if a.no_archive else load_archive(S, R)
    if archive:
        print(f"using {len(archive)} archived run(s): {', '.join(n for n, _, _ in archive)}")
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    ctx = section1(S, ev, a.tol_keV, archive)
    ctx5 = {}
    if full:
        section2(S, ev, ctx)
        section3(S, ev, ctx, R)
        section4(S, ev, ctx, R)
        ctx5 = section5(S, ev, ctx, archive)
    html_text = build_report(S, ctx5, archive)
    open(os.path.join(R, "report.html"), "w", encoding="utf-8").write(html_text)
    out = dict(mode=S["run"]["mode_effective"], metrics=METRICS, notes=NOTES, plots={k: [p["file"] for p in v] for k, v in PLOTS.items()},
               photopeak={k: v for k, v in ctx5.get("fit", {}).items() if k not in ("fit_hist",)} if ctx5 else None, linearity=ctx5.get("linearity") if ctx5 else None)
    json.dump(out, open(os.path.join(R, "analysis_metrics.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    nplots = sum(len(v) for v in PLOTS.values())
    print(f"analysis done: {nplots} plots, report -> {os.path.join(R, 'report.html')}")
    if full and ctx5.get("fit", {}).get("ok"):
        f = ctx5["fit"]; print(f"photopeak: {f['mu']:.1f} {ctx5['unit']}, resolution {f['resolution_pct']:.2f} % (n_fullE={f['n_truth']})")


if __name__ == "__main__":
    main()
