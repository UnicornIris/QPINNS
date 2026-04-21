# Research Roadmap

## Target Venues (in order of fit)

| Venue | Deadline | Fit | Why |
|---|---|---|---|
| **IEEE QCE (Quantum Computing & Engineering)** | ~April 2026 | Best | Explicitly wants quantum+application papers, hardware results valued |
| **QTML (Quantum Techniques in Machine Learning)** | ~July 2026 | Strong | QML-focused workshop, noise studies are welcome |
| **IEEE ICQC** | ~TBD | Good | Newer venue, quantum computing applications |
| **Scientific Reports** (journal) | Rolling | Backup | Where QCPINN was published, more relaxed standards |
| **Quantum Science and Technology** (journal) | Rolling | Stretch | Higher prestige, needs strong results |

**Realistic first target**: IEEE QCE 2026 (workshop or short paper).

---

## Week-by-Week Plan

### Week 1: Validate the pipeline and run initial experiments
**Goal**: Confirm the code works end-to-end and get preliminary results.

- [ ] Run `python tests/test_all.py` — verify 16/16 pass
- [ ] Run `python -m qcpinn --suite helmholtz_comparison --epochs 3000` (~1 hr)
- [ ] Run `python -m qcpinn.cross_noise --epochs 500` as a quick preview (~1 hr)
- [ ] Review output plots in `experiments/` — do the curves make sense?
- [ ] Document any issues, discuss with professor

**Deliverable**: Preliminary comparison plots showing baseline vs TE vs classical.

### Week 2: Run the main noise study
**Goal**: Generate Table 1 and Figure 1 — the core results.

- [ ] Run the full cross-noise study: `python -m qcpinn.cross_noise --epochs 5000` (~6 hrs)
- [ ] Examine `experiments/cross_noise/cross_noise_main.pdf` — this is Figure 1
- [ ] Examine `experiments/cross_noise/results_table.tex` — this is Table 1
- [ ] If TE noise-aware doesn't show improvement, try:
  - Increase qubits: `--num-qubits 6`
  - Try L-BFGS: add `optimizer: lbfgs` support
  - Adjust noise range: `--train-noise 0.0 0.005 0.01 --test-noise 0.0 0.001 0.003 0.005 0.01`
- [ ] Start writing Section 3 (Methodology) of the paper

**Deliverable**: Main cross-noise figure and results table.

### Week 3: Multi-PDE generalization + ablations
**Goal**: Show the finding generalizes beyond Helmholtz.

- [ ] Run `python -m qcpinn --suite multi_pde --epochs 5000` (~3 hrs)
- [ ] Run wave cross-noise: `python -m qcpinn.cross_noise --problem wave --epochs 5000`
- [ ] Run Klein-Gordon: `python -m qcpinn.cross_noise --problem klein_gordon --epochs 5000`
- [ ] Run ablation: vary number of qubits (2, 4, 6) at fixed noise
- [ ] Run ablation: vary TE width (16, 32, 64)
- [ ] Write Section 2 (Background) and Section 4 (Experiments)

**Deliverable**: Multi-PDE results + ablation studies.

### Week 4: IonQ hardware validation
**Goal**: Get at least one real-hardware data point.

- [ ] Get IonQ API key (if not already)
- [ ] Test on IonQ simulator first: `python -m qcpinn.hardware --checkpoint <best_te_model> --backend simulator --grid 10`
- [ ] Run on IonQ QPU (5x5 grid = 25 circuits, manageable cost): `python -m qcpinn.hardware --checkpoint <best_te_model> --backend qpu.harmony --grid 5 --shots 1024`
- [ ] Run both baseline and TE models on hardware — compare degradation
- [ ] Write Section 5 (Hardware Validation)

**Deliverable**: Sim-vs-hardware comparison figures.

### Week 5: Paper writing
**Goal**: Complete draft.

- [ ] Write Section 1 (Introduction) — position vs QCPINN, TE-QPINN, x-TE-QPINN
- [ ] Write Section 6 (Discussion) — honest about limitations
- [ ] Write Abstract and Conclusion
- [ ] Create all final figures (use PDF outputs from experiments/)
- [ ] Professor review

**Deliverable**: Complete paper draft.

### Week 6: Revision and submission
**Goal**: Submit.

- [ ] Incorporate professor feedback
- [ ] Re-run any experiments if needed
- [ ] Polish figures (fonts, labels, consistent colors)
- [ ] Submit to target venue

---

## What To Tell Your Professor

> "I've pivoted the project from reimplementing TE-QPINN (which was published while I was working on it) to studying a question nobody has answered yet: **how does quantum hardware noise affect QPINN accuracy, and can trainable embeddings learn to compensate?**
>
> The experiment trains QPINNs at different noise levels and cross-evaluates them. If TE embeddings show slower degradation under noise, that's a novel finding. If they don't, we still have the first systematic noise characterization study for QPINNs, which is publishable as a benchmark paper.
>
> The code is complete: I have a clean codebase with 16 passing tests, automated experiment suites, and IonQ hardware integration. I can generate all paper figures by running three commands."

---

## Risk Mitigation

**Risk 1: TE doesn't help with noise at all.**
- Mitigation: The *characterization study itself* is publishable. "We show that QPINNs degrade as X under depolarizing noise, with a critical threshold at p≈Y" is a useful finding even without a solution.

**Risk 2: IonQ hardware is too expensive or slow.**
- Mitigation: Even a single 5x5 grid evaluation (~25 circuits) is sufficient for a "hardware validation" section. Use IonQ simulator (free) for development, real QPU for just the final figure.

**Risk 3: Results are too similar to x-TE-QPINN.**
- Mitigation: We're asking a different question (noise robustness vs convergence). As long as we don't claim their contribution (TE improves convergence) as ours, we're fine.
