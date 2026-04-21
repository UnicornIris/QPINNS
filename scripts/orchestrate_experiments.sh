#!/bin/bash
# Full autonomous experiment orchestrator
# Phase 2 noise training + IonQ simulator + QPU demo
set -euo pipefail

PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/qcpinn-modern/bin/python
PROJECT=/Users/pranavbykampadi/QCPINN-1
LOG=$PROJECT/experiments/orchestrator.log
export PYTHONUNBUFFERED=1

cd "$PROJECT"

# Load local secrets if present, otherwise require them from the environment.
if [ -f "$PROJECT/.env" ]; then
    set -a
    . "$PROJECT/.env"
    set +a
fi

: "${IONQ_API_KEY:?Set IONQ_API_KEY in the environment or .env before running this script.}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== Orchestrator started ==="

# -------------------------------------------------------
# Step 1: Phase 2 noise-aware training
# (noiseless models use L-BFGS, noisy models use Adam)
# -------------------------------------------------------
log "Step 1: Running Phase 2 noise-aware training..."

$PYTHON scripts/noise_robustness.py \
    --phase 2 \
    --train-noise 0.0 0.01 \
    --test-noise 0.0 0.001 0.005 0.01 0.02 0.05 \
    --epochs 300 \
    --grid-points 20 \
    --output-dir experiments/noise_robustness \
    2>&1 | tee -a "$LOG"

log "Phase 2 complete."

# -------------------------------------------------------
# Step 2: IonQ Forte-1 simulator runs (TE + Baseline)
# -------------------------------------------------------
log "Step 2: IonQ Forte-1 simulator (8x8 interior grid)..."

log "  Running TE-QPINN on IonQ simulator..."
$PYTHON -m qcpinn.hardware \
    --checkpoint experiments/full_comparison_fixed/te_qpinn_4q5l/best_val_model.pth \
    --backend simulator --noise-model forte-1 --shots 1024 --grid 8 --grid-scheme interior \
    --output-dir experiments/ionq_forte1_te_8x8_interior \
    2>&1 | grep -v "RuntimeWarning\|warnings.warn\|UserWarning" | tee -a "$LOG"

log "  Running Baseline QPINN on IonQ simulator..."
$PYTHON -m qcpinn.hardware \
    --checkpoint experiments/full_comparison_fixed/baseline_qpinn_4q5l/best_val_model.pth \
    --backend simulator --noise-model forte-1 --shots 1024 --grid 8 --grid-scheme interior \
    --output-dir experiments/ionq_forte1_baseline_8x8_interior \
    2>&1 | grep -v "RuntimeWarning\|warnings.warn\|UserWarning" | tee -a "$LOG"

log "IonQ simulator runs complete."

# -------------------------------------------------------
# Step 3: Real IonQ QPU demo run (small interior grid)
# -------------------------------------------------------
log "Step 3: Real IonQ QPU demo (4x4 interior grid, debiased, TE-QPINN only)..."

$PYTHON -m qcpinn.hardware \
    --checkpoint experiments/full_comparison_fixed/te_qpinn_4q5l/best_val_model.pth \
    --backend qpu --shots 512 --grid 4 --grid-scheme interior --error-mitigation debias \
    --output-dir experiments/ionq_qpu_te_4x4_interior_debias512 \
    2>&1 | grep -v "RuntimeWarning\|warnings.warn\|UserWarning" | tee -a "$LOG"

log "QPU demo complete."

# -------------------------------------------------------
# Step 4: Full summary
# -------------------------------------------------------
log "=== FULL EXPERIMENT SUMMARY ==="

log "--- Phase 1 (inference-only noise robustness) ---"
$PYTHON -c "
import json
with open('experiments/noise_robustness/phase1_results.json') as f:
    r = json.load(f)
print(f'{\"Noise\":<10} {\"TE L2%\":<12} {\"Baseline L2%\":<14} {\"Advantage\":<10}')
print('-'*46)
for n in sorted(r['te_clean'].keys(), key=float):
    te = r['te_clean'][n]['rel_l2_u']
    bl = r['baseline_clean'][n]['rel_l2_u']
    print(f'{float(n):<10.3f} {te:<12.2f} {bl:<14.2f} {bl/max(te,0.01):.1f}x')
" 2>&1 | tee -a "$LOG"

log "--- Phase 2 (cross-noise training) ---"
if [ -f experiments/noise_robustness/phase2_results.json ]; then
    $PYTHON -c "
import json
with open('experiments/noise_robustness/phase2_results.json') as f:
    r = json.load(f)
for mode in r:
    for tn_train in r[mode]:
        print(f'\n{mode} (trained at noise={tn_train}):')
        for tn_test in sorted(r[mode][tn_train].keys(), key=float):
            m = r[mode][tn_train][tn_test]
            print(f'  test_noise={float(tn_test):.3f}: L2={m[\"rel_l2_u\"]:.2f}%')
" 2>&1 | tee -a "$LOG"
else
    log "Phase 2 results not found!"
fi

log "--- IonQ Results ---"
for d in experiments/ionq_*; do
    if [ -f "$d/hardware_results.json" ]; then
        log "$(basename $d):"
        $PYTHON -c "
import json
with open('$d/hardware_results.json') as f:
    r = json.load(f)
print(f'  Backend: {r[\"backend\"]} | Noise model: {r.get(\"noise_model\",\"none\")}')
print(f'  Grid: {r[\"grid_points\"]}x{r[\"grid_points\"]} = {r[\"total_circuits\"]} points')
print(f'  MAE sim vs hw: {r[\"mae_sim_vs_hw\"]:.4f}')
print(f'  Correlation: {r[\"correlation\"]:.4f}')
print(f'  Total time: {r[\"total_time_sec\"]:.1f}s')
if 'rel_l2_hw_vs_exact' in r:
    print(f'  HW vs exact L2: {r[\"rel_l2_hw_vs_exact\"]:.2f}%')
" 2>&1 | tee -a "$LOG"
    fi
done

log "=== ALL EXPERIMENTS COMPLETE ==="
log "Results in: experiments/noise_robustness/, experiments/ionq_*/"
