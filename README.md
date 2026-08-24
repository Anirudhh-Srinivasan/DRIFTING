# Drift-Type-Aware Retraining with Model Reuse (NIDS)

Research prototype for a drift-type-aware retraining framework for network
intrusion detection. The framework detects concept drift in a streaming
classification setting, **classifies the type** of drift (sudden / gradual /
incremental / recurring), and selects a retraining strategy **proportional to the
type** — instead of always doing a full retrain. The headline comparison is
compute/time saved (especially via cache-reuse for recurring drift) vs. a naive
always-retrain baseline, at equal or near-equal detection accuracy.

## Status

| Component | State |
|---|---|
| Synthetic CICIDS2017-like data + drift simulation | ✅ implemented |
| Real CICIDS2017 loader (same interface) | 🔲 stubbed, drop-in later |
| Drift detectors (ADWIN, DDM, EDDM, KSWIN via `river`) | ✅ implemented |
| Naive always-retrain baseline (end-to-end) | ✅ implemented & running |
| Evaluation + accuracy-over-stream plot | ✅ implemented |
| Concept fingerprint + symmetric-KL similarity | ✅ implemented |
| Model-reuse cache + recurring detection | ✅ implemented & validated |
| Cache-reuse pipeline (detect→reuse/retrain) | ✅ implemented & validated |
| Controlled ablation (isolates cache effect) | ✅ implemented & validated |
| Drift feature extractor | ✅ implemented |
| Rule-based drift-type classifier (calibrated) | ✅ implemented & validated |
| ML drift-type classifier (comparison) | 🔲 interface only — deferred |
| Full type-aware strategy pipeline (end-to-end) | ✅ implemented & validated |

All core novel components are implemented and validated end-to-end. Remaining
work: the real CICIDS2017 loader (stubbed behind the same interface), the ML
drift-type classifier for the rule-vs-ML comparison (stubbed), and — if a cleaner
gradual/incremental separation is wanted — richer transition-shape features.

## Library choice

`river` (0.26.x) is the streaming-ML detector layer — it is the actively
maintained standard and ships all four requested detectors with a consistent
interface. The older `scikit-multiflow` is deprecated/folded into `river`.
Underlying task classifiers use `scikit-learn` (LogisticRegression /
RandomForest), batch-retrained on windows; "streaming" is simulated by processing
the stream in order (prequential test-then-train).

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the baseline

```bash
.venv/bin/python -m experiments.run_baseline                     # ADWIN + RandomForest
.venv/bin/python -m experiments.run_baseline --detector ddm --model logreg
```

Writes to `results/`:
- `baseline_metrics.json` — detection recall, delay, false-alarm rate, recovery
  time, retrain count + wall-clock cost
- `baseline_accuracy.png` — rolling accuracy with drift onsets marked
- `stream_summary.json` — ground-truth drift events

### Example result (default config, ADWIN + RandomForest, seed 42)

26 000 samples, 5 injected drifts: detection recall **1.0**, mean detection delay
**205** samples, **2** false alarms, **8** retrains (0.59 s total), overall
accuracy **0.954**. Per-event delays show the expected signature — sudden /
recurring detected in 15–31 samples, gradual / incremental in 431 / 519 — which
is exactly the signal the drift-type classifier will exploit.

## Run the cache-reuse validation

```bash
.venv/bin/python -m experiments.run_cache_reuse
.venv/bin/python -m experiments.run_cache_reuse --threshold 1.0 --detector adwin
```

Runs the naive baseline and the cache-reuse pipeline on the same stream and
reports the per-detection concept-distance table (calibration evidence), cache
hit rate, compute saved, retrains avoided, and accuracy parity. Writes
`cache_reuse_metrics.json` and `cache_reuse_accuracy.png`.

**Cache design (agreed):** concept fingerprint = per-class feature mean +
covariance; similarity = symmetric KL between per-class diagonal Gaussians;
recurrence when the nearest cached concept's distance ≤ `distance_threshold`.
Calibration is clean — recurrences sit at distance ≈ 0.09–0.18 while novel
concepts are ≈ 3–27, so the threshold (default 1.0) lies in a wide gap. The new
concept is fingerprinted from strictly post-detection samples (`settle=150`) so
the window never bleeds into the old concept.

### Example result (default config, ADWIN + RandomForest, seed 42)

| | baseline (always retrain) | cache-reuse |
|---|---|---|
| retrains | 8 | 5 |
| cache reuses (correct/total recurrences) | — | 2 / 2 |
| retrain wall-clock | 1.10 s | 0.81 s (saved 0.28 s) |
| overall accuracy | 0.9537 | **0.9647** |

Cache-reuse matches-or-beats baseline accuracy while skipping retraining on both
recurrences: reusing the cleanly-trained prior model is actually *better* than
the baseline's retrain on a window that straddles the drift onset.

*Note:* `run_cache_reuse` compares against the naive baseline as literally
defined (immediate retrain, its own detector), so its retrain counts also reflect
detector-timing differences. For the exact, confound-free cache effect use the
controlled ablation below.

## Run the controlled ablation (isolates the cache effect)

```bash
.venv/bin/python -m experiments.run_controlled_ablation
```

Captures the drift schedule once from the always-retrain policy and **replays the
identical schedule** under both policies, so detection points match exactly and
the *only* difference is the action taken at each point. Therefore
`retrains_avoided == reuses` (pure caching) and `compute_saved` is the baseline's
actual retrain wall-clock at the exact points where the cache reused.

### Example result (default config, ADWIN + RandomForest, seed 42)

| (identical 7-point schedule) | always-retrain | cache-reuse |
|---|---|---|
| retrains | 7 | 5 |
| reuses (correct / total, wrong) | — | 2 / 2, 0 wrong |
| retrain wall-clock | 1.133 s | 0.856 s |
| accuracy | 0.9653 | 0.9647 |

**Retrains avoided = 2 = reuses** (both genuine recurrences, zero false reuses).
**Exact compute saved = 0.271 s** (~24% of retrain wall-clock). **Accuracy delta
= −0.0006** — parity. This is the confound-free number for the paper's headline
table. Writes `controlled_ablation_metrics.json` and
`controlled_ablation_accuracy.png`.

## Run the drift-type classifier (calibrate + confusion matrix)

```bash
.venv/bin/python -m experiments.run_drift_classification --calib 1-6 --test 101-104
```

Generates balanced single-drift streams per type (features extracted at the
matched detection, labelled by ground truth), calibrates a 3-level rule tree on
the **calibration** seeds, and evaluates on **held-out** seeds. Thresholds and the
feature at each node are chosen by optimal 1-D splits on the ground truth — not
hand-picked. Writes `drift_classification_metrics.json` and
`drift_classification_confusion.png`.

**Rule tree** (order matters): `concept_distance ≤ τ` → recurring (a recurrence is
dynamically sudden-like, so the distance test must come first); else `error_jump ≥
τ` → sudden; else split gradual/incremental on the most discriminative feature.

### Example result (calib seeds 1–6, test seeds 101–104)

Calibrated: recurring `concept_distance ≤ 3.79`, sudden `error_jump ≥ 0.287`,
gradual/incremental `ramp_length ≥ 113 → gradual`.

| true \ pred | sudden | gradual | incremental | recurring |
|---|---|---|---|---|
| **sudden** | 4 | 0 | 0 | 0 |
| **gradual** | 0 | 4 | 0 | 0 |
| **incremental** | 0 | 2 | 2 | 0 |
| **recurring** | 0 | 0 | 0 | 4 |

Per-class: sudden 1.00/1.00, gradual 0.67/1.00, incremental 1.00/0.50, recurring
1.00/1.00. **Overall accuracy 0.875.** Sudden and recurring are classified
perfectly; the only confusions are incremental→gradual — the two are near-identical
in the error signal (a well-known hard distinction). The confusion matrix is
reported in full so this is visible rather than hidden in an aggregate. Candidate
improvements: richer transition-shape features, or the ML classifier over the same
features (`drift_classifier/ml_classifier.py`, currently a stub).

## Run the full type-aware pipeline (headline experiment)

```bash
.venv/bin/python -m experiments.run_full_pipeline --calib 1-6
.venv/bin/python -m experiments.run_full_pipeline --calib 1-6 --model logreg
```

Calibrates the classifier, runs the naive baseline live to capture the drift
schedule, classifies each detected drift, then replays the **same schedule** under
the type-aware pipeline:

* **sudden** → full retrain (fresh model, `retrain_window` samples)
* **gradual / incremental** → fresh retrain on a *smaller* recent window
  (`incremental_window`) — same model class, fewer samples, so it's cheaper yet
  still learns the union of concepts present and recovers accuracy
* **recurring** → cache retrieval (near-zero cost); full-retrain fallback on a miss

Because both pipelines replay the identical schedule, the difference is purely the
type-aware strategy selection.

### Headline result (default config, ADWIN + RandomForest, seed 42)

| | naive (always retrain) | type-aware |
|---|---|---|
| strategy mix | 7 full retrains | 1 full · 4 incremental · 2 cache |
| retrain wall-clock | 1.126 s | 0.588 s |
| overall accuracy | 0.9653 | 0.9640 |

**≈48% compute reduction (1.9× speedup) at statistically equal accuracy (Δ
−0.0013).** Sudden gets a full retrain, the two recurrences are served from cache
(2/2 hits, near-zero cost), and gradual/incremental use the cheaper fresh
small-window retrain. The type-aware and baseline accuracy curves overlap across
the whole stream (`results/full_pipeline_accuracy.png`).

*Design note:* a warm-start partial update was evaluated first and rejected —
sklearn's RandomForest warm-start can only *add* trees, so a stale-tree majority
out-voted the new trees on large shifts (RF −0.22, LogReg −0.12 accuracy).
Fresh-retrain-on-a-smaller-window keeps the same model class and the
smaller-recent-window intent while restoring parity.

### Drift-type classification within the pipeline

The pipeline also reports how the classifier labelled each detected drift on the
main stream (feeding the routing). Sudden and recurring are labelled correctly;
the residual errors are incremental→gradual — the known-hard pair (see the
classifier section) — which route to the *same* incremental strategy, so those
particular confusions don't change the action taken.

## Layout

```
drift-retraining/
├── config.py              # single source of truth: seeds, scenario, hyper-params
├── models.py              # sklearn task classifiers (LogReg / RandomForest)
├── data/                  # synthetic CICIDS-like generator + drift simulation
│   ├── stream.py          #   DriftStream / DriftEvent (the data interface seam)
│   ├── synthetic.py       #   concept space (affine-transformed class centroids)
│   ├── drift_simulation.py#   sudden / gradual / incremental / recurring builder
│   ├── cicids_loader.py   #   real-CICIDS loader stub (same interface)
│   └── __init__.py        #   load_stream(source="synthetic"|"cicids")
├── detectors/             # ADWIN / DDM / EDDM / KSWIN wrappers → common interface
├── drift_classifier/      # rule-based + ML drift-type classifier (stubs)
├── model_cache/           # model-reuse cache (stub)
├── strategy/              # strategy selector (routing defined, actions stub)
├── baseline/              # naive detect→always-full-retrain pipeline
├── experiments/           # runner + evaluation metrics
├── results/               # metrics, logs, plots
└── notebooks/             # exploratory analysis / paper plots
```

## Reproducibility

A single `seed` in `config.py` drives all randomness (concept geometry, sampling,
model training). Every component reads parameters from `ExperimentConfig` rather
than hard-coding them, so experiments are config-driven and repeatable.

## Design decisions on record (for the Methodology section)

- **Labels:** multi-class by attack family; binary benign/attack is derivable
  (`DriftStream.binary_labels()`, class 0 == BENIGN).
- **Drift mechanism:** each *concept* applies a deterministic affine transform
  (rotation + translation) to fixed per-class centroids. A model trained under
  one concept degrades under another → the prequential error rises → detectors
  fire. Determinism makes recurrence *genuine* (the cache can legitimately hit).
- **Detector input:** the prequential 0/1 error stream, fed identically to all
  four detectors for an apples-to-apples comparison.
- **Default scenario:** warm-up on concept 0, then one drift of each type, then
  two recurring concepts to exercise the model-reuse cache.

## Next (needs sign-off before building)

1. Model-cache fingerprint representation + similarity metric + match threshold.
2. Drift-type classifier: rule thresholds (rule-based) and feature vector +
   training protocol (ML), then compare the two.
3. Strategy selector actions: incremental/partial update mechanism + cache
   retrieval wiring; then the proposed-vs-baseline headline comparison.
```
