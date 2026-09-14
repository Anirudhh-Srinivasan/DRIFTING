"""Natural, chronological CICIDS2017 baseline (no drift-type ground truth)."""
from __future__ import annotations
import argparse, json, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import default_config
from data import load_stream
from baseline.naive import run_naive_baseline

def main():
    started = time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", required=True); p.add_argument("--max-rows", type=int)
    p.add_argument("--warmup", type=int, default=None); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model", default=None); p.add_argument("--detector", default=None)
    p.add_argument("--results-dir", default=None); args=p.parse_args()
    cfg=default_config(); cfg.cicids.csv_dir=args.csv_dir; cfg.cicids.max_rows=args.max_rows
    if args.warmup is not None: cfg.eval.warmup_size=args.warmup; cfg.eval.retrain_window=min(2000, args.warmup)
    if args.seed is not None: cfg.seed=args.seed
    if args.model: cfg.model.kind=args.model
    if args.detector: cfg.primary_detector=args.detector
    if args.results_dir: cfg.results_dir=args.results_dir
    os.makedirs(cfg.results_dir, exist_ok=True)
    print(f"Configuration: source=natural CICIDS2017, max_rows={cfg.cicids.max_rows}, "
          f"warmup={cfg.eval.warmup_size}, model={cfg.model.kind}, detector={cfg.primary_detector}, "
          "shuffle=False, preprocessing=warm-up-only")
    stream=load_stream(cfg, source="cicids", controlled=False)
    if stream.n_samples <= cfg.eval.warmup_size: raise ValueError("max_rows must exceed --warmup")
    print(f"Loaded chronological natural CICIDS stream: {stream.n_samples} rows, {stream.n_features} features")
    print("Raw label -> mapped class -> samples:")
    for r in stream.metadata["raw_label_mapping"]: print(f"  {r['raw_label']} -> {r['mapped_class']} -> {r['sample_count']}")
    result=run_naive_baseline(stream,cfg); truth=stream.y[cfg.eval.warmup_size:]
    prec, rec, f1, support=precision_recall_fscore_support(truth,result.predictions,labels=range(len(stream.class_names)),zero_division=0)
    def context(index):
        file_info = next((b for b in stream.metadata.get("file_boundaries", [])
                          if b["start"] <= index <= b["end"]), None)
        lo, hi = max(0, index - 1000), min(stream.n_samples, index + 1001)
        vals, counts = __import__('numpy').unique(stream.y[lo:hi], return_counts=True)
        return {"index": int(index), "file": file_info["file"] if file_info else None,
                "local_class_counts_±1000": {stream.class_names[int(v)]: int(c) for v, c in zip(vals, counts)},
                "interpretation": "detector event only; natural CICIDS provides no drift-type ground truth"}
    metrics={"source":"natural CICIDS2017; no asserted drift-type ground truth", "accuracy":round(accuracy_score(truth,result.predictions),4),
      "precision_macro":round(float(prec.mean()),4),"recall_macro":round(float(rec.mean()),4),"f1_macro":round(float(f1.mean()),4),
      "per_class":[{"class":n,"precision":round(float(a),4),"recall":round(float(b),4),"f1":round(float(c),4),"support":int(d)} for n,a,b,c,d in zip(stream.class_names,prec,rec,f1,support)],
      "detections":result.detections,"detection_context":[context(i) for i in result.detections], "n_detections":len(result.detections),"retraining_count":result.n_retrains,"retraining_time_s":round(result.total_retrain_time_s,4),
      "total_runtime_s":round(time.perf_counter()-started,4), "file_boundaries":stream.metadata.get("file_boundaries", []),
      "feature_names":stream.feature_names,"label_mapping":stream.metadata["raw_label_mapping"],"preprocessing":stream.metadata["preprocessing"]}
    with open(os.path.join(cfg.results_dir,"cicids_baseline_metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    fig,ax=plt.subplots(figsize=(10,3.5)); ax.plot(result.step_index, result.correct.cumsum()/range(1,len(result.correct)+1)); ax.set(xlabel="chronological stream index",ylabel="cumulative prequential accuracy",title="Natural CICIDS2017 baseline"); fig.tight_layout(); fig.savefig(os.path.join(cfg.results_dir,"cicids_baseline_accuracy.png"),dpi=130); plt.close(fig)
    print(json.dumps({k:metrics[k] for k in ("accuracy","precision_macro","recall_macro","f1_macro","n_detections","retraining_count","retraining_time_s")},indent=2))
if __name__=="__main__": main()
