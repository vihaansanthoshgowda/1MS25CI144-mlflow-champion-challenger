"""
evaluate.py
Runs Model A (Champion) and Model B (Challenger) on the SAME 200 SST-2 sentences,
computes metrics + latency, and logs everything to MLflow:
  - compare_sentiment            (parent / summary run)
      - A_sentiment_distilbert-sst2   (child run)
      - B_sentiment_bert-sst2         (child run)
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import pandas as pd
import sklearn
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

# ---------------- settings ----------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
STUDENT_USN = "1MS25CI144"
TASK = "sentiment"
SEED = 42
EVAL_SAMPLES = 200
BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = "cpu"

MODELS = {
    "A": {"hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
          "short": "distilbert-sst2", "role": "champion"},
    "B": {"hf_model_id": "textattack/bert-base-uncased-SST-2",
          "short": "bert-sst2", "role": "challenger"},
}

# Champion rule (same as README, written before the run)
PRIMARY_METRIC = "macro_f1"   # higher is better
MARGIN = 0.02                 # Challenger must win by >= 2 percentage points
MAX_LATENCY_RATIO = 5.0       # guardrail: Challenger may not be > 5x slower

# Both models must speak the same language: NEGATIVE / POSITIVE
LABEL_MAP = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE",
             "NEGATIVE": "NEGATIVE", "POSITIVE": "POSITIVE"}

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ---------------- MLflow model wrapper (Lab 9 style) ----------------
class HFSentimentModel(mlflow.pyfunc.PythonModel):
    """Wraps a Hugging Face sentiment pipeline so MLflow can save and load it.
    Input : DataFrame with a 'text' column
    Output: DataFrame with text, label (NEGATIVE/POSITIVE), score"""

    def __init__(self, hf_model_id, batch_size=16, max_length=512):
        self.hf_model_id = hf_model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.pipe = None

    def load_context(self, context):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=self.hf_model_id,
                             device=-1, truncation=True, max_length=self.max_length)

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        outputs = self.pipe(texts, batch_size=self.batch_size)
        return pd.DataFrame({
            "text": texts,
            "label": [LABEL_MAP.get(o["label"], o["label"]) for o in outputs],
            "score": [round(float(o["score"]), 4) for o in outputs],
        })


# ---------------- helpers ----------------
def compute_metrics(refs, preds):
    return {
        "accuracy": accuracy_score(refs, preds),
        "macro_f1": f1_score(refs, preds, average="macro"),
        "precision_macro": precision_score(refs, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(refs, preds, average="macro", zero_division=0),
    }


def confusion_plot(refs, preds, title):
    labels = ["NEGATIVE", "POSITIVE"]
    cm = confusion_matrix(refs, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def evaluate_one(key, eval_df, library_versions):
    cfg = MODELS[key]
    run_name = f"{key}_{TASK}_{cfg['short']}"
    print(f"\n=== {run_name}: {cfg['hf_model_id']} ===")

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params({
            "hf_model_id": cfg["hf_model_id"], "task": TASK,
            "role": f"{cfg['role']} candidate", "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH, "truncation": True,
            "eval_samples": len(eval_df), "seed": SEED, "device": DEVICE,
        })
        mlflow.set_tags({
            "student_usn": STUDENT_USN, "library_versions": library_versions,
            "dataset_name": "glue/sst2 (validation)", "hf_model_id": cfg["hf_model_id"],
        })

        # load model (first time downloads it) + one warm-up call, NOT timed
        runner = HFSentimentModel(cfg["hf_model_id"], BATCH_SIZE, MAX_LENGTH)
        runner.load_context(None)
        runner.predict(None, eval_df[["text"]].head(2))

        # timed prediction on all 200 sentences
        start = time.perf_counter()
        out = runner.predict(None, eval_df[["text"]])
        total_s = time.perf_counter() - start

        preds_df = pd.DataFrame({
            "idx": eval_df["idx"], "text": eval_df["text"],
            "reference": eval_df["reference"], "prediction": out["label"],
            "score": out["score"],
        })
        preds_df["correct"] = preds_df["reference"] == preds_df["prediction"]

        metrics = compute_metrics(preds_df["reference"], preds_df["prediction"])
        metrics["total_inference_s"] = total_s
        metrics["avg_latency_ms"] = total_s / len(eval_df) * 1000
        metrics = {k: round(float(v), 4) for k, v in metrics.items()}
        mlflow.log_metrics(metrics)
        print(metrics)

        # artifacts
        pred_path = f"outputs/predictions_{cfg['short']}.csv"
        preds_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path)
        mlflow.log_artifact("data/eval_inputs.csv")

        metrics_path = f"outputs/metrics_summary_{cfg['short']}.json"
        with open(metrics_path, "w") as f:
            json.dump({"run_name": run_name, "hf_model_id": cfg["hf_model_id"], **metrics}, f, indent=2)
        mlflow.log_artifact(metrics_path)

        fig = confusion_plot(preds_df["reference"], preds_df["prediction"], f"Confusion matrix – {key}")
        fig.savefig(f"outputs/confusion_matrix_{cfg['short']}.png", dpi=120)
        mlflow.log_figure(fig, f"confusion_matrix_{cfg['short']}.png")
        plt.close(fig)

        # log the model itself (fresh wrapper, so the big pipeline object isn't pickled)
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=HFSentimentModel(cfg["hf_model_id"], BATCH_SIZE, MAX_LENGTH),
            input_example=pd.DataFrame({"text": ["I loved this movie!", "Terrible acting."]}),
            pip_requirements=[f"mlflow=={mlflow.__version__}", f"transformers=={transformers.__version__}",
                              f"torch=={torch.__version__.split('+')[0]}", "pandas"],
            metadata={"hf_model_id": cfg["hf_model_id"], "task": TASK},
        )
        mlflow.set_tag("model_uri", model_info.model_uri)

        return {"key": key, "run_id": run.info.run_id, "run_name": run_name,
                "metrics": metrics, "preds": preds_df, **cfg}


# ---------------- main ----------------
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    torch.manual_seed(SEED)

    library_versions = (f"transformers={transformers.__version__}, torch={torch.__version__}, "
                        f"mlflow={mlflow.__version__}, scikit-learn={sklearn.__version__}")

    # 1. shared evaluation set: 200 random SST-2 validation sentences, fixed seed
    ds = load_dataset("nyu-mll/glue", "sst2", split="validation").shuffle(seed=SEED).select(range(EVAL_SAMPLES))
    eval_df = pd.DataFrame({
        "idx": ds["idx"],
        "text": [s.strip() for s in ds["sentence"]],
        "reference": ["POSITIVE" if l == 1 else "NEGATIVE" for l in ds["label"]],
    })
    eval_df.to_csv("data/eval_inputs.csv", index=False)
    print(f"Saved {len(eval_df)} evaluation sentences to data/eval_inputs.csv")

    with mlflow.start_run(run_name=f"compare_{TASK}") as parent:
        mlflow.set_tags({"student_usn": STUDENT_USN, "library_versions": library_versions,
                         "dataset_name": "glue/sst2 (validation)"})

        A = evaluate_one("A", eval_df, library_versions)
        B = evaluate_one("B", eval_df, library_versions)

        # 2. apply the Champion rule
        delta = B["metrics"][PRIMARY_METRIC] - A["metrics"][PRIMARY_METRIC]
        latency_ratio = B["metrics"]["avg_latency_ms"] / A["metrics"]["avg_latency_ms"]
        beats_margin = delta >= MARGIN
        passes_guardrail = latency_ratio <= MAX_LATENCY_RATIO
        winner = "B" if (beats_margin and passes_guardrail) else "A"
        winner_cfg = MODELS[winner]

        if winner == "B":
            reason = f"Challenger beat Champion by {delta:+.4f} (>= {MARGIN}) and latency ratio {latency_ratio:.2f}x <= {MAX_LATENCY_RATIO}x"
        elif not beats_margin:
            reason = f"Challenger difference {delta:+.4f} is below the {MARGIN} margin, so the Champion stays"
        else:
            reason = f"Challenger is {latency_ratio:.2f}x slower (> {MAX_LATENCY_RATIO}x guardrail), so the Champion stays"
        print(f"\nDECISION: winner = Model {winner} ({winner_cfg['hf_model_id']})\nReason: {reason}")

        mlflow.log_params({"primary_metric": PRIMARY_METRIC, "margin": MARGIN,
                           "max_latency_ratio": MAX_LATENCY_RATIO,
                           "model_A": A["hf_model_id"], "model_B": B["hf_model_id"],
                           "eval_samples": len(eval_df), "seed": SEED})
        mlflow.log_metrics({"delta_primary_metric": round(delta, 4),
                            "delta_accuracy": round(B["metrics"]["accuracy"] - A["metrics"]["accuracy"], 4),
                            "latency_ratio_B_over_A": round(latency_ratio, 4)})
        mlflow.set_tags({"winner": f"{winner}_{winner_cfg['short']}",
                         "winner_hf_model_id": winner_cfg["hf_model_id"],
                         "winner_run_id": (A if winner == "A" else B)["run_id"],
                         "loser_run_id": (B if winner == "A" else A)["run_id"],
                         "decision_reason": reason})

        # 3. combined metrics file + results table
        summary = {"A": {"hf_model_id": A["hf_model_id"], **A["metrics"]},
                   "B": {"hf_model_id": B["hf_model_id"], **B["metrics"]},
                   "delta_B_minus_A": {k: round(B["metrics"][k] - A["metrics"][k], 4) for k in A["metrics"]},
                   "decision": {"primary_metric": PRIMARY_METRIC, "margin": MARGIN,
                                "winner": winner, "reason": reason}}
        with open("outputs/metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact("outputs/metrics_summary.json")

        table = pd.DataFrame({"metric": list(A["metrics"].keys()),
                              "A_champion": list(A["metrics"].values()),
                              "B_challenger": list(B["metrics"].values())})
        table["difference_B_minus_A"] = (table["B_challenger"] - table["A_champion"]).round(4)
        table.to_csv("outputs/results_table.csv", index=False)
        mlflow.log_artifact("outputs/results_table.csv")
        print("\n" + table.to_string(index=False))

        # 4. comparison plots
        quality = ["accuracy", "macro_f1", "precision_macro", "recall_macro"]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(quality))
        ax.bar([i - 0.2 for i in x], [A["metrics"][m] for m in quality], width=0.4, label="A (Champion)")
        ax.bar([i + 0.2 for i in x], [B["metrics"][m] for m in quality], width=0.4, label="B (Challenger)")
        ax.set_xticks(list(x)); ax.set_xticklabels(quality)
        ax.set_ylim(0.5, 1.0); ax.set_ylabel("score (higher is better)")
        ax.set_title("Champion vs Challenger – quality metrics"); ax.legend()
        fig.tight_layout(); fig.savefig("outputs/compare_metrics.png", dpi=120)
        mlflow.log_figure(fig, "compare_metrics.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["A (Champion)", "B (Challenger)"],
               [A["metrics"]["avg_latency_ms"], B["metrics"]["avg_latency_ms"]], color=["tab:blue", "tab:orange"])
        ax.set_ylabel("avg latency per sentence (ms, lower is better)")
        ax.set_title("Inference speed")
        fig.tight_layout(); fig.savefig("outputs/compare_latency.png", dpi=120)
        mlflow.log_figure(fig, "compare_latency.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(A["preds"]["score"], bins=20, alpha=0.6, label="A (Champion)")
        ax.hist(B["preds"]["score"], bins=20, alpha=0.6, label="B (Challenger)")
        ax.set_xlabel("confidence score of predicted label"); ax.set_ylabel("count")
        ax.set_title("Confidence distribution"); ax.legend()
        fig.tight_layout(); fig.savefig("outputs/confidence_hist.png", dpi=120)
        mlflow.log_figure(fig, "confidence_hist.png"); plt.close(fig)

        # 5. cases where the two models disagree (add your comments in the 'comment' column later)
        both = A["preds"][["idx", "text", "reference"]].copy()
        both["pred_A"] = A["preds"]["prediction"].values
        both["pred_B"] = B["preds"]["prediction"].values
        disagree = both[both["pred_A"] != both["pred_B"]].copy()
        disagree["comment"] = ""
        disagree.to_csv("outputs/disagreements.csv", index=False)
        mlflow.log_artifact("outputs/disagreements.csv")
        print(f"\nModels disagree on {len(disagree)} of {len(both)} sentences (see outputs/disagreements.csv)")

        for f in ["README.md", "requirements.txt"]:
            if os.path.exists(f):
                mlflow.log_artifact(f)

    print("\nDone! Open http://127.0.0.1:5000 to see the runs.")


if __name__ == "__main__":
    main()
