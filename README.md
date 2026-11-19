# Champion–Challenger Model Comparison (Hugging Face + MLflow)
Vihaan Santhosh Gowda · 1MS25CI144 · Skill Lab (MLOps with MLflow)

## 1. Task and models
- **Task:** Sentiment classification (positive / negative)
- **Dataset:** GLUE SST-2 validation split (`nyu-mll/glue`, `sst2`), 200 sentences, shuffled with seed 42 and saved to `data/eval_inputs.csv`
- **Model A (Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters)
- **Model B (Challenger):** `textattack/bert-base-uncased-SST-2` (~110M parameters). It outputs `LABEL_0` / `LABEL_1`, which are mapped to `NEGATIVE` / `POSITIVE` so that both models use the same labels.
- Both models run on CPU with the same settings (batch_size 16, truncation, max_length 512) on the same 200 inputs in the same order.

## 2. How to run
```bash
# 1. set up (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. start the MLflow tracking server (keep this terminal open)
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# 3. in a second terminal: run both models and log everything
venv\Scripts\python evaluate.py

# 4. register both models and set the @champion / @challenger aliases
venv\Scripts\python register_champion.py

# 5. load the Champion through MLflow and predict on new data
venv\Scripts\python predict_from_mlflow.py
# optional: load the Challenger instead
venv\Scripts\python predict_from_mlflow.py challenger
```
MLflow UI: http://127.0.0.1:5000 · Experiment: `hf-sentiment-champion-challenger` · Registered model: `sentiment-model`

## 3. Champion rule (written before the final run)
1. **Primary metric:** macro-F1 (higher is better).
2. **Margin:** the Challenger must beat the Champion by at least 2 percentage points (0.02) in macro-F1, otherwise the Champion stays. With only 200 samples, 1 point = 2 sentences, which could be random noise, so a 2-point margin (4 sentences) is safer.
3. **Guardrail:** the Challenger is rejected if its average latency is more than 5x the Champion's, even if it is more accurate.
4. **Tie-breaker:** if the difference is within the margin, the Champion is kept (it is smaller and faster).
5. **Directions:** accuracy, macro-F1, precision, recall → higher is better. avg_latency_ms, total_inference_s → lower is better.

## 4. Results (200 SST-2 validation sentences)

| Metric | A – DistilBERT (Champion) | B – BERT-base (Challenger) | Difference (B − A) | Better is |
|---|---|---|---|---|
| Accuracy | 0.9150 | 0.9500 | +0.0350 | higher |
| **Macro-F1 (primary)** | **0.9141** | **0.9497** | **+0.0356** | higher |
| Precision (macro) | 0.9183 | 0.9505 | +0.0322 | higher |
| Recall (macro) | 0.9121 | 0.9491 | +0.0370 | higher |
| Avg latency (ms / sentence) | 33.66 | 43.60 | +9.95 (1.30x) | lower |
| Total inference time (s) | 6.73 | 8.72 | +1.99 | lower |

- Correct predictions: A = 183 / 200, B = 190 / 200
- The two models disagree on 11 of the 200 sentences (`outputs/disagreements.csv`)

### Qualitative examples: where the models disagree
The models disagree on 11 sentences. BERT (B) is right on 9 and DistilBERT (A) is right on 2. All 11, with comments, are in `outputs/disagreements.csv`. Five examples:

| # | Sentence (SST-2) | Truth | A | B | Comment |
|---|---|---|---|---|---|
| 770 | "i do n't think i laughed out loud once ." | NEG | POS | NEG | Simple negation. A misses the "do n't", B handles it. |
| 107 | "what the director ca n't do is make either of val kilmer 's two personas interesting or worth caring about ." | NEG | POS | NEG | Negation over positive words ("interesting", "worth caring"). A is fooled by the positive words. |
| 356 | "jason x is positively anti-darwinian : nine sequels and 400 years later , the teens are none the wiser ..." | NEG | POS | NEG | Sarcasm: "positively" is ironic. B catches it. |
| 143 | "a solid film ... but more conscientious than it is truly stirring ." | POS | POS | NEG | Mixed review. A is right; B gives too much weight to the "but" clause. |
| 422 | "it moves quickly , adroitly , and without fuss ; it does n't give you time to reflect on the inanity ..." | POS | POS | NEG | Praise followed by a mild complaint. A is right; B over-weights "inanity". |

Pattern: the larger BERT model is better at negation and sarcasm, but it can over-react to a negative clause inside an otherwise positive review.

MLflow runs: `compare_sentiment` (parent / summary run) with nested runs `A_sentiment_distilbert-sst2` and `B_sentiment_bert-sst2`.
Registry: `sentiment-model` version 1 (BERT-base) = **@champion**, version 2 (DistilBERT) = **@challenger**.

## 5. Decision
**New Champion: Model B, `textattack/bert-base-uncased-SST-2`.**

Applying the rule exactly as written, the Challenger improves macro-F1 from 0.9141 to 0.9497, a gain of +3.56 percentage points. This is above the 2-point margin. Its average latency is 43.6 ms per sentence against 33.7 ms for DistilBERT, which makes it only 1.30x slower, well inside the 5x guardrail. Both conditions are met, so BERT-base replaces DistilBERT as the Champion. The decision is logged in the `compare_sentiment` run (`winner`, `delta_primary_metric = 0.0356`), and the `@champion` alias was moved to it in the registry.

Is the difference meaningful? With 200 sentences, the gap is 7 sentences (190 vs 183 correct). The models disagree on only 11 sentences: BERT is right on 9 of them (mostly negation and sarcasm) and DistilBERT on 2. A sign test on these 11 gives p ≈ 0.065. This is a consistent improvement across accuracy, precision, recall and F1, but it is not conclusive at the usual 5% level, and a larger evaluation set would be needed to confirm it.

Trade-offs: BERT-base is about 1.6x larger (110M vs 67M parameters) and about 1.3x slower on this CPU. For a batch or low-traffic use case this cost is acceptable for ~3.5 points of accuracy. For a latency-critical service, DistilBERT stays a strong fallback, which is why it is kept as `@challenger`.

## 6. Limitations and declaration
- **Sample size:** only 200 sentences, so a few sentences change the metrics by whole percentage points (see the significance note above).
- **Same-domain data:** both models were fine-tuned on SST-2 training data and are evaluated on the SST-2 validation split (movie-review sentences). Results may differ on other domains (tweets, product reviews, Indian English, etc.).
- **Binary only:** SST-2 has no neutral class, so neutral or mixed sentences are always forced into positive or negative.
- **Latency** was measured once on a single laptop CPU after a warm-up call. It depends on hardware and background load, so only the ratio between the models is meaningful.
- **New data** (`data/new_data.csv`) is 8 sentences written by me, not taken from the evaluation set. The Champion's predictions are in `outputs/new_data_predictions.csv`.
- **AI assistance:** I used Claude (Anthropic) as an AI assistant to help plan the project and write the scripts. I have gone through the code and can explain every part of it.

## Repository structure
```
README.md
requirements.txt
evaluate.py              # runs both models, logs runs, metrics, artifacts and models to MLflow
register_champion.py     # registers both models, sets @champion / @challenger aliases
predict_from_mlflow.py   # loads models:/sentiment-model@champion and predicts on new data
data/       eval_inputs.csv, new_data.csv
outputs/    predictions_*.csv, metrics_summary*.json, results_table.csv, disagreements.csv,
            confusion matrices, comparison plots, new_data_predictions.csv
screenshots/ mlflow_compare.png, registry_aliases.png, predict_output.png
```
