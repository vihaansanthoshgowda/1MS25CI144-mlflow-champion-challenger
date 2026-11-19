# Champion–Challenger Model Comparison (Hugging Face + MLflow)
Vihaan Santhosh Gowda · 1MS25CI144 · Skill Lab (MLOps with MLflow)

## 1. Task and models
- **Task:** Sentiment classification (positive / negative)
- **Dataset:** GLUE SST-2 validation split (`nyu-mll/glue`, `sst2`), 200 sentences, shuffled with seed 42 and saved to `data/eval_inputs.csv`
- **Model A (Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters)
- **Model B (Challenger):** `textattack/bert-base-uncased-SST-2` (~110M parameters). It outputs `LABEL_0` / `LABEL_1`, which are mapped to `NEGATIVE` / `POSITIVE` so that both models use the same labels.
- Both models run on CPU with the same settings (batch_size 16, truncation, max_length 512) on the same 200 inputs in the same order.

## 2. How to run
(to be filled in)

## 3. Champion rule (written before the final run)
1. **Primary metric:** macro-F1 (higher is better).
2. **Margin:** the Challenger must beat the Champion by at least 2 percentage points (0.02) in macro-F1, otherwise the Champion stays. With only 200 samples, 1 point = 2 sentences, which could be random noise, so a 2-point margin (4 sentences) is safer.
3. **Guardrail:** the Challenger is rejected if its average latency is more than 5x the Champion's, even if it is more accurate.
4. **Tie-breaker:** if the difference is within the margin, the Champion is kept (it is smaller and faster).
5. **Directions:** accuracy, macro-F1, precision, recall → higher is better. avg_latency_ms, total_inference_s → lower is better.

## 4. Results
(to be filled in)

## 5. Decision
(to be filled in)

## 6. Limitations and declaration
(to be filled in)
