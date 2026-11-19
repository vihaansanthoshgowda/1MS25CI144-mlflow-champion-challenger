"""
predict_from_mlflow.py
Loads whichever model currently has the @champion alias and predicts on NEW data.
No Hugging Face import here: everything goes through MLflow.
"""
import sys

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-model"
ALIAS = sys.argv[1] if len(sys.argv) > 1 else "champion"   # optional: python predict_from_mlflow.py challenger

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# 1. which version does the alias point to right now?
mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS)
model_uri = f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}"
print(f"Loading {model_uri}")
print(f"  -> registered version : {mv.version}")
print(f"  -> Hugging Face model : {mv.tags.get('hf_model_id', 'unknown')}")
print(f"  -> source run         : {mv.tags.get('source_run', mv.run_id)}")

# 2. load through MLflow only
model = mlflow.pyfunc.load_model(model_uri)
meta = (model.metadata.metadata or {}) if model.metadata else {}
print(f"  -> model metadata     : {meta}\n")

# 3. predict on brand-new sentences (not from the evaluation set)
new_df = pd.read_csv("data/new_data.csv")
result = model.predict(new_df[["text"]])
result.insert(0, "model_version", mv.version)
result.insert(1, "alias", ALIAS)

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 160)
print(result[["text", "label", "score"]].to_string(index=False))

out_path = "outputs/new_data_predictions.csv"
result.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")
