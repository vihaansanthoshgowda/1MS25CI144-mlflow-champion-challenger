"""
register_champion.py
Finds the latest compare_sentiment run, reads the decision (winner tag) and
registers both models under one name with aliases @champion and @challenger.
No run IDs are hard-coded: they are looked up from MLflow.
"""
import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# 1. latest summary run
runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                          filter_string="tags.mlflow.runName = 'compare_sentiment'",
                          order_by=["start_time DESC"], max_results=1, output_format="list")
if not runs:
    raise SystemExit("No compare_sentiment run found. Run evaluate.py first.")
parent = runs[0]
winner_run_id = parent.data.tags["winner_run_id"]
loser_run_id = parent.data.tags["loser_run_id"]
print(f"Summary run: {parent.info.run_id}")
print(f"Winner: {parent.data.tags['winner']}  ({parent.data.tags['decision_reason']})")

# 2. register both models and set aliases
for alias, run_id in [("champion", winner_run_id), ("challenger", loser_run_id)]:
    run = client.get_run(run_id)
    model_uri = run.data.tags["model_uri"]
    hf_id = run.data.params["hf_model_id"]

    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "hf_model_id", hf_id)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "source_run", run.info.run_name)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, mv.version)
    print(f"Registered {run.info.run_name} ({hf_id}) as version {mv.version} -> @{alias}")

print(f"\nDone. Check the 'Models' tab at {TRACKING_URI}")
