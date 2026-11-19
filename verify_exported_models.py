import json
import joblib
from pathlib import Path

folder = Path("boreman-ai-api/models")
manifest = json.loads((folder / "model_manifest.json").read_text(encoding="utf-8"))

for name, info in manifest["models"].items():
    model = joblib.load(folder / info["file"])
    expected = list(model.feature_names_in_)
    declared = info["features"]
    print(
        name,
        "| feature count:", len(expected),
        "| manifest match:", expected == declared,
        "| model loaded:", model is not None
    )
