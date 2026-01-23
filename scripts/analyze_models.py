import json
import os
from typing import Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def max_with_index(values):
    """
    Returns (index, value) of the maximum element.
    Supports scalar or list.
    """
    if isinstance(values, list):
        max_val = max(values)
        return values.index(max_val), max_val
    else:
        return None, values


def extract_best_metrics(logs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Finds the highest value across all keys starting with:
    - train_acc_
    - val_acc_
    - test_acc_
    """
    result = {}

    for prefix in ("train_acc", "val_acc", "test_acc"):
        best_value = None
        best_index = None

        for key, value in logs.items():
            if not key.startswith(prefix + "_"):
                continue

            idx, val = max_with_index(value)

            if best_value is None or val > best_value:
                best_value = val
                best_index = idx

        if best_value is not None:
            result[prefix] = {
                "index": best_index,
                "value": best_value,
            }

    return result


def process_directory(dir_path: str) -> Dict[str, Any] | None:
    metadata_path = os.path.join(dir_path, "metadata.json")
    logs_path = os.path.join(dir_path, "logs.json")

    if not os.path.isfile(metadata_path) or not os.path.isfile(logs_path):
        return None

    metadata = load_json(metadata_path)
    logs = load_json(logs_path)

    # Model name = subdirectory name
    model_name = os.path.basename(os.path.normpath(dir_path))

    # ---- source models ----
    source_models = []
    for model in metadata.get("source_models", []):
        source_models.append({
            "val_acc": model.get("val_acc"),
            "test_acc": model.get("test_acc"),
        })

    # ---- destination (merged) metrics ----
    dest_metrics = extract_best_metrics(logs)

    return {
        "model": model_name,
        "source": source_models,
        "dest": dest_metrics,
    }


def collect_results(root_dir: str):
    """
    Walks all subdirectories and collects results.
    Returns a list of per-model records.
    """
    results = []

    for root, _, _ in os.walk(root_dir):
        result = process_directory(root)
        if result is not None:
            results.append(result)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to search from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="collated_results.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    all_results = collect_results(args.root)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Collected results from {len(all_results)} models")
    print(f"Saved to {args.output}")
