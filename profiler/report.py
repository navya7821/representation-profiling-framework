import json


def build_report(results):
    """
    Takes raw results from engine and returns structured report (dict)
    """
    report = {}

    for aug_name, data in results.items():
        report[aug_name] = {
            "top_sensitive_layers": data["top_sensitive_layers"],
            "sensitivity": data["sensitivity"],
            "stability": data["stability"],
            "embedding_robustness": data["embedding_robustness"],
        }

    return report


def print_report(report):
    """
    Clean console output (replaces messy prints in main.py)
    """
    for aug_name, data in report.items():
        print(f"\n=== AUGMENTATION: {aug_name} ===")

        print("\nTop Sensitive Layers:")
        for layer in data["top_sensitive_layers"]:
            print(f"  - {layer}")

        print("\nSensitivity:")
        for layer, vals in data["sensitivity"].items():
            print(
                f"  {layer}: "
                f"CKA={vals['cka']:.4f}, "
                f"L2_norm={vals['l2_normalised']:.4f}, "
                f"Score={vals['score']:.4f}"
            )

        print("\nStability:")
        for layer, vals in data["stability"].items():
            gram_val = (
                f"{vals['gram']:.4f}" if vals["gram"] is not None else "N/A"
            )
            print(
                f"  {layer}: "
                f"Cosine={vals['cosine']:.4f}, "
                f"Gram={gram_val}"
            )

        r = data["embedding_robustness"]
        print("\nEmbedding Robustness:")
        print(
            f"  Cosine={r['cosine']:.4f}, "
            f"L2_norm={r['l2_normalised']:.4f}, "
            f"Score={r['score']:.4f}"
        )


def save_report(report, path="report.json"):
    """
    Save report to JSON file
    """
    with open(path, "w") as f:
        json.dump(report, f, indent=4)