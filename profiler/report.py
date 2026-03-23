import json


def build_report(df):
    """
    Build a structured report from profiler DataFrame.

    Args:
        df: pandas DataFrame from ModelProfiler (p.df)

    Returns:
        dict grouped by augmentation (tag)
    """
    if df is None or len(df) == 0:
        return {}

    report = {}

    # group by augmentation tag
    if "tag" in df.columns:
        grouped = df.groupby("tag")
    else:
        # fallback if no tags present
        grouped = [("no_tag", df)]

    for tag, group in grouped:
        # average across rows for that augmentation
        mean_vals = group.mean(numeric_only=True).to_dict()

        report[tag] = mean_vals

    return report


def print_report(report):
    """
    Clean console output
    """
    if not report:
        print("No report data available.")
        return

    for tag, metrics in report.items():
        print(f"\n=== AUGMENTATION: {tag} ===")

        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


def save_report(report, path="report.json"):
    """
    Save report to JSON file
    """
    with open(path, "w") as f:
        json.dump(report, f, indent=4)