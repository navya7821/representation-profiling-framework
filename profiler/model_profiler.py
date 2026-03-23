import torch
from collections import defaultdict

from hooks import FeatureExtractor

# Lazy pandas import (Kornia style)
from kornia.core.external import LazyLoader
pandas = LazyLoader("pandas")

# Metric registry (clean + scalable)
from profiler.metrics.cosine import cosine_similarity
from profiler.metrics.linear import linear_similarity

METRIC_REGISTRY = {
    "cosine": cosine_similarity,
    "linear": linear_similarity,
}


class ModelProfiler:
    def __init__(self, model, layers=None, processing="flatten"):
        self.model = model.eval()
        self.layers = layers
        self.processing = processing

        # storage: group -> list of dicts {"features": ..., "tag": ...}
        self.storage = defaultdict(list)

        # results
        self.results = None
        self._df = None

        self.extractor = None

    def __enter__(self):
        self.extractor = FeatureExtractor(
            self.model,
            self.layers,
            processing=self.processing
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.extractor, "remove_hooks"):
            self.extractor.remove_hooks()

    def __call__(self, group="default", tag=None, **inputs):
        """
        Run model and store features under a group

        Args:
            group: group name (e.g., "group_a", "group_b")
            tag: optional metadata (e.g., augmentation name)
            **inputs: model inputs (supports multi-input models)
        """
        with torch.no_grad():
            features = self.extractor(**inputs)

        self.storage[group].append({
            "features": features,
            "tag": tag
        })

    def compute(self, metrics, groups):
        """
        Compute metrics between groups
        Example: groups=["group_a", "group_b"]
        """
        assert len(groups) == 2, "Currently supports pairwise group comparison"

        g1, g2 = groups
        feats1 = self.storage[g1]
        feats2 = self.storage[g2]

        if len(feats1) == 0 or len(feats2) == 0:
            raise RuntimeError("One of the groups has no data.")

        #  Safe pairing
        min_len = min(len(feats1), len(feats2))

        results = []

        for i in range(min_len):
            f1_entry = feats1[i]
            f2_entry = feats2[i]

            f1 = f1_entry["features"]
            f2 = f2_entry["features"]

            # Prefer tag from group_b (augmentation), fallback to group_a
            tag = f2_entry.get("tag", None) or f1_entry.get("tag", None)

            row = {}

            #  Attach metadata
            if tag is not None:
                row["tag"] = tag

            for metric_name in metrics:
                if metric_name not in METRIC_REGISTRY:
                    raise ValueError(f"Unknown metric: {metric_name}")

                metric_fn = METRIC_REGISTRY[metric_name]
                metric_output = metric_fn(f1, f2)  # dict: {layer: value}

                #  Flatten layer-wise outputs
                for layer, val in metric_output.items():
                    key = f"{metric_name}_{layer}"
                    row[key] = val

            results.append(row)

        self.results = results

        #  Create DataFrame
        self._df = pandas.DataFrame(results)

    def print(self):
        if self._df is not None:
            print(self._df)
        else:
            print("No results computed yet.")

    def save_as_report(self, path):
        if self._df is None:
            raise RuntimeError("Run compute() before saving report.")

        if path.endswith(".csv"):
            self._df.to_csv(path, index=False)
        elif path.endswith(".json"):
            self._df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError("Unsupported format. Use .csv or .json")

    @property
    def df(self):
        return self._df