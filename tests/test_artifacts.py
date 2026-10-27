import csv
import hashlib
import json
import statistics
import tempfile
import unittest
from pathlib import Path

import yaml

from src.data import generate_dataset


ROOT = Path(__file__).resolve().parents[1]


class ArtifactIntegrityTests(unittest.TestCase):
    def test_confirmatory_matrix_is_complete_and_unique(self):
        files = sorted((ROOT / "artifacts/results").glob("rank*_seed*.json"))
        records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        pairs = {(int(row["rank"]), int(row["seed"])) for row in records}
        expected = {(rank, seed) for rank in (4, 8, 16) for seed in (101, 202, 303, 404, 505)}
        self.assertEqual(len(records), 15)
        self.assertEqual(pairs, expected)

    def test_seed_csv_and_summary_match_raw_records(self):
        result_dir = ROOT / "artifacts/results"
        raw = {
            (int(row["rank"]), int(row["seed"])): row
            for row in (
                json.loads(path.read_text(encoding="utf-8"))
                for path in result_dir.glob("rank*_seed*.json")
            )
        }
        with (result_dir / "seed_level_results.csv").open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 15)
        for row in csv_rows:
            source = raw[(int(row["rank"]), int(row["seed"]))]
            for key in ("mape_percent", "mae_kw", "rmse_kw", "training_seconds", "peak_gpu_memory_mb"):
                self.assertAlmostEqual(float(row[key]), float(source[key]), places=9)

        with (result_dir / "summary_by_rank.csv").open(newline="", encoding="utf-8") as handle:
            summaries = {int(row["rank"]): row for row in csv.DictReader(handle)}
        for rank in (4, 8, 16):
            values = [float(row["mape_percent"]) for row in csv_rows if int(row["rank"]) == rank]
            summary = summaries[rank]
            self.assertEqual(int(summary["n_seeds"]), 5)
            self.assertAlmostEqual(float(summary["mean_mape_percent"]), statistics.mean(values), places=9)
            self.assertAlmostEqual(float(summary["std_mape_percent"]), statistics.stdev(values), places=9)
            self.assertAlmostEqual(float(summary["min_mape_percent"]), min(values), places=9)
            self.assertAlmostEqual(float(summary["max_mape_percent"]), max(values), places=9)

    def test_dataset_hash_matches_provenance(self):
        data_path = ROOT / "artifacts/data/synthetic_demand.npz"
        metadata = json.loads((ROOT / "artifacts/data/synthetic_demand_metadata.json").read_text(encoding="utf-8"))
        if data_path.exists():
            digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        else:
            config = yaml.safe_load((ROOT / "configs/ablation.yaml").read_text(encoding="utf-8"))
            with tempfile.TemporaryDirectory() as directory:
                temporary_root = Path(directory)
                generated = generate_dataset(
                    config["data"],
                    temporary_root / "synthetic_demand.npz",
                    temporary_root / "synthetic_demand_metadata.json",
                )
                digest = generated["sha256"]
        self.assertEqual(digest, metadata["sha256"])


if __name__ == "__main__":
    unittest.main()
