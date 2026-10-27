import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data import generate_dataset, load_datasets
from src.model import ForecastTransformer, freeze_for_lora


class PipelineTests(unittest.TestCase):
    def test_dataset_is_deterministic(self):
        config = {"generation_seed": 1, "hours": 200, "lookback": 24, "horizon": 3, "source_coops": 1,
                  "train_fraction": 0.7, "validation_fraction": 0.15}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, first_meta = root / "one.npz", root / "one.json"
            second, second_meta = root / "two.npz", root / "two.json"
            one = generate_dataset(config, first, first_meta)
            two = generate_dataset(config, second, second_meta)
            self.assertEqual(one["sha256"], two["sha256"])
            datasets, scaling = load_datasets(first)
            self.assertGreater(len(datasets["train"]), 0)
            self.assertGreater(scaling["target_std"], 0)

    def test_only_lora_parameters_train_after_freeze(self):
        model = ForecastTransformer(4, 6, 32, 4, 1, 64, 48, 0.1, 4)
        freeze_for_lora(model)
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(names)
        self.assertTrue(all("lora_" in name for name in names))
        self.assertEqual(model(torch.zeros(2, 48, 4)).shape, (2, 6))


if __name__ == "__main__":
    unittest.main()
