import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pcf import freeze, model


class FreezeTests(unittest.TestCase):
    def test_source_drift_invalidates_the_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.py"
            source.write_text("original")
            spec = Path(directory) / "spec.json"
            spec.write_text(json.dumps({
                "source_sha256": {str(source): freeze.file_hash(source)}, "validation_sha256": {},
            }))
            with patch.object(freeze, "SPEC", spec):
                freeze.load_frozen_spec()
                source.write_text("changed")
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    freeze.load_frozen_spec()

    def test_missing_freeze_prevents_reading_holdout_data(self):
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(freeze, "SPEC", Path(directory) / "missing.json"),
                  patch.object(model.pd, "read_parquet") as read):
                with self.assertRaisesRegex(ValueError, "Freeze the validated"):
                    model.run(True)
                read.assert_not_called()

    def test_existing_holdout_output_prevents_reading_it_again(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            (artifacts / "forecasts_holdout_2025.parquet").touch()
            with (patch.object(model, "ARTIFACTS", artifacts),
                  patch.object(freeze, "load_frozen_spec", return_value={}),
                  patch.object(model.pd, "read_parquet") as read):
                with self.assertRaisesRegex(FileExistsError, "already exist"):
                    model.run(True)
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
