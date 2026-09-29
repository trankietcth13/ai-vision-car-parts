import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline.fetch_vehicle_datasets import write_dataset_provenance


class TestDatasetProvenance(unittest.TestCase):
    def test_writes_image_source_and_yolo_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = output / "sample"
            image = root / "train" / "images" / "part.jpg"
            label = root / "train" / "labels" / "part.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"test-image")
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            result = write_dataset_provenance(
                output,
                {
                    "id": "sample",
                    "name": "Sample",
                    "source_page": "https://example.test/dataset",
                    "license": "MIT",
                    "citation": "Example",
                    "classes": ["part"],
                    "version": 1,
                },
            )

            source = json.loads((root / "DATASET_SOURCE.json").read_text(encoding="utf-8"))
            record = json.loads((root / "IMAGE_PROVENANCE.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(result["image_count"], 1)
            self.assertEqual(source["source_page"], "https://example.test/dataset")
            self.assertEqual(record["image_path"].replace("\\", "/"), "train/images/part.jpg")
            self.assertEqual(
                record["annotation_path"].replace("\\", "/"), "train/labels/part.txt"
            )
            self.assertEqual(len(record["sha256"]), 64)

    def test_includes_extensionless_images_by_magic_bytes_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = output / "sample"
            image = root / "images" / "augmentation_001"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"\xff\xd8\xff\xe0" + b"jpeg-payload")
            (root / "not_an_image").write_text("checkpoint", encoding="utf-8")

            result = write_dataset_provenance(
                output,
                {
                    "id": "sample",
                    "name": "Sample",
                    "source_page": "https://example.test/dataset",
                    "license": "MIT",
                    "include_extensionless_images": True,
                },
            )

            records = (root / "IMAGE_PROVENANCE.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(result["image_count"], 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(
                json.loads(records[0])["image_path"].replace("\\", "/"),
                "images/augmentation_001",
            )


if __name__ == "__main__":
    unittest.main()
