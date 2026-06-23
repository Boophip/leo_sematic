from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.proxy.image_features import (
    IMAGE_FEATURE_COLUMNS,
    extract_image_quality_features,
    extract_image_quality_features_from_path,
)


class ImageFeatureTests(unittest.TestCase):
    def test_solid_image_has_low_texture_features(self) -> None:
        image = Image.new("RGB", (16, 8), color=(128, 128, 128))

        features = extract_image_quality_features(image)

        self.assertEqual(set(features), set(IMAGE_FEATURE_COLUMNS))
        self.assertEqual(features["crop_width"], 16.0)
        self.assertEqual(features["crop_height"], 8.0)
        self.assertAlmostEqual(features["crop_aspect_ratio"], 2.0)
        self.assertAlmostEqual(features["brightness_std"], 0.0)
        self.assertAlmostEqual(features["laplacian_var"], 0.0)
        self.assertAlmostEqual(features["edge_density"], 0.0)
        self.assertAlmostEqual(features["entropy"], 0.0)

    def test_checkerboard_has_more_edges_and_entropy_than_solid_image(self) -> None:
        solid = Image.new("RGB", (16, 16), color=(128, 128, 128))
        checker = Image.new("RGB", (16, 16), color=(0, 0, 0))
        pixels = checker.load()
        for y in range(16):
            for x in range(16):
                if (x + y) % 2 == 0:
                    pixels[x, y] = (255, 255, 255)

        solid_features = extract_image_quality_features(solid)
        checker_features = extract_image_quality_features(checker)

        self.assertGreater(checker_features["brightness_std"], solid_features["brightness_std"])
        self.assertGreater(checker_features["laplacian_var"], solid_features["laplacian_var"])
        self.assertGreater(checker_features["edge_density"], solid_features["edge_density"])
        self.assertGreater(checker_features["entropy"], solid_features["entropy"])

    def test_features_can_be_extracted_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crop.png"
            Image.new("RGB", (10, 20), color=(10, 20, 30)).save(path)

            features = extract_image_quality_features_from_path(path)

            self.assertEqual(features["crop_width"], 10.0)
            self.assertEqual(features["crop_height"], 20.0)
            self.assertAlmostEqual(features["crop_aspect_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
