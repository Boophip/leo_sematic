from __future__ import annotations

import unittest

from PIL import Image

from src.profiling.compression import compress_image, resolve_compression_spec


class RoiCompressionTests(unittest.TestCase):
    def test_local_has_no_communication_payload(self) -> None:
        image = Image.new("RGB", (12, 40), color=(20, 90, 140))

        result = compress_image(image, "local")

        self.assertEqual(result.compression_level, "local")
        self.assertEqual(result.encoded_format, "none")
        self.assertEqual(result.compressed_bytes, 0)
        self.assertEqual(result.encode_ms, 0.0)
        self.assertEqual(result.decode_ms, 0.0)
        self.assertEqual((result.output_width, result.output_height), (12, 40))

    def test_beta_levels_have_expected_formats_and_size_rules(self) -> None:
        image = Image.new("RGB", (12, 40), color=(20, 90, 140))

        beta0 = compress_image(image, 0)
        beta1 = compress_image(image, "beta_1")
        beta2 = compress_image(image, 2)
        beta3 = compress_image(image, 3)

        self.assertEqual(beta0.encoded_format, "PNG")
        self.assertEqual((beta0.output_width, beta0.output_height), (12, 40))
        self.assertEqual(beta1.encoded_format, "JPEG")
        self.assertEqual((beta1.output_width, beta1.output_height), (12, 40))
        self.assertEqual(beta2.encoded_format, "JPEG")
        self.assertGreaterEqual(min(beta2.output_width, beta2.output_height), 16)
        self.assertEqual(beta3.encoded_format, "JPEG")
        self.assertGreaterEqual(min(beta3.output_width, beta3.output_height), 8)
        for result in (beta0, beta1, beta2, beta3):
            self.assertGreater(result.compressed_bytes, 0)
            self.assertGreater(result.output_width, 0)
            self.assertGreater(result.output_height, 0)

    def test_invalid_level_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_compression_spec("beta_9")


if __name__ == "__main__":
    unittest.main()
