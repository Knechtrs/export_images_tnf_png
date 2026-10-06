import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from lif_pipeline import (
    add_scale_bar,
    contrast_image,
    parse_ome_pixel_sizes_um,
    process_tiff,
    read_pixel_size_um,
    save_metadata_json,
)


class LifPipelineTests(unittest.TestCase):
    def test_contrast_stretch_uses_requested_percentiles(self) -> None:
        image = contrast_image(np.array([[0, 10], [100, 1000]], dtype=np.uint16), 0, 100)
        self.assertEqual(image.getpixel((0, 0)), 0)
        self.assertEqual(image.getpixel((1, 1)), 255)

    def test_reads_ome_physical_size_in_micrometers(self) -> None:
        ome_xml = (
            '<OME><Image><Pixels PhysicalSizeX="0.25" '
            'PhysicalSizeXUnit="µm"/></Image></OME>'
        )
        self.assertEqual(read_pixel_size_um(ome_xml), 0.25)
        self.assertIsNone(read_pixel_size_um(None))

    def test_parses_per_series_calibration_from_lif_ome_metadata(self) -> None:
        ome_xml = (
            '<OME><Image Name="series one"><Pixels PhysicalSizeX="0.2271617" '
            'PhysicalSizeXUnit="µm"/></Image></OME>'
        )
        self.assertEqual(parse_ome_pixel_sizes_um(ome_xml), {"series one": 0.2271617})

    def test_scale_bar_has_no_text_above_it(self) -> None:
        image = add_scale_bar(Image.new("L", (128, 96)), 0.25, 10)
        self.assertEqual(np.asarray(image.crop((0, 0, 128, 70))).max(), 0)

    def test_creates_png_and_scale_bar_display_tiff_without_changing_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "series.tif"
            original = np.arange(128 * 96, dtype=np.uint16).reshape(96, 128)
            tifffile.imwrite(
                raw,
                original,
                ome=True,
                metadata={"axes": "YX", "PhysicalSizeX": 0.25, "PhysicalSizeXUnit": "µm"},
            )

            png_dir = root / "png"
            display_tiff_dir = root / "tiff_display"
            count, pixel_size_um = process_tiff(raw, png_dir, display_tiff_dir, 0.5, 99.9, 5)

            self.assertEqual(count, 1)
            self.assertEqual(pixel_size_um, 0.25)
            self.assertTrue((png_dir / "series.png").is_file())
            self.assertTrue((display_tiff_dir / "series.tif").is_file())
            np.testing.assert_array_equal(tifffile.imread(raw), original)

    def test_writes_metadata_json_with_scale_bar_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = save_metadata_json(
                root,
                root / "sample.lif",
                "<OME />",
                100,
                0.5,
                99.9,
                [{"name": "series one", "scale_bar": {"length_um": 100}}],
            )
            document = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(document["scale_bar"]["length_um"], 100)
            self.assertEqual(document["series"][0]["name"], "series one")
            self.assertEqual(document["source_lif_ome_xml"], "<OME />")


if __name__ == "__main__":
    unittest.main()