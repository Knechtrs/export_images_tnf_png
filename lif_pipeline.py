from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw


DEFAULT_BFCONVERT = os.environ.get("BFCONVERT", "bfconvert.bat")
UNIT_TO_UM = {
    "m": 1_000_000,
    "mm": 1_000,
    "um": 1,
    "µm": 1,
    "μm": 1,
    "micrometer": 1,
    "micrometers": 1,
    "nm": 0.001,
}


def read_pixel_size_um(ome_xml: str | None) -> float | None:
    if not ome_xml:
        return None
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None

    pixels = next((node for node in root.iter() if node.tag.endswith("Pixels")), None)
    if pixels is None:
        return None
    return physical_size_x_um(pixels.attrib)


def physical_size_x_um(attributes: dict[str, str]) -> float | None:
    try:
        size_x = float(attributes["PhysicalSizeX"])
    except (KeyError, ValueError):
        return None
    unit = attributes.get("PhysicalSizeXUnit", "µm").lower()
    factor = UNIT_TO_UM.get(unit)
    return size_x * factor if factor is not None and size_x > 0 else None


def extract_ome_xml(output: str) -> str | None:
    xml_start = output.find("<OME")
    xml_end = output.rfind("</OME>")
    if xml_start < 0 or xml_end < xml_start:
        return None
    return output[xml_start : xml_end + len("</OME>")]


def parse_ome_series_metadata(ome_xml: str | None) -> dict[str, dict[str, object]]:
    if ome_xml is None:
        return {}
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return {}

    series_metadata: dict[str, dict[str, object]] = {}
    for image in root.iter():
        if image.tag.rsplit("}", 1)[-1] != "Image":
            continue
        name = image.attrib.get("Name")
        pixels = next((node for node in image if node.tag.endswith("Pixels")), None)
        if name is None or pixels is None:
            continue
        metadata: dict[str, object] = {
            "name": name,
            "ome_image_id": image.attrib.get("ID"),
            "pixels": dict(pixels.attrib),
        }
        size_x_um = physical_size_x_um(pixels.attrib)
        if size_x_um is not None:
            metadata["physical_size_x_um"] = size_x_um
        series_metadata[name.casefold()] = metadata
    return series_metadata


def parse_ome_pixel_sizes_um(output: str) -> dict[str, float]:
    ome_xml = extract_ome_xml(output)
    series_metadata = parse_ome_series_metadata(ome_xml)
    pixel_sizes = {}
    for name, metadata in series_metadata.items():
        size_x_um = metadata.get("physical_size_x_um")
        if isinstance(size_x_um, (int, float)):
            pixel_sizes[name] = float(size_x_um)
    return pixel_sizes


def read_lif_metadata(
    lif_path: Path, bfconvert: Path
) -> tuple[str | None, dict[str, dict[str, object]]]:
    showinf = bfconvert.with_name("showinf" + bfconvert.suffix)
    if not showinf.is_file():
        logging.warning("%s not found; source metadata unavailable for %s", showinf.name, lif_path.name)
        return None, {}
    result = subprocess.run(
        [str(showinf), "-nopix", "-omexml-only", str(lif_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ome_xml = extract_ome_xml(result.stdout)
    return ome_xml, parse_ome_series_metadata(ome_xml)


def contrast_image(plane: np.ndarray, lower: float, upper: float) -> Image.Image:
    plane = np.squeeze(np.asarray(plane))
    if plane.ndim == 3 and plane.shape[0] in (3, 4) and plane.shape[-1] not in (3, 4):
        plane = np.moveaxis(plane, 0, -1)
    if plane.ndim not in (2, 3) or (plane.ndim == 3 and plane.shape[-1] not in (3, 4)):
        raise ValueError(f"Expected a grayscale or RGB image plane, got shape {plane.shape}")
    if plane.ndim == 3 and plane.shape[-1] == 4:
        plane = plane[..., :3]

    values = plane.astype(np.float64, copy=False)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Image plane contains no finite pixel values")
    low, high = np.percentile(finite_values, (lower, upper))
    if high <= low:
        display = np.zeros(values.shape, dtype=np.uint8)
    else:
        scaled = (values - low) * (255.0 / (high - low))
        display = np.clip(np.nan_to_num(scaled, nan=0, posinf=255, neginf=0), 0, 255)
        display = display.astype(np.uint8)
    return Image.fromarray(display)


def add_scale_bar(image: Image.Image, pixel_size_um: float, length_um: float) -> Image.Image:
    bar_width = round(length_um / pixel_size_um)
    width, height = image.size
    margin = max(10, round(min(width, height) * 0.03))
    if bar_width < 1 or bar_width + margin * 2 > width:
        raise ValueError(
            f"A {length_um:g} um scale bar does not fit in the {width}-pixel image width"
        )

    result = image.convert("RGB") if image.mode not in ("L", "RGB") else image.copy()
    draw = ImageDraw.Draw(result)
    thickness = max(3, round(min(width, height) * 0.008))
    right = width - margin
    left = right - bar_width
    baseline = height - margin
    draw.line((left, baseline, right, baseline), fill=0, width=thickness + 2)
    draw.line((left, baseline, right, baseline), fill=255, width=thickness)
    return result


def center_crop(image: Image.Image, width_px: int, height_px: int) -> Image.Image:
    width, height = image.size
    crop_w, crop_h = min(width_px, width), min(height_px, height)
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


def convert_lif(lif_path: Path, raw_dir: Path, bfconvert: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = raw_dir / "%n.tif"
    subprocess.run([str(bfconvert), str(lif_path), str(output_pattern)], check=True)
    return sorted(raw_dir.glob("*.tif"))


def process_tiff(
    tiff_path: Path,
    png_dir: Path,
    display_tiff_dir: Path,
    lower: float,
    upper: float,
    scale_bar_um: float | None,
    lif_pixel_size_um: float | None = None,
    crop_dir: Path | None = None,
    crop_tiff_dir: Path | None = None,
    crop_px: int | None = None,
    crop_um: float | None = None,
    crop_scale_bar_um: float | None = None,
) -> tuple[int, float | None]:
    png_dir.mkdir(parents=True, exist_ok=True)
    if scale_bar_um is not None:
        display_tiff_dir.mkdir(parents=True, exist_ok=True)
    do_crop = crop_dir is not None and (crop_px is not None or crop_um is not None)
    if crop_scale_bar_um is None:
        crop_scale_bar_um = scale_bar_um
    if do_crop:
        crop_dir.mkdir(parents=True, exist_ok=True)
        if crop_scale_bar_um is not None:
            crop_tiff_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with tifffile.TiffFile(tiff_path) as tif:
        pixel_size_um = read_pixel_size_um(tif.ome_metadata) or lif_pixel_size_um
        pages = list(tif.pages)
        if not pages:
            raise ValueError("TIFF contains no image pages")
        if scale_bar_um is not None and pixel_size_um is None:
            logging.warning("No physical pixel size in OME metadata; scale bar omitted for %s", tiff_path)

        crop_size = crop_px
        if do_crop and crop_size is None:
            if pixel_size_um is None:
                logging.warning("No physical pixel size; --crop-um skipped for %s", tiff_path)
                do_crop = False
            else:
                crop_size = round(crop_um / pixel_size_um)

        for index, page in enumerate(pages, start=1):
            image = contrast_image(page.asarray(), lower, upper)
            if do_crop:
                # Contrast is computed on the full plane so crops match the uncropped PNGs.
                cropped = center_crop(image, crop_size, crop_size)
                if crop_scale_bar_um is not None and pixel_size_um is not None:
                    cropped = add_scale_bar(cropped, pixel_size_um, crop_scale_bar_um)
                plane = f"_plane_{index:04d}" if len(pages) > 1 else ""
                crop_stem = f"{tiff_path.stem}{plane}_crop"
                cropped.save(crop_dir / f"{crop_stem}.png")
                if crop_scale_bar_um is not None and pixel_size_um is not None:
                    cropped.save(crop_tiff_dir / f"{crop_stem}.tif", format="TIFF")
            if scale_bar_um is not None and pixel_size_um is not None:
                image = add_scale_bar(image, pixel_size_um, scale_bar_um)
            suffix = f"_plane_{index:04d}" if len(pages) > 1 else ""
            output_stem = f"{tiff_path.stem}{suffix}"
            image.save(png_dir / f"{output_stem}.png")
            if scale_bar_um is not None and pixel_size_um is not None:
                image.save(display_tiff_dir / f"{output_stem}.tif", format="TIFF")
            written += 1
    return written, pixel_size_um


def plane_filenames(stem: str, count: int, extension: str) -> list[str]:
    if count == 1:
        return [f"{stem}{extension}"]
    return [f"{stem}_plane_{index:04d}{extension}" for index in range(1, count + 1)]


def save_metadata_json(
    sample_dir: Path,
    source_file: Path,
    source_ome_xml: str | None,
    scale_bar_um: float | None,
    lower_percentile: float,
    upper_percentile: float,
    series: list[dict[str, object]],
) -> Path:
    document = {
        "source_file": str(source_file),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contrast_stretch_percentiles": {
            "lower": lower_percentile,
            "upper": upper_percentile,
        },
        "scale_bar": {
            "length_um": scale_bar_um,
            "drawn_without_text_label": True,
        }
        if scale_bar_um is not None
        else None,
        "series": series,
        "source_lif_ome_xml": source_ome_xml,
    }
    metadata_path = sample_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def collect_lif_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".lif":
            raise ValueError(f"Input file is not a .lif file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")
    pattern = "**/*.lif" if recursive else "*.lif"
    return sorted(path for path in input_path.glob(pattern) if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert Leica .lif files to raw TIFF and contrast-adjusted PNG."
    )
    parser.add_argument("--input", required=True, type=Path, help="A .lif file or folder of .lif files")
    parser.add_argument("--output", required=True, type=Path, help="Output folder")
    parser.add_argument("--bfconvert", type=Path, default=Path(DEFAULT_BFCONVERT), help="Path to bfconvert.bat")
    parser.add_argument("--recursive", action="store_true", help="Search input folders recursively")
    parser.add_argument("--scale-bar-um", type=float, help="Add a calibrated scale bar of this length in um")
    parser.add_argument("--crop-scale-bar-um", type=float, help="Scale bar length in um for cropped images (default: --scale-bar-um)")
    parser.add_argument("--crop-px", type=int, help="Also export a center crop of N x N pixels")
    parser.add_argument("--crop-um", type=float, help="Also export a center crop of N x N um (needs pixel size metadata)")
    parser.add_argument("--lower-percentile", type=float, default=0.5, help="PNG contrast lower percentile (default: 0.5)")
    parser.add_argument("--upper-percentile", type=float, default=99.9, help="PNG contrast upper percentile (default: 99.9)")
    args = parser.parse_args()
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("Percentiles must satisfy 0 <= lower < upper <= 100")
    if args.scale_bar_um is not None and args.scale_bar_um <= 0:
        parser.error("--scale-bar-um must be greater than zero")
    if args.crop_scale_bar_um is not None and args.crop_scale_bar_um <= 0:
        parser.error("--crop-scale-bar-um must be greater than zero")
    if args.crop_px is not None and args.crop_um is not None:
        parser.error("Use only one of --crop-px and --crop-um")
    if (args.crop_px is not None and args.crop_px < 1) or (args.crop_um is not None and args.crop_um <= 0):
        parser.error("Crop size must be greater than zero")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        lif_files = collect_lif_files(args.input, args.recursive)
    except ValueError as error:
        logging.error("%s", error)
        return 2
    if not lif_files:
        logging.error("No .lif files found in %s", args.input)
        return 2
    if not args.bfconvert.is_file():
        converter_on_path = shutil.which(str(args.bfconvert))
        if converter_on_path is None:
            logging.error("bfconvert.bat not found: %s", args.bfconvert)
            return 2
        args.bfconvert = Path(converter_on_path)

    failed = 0
    for lif_path in lif_files:
        sample_dir = args.output / lif_path.stem
        raw_dir = sample_dir / "tiff_raw"
        png_dir = sample_dir / "png_contrast"
        display_tiff_dir = sample_dir / "tiff_display"
        crop_dir = sample_dir / "png_cropped"
        crop_tiff_dir = sample_dir / "tiff_display_cropped"
        logging.info("Processing %s", lif_path)
        try:
            try:
                source_ome_xml, lif_series_metadata = read_lif_metadata(lif_path, args.bfconvert)
            except (OSError, subprocess.CalledProcessError) as error:
                logging.warning("Could not read metadata from LIF: %s", error)
                source_ome_xml, lif_series_metadata = None, {}
            tiff_files = convert_lif(lif_path, raw_dir, args.bfconvert)
            if not tiff_files:
                raise RuntimeError("bfconvert completed but produced no TIFF files")
            output_series = []
            for tiff_path in tiff_files:
                source_series = lif_series_metadata.get(tiff_path.stem.casefold(), {})
                source_pixel_size = source_series.get("physical_size_x_um")
                lif_pixel_size = (
                    float(source_pixel_size)
                    if isinstance(source_pixel_size, (int, float))
                    else None
                )
                count, pixel_size_um = process_tiff(
                    tiff_path,
                    png_dir,
                    display_tiff_dir,
                    args.lower_percentile,
                    args.upper_percentile,
                    args.scale_bar_um,
                    lif_pixel_size,
                    crop_dir,
                    crop_tiff_dir,
                    args.crop_px,
                    args.crop_um,
                    args.crop_scale_bar_um,
                )
                logging.info("  %s: wrote %d PNG plane(s)", tiff_path.name, count)
                series_record = dict(source_series)
                series_record.update(
                    {
                        "name": source_series.get("name", tiff_path.stem),
                        "physical_size_x_um": pixel_size_um,
                        "raw_tiff": str(tiff_path.relative_to(sample_dir)),
                        "png_files": [
                            f"png_contrast/{name}"
                            for name in plane_filenames(tiff_path.stem, count, ".png")
                        ],
                        "display_tiff_files": [
                            f"tiff_display/{name}"
                            for name in plane_filenames(tiff_path.stem, count, ".tif")
                        ]
                        if args.scale_bar_um is not None and pixel_size_um is not None
                        else [],
                        "scale_bar": {
                            "length_um": args.scale_bar_um,
                            "width_px": round(args.scale_bar_um / pixel_size_um)
                            if args.scale_bar_um is not None and pixel_size_um is not None
                            else None,
                            "drawn": args.scale_bar_um is not None and pixel_size_um is not None,
                        }
                        if args.scale_bar_um is not None
                        else None,
                    }
                )
                crop_size_px = args.crop_px
                if crop_size_px is None and args.crop_um is not None and pixel_size_um is not None:
                    crop_size_px = round(args.crop_um / pixel_size_um)
                if crop_size_px is not None:
                    crop_bar_um = (
                        args.crop_scale_bar_um if args.crop_scale_bar_um is not None else args.scale_bar_um
                    )
                    drawn = crop_bar_um is not None and pixel_size_um is not None
                    series_record["crop"] = {
                        "mode": "center",
                        "scale_bar_um": crop_bar_um,
                        "requested_px": args.crop_px,
                        "requested_um": args.crop_um,
                        "size_px": crop_size_px,
                        "png_files": [
                            f"png_cropped/{name}"
                            for name in plane_filenames(f"{tiff_path.stem}_crop", count, ".png")
                        ],
                        "display_tiff_files": [
                            f"tiff_display_cropped/{name}"
                            for name in plane_filenames(f"{tiff_path.stem}_crop", count, ".tif")
                        ]
                        if drawn
                        else [],
                    }
                output_series.append(series_record)
            metadata_path = save_metadata_json(
                sample_dir,
                lif_path,
                source_ome_xml,
                args.scale_bar_um,
                args.lower_percentile,
                args.upper_percentile,
                output_series,
            )
            logging.info("  Saved metadata: %s", metadata_path)
        except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
            failed += 1
            logging.error("Failed on %s: %s", lif_path.name, error)

    logging.info("Finished: %d input file(s), %d failed", len(lif_files), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())