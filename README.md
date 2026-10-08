# LIF Batch Export

Batch-convert Leica `.lif` files with Bio-Formats, preserve the converted raw TIFFs, and create contrast-adjusted PNGs. Optionally add a calibrated scale bar to the PNGs and to separate display TIFF copies.

## Setup

Install Python dependencies in the interpreter selected for this VS Code workspace:

```powershell
python -m pip install -r requirements.txt
```

Install Bio-Formats tools separately and make `bfconvert.bat` available on `PATH`, or set the `BFCONVERT` environment variable / pass its location with `--bfconvert`.

## Run

From the integrated terminal:

```powershell
python .\lif_pipeline.py --input "E:\images\lif" --output "E:\images\processed"
```

Add `--scale-bar-um 100` to draw a 100 um bar. Pixel size is read from the TIFF OME metadata, with a fallback to the original LIF metadata via Bio-Formats `showinf`. If calibration is missing, the images are still exported and a warning is logged; no uncalibrated scale bar is drawn. Choose a bar length that fits within the image field of view.

Useful options:

- `--recursive` searches subfolders for `.lif` files.
- `--lower-percentile 0.5 --upper-percentile 99.9` controls PNG contrast stretching.
- `--crop-px 512` also exports a centered 512 x 512 pixel crop; `--crop-um 100` does the same for 100 x 100 um (needs pixel-size metadata). Use only one. Pixels are more robust. Crops reuse the full-image contrast stretch, get the scale bar if `--scale-bar-um` is set, and are clamped to the image size.
- `--bfconvert "C:\path\to\bfconvert.bat"` selects another Bio-Formats installation.

## Output

Each input file gets its own folder under the output directory:

- `tiff_raw/` contains Bio-Formats TIFF output and remains unmodified.
- `png_contrast/` contains 8-bit percentile-stretched PNG planes.
- `tiff_display/` is created only when a scale bar is requested and contains display copies with the unlabeled scale bar baked in. These copies are not quantitative raw data.
- `png_cropped/` and `tiff_display_cropped/` (the latter only with a scale bar) hold the `*_crop` files when a crop is requested.
- `metadata.json` records the processing timestamp, contrast settings, scale-bar size, per-series dimensions and calibration, output filenames, and original LIF OME-XML metadata.

Multi-page TIFF series are exported one plane per PNG. Re-running the pipeline overwrites generated files in the output folder.

In VS Code, run **Terminal: Run Task** and choose one of the **LIF Batch Export** tasks. The scale-bar task prompts for a bar length in micrometers; the crop task also prompts for a crop size in pixels.

Example: `python lif_pipeline.py --input in --output out --crop-px 512 --scale-bar-um 50`

## Public Repository Safety

The `.gitignore` excludes LIF files, exported images, generated metadata sidecars, local secrets, and Python caches. Do not commit microscopy data or generated `metadata.json` files: OME metadata can contain instrument identifiers and acquisition details. Review all files and the staged changes before publishing.