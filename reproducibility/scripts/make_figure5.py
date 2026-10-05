#!/usr/bin/env python3
"""Regenerate Figure 5 from released snapshots in the production layout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, NumberObject
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
INDICES = (0, 50, 100, 150, 200)
ERROR_LIMITS = (1e-5, 0.07, 0.26, 0.30, 0.30)
ERROR_IMAGES = ((15, 16), (22, 23), (24, 25), (26, 27), (28, 29))


def image_stream(field, cmap, upper):
    size = 1000
    axis = np.clip((np.arange(size) + 0.5) * 41 / size - 0.5, 0, 40)
    xx, yy = np.meshgrid(axis, axis[::-1])
    values = map_coordinates(field, [yy, xx], order=1, mode="nearest")
    rgb = colormaps[cmap](np.clip(values / upper, 0, 1), bytes=True)[:, :, :3]
    image = DecodedStreamObject()
    image.set_data(rgb.tobytes())
    for key, value in {
        "/Type": NameObject("/XObject"), "/Subtype": NameObject("/Image"),
        "/Width": NumberObject(size), "/Height": NumberObject(size),
        "/ColorSpace": NameObject("/DeviceRGB"), "/BitsPerComponent": NumberObject(8),
    }.items():
        image[NameObject(key)] = value
    return image.flate_encode()


def generate(output):
    template = ROOT / "figure_data/two_d_burgers/figure5_layout.pdf"
    writer = PdfWriter()
    page = writer.add_page(PdfReader(template).pages[0])
    images = page["/Resources"]["/XObject"]
    records = []
    for row, index in enumerate(INDICES):
        frames = []
        hashes = {}
        for kind in ("reference", "observed"):
            path = ROOT / f"data/two_d_burgers/{kind}/uv_times_{index:04d}.txt"
            data = np.loadtxt(path)
            if data.shape != (1681, 4) or not np.isfinite(data).all():
                raise ValueError(f"Invalid snapshot: {path}")
            x, y = np.meshgrid(np.linspace(0, 2, 41), np.linspace(0, 2, 41))
            if not np.allclose(data[:, :2], np.column_stack((x.ravel(), y.ravel())), atol=1e-12, rtol=0):
                raise ValueError(f"Unexpected grid ordering: {path}")
            frames.append(data[:, 2:].reshape(41, 41, 2))
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        reference, observed = frames
        errors = np.abs(observed - reference)
        maxima = errors.max(axis=(0, 1))
        if np.any(maxima > ERROR_LIMITS[row]):
            raise ValueError(f"Error color scale would clip snapshot {index}")
        fields = (reference[:, :, 0], observed[:, :, 0], reference[:, :, 1], observed[:, :, 1])
        for column, field in enumerate(fields):
            name = NameObject(f"/PublicField{column}_{row}")
            images[name] = writer._add_object(image_stream(field, "turbo", 1.0 if column < 2 else 0.8))
        for component, number in enumerate(ERROR_IMAGES[row]):
            images[NameObject(f"/Im{number}")] = writer._add_object(
                image_stream(errors[:, :, component], "RdBu_r", ERROR_LIMITS[row]))
        records.append({"time": index / 100, "error_limit": ERROR_LIMITS[row],
                        "maximum_absolute_error_uv": maxima.tolist(), "input_sha256": hashes})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)
    report = {"figure": 5, "panels_regenerated": 30, "snapshots": records,
              "display_interpolation": "bilinear; raw data unchanged",
              "pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "work/figures/Figure5.pdf")
    args = parser.parse_args()
    generate(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
