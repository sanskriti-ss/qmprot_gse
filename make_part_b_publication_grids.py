"""
Create publication-style panel grids from per-molecule Part B PNG outputs.

Generates two high-resolution grids in the specified output root:
- convergence_by_init_all_molecules_grid.png
- final_energy_by_init_all_molecules_grid.png

Usage:
    python make_part_b_publication_grids.py \
        --output-root framework/experiments/results/barren_plateau/<timestamp>_all_molecules_part_b_bayesian
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_molecule_order(output_root: Path) -> List[str]:
    manifest = output_root / "batch_manifest.json"
    if manifest.is_file():
        with open(manifest) as handle:
            data = json.load(handle)
        succeeded = data.get("molecules_succeeded") or []
        if succeeded:
            return list(succeeded)
        selected = data.get("molecules_selected") or []
        if selected:
            return list(selected)

    return sorted(p.name for p in output_root.iterdir() if p.is_dir())


def _collect_images(
    output_root: Path,
    molecules: Sequence[str],
    plot_filename: str,
) -> List[Tuple[str, Path]]:
    collected: List[Tuple[str, Path]] = []
    for molecule in molecules:
        img_path = output_root / molecule / plot_filename
        if img_path.is_file():
            collected.append((molecule, img_path))
    if not collected:
        raise FileNotFoundError(
            f"No files named {plot_filename!r} found under molecule folders in {output_root}"
        )
    return collected


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = left + max(0, (right - left - w) // 2)
    y = top + max(0, (bottom - top - h) // 2)
    draw.text((x, y), text, font=font, fill=fill)


def make_grid(
    output_root: Path,
    items: Sequence[Tuple[str, Path]],
    *,
    title: str,
    out_name: str,
    columns: int = 4,
    tile_w: int = 900,
    tile_h: int = 580,
    label_h: int = 50,
    title_h: int = 80,
    margin: int = 30,
    gap_x: int = 24,
    gap_y: int = 26,
) -> Path:
    rows = math.ceil(len(items) / columns)

    canvas_w = margin * 2 + columns * tile_w + (columns - 1) * gap_x
    grid_h = rows * (label_h + tile_h) + (rows - 1) * gap_y
    canvas_h = margin * 2 + title_h + grid_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    title_font = _get_font(36)
    label_font = _get_font(24)

    _draw_centered_text(
        draw,
        (margin, margin, canvas_w - margin, margin + title_h),
        title,
        title_font,
        (20, 20, 20),
    )

    y0 = margin + title_h
    for idx, (molecule, img_path) in enumerate(items):
        row = idx // columns
        col = idx % columns

        x = margin + col * (tile_w + gap_x)
        y = y0 + row * (label_h + tile_h + gap_y)

        # label
        _draw_centered_text(
            draw,
            (x, y, x + tile_w, y + label_h),
            molecule,
            label_font,
            (35, 35, 35),
        )

        # image tile
        image_top = y + label_h
        img = Image.open(img_path).convert("RGB")
        fitted = ImageOps.contain(img, (tile_w, tile_h), Image.Resampling.LANCZOS)

        # white panel background
        draw.rectangle(
            (x, image_top, x + tile_w, image_top + tile_h),
            fill=(255, 255, 255),
            outline=(210, 210, 210),
            width=1,
        )

        px = x + (tile_w - fitted.width) // 2
        py = image_top + (tile_h - fitted.height) // 2
        canvas.paste(fitted, (px, py))

    out_path = output_root / out_name
    canvas.save(out_path, dpi=(300, 300))
    return out_path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create publication-style grids for Part B per-molecule PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    output_root = args.output_root

    if not output_root.is_dir():
        raise NotADirectoryError(f"Output root not found: {output_root}")

    molecules = _load_molecule_order(output_root)

    convergence_items = _collect_images(output_root, molecules, "convergence_by_init.png")
    final_energy_items = _collect_images(output_root, molecules, "final_energy_by_init.png")

    out1 = make_grid(
        output_root,
        convergence_items,
        title="Part B Convergence by Initialization (All Molecules)",
        out_name="convergence_by_init_all_molecules_grid.png",
        columns=args.columns,
    )
    out2 = make_grid(
        output_root,
        final_energy_items,
        title="Part B Final Energy by Initialization (All Molecules)",
        out_name="final_energy_by_init_all_molecules_grid.png",
        columns=args.columns,
    )

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
