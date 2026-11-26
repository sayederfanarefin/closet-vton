"""Rebuild all worn qualitative panels using released predictions and external photos."""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageChops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True, help='VITON-HD root containing test/')
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    data = args.data_root.resolve()
    # The released layout supplies only headings/gutters; every photograph/output panel is replaced.
    template = Image.open(data / 'figures/fig_qualitative_selected.png').convert('RGB')
    figure = template.copy()
    with (data / 'manifests/qualitative_worn.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    xs = [11, 404, 797, 1191, 1591, 1984, 2377, 2771]
    ys = [84, 579, 1074]
    for index, row in enumerate(rows):
        y_index, group = divmod(index, 2)
        source = args.source_root.resolve()
        paths = [source / Path(*Path(row['target']).parts[2:]),
                 source / Path(*Path(row['reference']).parts[2:]),
                 data / row['closetvton_pred'], data / row['catvton_pred']]
        for panel, path in enumerate(paths):
            dx, dy = (1, 2) if y_index == 2 and group == 1 else (0, 0)
            x, y = xs[group * 4 + panel] + dx, ys[y_index] + dy
            with Image.open(path) as image:
                if panel < 2: assert image.size == (768, 1024), path
                figure.paste(image.convert('RGB').resize((360, 480), Image.Resampling.LANCZOS), (x, y))
    assert ImageChops.difference(figure, template).getbbox() is None, 'Reconstructed panels differ from released figure'
    args.outdir.mkdir(parents=True, exist_ok=False)
    figure.save(args.outdir / 'fig_qualitative_selected.png')
    print('All 24 panels reconstructed; pixels match the released figure exactly.')


if __name__ == '__main__': main()
