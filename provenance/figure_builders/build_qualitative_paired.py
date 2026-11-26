"""Assemble final accepted paired predictions at native resolution; no inference."""
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageChops

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/results/segmentation_rescue_20261122'
OUT = ROOT / 'figures'
COLUMNS = [('Worst', ['12651_00', '11985_00']),
           ('Median', ['11851_00', '11127_00']),
           ('Best', ['04766_00', '09969_00'])]


def main():
    metrics_path = RESULTS / 'paired_per_sample_metrics.csv'
    records_path = RESULTS / 'merged_runs/paired/results.jsonl'
    metrics = {r['id']: r for r in csv.DictReader(metrics_path.open())}
    records = {r['id']: r for r in map(json.loads, records_path.read_text().splitlines())}
    # Images are pasted 1:1; only the physical print scale is set using PNG DPI.
    panel_w, panel_h, inner_gap, outer_gap, margin = 768, 1024, 16, 72, 40
    case_w, case_h = 2 * panel_w + inner_gap, 2240
    width = 2 * margin + 3 * case_w + 2 * outer_gap
    height = 2 * margin + 90 + 2 * case_h + 60
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 36)
    heading = ImageFont.truetype(font_path.replace('.ttf', '-Bold.ttf'), 54)
    rows, regions, hashes = [], [], {}

    def center(text, x, y, w, face=font):
        draw.text((x + w / 2, y), text, fill='black', font=face, anchor='mt')

    for col, (group, ids) in enumerate(COLUMNS):
        x = margin + col * (case_w + outer_gap)
        center(group, x, margin, case_w, heading)
        for row_index, image_id in enumerate(ids):
            record = records[image_id]
            assert record['status'] == 'ok' and record['telemetry']['valid']
            attempt = int(record['telemetry']['attempt_number'])
            assert attempt in (1, 2, 3)
            score = metrics[image_id]['ssim_masked_after']
            pred = RESULTS / 'merged_runs/paired/images/pred' / (image_id + '.png')
            assert pred.resolve() == Path(record['pred_path']).resolve()
            y = margin + 90 + row_index * (case_h + 60)
            center(f'{image_id} | Attempt {attempt} | Masked SSIM {float(score):.4f}', x, y, case_w)
            paths = [Path(record['person_path']), Path(record['cloth_path']), pred, Path(record['gt_path'])]
            labels = ['Target person', 'Catalog garment', 'ClosetVTON final output', 'Ground truth']
            for index, (path, label) in enumerate(zip(paths, labels)):
                px = x + (index % 2) * (panel_w + inner_gap)
                label_y = y + 62 + (index // 2) * (panel_h + 62)
                center(label, px, label_y, panel_w)
                py = label_y + 48
                hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as im:
                    assert im.size == (panel_w, panel_h), (path, im.size)
                    canvas.paste(im.convert('RGB'), (px, py))
                regions.append((path, (px, py, px + panel_w, py + panel_h)))
            rows.append(dict(id=image_id, attempt_number=attempt, pred_path=str(pred),
                             target_path=record['person_path'], reference_path=record['cloth_path'],
                             ssim_masked=score))
    png = OUT / 'fig_qualitative_paired.png'
    canvas.save(png, dpi=(width / 7, width / 7))
    with Image.open(png) as saved:
        for path, box in regions:
            with Image.open(path) as source:
                assert ImageChops.difference(saved.crop(box), source.convert('RGB')).getbbox() is None
        assert saved.info['dpi'][0] >= 300
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    with (OUT / 'fig_qualitative_paired_manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'png': str(png), 'pixels': [width, height], 'dpi': width / 7,
                      'width_inches': 7, 'cases': rows, 'native_panel_pixels_verified': 24}, indent=2))


if __name__ == '__main__':
    main()
