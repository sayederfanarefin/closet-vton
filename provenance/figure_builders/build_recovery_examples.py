"""Read-only evidence audit and native-resolution recovery figure assembly."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
RES = ROOT / 'experiments/results/segmentation_rescue_20261122'


def readcsv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    compact = Path('/tmp/closetvton-publish-20261123/experiments/renderer_recovery_20261123') / RES.relative_to(ROOT) / 'request_attempts_compact.csv'
    attempts_path = RES / 'request_attempts.csv'
    attempts = readcsv(attempts_path)
    compact_rows = readcsv(compact) if compact.exists() else attempts
    choices = [('00095_00', 'empty_target_mask', 2, 'target'),
               ('00330_00', 'empty_target_mask', 3, 'target'),
               ('00273_00', 'invalid_segmented_reference', 2, 'reference'),
               ('00991_00', 'invalid_segmented_reference', 3, 'reference')]
    hashes = {attempts_path: digest(attempts_path)}
    rows, mask_evidence, swaps = [], [], []
    width, row_h, margin, gap = 3232, 1240, 40, 32
    canvas = Image.new('RGB', (width, 80 + 4 * row_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 31)
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 35)

    def text(s, x, y, face=font):
        draw.text((x + 384, y), s, fill='black', font=face, anchor='mt', align='center', spacing=9)

    for col, label in enumerate(['Target', 'Reference garment', 'Primary rejection / mask evidence', 'Final accepted output']):
        text(label, margin + col * (768 + gap), 18, font)
    for index, (requested, category, attempt, role) in enumerate(choices):
        image_id = requested

        def matches(i):
            aa = [x for x in compact_rows if x['protocol'] == 'paired' and x['id'] == i]
            return any(x['attempt_number'] == '1' and x['category'] == category for x in aa) and any(x['attempt_number'] == str(attempt) and x['status'] == 'ok' and x['valid'].lower() == 'true' for x in aa)

        if not matches(image_id):
            image_id = next(i for i in sorted({x['id'] for x in compact_rows if x['protocol'] == 'paired'}) if matches(i))
            swaps.append({'requested': requested, 'used': image_id})
        initial = next(x for x in attempts if x['protocol'] == 'paired' and x['id'] == image_id and x['attempt_number'] == '1')
        final = next(x for x in attempts if x['protocol'] == 'paired' and x['id'] == image_id and x['attempt_number'] == str(attempt))
        pred = RES / 'merged_runs/paired/images/pred' / (image_id + '.png')
        assert pred.resolve() == Path(final['pred_path']).resolve()
        y = 80 + index * row_h
        text(image_id, margin, y, bold)
        text(f'Fallback {attempt - 1} / attempt {attempt}', margin + 3 * (768 + gap), y, bold)
        image_y = y + 66
        for col, path in [(0, Path(final['target_path'])), (1, Path(final['reference_path'])), (3, pred)]:
            hashes[path] = digest(path)
            with Image.open(path) as im:
                assert im.size == (768, 1024)
                canvas.paste(im.convert('RGB'), (margin + col * (768 + gap), image_y))
        cx = margin + 2 * (768 + gap)
        text(category.replace('_', ' '), cx, y + 65)
        primary = initial['pred_path']
        assert not primary or not Path(primary).is_file(), 'Unexpected primary image; inspect before proceeding'
        text('No image returned', cx, y + 113)
        mask_path = ''
        engine = ''
        if attempt == 3:
            diagnostic = json.loads(final['segmentation_json'])[role]
            engine = diagnostic['selected']
            p = RES / 'pilot/mask' / f'{role}_{image_id}.png'
            with Image.open(p) as im:
                mask = np.asarray(im.convert('L')) > 0
            raw_hash = hashlib.sha256(mask.astype('uint8').tobytes()).hexdigest()
            assert raw_hash == diagnostic['selected_mask_sha256']
            source = Path(final['target_path'] if role == 'target' else final['reference_path'])
            with Image.open(source) as im:
                overlay = im.convert('RGBA')
            # Draw only a translucent contour, on a distinct diagnostic panel.
            binary = Image.fromarray(mask.astype('uint8') * 255)
            edge = ImageChops.subtract(binary.filter(ImageFilter.MaxFilter(7)), binary.filter(ImageFilter.MinFilter(7)))
            layer = Image.new('RGBA', overlay.size, (0, 175, 245, 0))
            layer.putalpha(edge.point(lambda v: round(v * .70)))
            overlay = Image.alpha_composite(overlay, layer).convert('RGB')
            text(f'Accepted {role} mask: SCHP', cx, y + 162)
            # Diagnostic occupies the same native-size cell, below the rejection caption.
            canvas.paste(overlay, (cx, y + 210))
            mask_path = str(p)
            hashes[p] = digest(p)
            mask_evidence.append({'id': image_id, 'role': role, 'engine': engine, 'mask_path': mask_path,
                                  'binary_sha256': raw_hash, 'matches_accepted_request_hash': True,
                                  'origin': 'Saved segmentation pilot; binary hash equals final request selected_mask_sha256'})
        rows.append(dict(id=image_id, category=category, attempt=attempt, target_path=final['target_path'],
                         reference_path=final['reference_path'], primary_pred_path=primary,
                         final_pred_path=str(pred), mask_path=mask_path, mask_engine=engine))
    png = OUT / 'fig_recovery_examples.png'
    canvas.save(png, dpi=(width / 7, width / 7))
    with (OUT / 'fig_recovery_examples_manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # Compare protocol + ID, not just target IDs (five worn-reference pairs are outside the final set).
    original = ROOT / 'fig_qualitative_selected.png'
    selected = OUT / 'fig_qualitative_selected.png'
    if not selected.exists():
        shutil.copy2(original, selected)
    figure = Image.open(selected).convert('RGB')
    provenance = readcsv(ROOT / 'experiments/results/qualitative_provenance_20261119/example_provenance.csv')
    worn = {r['id']: r for r in map(json.loads, (RES / 'merged_runs/worn_1000/results.jsonl').read_text().splitlines())}
    audit = []
    changed = False
    for index, p in enumerate(provenance):
        row, col = divmod(index, 2)
        x = [797, 2377][col] + (1 if row == 2 and col == 1 else 0)
        y = [84, 579, 1074][row] + (2 if row == 2 and col == 1 else 0)
        box = (x, y, x + 360, y + 480)
        old_path = ROOT / p['closetvton_output_path']
        with Image.open(old_path) as im:
            archived_equal = np.array_equal(np.asarray(figure.crop(box)), np.asarray(im.convert('RGB').resize((360, 480), Image.Resampling.LANCZOS)))
        r = worn.get(p['pair_id'])
        info = dict(pair_id=p['pair_id'], position=p['example_position'], in_final_worn_1000=r is not None,
                    archived_path=str(old_path), pixels_match_archived_output=bool(archived_equal),
                    final_path='', attempt_number='', path_in_final_run='not found', pixels_identical_to_final='not verifiable')
        if r:
            path = Path(r['pred_path']); hashes[path] = digest(path)
            with Image.open(path) as im:
                panel = im.convert('RGB').resize((360, 480), Image.Resampling.LANCZOS)
            equal = np.array_equal(np.asarray(figure.crop(box)), np.asarray(panel))
            attempt = int(r['telemetry']['attempt_number'])
            info.update(final_path=str(path), attempt_number=attempt,
                        path_in_final_run='primary' if attempt == 1 else f'fallback {attempt - 1}',
                        pixels_identical_to_final=bool(equal))
            if not equal:
                figure.paste(panel, box); changed = True
        audit.append(info)
    if changed:
        figure.save(selected)
    with (OUT / 'fig_qualitative_selected_verification.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(audit[0])); w.writeheader(); w.writerows(audit)

    overlaps = []
    for protocol, run in [('paired', '20260131_055842_a14987_vitonhd_test_paired'), ('unpaired', '20260131_125119_2a93d2_vitonhd_test_unpaired')]:
        path = ROOT / 'runs' / run / 'results.jsonl'
        old = [json.loads(l) for l in path.read_text().splitlines()]
        errors = {r['id'] for r in old if r['status'] != 'ok' and 'AddMaskForICLora' in r.get('error', '') and 'not enough values to unpack (expected 3, got 2)' in r.get('error', '')}
        primary = {r['id'] for r in attempts if r['protocol'] == protocol and r['attempt_number'] == '1' and r['category'] == 'invalid_segmented_reference'}
        for image_id in sorted(primary | errors):
            overlaps.append(dict(protocol=protocol, id=image_id, invalid_segmented_reference=image_id in primary,
                                 archived_dimensional_unpacking_error=image_id in errors, archived_results_path=str(path)))
    with (OUT / 'primary_reference_failure_overlap.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(overlaps[0])); w.writeheader(); w.writerows(overlaps)
    assert all(digest(path) == h for path, h in hashes.items())
    evidence = dict(recovery_figure=str(png), pixels=list(canvas.size), dpi=width / 7, swaps=swaps,
                    attempt_evidence=str(attempts_path), compact_attempt_evidence=str(compact) if compact.exists() else None,
                    masks=mask_evidence, worn_verification=audit, worn_figure_regenerated=changed,
                    worn_note='Only one of six exact target/reference pairs is in the final worn-reference run. Five final outputs are unavailable; archived pixels verified independently.',
                    overlap_total=sum(r['invalid_segmented_reference'] and r['archived_dimensional_unpacking_error'] for r in overlaps),
                    source_files_unchanged=True)
    (OUT / 'recovery_figure_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
