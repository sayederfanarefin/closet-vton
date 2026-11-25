"""Resolve exported data paths and construct original evaluation input manifests."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, help='VITON-HD root containing test/, if stored separately')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--allow-missing-source', action='store_true', help='Prepare paths for inspection without source files; scoring still requires them')
    args = p.parse_args()
    data = args.data_root.resolve()
    missing, counts = set(), {}
    all_rows = {}
    for protocol in ['paired', 'unpaired', 'worn_1000']:
        rows = [json.loads(l) for l in (data / 'manifests' / (protocol + '.jsonl')).read_text().splitlines()]
        for r in rows:
            for key in ['person', 'cloth', 'gt', 'mask', 'cloth_mask', 'pred']:
                if not r.get(key): continue
                rel = Path(r[key])
                assert not rel.is_absolute() and '..' not in rel.parts, rel
                if args.source_root and rel.parts[:2] == ('source', 'viton-hd'):
                    path = args.source_root.resolve() / Path(*rel.parts[2:])
                else: path = data / rel
                if not path.is_file():
                    if key == 'pred': raise FileNotFoundError(path)
                    missing.add(str(path))
                r[key] = str(path)
        counts[protocol] = len(rows)
        assert len({r['id'] for r in rows}) == len(rows)
        all_rows[protocol] = rows
    if missing and not args.allow_missing_source:
        raise SystemExit(f'{len(missing)} required VITON-HD files missing. Supply --source-root or install source/viton-hd/test/. No manifests written.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for protocol, rows in all_rows.items():
        (args.output_dir / (protocol + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    worn = {r['id']: r for r in all_rows['worn_1000']}
    baseline = list(csv.DictReader((data / 'manifests/baseline_predictions.csv').open()))
    for model in ['catvton', 'hrviton', 'idm']:
        overlap = []
        for b in baseline:
            if b['model'] != model: continue
            r = worn[b['id']]
            overlap.append(dict(pair_id=b['id'], run_a='closetvton_final', run_b=model,
                pred_path_closetvton_final=r['pred'], **{'pred_path_' + model: str(data / b['pred'])},
                person_path=r['person'], cloth_path_closetvton_final=r['cloth'],
                **{'cloth_path_' + model: r['cloth']}, mask_path=r['mask'],
                gt_path_closetvton_final='', **{'gt_path_' + model: ''}))
        assert len(overlap) == 1000
        with (args.output_dir / ('overlap_ours_vs_' + model + '.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(overlap[0])); w.writeheader(); w.writerows(overlap)
    print(json.dumps(dict(counts=counts, missing_source_files=len(missing), scoring_ready=not missing, output_dir=str(args.output_dir.resolve()))))


if __name__ == '__main__': main()
