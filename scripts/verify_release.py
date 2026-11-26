"""Validate Code/Data hashes and final cohort counts after transfer."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    checked = 0
    for folder in ['Code', 'Data']:
        base = root if folder == "Code" else args.data_root.resolve()
        for line in (base / 'checksums.sha256').read_text().splitlines():
            expected, rel = line.split('  ', 1)
            path = base / rel
            assert path.is_file() and not path.is_symlink(), path
            h = hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
            assert h.hexdigest() == expected, path
            checked += 1
    data_base = args.data_root.resolve()
    for protocol, n in [('paired', 2032), ('unpaired', 2032), ('worn_1000', 1000)]:
        rows = [json.loads(l) for l in (data_base / 'manifests' / (protocol + '.jsonl')).read_text().splitlines()]
        assert len(rows) == len({r['id'] for r in rows}) == n
        assert all(r['attempt_number'] in (1, 2, 3) and r['valid'] and (data_base / r['pred']).is_file() for r in rows)
    assert len(list(csv.DictReader((data_base / 'attempts/request_attempts_compact.csv').open()))) == 5823
    print(json.dumps(dict(passed=True, files_checked=checked, accepted_predictions=5064, attempts=5823)))


if __name__ == '__main__': main()
