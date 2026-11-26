"""Recompute completion, paired means and client timings from released records."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path):
    with path.open() as f: return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    data = args.data_root
    attempts = read(data / 'attempts/request_attempts_compact.csv')
    completion = {r['protocol']: r for r in read(data / 'metrics/final/completion_overview.csv')}
    groups = defaultdict(list)
    for r in attempts: groups[r['protocol'], r['id']].append(r)
    counts, accepted_times, cumulative_times = {}, defaultdict(list), defaultdict(list)
    for (protocol, image_id), records in groups.items():
        accepted = [r for r in records if r['status'] == 'ok' and r['valid'] == 'True']
        assert len(accepted) == 1, (protocol, image_id)
        counts.setdefault(protocol, {'primary': 0, 'fallback_1': 0, 'fallback_2': 0, 'total': 0})
        stage = int(accepted[0]['attempt_number'])
        counts[protocol][['primary', 'fallback_1', 'fallback_2'][stage - 1]] += 1
        counts[protocol]['total'] += 1
        accepted_times[protocol].append(float(accepted[0]['elapsed_sec']))
        cumulative_times[protocol].append(sum(float(r['elapsed_sec']) for r in records))
    for protocol, c in counts.items():
        saved = completion[protocol]
        assert c['primary'] == int(saved['guarded_valid'])
        assert c['primary'] + c['fallback_1'] == int(saved['fallback_valid'])
        assert c['total'] == int(saved['final_valid']) == int(saved['requests'])
    paired = read(data / 'metrics/recovery/paired_per_sample_metrics.csv')
    means = {m: float(np.mean([float(r[m + '_after']) for r in paired])) for m in ['psnr', 'ssim', 'lpips', 'psnr_masked', 'ssim_masked']}
    for row in read(data / 'metrics/final/paired_quality.csv'):
        assert np.isclose(means[row['metric']], float(row['final_mean']), rtol=0, atol=1e-12)
    latency = []
    for row in read(data / 'metrics/final/latency_views.csv'):
        view = row['timing_view']
        if view not in ['after_final_attempt', 'after_cumulative_attempts']: continue
        values = (accepted_times if view == 'after_final_attempt' else cumulative_times)[row['protocol']]
        stats = dict(mean=float(np.mean(values)))
        stats.update({f'p{q}': float(np.percentile(values, q)) for q in [25, 50, 75, 90, 95]})
        assert all(np.isclose(v, float(row[k]), rtol=0, atol=1e-12) for k, v in stats.items())
        latency.append(dict(protocol=row['protocol'], view=view, **stats))
    args.outdir.mkdir(parents=True, exist_ok=False)
    result = dict(completion=counts, paired_means=means, client_latency=latency, archived_values_match=True,
                  note='Client timings; service_elapsed_sec is a separate measurement. No inference performed.')
    (args.outdir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, attempts=len(attempts), accepted=len(groups), paired=len(paired))))


if __name__ == '__main__': main()
