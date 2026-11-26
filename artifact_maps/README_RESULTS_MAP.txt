ClosetVTON figure/table/result crosswalk — 2026-11-26
Code: https://github.com/sayederfanarefin/closet-vton
Data: https://www.kaggle.com/datasets/sayederfanarefin/closetvton-final-cascade-evaluation
Use data version 2; root is ClosetVTON_Data_v2/Data/ after Kaggle extraction.
C: means this code checkout; D: means the Data folder. Manuscript labels are stable;
numbers follow source declaration order. Source sections identify the manuscript location,
not an external repository. Conceptual/literature tables have no experimental rows.
Source photographs and weights are obtained separately. No original results are overwritten.

Table 1: Terminology [tab:terminology]
  Manuscript section: sections/background.tex:23
  Asset: Inline source / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): Manuscript/cited literature only
  Command/query: Read sections/background.tex
  Status: Conceptual table; no experimental rows
  Note: None

Table 2: Notation [tab:notation]
  Manuscript section: sections/system.tex:17
  Asset: Inline source / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): workflow_api.json; comfy_custom_nodes/crop_integrity/__init__.py; serve_workflow.py
  Command/query: Inspect node IDs and links in workflow_api.json
  Status: Implementation specification; no metric rows
  Note: None

Figure 1: Four-stage architecture and cascade [fig:architecture]
  Manuscript section: sections/system.tex:54
  Asset: figures/architecture.pdf
  Data (D:): Not applicable / not found; see status
  Code (C:): diagrams/architecture.tex; workflow_api.json; serve_workflow.py; comfy_custom_nodes/crop_integrity/segmentation_fallback.py; comfy_custom_nodes/crop_integrity/segmentation_rescue.py
  Command/query: pdflatex -output-directory /tmp/closet-diagram diagrams/architecture.tex
  Status: Source diagram available
  Note: SigLIP receives raw reference; canvas receives segmented reference. Crop coordinates are not the empty-mask root cause.

Table 3: Pipeline modules [tab:modules]
  Manuscript section: sections/system.tex:78
  Asset: Inline source / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): workflow_api.json; comfy_custom_nodes/crop_integrity/__init__.py; serve_workflow.py
  Command/query: Inspect node IDs and links in workflow_api.json
  Status: Implementation specification; no metric rows
  Note: None

Table 4: Operating configuration [tab:implementation]
  Manuscript section: sections/system.tex:198
  Asset: Inline source / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): workflow_api.json; provenance/environment.json; provenance/parser_provenance.json; provenance/render_settings_verification.json
  Command/query: Read workflow nodes 3/26/49/115/154/162 and provenance JSON
  Status: Configuration and recorded environment linked
  Note: Recorded source commits include a compatibility patch; fresh dependency resolution is not an exact environment lock.

Table 5: Input-contract comparison [tab:workflow]
  Manuscript section: sections/system.tex:235
  Asset: Inline source / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): Manuscript/cited literature only
  Command/query: Read sections/system.tex and cited references.bib entries
  Status: Literature/specification table; no experimental rows
  Note: External baseline input contracts are literature claims, not measurements in this data package.

Table 6: Metric definitions [tab:metric-defs]
  Manuscript section: sections/system.tex:320
  Asset: Inline source / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): evaluation/metrics.py; evaluation/lpips_metric.py; portable_eval_bundle/scripts/run_fidelity_comparison.py; portable_eval_bundle/scripts/run_deepfake_comparison.py; portable_eval_bundle/scripts/generate_nonstable_paper_report.py
  Command/query: Inspect compute_metrics, LPIPS scorer, pairwise proxies and pooled artifact normalization
  Status: Metric implementations linked
  Note: None

Table 7: Completion, reconstruction and median latency [tab:main-results]
  Manuscript section: sections/performance.tex:8
  Asset: Inline source / conceptual table
  Data (D:): metrics/final/completion_overview.csv; metrics/final/paired_quality.csv; metrics/final/latency_views.csv
  Code (C:): evaluation/metrics.py; evaluation/lpips_metric.py; scripts/prepare_data.py
  Command/query: Read final_valid; final_mean; timing_view=after_final_attempt, p50. Accepted records are status=ok and valid=True.
  Status: Values linked; caption mismatch identified
  Note: Table caption says median service time, but 10.75/11.34/11.07 match client p50, not service_p50. This mapping does not edit the manuscript.

Figure 2: Recovery funnel [fig:funnel]
  Manuscript section: sections/performance.tex:26
  Asset: figures/fig_recovery_funnel.pdf
  Data (D:): metrics/final/completion_overview.csv
  Code (C:): scripts/plot_results.py; comfy_custom_nodes/crop_integrity/segmentation_fallback.py; comfy_custom_nodes/crop_integrity/segmentation_rescue.py
  Command/query: python scripts/plot_results.py --data-root /path/to/Data --outdir results/figures
  Status: Portable plotting command verified
  Note: Plot reads guarded_valid, fallback_valid, final_valid; P:scripts/figure_data/final_20261122/completion_overview.csv is the frozen plot input.

Figure 3: Paired LPIPS histogram [fig:lpips-paired]
  Manuscript section: sections/performance.tex:36
  Asset: figures/fig_lpips_paired.pdf
  Data (D:): metrics/recovery/paired_per_sample_metrics.csv; metrics/final/paired_distribution.csv
  Code (C:): scripts/plot_results.py; evaluation/lpips_metric.py
  Command/query: python scripts/plot_results.py --data-root /path/to/Data --outdir results/figures
  Status: Portable plotting command verified
  Note: None

Figure 4: Paired PSNR/SSIM distributions [fig:metrics-hist]
  Manuscript section: sections/performance.tex:45
  Asset: figures/fig_metrics_hist_paired.pdf
  Data (D:): metrics/recovery/paired_per_sample_metrics.csv; metrics/recovery/paired_histograms.csv
  Code (C:): scripts/plot_results.py; evaluation/metrics.py
  Command/query: python scripts/plot_results.py --data-root /path/to/Data --outdir results/figures
  Status: Portable plotting command verified
  Note: The paper script uses 40-bin histograms; do not assume its edges equal the archived full-histogram CSV edges.

Table 8: Accepted-attempt and cumulative latency [tab:latency-percentiles]
  Manuscript section: sections/performance.tex:59
  Asset: Inline source / conceptual table
  Data (D:): metrics/final/latency_views.csv; attempts/request_attempts_compact.csv
  Code (C:): serve_workflow.py
  Command/query: Read timing_view=after_final_attempt or after_cumulative_attempts; columns mean,p25,p50,p75,p90,p95
  Status: Full-precision table source linked
  Note: Client elapsed_sec differs from service_elapsed_sec. Cumulative timing sums recorded failed attempts and accepted attempt per protocol/id.

Figure 5: Paired latency by accepted attempt [fig:runtime-paired]
  Manuscript section: sections/performance.tex:79
  Asset: figures/fig_runtime_paired.pdf
  Data (D:): attempts/request_attempts_compact.csv
  Code (C:): scripts/plot_results.py; serve_workflow.py
  Command/query: python scripts/plot_results.py --data-root /path/to/Data --outdir results/figures
  Status: Portable plotting command verified
  Note: P:scripts/figure_data/final_20261122/request_attempts_min.csv is the frozen reduced view. Plot clips values to 10.4–13.6 s; use unmodified records for statistics.

Figure 6: Worst/median/best paired examples [fig:qual-paired]
  Manuscript section: sections/performance.tex:88
  Asset: figures/fig_qualitative_paired.png
  Data (D:): figures/fig_qualitative_paired.png; figures/fig_qualitative_paired_manifest.csv; metrics/recovery/paired_per_sample_metrics.csv; manifests/paired.jsonl
  Code (C:): scripts/build_paired_figure.py
  Command/query: python scripts/build_paired_figure.py --data-root /path/to/Data --source-root /path/to/zalando-hd-resized --outdir results/paired_figure
  Status: Pixel relationship verified
  Note: All six are accepted attempt 1. Rebuilt pixels match the released full-resolution figure exactly. Manuscript image is its exact LANCZOS resize. External source photographs required.

Table 9: ClosetVTON versus CatVTON, 1000 worn pairs [tab:catvton]
  Manuscript section: sections/performance.tex:103
  Asset: Inline source / conceptual table
  Data (D:): metrics/final/closetvton_vs_catvton.csv; metrics/comparison/ours_vs_catvton/fidelity/fidelity_per_pair.csv; metrics/comparison/ours_vs_catvton/deepfake/deepfake_per_pair.csv
  Code (C:): portable_eval_bundle/scripts/run_fidelity_comparison.py; portable_eval_bundle/scripts/run_deepfake_comparison.py
  Command/query: Read metric, closetvton_mean, catvton_mean, delta, closetvton_win_rate, wilcoxon_p; inspect original two-sided signed-rank code
  Status: Full-precision table and per-pair inputs linked
  Note: One repeated reference; diagnostic comparison, not a uniform official benchmark. Original nominal p-values are not multiple-testing adjusted.

Figure 7: Six selected worn-reference pairs [fig:qualitative]
  Manuscript section: sections/performance.tex:133
  Asset: figures/fig_qualitative_selected.png
  Data (D:): figures/fig_qualitative_selected.png; figures/fig_qualitative_selected_verification.csv; manifests/qualitative_worn.csv
  Code (C:): scripts/build_worn_figure.py
  Command/query: python scripts/build_worn_figure.py --data-root /path/to/Data --source-root /path/to/zalando-hd-resized --outdir results/worn_figure
  Status: All 24 reconstructed panels pixel-identical
  Note: Only 00006_00__03173_00 is in final worn_1000. Ten outside-cohort predictions in qualitative_archive/ reconstruct the other five illustrations, not extra evaluation cases. Frozen template supplies headings/gutters only.

Table 10: Four-model fidelity and artifact diagnostics [tab:diagnostic]
  Manuscript section: sections/performance.tex:147
  Asset: Inline source / conceptual table
  Data (D:): metrics/final/four_model_diagnostics.csv; metrics/recovery/artifact_normalization_statistics.csv; metrics/recovery/artifact_normalization_sensitivity.csv
  Code (C:): portable_eval_bundle/scripts/generate_nonstable_paper_report.py
  Command/query: Read fidelity_composite_mean, combined_z_mean, combined_z_flag_rate and proxy means
  Status: Full-precision table source linked
  Note: Final four-model artifact z-scores use the pooled four-model cohort. Do not substitute the two-model z-score from pairwise comparisons.

Figure 8: Four-model proxy/artifact bar charts [fig:diagnostic-means]
  Manuscript section: sections/performance.tex:167
  Asset: figures/fig_diagnostic_means.pdf
  Data (D:): metrics/final/four_model_diagnostics.csv; metrics/recovery/four_model_artifact_means.csv
  Code (C:): scripts/plot_results.py; portable_eval_bundle/scripts/generate_nonstable_paper_report.py
  Command/query: python scripts/plot_results.py --data-root /path/to/Data --outdir results/figures
  Status: Portable plotting command verified
  Note: Paper plot starts proxy axis at 0.65 and omits chroma from artifact bars because of scale; CSV contains chroma.

Result 1: 5064 accepted outputs and cascade stage counts [result:completion]
  Manuscript section: sections/performance.tex:1
  Asset: Inline source / conceptual table
  Data (D:): metrics/final/completion_overview.csv; attempts/request_attempts_compact.csv
  Code (C:): comfy_custom_nodes/crop_integrity/segmentation_fallback.py; comfy_custom_nodes/crop_integrity/segmentation_rescue.py
  Command/query: Filter status=ok,valid=True; group protocol and attempt_number. Totals: primary=4391, F1=587, F2=86; all attempts=5823.
  Status: Headline counts checked against records
  Note: None

Result 2: Paired means, ranges, medians and retained/recovered cohorts [result:reconstruction]
  Manuscript section: sections/performance.tex:1
  Asset: Inline source / conceptual table
  Data (D:): metrics/recovery/paired_per_sample_metrics.csv; metrics/final/paired_quality.csv; metrics/final/paired_distribution.csv
  Code (C:): evaluation/metrics.py; evaluation/lpips_metric.py
  Command/query: Join paired attempts on id to separate accepted attempt=1 from >1; summarize *_after columns; preserve original before/after cohort definitions
  Status: Narrative source rows linked
  Note: Unpaired/worn-reference have no paired GT. Do not score a different target/reference pair or mistake before fields for the same historical cohort in every CSV.

Result 3: Primary mask/reference failures and remaining fallback rejections [result:rejections]
  Manuscript section: sections/performance.tex:1
  Asset: Inline source / conceptual table
  Data (D:): attempts/request_attempts_compact.csv; figures/primary_reference_failure_overlap.csv
  Code (C:): comfy_custom_nodes/crop_integrity/__init__.py; comfy_custom_nodes/crop_integrity/segmentation_fallback.py
  Command/query: Group rejected records by protocol, attempt_number, category; 44 paired + 44 unpaired invalid reference requests overlap the archived dimensional-unpacking failures
  Status: Category evidence linked
  Note: Original archived unpaired garment-panel count 231 is not the same definition as 232 empty-target-mask rejections in the guarded run.

Result 4: Latency, rejection overhead and accepted/cumulative distinction [result:latency]
  Manuscript section: sections/performance.tex:1
  Asset: Inline source / conceptual table
  Data (D:): attempts/request_attempts_compact.csv; metrics/final/latency_views.csv
  Code (C:): serve_workflow.py
  Command/query: For rejected attempts summarize elapsed_sec; for accepted/cumulative follow tab:latency-percentiles. Compare p50 versus service_p50 explicitly
  Status: Client/service fields separately available
  Note: None

Result 5: Four-model normalization, flags and pairwise signs [result:artifacts]
  Manuscript section: sections/performance.tex:1
  Asset: Inline source / conceptual table
  Data (D:): metrics/recovery/artifact_normalization_statistics.csv; metrics/recovery/artifact_normalization_sensitivity.csv; metrics/final/four_model_diagnostics.csv; metrics/final/closetvton_vs_catvton.csv
  Code (C:): portable_eval_bundle/scripts/generate_nonstable_paper_report.py
  Command/query: Use original pooled z-score and flag code; inspect normalization_scope before comparing values
  Status: Normalization scope linked
  Note: Unchanged baseline images can receive changed normalized z-scores when the pooled cohort changes.
