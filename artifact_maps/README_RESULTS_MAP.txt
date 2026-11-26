Paper figure, table and result crosswalk — closet — 2026-11-25

Scope: active figures/tables recursively included by main-access.tex.
Numbers follow source declaration order; LaTeX labels are authoritative across revisions.
P: means the manuscript source folder. C: means the code root. D: means the extracted Data folder.
Commands marked P: run in the manuscript source folder; functions marked C: are implementation entry points.
The mapping is evidence linkage, not a claim that every historical pixel or every manuscript sentence
has been independently regenerated. Explicitly partial/missing links are recorded below.
No prediction, metric, frozen result file or manuscript quantitative content was modified.

Code: https://github.com/sayederfanarefin/closet-vton (also snapshotted under paper Code/)
Data: https://www.kaggle.com/datasets/sayederfanarefin/closetvton-final-cascade-evaluation
Kaggle version 1 is private as uploaded. Extracted path is ClosetVTON_Data_20261125/Data/.
5064 accepted outputs = 2032 paired + 2032 unpaired + 1000 worn-reference.
Final records retain accepted outputs and add successful retries; they are not fresh first-pass runs.


Table 1: Terminology [tab:terminology]
  Manuscript: P:sections/background.tex:23
  Asset: Inline TeX / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): Manuscript/cited literature only
  Command/query: Read sections/background.tex
  Status: Conceptual table; no experimental rows
  Note: None

Table 2: Notation [tab:notation]
  Manuscript: P:sections/system.tex:17
  Asset: Inline TeX / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): workflow_api.json; comfy_custom_nodes/crop_integrity/__init__.py; serve_workflow.py
  Command/query: Inspect node IDs and links in workflow_api.json
  Status: Implementation specification; no metric rows
  Note: None

Figure 1: Four-stage architecture and cascade [fig:architecture]
  Manuscript: P:sections/system.tex:54
  Asset: figures/architecture.pdf
  Data (D:): Not applicable / not found; see status
  Code (C:): workflow_api.json; serve_workflow.py; comfy_custom_nodes/crop_integrity/segmentation_fallback.py; comfy_custom_nodes/crop_integrity/segmentation_rescue.py
  Command/query: Compile P:figures/architecture.tex with pdflatex; inspect C:workflow_api.json
  Status: Source diagram available
  Note: SigLIP receives raw reference; canvas receives segmented reference. Crop coordinates are not the empty-mask root cause.

Table 3: Pipeline modules [tab:modules]
  Manuscript: P:sections/system.tex:78
  Asset: Inline TeX / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): workflow_api.json; comfy_custom_nodes/crop_integrity/__init__.py; serve_workflow.py
  Command/query: Inspect node IDs and links in workflow_api.json
  Status: Implementation specification; no metric rows
  Note: None

Table 4: Operating configuration [tab:implementation]
  Manuscript: P:sections/system.tex:198
  Asset: Inline TeX / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): workflow_api.json; provenance/environment.json; provenance/parser_provenance.json; provenance/render_settings_verification.json
  Command/query: Read workflow nodes 3/26/49/115/154/162 and provenance JSON
  Status: Configuration and recorded environment linked
  Note: Recorded source commits include a compatibility patch; fresh dependency resolution is not an exact environment lock.

Table 5: Input-contract comparison [tab:workflow]
  Manuscript: P:sections/system.tex:235
  Asset: Inline TeX / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): Manuscript/cited literature only
  Command/query: Read sections/system.tex and cited references.bib entries
  Status: Literature/specification table; no experimental rows
  Note: External baseline input contracts are literature claims, not measurements in this data package.

Table 6: Metric definitions [tab:metric-defs]
  Manuscript: P:sections/system.tex:320
  Asset: Inline TeX / conceptual table
  Data (D:): Not applicable / not found; see status
  Code (C:): evaluation/metrics.py; evaluation/lpips_metric.py; portable_eval_bundle/scripts/run_fidelity_comparison.py; portable_eval_bundle/scripts/run_deepfake_comparison.py; portable_eval_bundle/scripts/generate_nonstable_paper_report.py
  Command/query: Inspect compute_metrics, LPIPS scorer, pairwise proxies and pooled artifact normalization
  Status: Metric implementations linked
  Note: None

Table 7: Completion, reconstruction and median latency [tab:main-results]
  Manuscript: P:sections/performance.tex:8
  Asset: Inline TeX / conceptual table
  Data (D:): metrics/final/completion_overview.csv; metrics/final/paired_quality.csv; metrics/final/latency_views.csv
  Code (C:): evaluation/metrics.py; evaluation/lpips_metric.py; scripts/prepare_data.py
  Command/query: Read final_valid; final_mean; timing_view=after_final_attempt, p50. Accepted records are status=ok and valid=True.
  Status: Values linked; caption mismatch identified
  Note: Table caption says median service time, but 10.75/11.34/11.07 match client p50, not service_p50. This mapping does not edit the manuscript.

Figure 2: Recovery funnel [fig:funnel]
  Manuscript: P:sections/performance.tex:26
  Asset: figures/fig_recovery_funnel.pdf
  Data (D:): metrics/final/completion_overview.csv
  Code (C:): comfy_custom_nodes/crop_integrity/segmentation_fallback.py; comfy_custom_nodes/crop_integrity/segmentation_rescue.py
  Command/query: P: python scripts/make_figures_final.py /tmp/closet-paper-figures
  Status: Exact plotting script available in paper repository
  Note: Plot reads guarded_valid, fallback_valid, final_valid; P:scripts/figure_data/final_20261122/completion_overview.csv is the frozen plot input.

Figure 3: Paired LPIPS histogram [fig:lpips-paired]
  Manuscript: P:sections/performance.tex:36
  Asset: figures/fig_lpips_paired.pdf
  Data (D:): metrics/recovery/paired_per_sample_metrics.csv; metrics/final/paired_distribution.csv
  Code (C:): evaluation/lpips_metric.py
  Command/query: P: python scripts/make_figures_final.py /tmp/closet-paper-figures; histogram lpips_after with 40 bins
  Status: Exact plotting script available in paper repository
  Note: None

Figure 4: Paired PSNR/SSIM distributions [fig:metrics-hist]
  Manuscript: P:sections/performance.tex:45
  Asset: figures/fig_metrics_hist_paired.pdf
  Data (D:): metrics/recovery/paired_per_sample_metrics.csv; metrics/recovery/paired_histograms.csv
  Code (C:): evaluation/metrics.py
  Command/query: P: python scripts/make_figures_final.py /tmp/closet-paper-figures; use psnr_after, ssim_after, psnr_masked_after, ssim_masked_after
  Status: Exact plotting script available in paper repository
  Note: The paper script uses 40-bin histograms; do not assume its edges equal the archived full-histogram CSV edges.

Table 8: Accepted-attempt and cumulative latency [tab:latency-percentiles]
  Manuscript: P:sections/performance.tex:59
  Asset: Inline TeX / conceptual table
  Data (D:): metrics/final/latency_views.csv; attempts/request_attempts_compact.csv
  Code (C:): serve_workflow.py
  Command/query: Read timing_view=after_final_attempt or after_cumulative_attempts; columns mean,p25,p50,p75,p90,p95
  Status: Full-precision table source linked
  Note: Client elapsed_sec differs from service_elapsed_sec. Cumulative timing sums recorded failed attempts and accepted attempt per protocol/id.

Figure 5: Paired latency by accepted attempt [fig:runtime-paired]
  Manuscript: P:sections/performance.tex:79
  Asset: figures/fig_runtime_paired.pdf
  Data (D:): attempts/request_attempts_compact.csv
  Code (C:): serve_workflow.py
  Command/query: P: python scripts/make_figures_final.py /tmp/closet-paper-figures; filter protocol=paired and valid=True; group attempt_number
  Status: Exact plotting script available in paper repository
  Note: P:scripts/figure_data/final_20261122/request_attempts_min.csv is the frozen reduced view. Plot clips values to 10.4–13.6 s; use unmodified records for statistics.

Figure 6: Worst/median/best paired examples [fig:qual-paired]
  Manuscript: P:sections/performance.tex:88
  Asset: figures/fig_qualitative_paired.png
  Data (D:): figures/fig_qualitative_paired.png; figures/fig_qualitative_paired_manifest.csv; metrics/recovery/paired_per_sample_metrics.csv; manifests/paired.jsonl
  Code (C:): provenance/figure_builders/build_qualitative_paired.py
  Command/query: Inspect manifest IDs and ssim_masked_after; original builder: python figures/build_qualitative_paired.py
  Status: Pixel relationship verified
  Note: Paper PNG (2400x2316) equals the Data PNG (4880x4710) resized using Pillow LANCZOS, RGB MAE=0. All six are accepted attempt 1. Builder snapshot requires original workspace layout and external source photos.

Table 9: ClosetVTON versus CatVTON, 1000 worn pairs [tab:catvton]
  Manuscript: P:sections/performance.tex:103
  Asset: Inline TeX / conceptual table
  Data (D:): metrics/final/closetvton_vs_catvton.csv; metrics/comparison/ours_vs_catvton/fidelity/fidelity_per_pair.csv; metrics/comparison/ours_vs_catvton/deepfake/deepfake_per_pair.csv
  Code (C:): portable_eval_bundle/scripts/run_fidelity_comparison.py; portable_eval_bundle/scripts/run_deepfake_comparison.py
  Command/query: Read metric, closetvton_mean, catvton_mean, delta, closetvton_win_rate, wilcoxon_p; inspect original two-sided signed-rank code
  Status: Full-precision table and per-pair inputs linked
  Note: One repeated reference; diagnostic comparison, not a uniform official benchmark. Original nominal p-values are not multiple-testing adjusted.

Figure 7: Six selected worn-reference pairs [fig:qualitative]
  Manuscript: P:sections/performance.tex:133
  Asset: figures/fig_qualitative_selected.png
  Data (D:): figures/fig_qualitative_selected.png; figures/fig_qualitative_selected_verification.csv; figures/recovery_figure_evidence.json
  Code (C:): provenance/figure_builders/build_recovery_examples.py
  Command/query: Read six verification rows; compare recorded final_path and pixels_identical_to_final
  Status: Paper/Data PNG SHA-256 identical
  Note: Only 00006_00__03173_00 is in final worn_1000 and is primary-path/pixel-identical. Other five come from the wider archived primary run; their final-run counterparts are absent. Their raw outputs/source photos are not all in this derived export.

Table 10: Four-model fidelity and artifact diagnostics [tab:diagnostic]
  Manuscript: P:sections/performance.tex:147
  Asset: Inline TeX / conceptual table
  Data (D:): metrics/final/four_model_diagnostics.csv; metrics/recovery/artifact_normalization_statistics.csv; metrics/recovery/artifact_normalization_sensitivity.csv
  Code (C:): portable_eval_bundle/scripts/generate_nonstable_paper_report.py
  Command/query: Read fidelity_composite_mean, combined_z_mean, combined_z_flag_rate and proxy means
  Status: Full-precision table source linked
  Note: Final four-model artifact z-scores use the pooled four-model cohort. Do not substitute the two-model z-score from pairwise comparisons.

Figure 8: Four-model proxy/artifact bar charts [fig:diagnostic-means]
  Manuscript: P:sections/performance.tex:167
  Asset: figures/fig_diagnostic_means.pdf
  Data (D:): metrics/final/four_model_diagnostics.csv; metrics/recovery/four_model_artifact_means.csv
  Code (C:): portable_eval_bundle/scripts/generate_nonstable_paper_report.py
  Command/query: P: python scripts/make_figures_final.py /tmp/closet-paper-figures
  Status: Exact plotting script available in paper repository
  Note: Paper plot starts proxy axis at 0.65 and omits chroma from artifact bars because of scale; CSV contains chroma.

Result 1: 5064 accepted outputs and cascade stage counts [result:completion]
  Manuscript: P:sections/performance.tex:1
  Asset: Inline TeX / conceptual table
  Data (D:): metrics/final/completion_overview.csv; attempts/request_attempts_compact.csv
  Code (C:): comfy_custom_nodes/crop_integrity/segmentation_fallback.py; comfy_custom_nodes/crop_integrity/segmentation_rescue.py
  Command/query: Filter status=ok,valid=True; group protocol and attempt_number. Totals: primary=4391, F1=587, F2=86; all attempts=5823.
  Status: Headline counts checked against records
  Note: None

Result 2: Paired means, ranges, medians and retained/recovered cohorts [result:reconstruction]
  Manuscript: P:sections/performance.tex:1
  Asset: Inline TeX / conceptual table
  Data (D:): metrics/recovery/paired_per_sample_metrics.csv; metrics/final/paired_quality.csv; metrics/final/paired_distribution.csv
  Code (C:): evaluation/metrics.py; evaluation/lpips_metric.py
  Command/query: Join paired attempts on id to separate accepted attempt=1 from >1; summarize *_after columns; preserve original before/after cohort definitions
  Status: Narrative source rows linked
  Note: Unpaired/worn-reference have no paired GT. Do not score a different target/reference pair or mistake before fields for the same historical cohort in every CSV.

Result 3: Primary mask/reference failures and remaining fallback rejections [result:rejections]
  Manuscript: P:sections/performance.tex:1
  Asset: Inline TeX / conceptual table
  Data (D:): attempts/request_attempts_compact.csv; figures/primary_reference_failure_overlap.csv
  Code (C:): comfy_custom_nodes/crop_integrity/__init__.py; comfy_custom_nodes/crop_integrity/segmentation_fallback.py
  Command/query: Group rejected records by protocol, attempt_number, category; 44 paired + 44 unpaired invalid reference requests overlap the archived dimensional-unpacking failures
  Status: Category evidence linked
  Note: Original archived unpaired garment-panel count 231 is not the same definition as 232 empty-target-mask rejections in the guarded run.

Result 4: Latency, rejection overhead and accepted/cumulative distinction [result:latency]
  Manuscript: P:sections/performance.tex:1
  Asset: Inline TeX / conceptual table
  Data (D:): attempts/request_attempts_compact.csv; metrics/final/latency_views.csv
  Code (C:): serve_workflow.py
  Command/query: For rejected attempts summarize elapsed_sec; for accepted/cumulative follow tab:latency-percentiles. Compare p50 versus service_p50 explicitly
  Status: Client/service fields separately available
  Note: None

Result 5: Four-model normalization, flags and pairwise signs [result:artifacts]
  Manuscript: P:sections/performance.tex:1
  Asset: Inline TeX / conceptual table
  Data (D:): metrics/recovery/artifact_normalization_statistics.csv; metrics/recovery/artifact_normalization_sensitivity.csv; metrics/final/four_model_diagnostics.csv; metrics/final/closetvton_vs_catvton.csv
  Code (C:): portable_eval_bundle/scripts/generate_nonstable_paper_report.py
  Command/query: Use original pooled z-score and flag code; inspect normalization_scope before comparing values
  Status: Normalization scope linked
  Note: Unchanged baseline images can receive changed normalized z-scores when the pooled cohort changes.

Result 6: Four before/rejection/after cascade examples (supporting asset, not active manuscript figure) [result:recovery-examples]
  Manuscript: P:sections/performance.tex:1
  Asset: Inline TeX / conceptual table
  Data (D:): figures/fig_recovery_examples.png; figures/fig_recovery_examples_manifest.csv; figures/recovery_figure_evidence.json
  Code (C:): provenance/figure_builders/build_recovery_examples.py
  Command/query: Read paired IDs 00095_00,00330_00,00273_00,00991_00; attempts 2,3,2,3; compare saved mask binary hashes to accepted diagnostics
  Status: Supporting data asset linked
  Note: Both attempt-3 examples use SCHP masks. Primary attempts returned no image. This asset is not included in the active manuscript; do not assign it an active figure number.

Reproduction boundary for qualitative assets: source photographs are external. The raw pilot masks
for the recovery overlays are not in Kaggle version 1; the released JSON records their accepted-request
hash checks. The archived source builders require the original workspace/run layout. They are supplied
as provenance source snapshots, not advertised as one-command scripts against the derived Data alone.
