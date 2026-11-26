"""Reproduce quantitative figures from released data without rendering new predictions."""
import argparse, csv
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--outdir', type=Path, required=True)
args = parser.parse_args()
args.outdir.mkdir(parents=True, exist_ok=False)
out = str(args.outdir)
def data_file(name):
    if name == 'request_attempts_min.csv': return args.data_root / 'attempts/request_attempts_compact.csv'
    if name == 'paired_per_sample_metrics.csv': return args.data_root / 'metrics/recovery' / name
    return args.data_root / 'metrics/final' / name
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8,"axes.labelsize":8,
  "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"axes.spines.top":False,"axes.spines.right":False,
  "pdf.fonttype":42,"savefig.bbox":"tight","savefig.pad_inches":0.02})
BLUE="#3b6ea5"; GREY="#9a9a9a"; ORANGE="#e08e2b"; RED="#c0392b"; W1=3.45
S = list(csv.DictReader(open(data_file("paired_per_sample_metrics.csv"))))
A = list(csv.DictReader(open(data_file("request_attempts_min.csv"))))
col = lambda m: np.array([float(r[m+"_after"]) for r in S])

# LPIPS
l = col("lpips"); fig, ax = plt.subplots(figsize=(W1,1.8))
ax.hist(l, bins=40, color=BLUE, edgecolor="white", linewidth=0.3)
ax.axvline(l.mean(), ls="--", lw=0.8, color="k"); ax.axvline(np.median(l), lw=0.8, color="#2a8a3a")
ax.text(l.mean()+0.005, ax.get_ylim()[1]*0.92, f"mean {l.mean():.3f}", fontsize=6.5)
ax.text(l.mean()+0.005, ax.get_ylim()[1]*0.80, f"median {np.median(l):.3f}", fontsize=6.5, color="#2a8a3a")
ax.set_xlabel("Full-image LPIPS (paired, 2,032 outputs)"); ax.set_ylabel("Outputs")
fig.savefig(f"{out}/fig_lpips_paired.pdf"); plt.close(fig)

# metric histograms
fig, axs = plt.subplots(2,2,figsize=(7.0,3.6))
for ax,(k,lab) in zip(axs.flat,[("psnr","PSNR (dB)"),("ssim","SSIM"),("psnr_masked","Masked PSNR (dB)"),("ssim_masked","Masked SSIM")]):
    ax.hist(col(k), bins=40, color=BLUE, edgecolor="white", linewidth=0.3); ax.set_xlabel(lab); ax.set_ylabel("Outputs")
fig.tight_layout(h_pad=0.8, w_pad=1.2); fig.savefig(f"{out}/fig_metrics_hist_paired.pdf"); plt.close(fig)

# latency of the accepted attempt, stacked by attempt number
acc = {1:[],2:[],3:[]}
for r in A:
    if r["protocol"]=="paired" and r["valid"]=="True": acc[int(r["attempt_number"])].append(float(r["elapsed_sec"]))
bins = np.arange(10.4, 13.65, 0.05); fig, ax = plt.subplots(figsize=(W1,1.8))
ax.hist([np.clip(acc[k],10.4,13.6) for k in (1,2,3)], bins=bins, stacked=True, color=[BLUE,ORANGE,RED],
        label=["primary path (1,759)","fallback 1 (238)","fallback 2 (35)"], edgecolor="white", linewidth=0.2)
ax.set_xlabel("Latency of the accepted attempt, paired (s)"); ax.set_ylabel("Requests"); ax.legend(frameon=False, loc="upper right")
fig.savefig(f"{out}/fig_runtime_paired.pdf"); plt.close(fig)

# four-model diagnostics (1,000 pairs)
M = {r["model"]:r for r in csv.DictReader(open(data_file("four_model_diagnostics.csv")))}
names = ["ClosetVTON","CatVTON","HR-VITON","IDM-VTON"]; cols = [BLUE,RED,ORANGE,"#3c9a4a"]
fig, (a,b) = plt.subplots(1,2,figsize=(7.0,2.2), gridspec_kw={"width_ratios":[1.4,1]})
fam = [("cloth_score_mean","Garment"),("face_score_mean","Face"),("pose_score_mean","Pose"),("fidelity_composite_mean","Composite")]
x = np.arange(len(fam)); w = 0.2
for i,n in enumerate(names): a.bar(x+(i-1.5)*w, [float(M[n][k]) for k,_ in fam], w, color=cols[i], label=n)
a.set_xticks(x); a.set_xticklabels([f for _,f in fam]); a.set_ylim(0.65,0.80); a.set_ylabel("Proxy score"); a.legend(frameon=False, ncol=2, fontsize=6.5); a.set_title("(a) Fidelity proxies (higher is better)")
art = [("fft_peakiness_mean","FFT peak."),("jpeg_blockiness_mean","JPEG block."),("noise_inconsistency_mean","Noise incons.")]
x = np.arange(len(art))
for i,n in enumerate(names): b.bar(x+(i-1.5)*w, [float(M[n][k]) for k,_ in art], w, color=cols[i])
b.set_xticks(x); b.set_xticklabels([f for _,f in art]); b.set_title("(b) Artifact statistics (lower is better)")
fig.tight_layout(); fig.savefig(f"{out}/fig_diagnostic_means.pdf"); plt.close(fig)

# recovery funnel
C = list(csv.DictReader(open(data_file("completion_overview.csv"))))
fig, axs = plt.subplots(1,3,figsize=(7.0,1.9))
for ax,r,t in zip(axs,C,["Paired","Unpaired","Worn reference"]):
    n = int(r["requests"]); v = [int(r["guarded_valid"]),int(r["fallback_valid"]),int(r["final_valid"])]
    ax.bar(range(3), v, color=[BLUE,ORANGE,"#3c9a4a"], width=0.6); ax.axhline(n, ls=":", lw=0.7, color="k")
    for i,c in enumerate(v): ax.text(i, c+n*0.012, f"{c:,}\n({100*c/n:.1f}%)", ha="center", fontsize=6.5)
    ax.set_ylim(0, n*1.22); ax.set_xticks(range(3)); ax.set_xticklabels(["Primary\n(guarded)","+ Fallback 1","+ Fallback 2\n(final)"], fontsize=6.5)
    ax.set_title(f"{t} ({n:,} requests)"); axs[0].set_ylabel("Accepted outputs")
fig.tight_layout(); fig.savefig(f"{out}/fig_recovery_funnel.pdf"); plt.close(fig)
