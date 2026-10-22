"""Reproducible experiments for the Nalanda AI IEEE paper.

Every number and figure produced here is computed from real data:
  * In-distribution corpus: stored all-MiniLM-L6-v2 chunk embeddings of three
    Big Data Analytics lecture PDFs (Unit 2, BDA Unit 4, UNIT 5 BDA).
  * Out-of-distribution corpus: stored chunk embeddings of a different course
    (APDD UNIT 5) plus an author-constructed set of off-topic probe queries.
  * Answer-refinement results: the existing evaluation_report.json (N = 5 cases).

Experiments
  E1  PCA vs. undercomplete autoencoder: held-out reconstruction across k.
  E2  Autoencoder / VAE training dynamics (real loss curves).
  E3  Reconstruction-error out-of-distribution detection (AUROC, 3-sigma rule).
  E4  Cosine-similarity distribution of candidate distractors vs. the code band.
  E5  Per-case breakdown of the answer-refinement benchmark.

Run:  python3 backend/services/atml/evaluation/paper_experiments.py
"""
import glob
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.atml.autoencoder_models import (  # noqa: E402
    UndercompleteAutoencoder,
    VariationalAutoencoder,
)
from backend.services.atml.pca_benchmark import compute_pca_reconstruction  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = REPO_ROOT / "paper" / "results"
FIG_DIR = REPO_ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
KS = [8, 16, 32, 64, 128]
EPOCHS = 35          # matches AutoencoderOutlierDetector default
LR = 0.005           # matches service / benchmark default
WD = 1e-5
DEVICE = "cpu"       # deterministic, hardware-neutral reference device

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})
COL_W = 3.45  # IEEE single-column width (inches)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_folder(match_file: str):
    best = None
    for meta_path in glob.glob(str(REPO_ROOT / "storage/users/*/folders/*/vector_index/metadata.json")):
        meta = json.load(open(meta_path))
        chunks = meta.get("chunks", []) if isinstance(meta, dict) else meta
        names = {c.get("source_filename") for c in chunks if isinstance(c, dict)}
        if match_file in names and (best is None or len(chunks) > len(best[1])):
            best = (meta_path, chunks)
    if best is None:
        raise FileNotFoundError(match_file)
    emb = np.load(Path(best[0]).parent / "embeddings.npy").astype(np.float32)
    return emb, best[1]


def split(X, seed, frac=0.8):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n = int(frac * len(X))
    return X[idx[:n]], X[idx[n:]]


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def train_ae(Xtr, k, seed, epochs=EPOCHS, Xva=None):
    set_seed(seed)
    m = UndercompleteAutoencoder(input_dim=Xtr.shape[1], hidden_dim=128, latent_dim=k).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    crit = nn.MSELoss()
    xt = torch.from_numpy(Xtr).to(DEVICE)
    xv = torch.from_numpy(Xva).to(DEVICE) if Xva is not None else None
    hist_tr, hist_va = [], []
    t0 = time.perf_counter()
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        r, _ = m(xt)
        loss = crit(r, xt)
        loss.backward()
        opt.step()
        hist_tr.append(float(loss.item()))
        if xv is not None:
            m.eval()
            with torch.no_grad():
                hist_va.append(float(crit(m(xv)[0], xv).item()))
    train_ms = (time.perf_counter() - t0) * 1000
    m.eval()
    return m, train_ms, hist_tr, hist_va


def ae_recon(m, X):
    with torch.no_grad():
        return m(torch.from_numpy(X).to(DEVICE))[0].cpu().numpy()


def pca_fit(Xtr, k):
    mu = Xtr.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    Vk = Vt[:k].T
    evr = float((S[:k] ** 2).sum() / (S ** 2).sum())
    return mu, Vk, evr


def pca_recon(mu, Vk, X):
    return (X - mu) @ Vk @ Vk.T + mu


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(a * b, axis=1)))


def row_err(a, b):
    return np.mean((a - b) ** 2, axis=1)


def auroc(scores_in, scores_out):
    """Mann-Whitney AUROC: P(score_out > score_in)."""
    s_in, s_out = np.asarray(scores_in), np.asarray(scores_out)
    gt = (s_out[:, None] > s_in[None, :]).mean()
    eq = (s_out[:, None] == s_in[None, :]).mean()
    return float(gt + 0.5 * eq)


# ----------------------------------------------------------------------------
# E1 PCA vs AE
# ----------------------------------------------------------------------------
def exp1(X):
    rows = []
    for k in KS:
        rec = {"k": k, "pca_mse": [], "pca_cos": [], "pca_evr": [], "pca_ms": [],
               "ae_mse": [], "ae_cos": [], "ae_train_ms": []}
        for s in SEEDS:
            Xtr, Xte = split(X, s)
            kk = min(k, Xtr.shape[0] - 1)
            t0 = time.perf_counter()
            mu, Vk, evr = pca_fit(Xtr, kk)
            rec["pca_ms"].append((time.perf_counter() - t0) * 1000)
            P = pca_recon(mu, Vk, Xte)
            rec["pca_mse"].append(mse(Xte, P)); rec["pca_cos"].append(cos(Xte, P)); rec["pca_evr"].append(evr)
            m, tms, _, _ = train_ae(Xtr, kk, s)
            A = ae_recon(m, Xte)
            rec["ae_mse"].append(mse(Xte, A)); rec["ae_cos"].append(cos(Xte, A)); rec["ae_train_ms"].append(tms)
        rows.append({key: (v if key == "k" else [float(np.mean(v)), float(np.std(v))]) for key, v in rec.items()})
    # Extended-training sensitivity check at k = 32
    ext = {"epochs": [], "ae_mse": [], "ae_cos": []}
    for ep in [35, 200, 1000]:
        ms_, cs_ = [], []
        for s in SEEDS:
            Xtr, Xte = split(X, s)
            m, _, _, _ = train_ae(Xtr, 32, s, epochs=ep)
            A = ae_recon(m, Xte)
            ms_.append(mse(Xte, A)); cs_.append(cos(Xte, A))
        ext["epochs"].append(ep); ext["ae_mse"].append([float(np.mean(ms_)), float(np.std(ms_))])
        ext["ae_cos"].append([float(np.mean(cs_)), float(np.std(cs_))])
    return {"by_k": rows, "epoch_sensitivity_k32": ext}


def fig_exp1(r):
    ks = [row["k"] for row in r["by_k"]]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.4))
    ax = axes[0]
    for key, lab, mk in [("pca_mse", "PCA (SVD)", "o"), ("ae_mse", "Undercomplete AE (35 ep.)", "s")]:
        m = [row[key][0] for row in r["by_k"]]; s = [row[key][1] for row in r["by_k"]]
        ax.errorbar(ks, m, yerr=s, marker=mk, capsize=3, label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("Latent dimension $k$"); ax.set_ylabel("Held-out reconstruction MSE")
    ax.set_title("(a) Reconstruction error"); ax.legend()
    ax = axes[1]
    for key, lab, mk in [("pca_cos", "PCA (SVD)", "o"), ("ae_cos", "Undercomplete AE (35 ep.)", "s")]:
        m = [row[key][0] for row in r["by_k"]]; s = [row[key][1] for row in r["by_k"]]
        ax.errorbar(ks, m, yerr=s, marker=mk, capsize=3, label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("Latent dimension $k$"); ax.set_ylabel("Mean cosine (input, reconstruction)")
    ax.set_title("(b) Angular fidelity"); ax.set_ylim(0, 1.02); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig3_pca_vs_ae.png", bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------
# E2 training dynamics
# ----------------------------------------------------------------------------
def exp2(X):
    Xtr, Xva = split(X, 0)
    _, _, tr, va = train_ae(Xtr, 32, 0, epochs=200, Xva=Xva)
    set_seed(0)
    vae = VariationalAutoencoder(input_dim=X.shape[1], hidden_dim=128, latent_dim=32)
    opt = torch.optim.Adam(vae.parameters(), lr=LR, weight_decay=WD)
    xt = torch.from_numpy(Xtr)
    rec, kl, tot = [], [], []
    for _ in range(200):
        vae.train(); opt.zero_grad()
        r, mu, lv = vae(xt)
        L = VariationalAutoencoder.loss_function(r, xt, mu, lv, beta=1.0)
        L["loss"].backward(); opt.step()
        rec.append(float(L["recon_loss"])); kl.append(float(L["kl_loss"])); tot.append(float(L["loss"]))
    return {"ae_train": tr, "ae_val": va, "vae_recon": rec, "vae_kl": kl, "vae_total": tot}


def fig_exp2(r):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.3))
    ep = np.arange(1, len(r["ae_train"]) + 1)
    ax = axes[0]
    ax.plot(ep, r["ae_train"], label="Train MSE"); ax.plot(ep, r["ae_val"], "--", label="Held-out MSE")
    ax.axvline(EPOCHS, color="gray", lw=0.8, ls=":"); ax.text(EPOCHS + 4, 0.02, "35 epochs\n(deployed)", fontsize=7)
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (log scale)")
    ax.set_title("(a) Undercomplete AE ($k$=32)"); ax.legend()
    ax = axes[1]
    ax.plot(ep, r["vae_total"], label="Negative ELBO"); ax.plot(ep, r["vae_recon"], "-.", label="Reconstruction (MSE)")
    ax.plot(ep, r["vae_kl"], ":", label=r"$D_{KL}$")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title(r"(b) VAE ($k$=32, $\beta$=1)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig4_training_dynamics.png", bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------
# E3 OOD detection
# ----------------------------------------------------------------------------
IN_QUERIES = [
    "What is rack awareness in HDFS?", "Explain the role of the NameNode.",
    "What is the Secondary NameNode used for?", "How does HDFS replicate blocks?",
    "What are heartbeat and block reports?", "Explain the Hive job execution flow.",
    "What is the Hive metastore?", "Compare Hive and SQL.", "What is the Grunt shell in Pig?",
    "What are the features of Flume?", "What is Sqoop used for?", "What are the limitations of Hadoop?",
    "Explain the YARN resource manager.", "What is Big SQL architecture?",
    "What is data visualization?", "What are the steps for creating a visualization?",
    "Why is visualization needed in big data?", "What is a line chart used for?",
    "What are interactive visualizations?", "List the tools used for creating visualizations.",
    "What is the problem with many small files in HDFS?", "Explain PigStorage in Pig Latin.",
    "What are data processing operators in Pig?", "What is linear scalability in Hadoop?",
    "Explain exploratory analysis in visualization.",
]
OUT_QUERIES = [
    "How do I bake sourdough bread?", "Who won the 2018 FIFA World Cup?",
    "What is photosynthesis?", "Explain the causes of the French Revolution.",
    "How do vaccines train the immune system?", "What is the capital of Australia?",
    "How do I change a car tyre?", "What is quantum entanglement?",
    "Explain the plot of Hamlet.", "What are the rules of cricket?",
    "How does a refrigerator work?", "What is the boiling point of water at altitude?",
    "Describe the water cycle.", "How do I care for a houseplant?",
    "What is inflation in economics?", "Explain plate tectonics.",
    "What is the Pythagorean theorem?", "How do bees make honey?",
    "What is a black hole?", "Recommend a workout routine for beginners.",
    "How to write a cover letter?", "What causes rainbows?",
    "Explain the theory of evolution.", "What are the symptoms of the flu?",
    "How do I learn to play guitar?",
]


def exp3(X_in, X_apdd):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    q_in = st.encode(IN_QUERIES, normalize_embeddings=True).astype(np.float32)
    q_out = st.encode(OUT_QUERIES, normalize_embeddings=True).astype(np.float32)

    res = {"n_in_chunks_heldout": None, "n_apdd_chunks": int(len(X_apdd)),
           "n_in_queries": len(IN_QUERIES), "n_out_queries": len(OUT_QUERIES), "seeds": {}}
    agg = {}
    for s in SEEDS:
        Xtr, Xte = split(X_in, s)
        res["n_in_chunks_heldout"] = int(len(Xte))
        m, _, _, _ = train_ae(Xtr, 32, s)
        tr_err = row_err(Xtr, ae_recon(m, Xtr))
        thr = float(tr_err.mean() + 3.0 * max(tr_err.std(), 0.0005))  # identical to service rule
        mu, Vk, _ = pca_fit(Xtr, 32)
        cen = Xtr.mean(0); cen /= np.linalg.norm(cen)

        def scores(Z):
            return {"ae": row_err(Z, ae_recon(m, Z)),
                    "pca": row_err(Z, pca_recon(mu, Vk, Z)),
                    "centroid": 1 - Z @ cen}
        S_te, S_ap, S_qi, S_qo = scores(Xte), scores(X_apdd), scores(q_in), scores(q_out)
        r = {}
        for det in ["ae", "pca", "centroid"]:
            r[f"{det}_auroc_chunks"] = auroc(S_te[det], S_ap[det])
            r[f"{det}_auroc_queries"] = auroc(S_qi[det], S_qo[det])
        e_te, e_ap, e_qi, e_qo = S_te["ae"], S_ap["ae"], S_qi["ae"], S_qo["ae"]
        z = lambda e: (e - tr_err.mean()) / (tr_err.std() + 1e-6)
        flag = lambda e: (e > thr) | (z(e) >= 3.0)
        r["rule_fpr_heldout_chunks"] = float(flag(e_te).mean())
        r["rule_tpr_apdd_chunks"] = float(flag(e_ap).mean())
        r["rule_fpr_in_queries"] = float(flag(e_qi).mean())
        r["rule_tpr_out_queries"] = float(flag(e_qo).mean())
        # latency of a single-query check
        q = torch.from_numpy(q_out[:1]); ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            with torch.no_grad():
                rr = m(q)[0]; float(torch.mean((q - rr) ** 2))
            ts.append((time.perf_counter() - t0) * 1000)
        r["single_query_check_ms_median"] = float(np.median(ts))
        res["seeds"][s] = r
        if s == 0:
            res["dist_seed0"] = {"heldout": e_te.tolist(), "apdd": e_ap.tolist(),
                                 "q_in": e_qi.tolist(), "q_out": e_qo.tolist(), "threshold": thr}
        for kk, vv in r.items():
            agg.setdefault(kk, []).append(vv)
    res["mean_std"] = {kk: [float(np.mean(v)), float(np.std(v))] for kk, v in agg.items()}
    return res


def fig_exp3(r):
    d = r["dist_seed0"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.3))
    ax = axes[0]
    bins = np.linspace(0, max(max(d["apdd"]), max(d["heldout"])) * 1.05, 35)
    ax.hist(d["heldout"], bins=bins, alpha=0.6, density=True, label="Held-out BDA chunks (ID)")
    ax.hist(d["apdd"], bins=bins, alpha=0.6, density=True, label="APDD chunks (OOD)")
    ax.axvline(d["threshold"], color="k", ls="--", lw=1, label=r"3$\sigma$ threshold")
    ax.set_xlabel("Per-vector reconstruction MSE"); ax.set_ylabel("Density"); ax.set_title("(a) Document chunks"); ax.legend()
    ax = axes[1]
    allq = d["q_in"] + d["q_out"]
    bins = np.linspace(min(allq) * 0.95, max(allq) * 1.05, 25)
    ax.hist(d["q_in"], bins=bins, alpha=0.6, density=True, label="In-domain queries")
    ax.hist(d["q_out"], bins=bins, alpha=0.6, density=True, label="Off-topic queries")
    ax.axvline(d["threshold"], color="k", ls="--", lw=1, label=r"3$\sigma$ threshold")
    ax.set_xlabel("Per-vector reconstruction MSE"); ax.set_title("(b) User queries"); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig5_ood_detection.png", bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------
# E4 distractor similarity distribution
# ----------------------------------------------------------------------------
def exp4(chunks):
    from sentence_transformers import SentenceTransformer
    import re
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    titles = []
    for c in chunks:
        t = re.sub(r"^[#\s\d\.\•]+", "", str(c.get("section_title") or "")).strip()
        if 3 <= len(t) <= 60 and t.lower() not in {x.lower() for x in titles}:
            titles.append(t)
    E = st.encode(titles, normalize_embeddings=True)
    S = E @ E.T
    iu = np.triu_indices(len(titles), 1)
    sims = S[iu]
    lo, hi = 0.25, 0.92  # band implemented in quiz_generator.py
    return {"n_concepts": len(titles), "n_pairs": int(len(sims)),
            "frac_below": float((sims < lo).mean()), "frac_in_band": float(((sims >= lo) & (sims < hi)).mean()),
            "frac_above": float((sims >= hi).mean()), "mean": float(sims.mean()), "std": float(sims.std()),
            "sims": sims.tolist(), "band": [lo, hi]}


def fig_exp4(r):
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    lo, hi = r["band"]
    ax.hist(r["sims"], bins=50, color="#607D8B", alpha=0.85)
    ax.axvspan(-0.2, lo, color="#E57373", alpha=0.18, label=f"Rejected: cos < {lo}")
    ax.axvspan(lo, hi, color="#81C784", alpha=0.18, label=f"Eligible: {lo} ≤ cos < {hi}")
    ax.axvspan(hi, 1.0, color="#FFB74D", alpha=0.3, label=f"Rejected: cos ≥ {hi}")
    ax.set_xlim(min(r["sims"]) - 0.02, 1.0)
    ax.set_xlabel("Pairwise cosine similarity of lecture concepts"); ax.set_ylabel("Pairs")
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig6_distractor_similarity.png", bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------
# E5 refinement per-case breakdown
# ----------------------------------------------------------------------------
def exp5():
    rep = json.load(open(REPO_ROOT / "backend/services/atml/evaluation/evaluation_report.json"))
    out = {"summary_all5": rep["summary"], "answerable_4": {}, "per_case": {}}
    for sysname, runs in rep["detailed_runs"].items():
        ans = [x for x in runs if "insufficient" not in x["id"]]
        out["answerable_4"][sysname] = {m: round(float(np.mean([x[m] for x in ans])), 4)
                                        for m in ["f1", "rouge_l", "factual_consistency", "structural_score", "unsupported_entity_rate"]}
        out["per_case"][sysname] = {x["id"]: {"f1": x["f1"], "rouge_l": x["rouge_l"], "uer": x["unsupported_entity_rate"]} for x in runs}
    return out


def fig_exp5(r):
    systems = [("Direct_Retrieval", "Direct retrieval"), ("Transformer_Baseline", "Vanilla FLAN-T5"),
               ("Proposed_ATML_Pipeline", "Proposed pipeline")]
    cases = list(r["per_case"]["Direct_Retrieval"].keys())
    labels = ["TC1\ndef.", "TC2\nsteps", "TC3\napps.", "TC4\nreasons", "TC5\nunans."]
    x = np.arange(len(cases)); w = 0.26
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for i, (key, lab) in enumerate(systems):
        ax.bar(x + (i - 1) * w, [r["per_case"][key][c]["f1"] for c in cases], w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Token F1"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig7_refinement_per_case.png", bbox_inches="tight"); plt.close(fig)


def main():
    X_in, chunks = load_folder("UNIT 5 BDA.pdf")
    X_apdd, _ = load_folder("APDD UNIT 5.pdf")
    env = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "machine": platform.machine(), "platform": platform.platform(), "device": DEVICE,
           "n_in_chunks": int(len(X_in)), "dim": int(X_in.shape[1]),
           "in_sources": sorted({c["source_filename"] for c in chunks}), "seeds": SEEDS}
    print("E1 ..."); r1 = exp1(X_in); fig_exp1(r1)
    print("E2 ..."); r2 = exp2(X_in); fig_exp2(r2)
    print("E3 ..."); r3 = exp3(X_in, X_apdd); fig_exp3(r3)
    print("E4 ..."); r4 = exp4(chunks); fig_exp4(r4)
    print("E5 ..."); r5 = exp5(); fig_exp5(r5)
    r4_small = {k: v for k, v in r4.items() if k != "sims"}
    r3_small = {k: v for k, v in r3.items() if k != "dist_seed0"}
    json.dump({"env": env, "E1": r1, "E3": r3_small, "E4": r4_small, "E5": r5,
               "E2_final": {k: v[-1] for k, v in r2.items()}, "E2_at35": {k: v[EPOCHS - 1] for k, v in r2.items()}},
              open(OUT_DIR / "paper_results.json", "w"), indent=2)
    print(json.dumps({"env": env, "E1": r1, "E3": r3_small["mean_std"], "E4": r4_small, "E5_answerable": r5["answerable_4"],
                      "E2_at35": {k: v[EPOCHS - 1] for k, v in r2.items()}}, indent=1))


if __name__ == "__main__":
    main()
