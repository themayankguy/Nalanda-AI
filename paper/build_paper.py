"""Single-source builder for the Nalanda AI IEEE conference paper.

Generates, from ONE content definition:
  paper/nalanda_ai_ieee.tex   (official IEEEtran conference class)
  paper/nalanda_ai_ieee.docx  (editable Word version, two-column IEEE-style)
All table values are read from paper/results/paper_results.json and the
project evaluation report, so no number is hand-transcribed.

Run:  python3 paper/build_paper.py   then   tectonic paper/nalanda_ai_ieee.tex
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
R = json.load(open(ROOT / "results" / "paper_results.json"))
REP = json.load(open(REPO / "backend/services/atml/evaluation/evaluation_report.json"))

# ---------------------------------------------------------------------------
# Number helpers (all values come from experiment artifacts)
# ---------------------------------------------------------------------------
def ms(v, scale=1.0, d=3):
    return f"{v[0]*scale:.{d}f} $\\pm$ {v[1]*scale:.{d}f}"

E1 = {row["k"]: row for row in R["E1"]["by_k"]}
k32 = E1[32]
EPS = R["E1"]["epoch_sensitivity_k32"]
E3 = R["E3"]["mean_std"]
E4 = R["E4"]
E5A = R["E5"]["answerable_4"]
SUM = REP["summary"]

def pct(x, d=1):
    return f"{100*x:.{d}f}\\%"

# ---------------------------------------------------------------------------
# Paper metadata
# ---------------------------------------------------------------------------
TITLE = ("Nalanda AI: Grounded Sequence-to-Sequence Answer Refinement and "
         "Autoencoder-Based Scope Detection for Edge-Hosted Academic Study")
AFFIL = ["B.Tech. in Artificial Intelligence and Data Science (AI-DS)",
         "SVKM's Narsee Monjee Institute of Management Studies (NMIMS)",
         "Navi Mumbai, Maharashtra, India"]
AUTHORS = [("Sayali Rajput", "rajput174@nmims.in"),
           ("Mayank Sawant", "mayank.sawant049@nmims.in")]

ABSTRACT = (
    "Students increasingly revise from lecture slides with retrieval-augmented assistants, yet such systems can "
    "return fragmented passages, introduce content that is absent from the source, answer questions outside the "
    "course scope, and generate implausible self-test items. This paper presents Nalanda AI, a locally hosted study "
    "workspace that combines dense retrieval over all-MiniLM-L6-v2 embeddings, FLAN-T5-small answer refinement "
    "protected by a lexical and entity-level grounding check with extractive fallback, a per-folder undercomplete "
    "autoencoder that flags out-of-scope queries through reconstruction error, and a quiz generator that selects "
    "distractors inside a cosine-similarity band. The components are evaluated on 253 chunks from three Big Data "
    "Analytics lecture decks, with a separate course (62 chunks) and 50 probe queries serving as out-of-distribution "
    f"data. Across five random splits, principal component analysis with $k=32$ achieved lower held-out "
    f"reconstruction error than the deployed 35-epoch autoencoder ({k32['pca_mse'][0]*1e3:.2f} vs. "
    f"{k32['ae_mse'][0]*1e3:.2f} $\\times 10^{{-3}}$). The autoencoder detector separated off-topic from in-domain "
    f"queries with an AUROC of {E3['ae_auroc_queries'][0]:.3f}, although a PCA-residual detector performed at least "
    f"as well ({E3['pca_auroc_queries'][0]:.3f}). On a five-question answer-refinement benchmark, the pipeline produced "
    "no unsupported entities under the project's entity metric and correctly abstained on the unanswerable question; "
    f"however, on the four answerable questions its token F1 ({E5A['Proposed_ATML_Pipeline']['f1']:.3f}) was below "
    f"that of direct extraction ({E5A['Direct_Retrieval']['f1']:.3f}). All 104 automated software tests pass. The "
    "results support the scope-detection and abstention components, while indicating that generative refinement "
    "requires larger-scale evaluation before improvements in answer quality can be claimed."
)
KEYWORDS = ("Retrieval-augmented generation, sequence-to-sequence models, autoencoders, principal component analysis, "
            "out-of-distribution detection, question generation, educational technology")

# ---------------------------------------------------------------------------
# Bibliography (IEEE style), ordered automatically by first citation
# ---------------------------------------------------------------------------
BIB = {
 "roediger2006": r"H. L. Roediger and J. D. Karpicke, ``Test-enhanced learning: Taking memory tests improves long-term retention,'' \emph{Psychol. Sci.}, vol. 17, no. 3, pp. 249--255, 2006.",
 "rozenblit2002": r"L. Rozenblit and F. Keil, ``The misunderstood limits of folk science: An illusion of explanatory depth,'' \emph{Cogn. Sci.}, vol. 26, no. 5, pp. 521--562, 2002.",
 "lewis2020": r"P. Lewis \emph{et al.}, ``Retrieval-augmented generation for knowledge-intensive NLP tasks,'' in \emph{Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 33, 2020, pp. 9459--9474.",
 "gao2023": r"Y. Gao \emph{et al.}, ``Retrieval-augmented generation for large language models: A survey,'' \emph{arXiv preprint arXiv:2312.10997}, 2023.",
 "ji2023": r"Z. Ji \emph{et al.}, ``Survey of hallucination in natural language generation,'' \emph{ACM Comput. Surv.}, vol. 55, no. 12, Art. no. 248, 2023.",
 "maynez2020": r"J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, ``On faithfulness and factuality in abstractive summarization,'' in \emph{Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)}, 2020, pp. 1906--1919.",
 "kurdi2020": r"G. Kurdi, J. Leo, B. Parsia, U. Sattler, and S. Al-Emari, ``A systematic review of automatic question generation for educational purposes,'' \emph{Int. J. Artif. Intell. Educ.}, vol. 30, no. 1, pp. 121--204, 2020.",
 "mitkov2003": r"R. Mitkov and L. A. Ha, ``Computer-aided generation of multiple-choice tests,'' in \emph{Proc. HLT-NAACL Workshop Building Educational Applications Using Natural Language Processing}, 2003, pp. 17--22.",
 "karpukhin2020": r"V. Karpukhin \emph{et al.}, ``Dense passage retrieval for open-domain question answering,'' in \emph{Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)}, 2020, pp. 6769--6781.",
 "reimers2019": r"N. Reimers and I. Gurevych, ``Sentence-BERT: Sentence embeddings using Siamese BERT-networks,'' in \emph{Proc. EMNLP-IJCNLP}, 2019, pp. 3982--3992.",
 "wang2020": r"W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, ``MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,'' in \emph{Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 33, 2020, pp. 5776--5788.",
 "vaswani2017": r"A. Vaswani \emph{et al.}, ``Attention is all you need,'' in \emph{Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 30, 2017, pp. 5998--6008.",
 "raffel2020": r"C. Raffel \emph{et al.}, ``Exploring the limits of transfer learning with a unified text-to-text transformer,'' \emph{J. Mach. Learn. Res.}, vol. 21, no. 140, pp. 1--67, 2020.",
 "shaw2018": r"P. Shaw, J. Uszkoreit, and A. Vaswani, ``Self-attention with relative position representations,'' in \emph{Proc. NAACL-HLT}, 2018, pp. 464--468.",
 "chung2024": r"H. W. Chung \emph{et al.}, ``Scaling instruction-finetuned language models,'' \emph{J. Mach. Learn. Res.}, vol. 25, no. 70, pp. 1--53, 2024.",
 "pearson1901": r"K. Pearson, ``On lines and planes of closest fit to systems of points in space,'' \emph{Philos. Mag.}, vol. 2, no. 11, pp. 559--572, 1901.",
 "jolliffe2016": r"I. T. Jolliffe and J. Cadima, ``Principal component analysis: A review and recent developments,'' \emph{Philos. Trans. Roy. Soc. A}, vol. 374, no. 2065, Art. no. 20150202, 2016.",
 "eckart1936": r"C. Eckart and G. Young, ``The approximation of one matrix by another of lower rank,'' \emph{Psychometrika}, vol. 1, no. 3, pp. 211--218, 1936.",
 "hinton2006": r"G. E. Hinton and R. R. Salakhutdinov, ``Reducing the dimensionality of data with neural networks,'' \emph{Science}, vol. 313, no. 5786, pp. 504--507, 2006.",
 "baldi1989": r"P. Baldi and K. Hornik, ``Neural networks and principal component analysis: Learning from examples without local minima,'' \emph{Neural Netw.}, vol. 2, no. 1, pp. 53--58, 1989.",
 "kingma2014": r"D. P. Kingma and M. Welling, ``Auto-encoding variational Bayes,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2014.",
 "vincent2008": r"P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, ``Extracting and composing robust features with denoising autoencoders,'' in \emph{Proc. 25th Int. Conf. Mach. Learn. (ICML)}, 2008, pp. 1096--1103.",
 "chandola2009": r"V. Chandola, A. Banerjee, and V. Kumar, ``Anomaly detection: A survey,'' \emph{ACM Comput. Surv.}, vol. 41, no. 3, Art. no. 15, 2009.",
 "sakurada2014": r"M. Sakurada and T. Yairi, ``Anomaly detection using autoencoders with nonlinear dimensionality reduction,'' in \emph{Proc. 2nd Workshop Mach. Learn. Sensory Data Anal. (MLSDA)}, 2014, pp. 4--11.",
 "hendrycks2017": r"D. Hendrycks and K. Gimpel, ``A baseline for detecting misclassified and out-of-distribution examples in neural networks,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2017.",
 "lin2004": r"C.-Y. Lin, ``ROUGE: A package for automatic evaluation of summaries,'' in \emph{Text Summarization Branches Out}, 2004, pp. 74--81.",
 "rajpurkar2016": r"P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, ``SQuAD: 100,000+ questions for machine comprehension of text,'' in \emph{Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)}, 2016, pp. 2383--2392.",
 "kingma2015": r"D. P. Kingma and J. Ba, ``Adam: A method for stochastic optimization,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2015.",
 "ba2016": r"J. L. Ba, J. R. Kiros, and G. E. Hinton, ``Layer normalization,'' \emph{arXiv preprint arXiv:1607.06450}, 2016.",
 "fawcett2006": r"T. Fawcett, ``An introduction to ROC analysis,'' \emph{Pattern Recognit. Lett.}, vol. 27, no. 8, pp. 861--874, 2006.",
}

# ---------------------------------------------------------------------------
# Tables (values computed from artifacts)
# ---------------------------------------------------------------------------
T_PCA_ROWS = []
for k in [8, 16, 32, 64, 128]:
    r = E1[k]
    T_PCA_ROWS.append([str(k), ms(r["pca_mse"], 1e3), ms(r["ae_mse"], 1e3), ms(r["pca_cos"]), ms(r["ae_cos"]),
                       f"{r['pca_evr'][0]:.3f}", f"{r['pca_ms'][0]:.1f}", f"{r['ae_train_ms'][0]:.1f}"])

T_REPRO_ROWS = [
    ["PCA reconstruction MSE", "0.000502", f"{k32['pca_mse'][0]:.6f}"],
    ["AE reconstruction MSE", "0.002902", f"{k32['ae_mse'][0]:.6f}"],
    ["PCA cosine fidelity", "0.8983", f"{k32['pca_cos'][0]:.4f}"],
    ["AE cosine fidelity", "0.1600", f"{k32['ae_cos'][0]:.4f}"],
    ["PCA time (ms)", "4.73", f"{k32['pca_ms'][0]:.2f}"],
    ["AE training time (ms)", "452.03", f"{k32['ae_train_ms'][0]:.2f}"],
    ["Samples / evaluation", "48 / in-sample", "202 train, 51 held-out"],
]

T_OOD_ROWS = [
    ["AE reconstruction error", ms(E3["ae_auroc_chunks"]), ms(E3["ae_auroc_queries"])],
    ["PCA residual ($k$=32)", ms(E3["pca_auroc_chunks"]), ms(E3["pca_auroc_queries"])],
    ["Centroid cosine distance", ms(E3["centroid_auroc_chunks"]), ms(E3["centroid_auroc_queries"])],
]
T_RULE_ROWS = [
    ["Held-out BDA chunks (ID)", "FPR", pct(E3["rule_fpr_heldout_chunks"][0]) + " $\\pm$ " + pct(E3["rule_fpr_heldout_chunks"][1])],
    ["APDD chunks (OOD)", "TPR", pct(E3["rule_tpr_apdd_chunks"][0]) + " $\\pm$ " + pct(E3["rule_tpr_apdd_chunks"][1])],
    ["In-domain queries", "FPR", pct(E3["rule_fpr_in_queries"][0]) + " $\\pm$ " + pct(E3["rule_fpr_in_queries"][1])],
    ["Off-topic queries", "TPR", pct(E3["rule_tpr_out_queries"][0]) + " $\\pm$ " + pct(E3["rule_tpr_out_queries"][1])],
]

def sysrow(key, label):
    s, a = SUM[key], E5A[key]
    return [label, f"{s['avg_token_f1']:.3f}", f"{s['avg_rouge_l']:.3f}", f"{s['avg_factual_consistency']:.3f}",
            f"{s['avg_structural_score']:.2f}", f"{s['avg_unsupported_entity_rate']:.2f}",
            f"{s['avg_latency_ms']:.0f}", f"{a['f1']:.3f}", f"{a['rouge_l']:.3f}"]

T_REF_ROWS = [sysrow("Direct_Retrieval", "Direct extraction"),
              sysrow("Transformer_Baseline", "Vanilla FLAN-T5"),
              sysrow("Proposed_ATML_Pipeline", "Proposed pipeline")]

T_AUDIT_ROWS = [
    ["OOD threshold", r"$\mu + 2.5\sigma$", r"$\bar{e} + 3\max(s_e, 5{\times}10^{-4})$ or $z \geq 3$"],
    ["LeakyReLU slope", "0.1", "0.2"],
    ["FLAN-T5 attention heads", "8", "6 (FLAN-T5-small: 8+8 layers, $d$=512)"],
    ["Distractor band", "[0.35, 0.92)", "[0.25, 0.92); docstring states [0.40, 0.85]"],
    ["Grounding rejection", "any unsupported entity", r"ratio $>0.45$ with $\geq 2$ entities, or $\rho<0.35$, or $c<0.40$"],
    ["Denoising AE in ingestion", "applied to OCR text", "implemented and unit-tested; not called in ingestion path"],
    ["Explained variance at $k$=32", r"$\geq 85\%$", f"{k32['pca_evr'][0]*100:.1f}\\% (training split)"],
    ["Quiz distractors", "non-generative", "FLAN-T5 used as a fallback when the pool is insufficient"],
]

T_TEST_ROWS = [
    ["Autoencoders, PCA benchmark, OOD", "test\\_atml\\_autoencoders", "14"],
    ["Quiz anchors and distractors", "test\\_atml\\_quiz\\_generation", "18"],
    ["Incremental knowledge integration", "test\\_incremental\\_knowledge\\_integration", "13"],
    ["Answer refinement and grounding", "test\\_atml\\_refinement", "11"],
    ["Semantic search and isolation", "test\\_semantic\\_search", "9"],
    ["Notes generation", "test\\_atml\\_notes\\_generation", "8"],
    ["Notes subtopics and filtering", "test\\_atml\\_notes\\_subtopics", "8"],
    ["Chat normalization and coreference", "test\\_chatbot\\_upgrade", "8"],
    ["Multimodal retrieval", "test\\_multimodal\\_rag", "5"],
    ["Storage and validation", "test\\_storage\\_and\\_models", "4"],
    ["Extraction service", "test\\_extraction\\_service", "3"],
    ["API integration", "test\\_api\\_integration", "2"],
    ["End-to-end PDF extraction", "test\\_real\\_extraction\\_pipeline", "1"],
    ["\\textbf{Total}", "13 modules", "\\textbf{104}"],
]

# ---------------------------------------------------------------------------
# Content blocks
#   ("sec", title) ("sub", title) ("p", text) ("eq", latex, label)
#   ("list", [items], enumerated) ("fig", path|None, caption, label, wide, height_in)
#   ("tab", caption, label, header, rows, colspec, wide)
# ---------------------------------------------------------------------------
B = []
sec = lambda t: B.append(("sec", t))
sub = lambda t: B.append(("sub", t))
p = lambda t: B.append(("p", t))
eq = lambda t, l: B.append(("eq", t, l))
lst = lambda items, num=False: B.append(("list", items, num))
fig = lambda path, cap, lab, wide=False, h=2.0: B.append(("fig", path, cap, lab, wide, h))
tab = lambda cap, lab, head, rows, spec, wide=False: B.append(("tab", cap, lab, head, rows, spec, wide))

# I. INTRODUCTION ------------------------------------------------------------
sec("Introduction")
p(r"Active retrieval practice improves long-term retention more than re-reading \cite{roediger2006}, and learners "
  r"frequently overestimate how well they understand complex mechanisms until they are asked to explain them "
  r"\cite{rozenblit2002}. Study tools that let a student question their own lecture material, and then test "
  r"themselves on it, are therefore pedagogically attractive. Retrieval-augmented generation (RAG) provides a natural "
  r"technical basis for such tools: relevant passages are retrieved from a document collection and passed to a "
  r"generative model that composes an answer \cite{lewis2020,gao2023}.")
p(r"Applying RAG to university lecture slides exposes several practical problems. First, slide decks are terse and "
  r"fragmented, so the top-ranked chunk often contains only part of an enumeration or procedure. Second, generative "
  r"models can produce statements that are not supported by the retrieved evidence \cite{maynez2020,ji2023}, which is "
  r"particularly harmful when the student intends to reproduce the answer in an examination. Third, a general-purpose "
  r"model will usually answer questions unrelated to the course instead of reporting that the material does not cover "
  r"them. Fourth, automatically generated multiple-choice questions depend heavily on distractor quality: options that "
  r"are synonyms of the key make an item ambiguous, whereas options from an unrelated domain make it trivial "
  r"\cite{mitkov2003,kurdi2020}. Finally, many institutions prefer that course material and student interaction logs "
  r"remain on local infrastructure rather than being sent to commercial cloud services.")
p(r"This paper describes \emph{Nalanda AI}, a study workspace that addresses these problems with small models that run "
  r"on a laptop-class machine and with transparent, rule-based safeguards around them. The system does not introduce a "
  r"new learning algorithm; its contribution lies in the integration of established techniques and in an evaluation "
  r"that compares each learned component with a simpler baseline. Specifically, this paper makes the following "
  r"contributions:")
lst([r"A description of an end-to-end, document-scoped study system combining dense retrieval, FLAN-T5 answer "
     r"refinement with a grounding check and extractive fallback, autoencoder-based scope detection, and "
     r"cosine-constrained distractor selection, with each component traced to its implementation.",
     r"A reproducible comparison of principal component analysis (PCA) and the deployed undercomplete autoencoder on "
     r"real lecture embeddings, using held-out data, five random splits, and five latent sizes.",
     r"An evaluation of reconstruction-error out-of-distribution (OOD) detection against PCA-residual and "
     r"centroid-distance baselines on cross-course chunks and probe queries.",
     r"A per-case analysis of the project's answer-refinement benchmark that identifies where its aggregate gains "
     r"originate, together with an audit of discrepancies between the project documentation and the code."], True)
p(r"Section~II reviews related work and Section~III formulates the problem. Section~IV describes the architecture and "
  r"Section~V the experimental methodology. Section~VI defines the metrics, Section~VII reports results, and "
  r"Sections~VIII--X discuss the findings, limitations, and future work.")

# II. RELATED WORK -------------------------------------------------------------
sec("Related Work")
sub("Retrieval-Augmented Generation and Sentence Embeddings")
p(r"Lewis \emph{et al.} \cite{lewis2020} combined a dense retriever with a sequence-to-sequence generator and showed "
  r"gains on knowledge-intensive tasks; dense passage retrieval \cite{karpukhin2020} established bi-encoder retrieval "
  r"as a strong alternative to sparse methods, and Gao \emph{et al.} \cite{gao2023} survey the subsequent design space. "
  r"Sentence-BERT \cite{reimers2019} made bi-encoder sentence embeddings efficient enough for semantic search, and "
  r"MiniLM \cite{wang2020} distilled such encoders into compact models. Nalanda AI uses the all-MiniLM-L6-v2 checkpoint, "
  r"which follows this line of work, to embed both chunks and queries.")
sub("Encoder--Decoder Transformers")
p(r"The Transformer \cite{vaswani2017} replaced recurrence with multi-head attention. T5 \cite{raffel2020} cast NLP "
  r"tasks in a text-to-text format and replaced absolute position embeddings with learned, bucketed relative position "
  r"biases, a refinement of relative position representations \cite{shaw2018}. FLAN-T5 \cite{chung2024} further "
  r"instruction-tuned T5 on a large task mixture, improving zero-shot instruction following. The present system uses "
  r"the smallest FLAN-T5 checkpoint so that inference remains feasible without a discrete GPU.")
sub("Dimensionality Reduction and Autoencoders")
p(r"PCA \cite{pearson1901,jolliffe2016} projects data onto the directions of maximal variance; by the Eckart--Young "
  r"theorem \cite{eckart1936}, the truncated singular value decomposition gives the best rank-$k$ approximation in the "
  r"least-squares sense. Baldi and Hornik \cite{baldi1989} showed that a linear autoencoder trained with squared error "
  r"recovers the principal subspace, so nonlinear autoencoders \cite{hinton2006} can only improve on PCA by exploiting "
  r"curvature in the data manifold. Variational autoencoders \cite{kingma2014} impose a probabilistic latent prior, and "
  r"denoising autoencoders \cite{vincent2008} learn to map corrupted inputs back to clean ones.")
sub("Anomaly and Out-of-Distribution Detection")
p(r"Chandola \emph{et al.} \cite{chandola2009} review anomaly detection, including reconstruction-based approaches. "
  r"Sakurada and Yairi \cite{sakurada2014} used autoencoder reconstruction error to detect anomalies and compared it "
  r"with linear PCA, and Hendrycks and Gimpel \cite{hendrycks2017} established simple baselines and AUROC-based "
  r"evaluation for OOD detection. These works motivate the comparison of autoencoder and PCA residuals in Section~VII.")
sub("Question Generation and Distractor Selection")
p(r"Mitkov and Ha \cite{mitkov2003} generated multiple-choice items by selecting semantically close concepts as "
  r"distractors, and Kurdi \emph{et al.} \cite{kurdi2020} systematically reviewed automatic question generation for "
  r"education, noting that evaluation is often limited and rarely includes psychometric analysis. Nalanda AI follows "
  r"the similarity-based approach to distractor selection.")
sub("Faithfulness of Generated Text")
p(r"Maynez \emph{et al.} \cite{maynez2020} showed that abstractive models frequently produce unfaithful content, and "
  r"Ji \emph{et al.} \cite{ji2023} survey hallucination across generation tasks. Lexical overlap metrics such as token "
  r"F1 \cite{rajpurkar2016} and ROUGE-L \cite{lin2004} are inexpensive but correlate imperfectly with faithfulness. The "
  r"grounding check in this work is a lightweight lexical heuristic and should be interpreted in that light.")

# III. PROBLEM FORMULATION -----------------------------------------------------
sec("Problem Formulation and Research Objectives")
p(r"Let a course folder contain chunks $\mathcal{D}=\{c_1,\dots,c_N\}$ extracted from uploaded documents, each with a "
  r"unit-norm embedding $\mathbf{e}_i\in\mathbb{R}^{384}$. Given a student query $q$, the system must (i) return an "
  r"answer $a$ whose content is supported by retrieved evidence $\mathcal{E}\subset\mathcal{D}$, or abstain; (ii) "
  r"identify queries and documents that lie outside the scope of the folder; and (iii) produce multiple-choice items "
  r"whose distractors are plausible but not equivalent to the key. These tasks must run on a single local machine and "
  r"remain isolated per user and per folder. The evaluation addresses four research questions:")
lst([r"\textbf{RQ1:} How does linear PCA compare with the deployed undercomplete autoencoder in held-out reconstruction "
     r"fidelity and computational cost on lecture embeddings?",
     r"\textbf{RQ2:} Does autoencoder reconstruction error separate in-scope from out-of-scope chunks and queries, and "
     r"how does it compare with simpler detectors?",
     r"\textbf{RQ3:} Does grounded refinement reduce unsupported content relative to an unguarded sequence-to-sequence "
     r"baseline, and how does it affect lexical agreement with reference answers?",
     r"\textbf{RQ4:} How does the implemented cosine band partition the space of candidate distractor concepts?"])
p(r"Models are executed locally once downloaded; the model weights themselves are obtained from the Hugging Face Hub "
  r"on first use, so the system is not free of external dependencies at installation time.")

# IV. ARCHITECTURE -------------------------------------------------------------
sec("Proposed System Architecture")
p(r"Fig.~\ref{fig:arch} summarizes the architecture. A browser client communicates with a FastAPI server that manages "
  r"users, folders, and documents in SQLite through SQLAlchemy. Each folder has an isolated vector index. Retrieval "
  r"results flow to the answer-refinement, notes, and quiz services; the autoencoder detector operates on the same "
  r"embeddings. The following subsections describe each component as implemented in the project source code.")
fig(None, r"Overall architecture of Nalanda AI: client views, API layer, per-folder vector index, and the "
    r"refinement, scope-detection, notes, and quiz services.", "fig:arch", True, 2.6)

sub("Document Ingestion and Text Normalization")
p(r"Digital PDF pages are parsed with PyMuPDF, which extracts text, native tables, and raster figures; pages without an "
  r"extractable text layer, and standalone images, are processed with PaddleOCR (PP-StructureV3). Extraction runs in a "
  r"time-limited subprocess. Regular-expression normalization then repairs line-break hyphenation, unwraps soft line "
  r"breaks, removes repeated slide banners, and protects code blocks and tables. Text is grouped into section-aware "
  r"chunks of roughly 1{,}200--1{,}800 characters with a 150-character overlap, and each chunk retains its document "
  r"identifier, page number, section title, and links to figures and tables (Fig.~\ref{fig:pipeline}).")
fig(None, r"Document ingestion and retrieval pipeline, from upload and extraction to normalization, chunking, "
    r"embedding, and per-folder indexing.", "fig:pipeline", False, 2.3)

sub("Semantic Embedding and Retrieval")
p(r"Chunks and queries are encoded with all-MiniLM-L6-v2 \cite{wang2020,reimers2019} and L2-normalized, so that the dot "
  r"product equals cosine similarity. Embeddings are stored per folder as a NumPy matrix with a JSON metadata file, and "
  r"retrieval ranks chunks by dot product with the query vector. Adjacent chunks belonging to the same section are "
  r"stitched together before answer generation to reduce fragmentation.")

sub("Dimensionality Reduction with PCA")
p(r"For a matrix of $N$ embeddings $\mathbf{X}\in\mathbb{R}^{N\times D}$ with mean $\bar{\mathbf{x}}$, PCA is computed "
  r"from the singular value decomposition of the centered data:")
eq(r"\mathbf{X}_c=\mathbf{X}-\mathbf{1}\bar{\mathbf{x}}^{\top}=\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}", "eq:svd")
p(r"Retaining the first $k$ right singular vectors $\mathbf{V}_k$ yields the reconstruction and the explained-variance "
  r"ratio")
eq(r"\hat{\mathbf{X}}=\mathbf{X}_c\mathbf{V}_k\mathbf{V}_k^{\top}+\mathbf{1}\bar{\mathbf{x}}^{\top},\qquad "
   r"\eta_k=\frac{\sum_{i=1}^{k}\sigma_i^2}{\sum_{j}\sigma_j^2}", "eq:pca")
p(r"In the current implementation, PCA is provided as a benchmarking utility and is not part of the request-serving "
  r"path; Section~VII evaluates it both as a compressor and as an OOD scoring function.")

sub("Autoencoder-Based Scope Detection")
p(r"The deployed detector is an undercomplete autoencoder with encoder $384\rightarrow128\rightarrow k$ and a mirrored "
  r"decoder, using layer normalization \cite{ba2016}, LeakyReLU (slope 0.2), dropout of 0.05, and a layer-normalized "
  r"latent code with $k=32$. With encoder $f_\theta$ and decoder $g_\phi$,")
eq(r"\mathbf{z}=f_\theta(\mathbf{x}),\qquad \hat{\mathbf{x}}=g_\phi(\mathbf{z}),\qquad "
   r"\mathcal{L}_{\mathrm{AE}}=\frac{1}{D}\|\mathbf{x}-\hat{\mathbf{x}}\|_2^2", "eq:ae")
p(r"A separate model is trained for each folder on all of its chunk embeddings, using full-batch Adam \cite{kingma2015} "
  r"with a learning rate of $5\times10^{-3}$, weight decay of $10^{-5}$, and 35 epochs, and is cached on disk. For a "
  r"query embedding $\mathbf{q}$, the reconstruction error is")
eq(r"e(\mathbf{q})=\frac{1}{D}\|\mathbf{q}-g_\phi(f_\theta(\mathbf{q}))\|_2^2", "eq:err")
p(r"Let $\bar{e}$ and $s_e$ be the mean and standard deviation of the training errors. The query is flagged when")
eq(r"e(\mathbf{q})>\bar{e}+3\max(s_e,\,5\times10^{-4})\ \ \vee\ \ (e(\mathbf{q})-\bar{e})/s_e\geq 3", "eq:rule")
p(r"The search service returns an out-of-scope message only if the query is flagged \emph{and} the best retrieval "
  r"similarity is below 0.28. During ingestion, a new document is marked as mismatched in its metadata, without being "
  r"rejected, when at least 65\% of its chunks are flagged. The code base also contains a variational autoencoder "
  r"\cite{kingma2014} trained on the negative evidence lower bound")
eq(r"\mathcal{L}_{\mathrm{VAE}}=\frac{1}{D}\|\mathbf{x}-g_\phi(\mathbf{z})\|_2^2-\frac{\beta}{2}\sum_{j=1}^{k}"
   r"\left(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\right)", "eq:vae")
p(r"with $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$ and "
  r"$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, a denoising autoencoder \cite{vincent2008} (Gaussian "
  r"noise $\sigma=0.15$, feature masking $p=0.10$, latent size 48), and a sparse autoencoder with an L1 penalty. These "
  r"variants are implemented and unit-tested, but in the evaluated version none of them is invoked in the serving or "
  r"ingestion path.")

sub("Sequence-to-Sequence Answer Refinement")
p(r"Answer refinement uses FLAN-T5-small \cite{chung2024,raffel2020}: an encoder--decoder Transformer with eight "
  r"encoder and eight decoder layers, model width 512, six attention heads of dimension 64, gated-GELU feed-forward "
  r"layers, and approximately 77M parameters. T5 attention omits explicit $1/\sqrt{d_k}$ scaling and adds a learned "
  r"scalar bias that depends on the bucketed relative distance between positions (32 buckets):")
eq(r"\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V})=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top}+\mathbf{B})\mathbf{V},"
   r"\qquad B_{ij}=b_h(\mathrm{bkt}(j-i))", "eq:attn")
p(r"Up to six retrieved chunks form the context. Generation uses beam search with two beams, a no-repeat $n$-gram size "
  r"of 3, and minimum and maximum output lengths of 32 and 1{,}536 tokens; the configured maximum input length is "
  r"2{,}048 tokens, which exceeds the 512-token length used in T5 pre-training. Before generation, a deterministic "
  r"structurer classifies the query as procedural, taxonomic, comparative, or descriptive and formats the evidence as "
  r"numbered steps, bullet lists, comparison tables, or paragraphs respectively; this structured evidence also serves as "
  r"the extractive fallback.")

sub("Entity-Level Grounding and Extractive Fallback")
p(r"Each candidate answer $a$ is sanitized (conversational preambles, slide numbering, and OCR artifacts are removed) "
  r"and compared with the concatenated evidence. Let $T(\cdot)$ denote the set of non-stop-word content tokens and "
  r"$\mathcal{M}(a)$ the set of entities in $a$, where entities are acronyms or multi-word title-case phrases detected "
  r"by a regular expression. The check computes the token overlap and the unsupported-entity ratio")
eq(r"\rho=\frac{|T(a)\cap T(\mathcal{E})|}{|T(a)|},\qquad u=\frac{|\{m\in\mathcal{M}(a):m\notin\mathcal{E}\}|}"
   r"{|\mathcal{M}(a)|}", "eq:ground")
p(r"and a composite confidence that also uses the retrieval confidence $r$:")
eq(r"c=\left(0.6\,\rho+0.4\,(1-u)\right)\min(1,\,r+0.2)", "eq:conf")
p(r"An answer is accepted when $\rho\geq0.35$, $c\geq0.40$, no repetition loop is detected, and it is not the case that "
  r"$u>0.45$ with at least two unsupported entities. Otherwise, the structured extractive answer is validated in the same "
  r"way and returned if its confidence is higher. If $r<0.30$ or fewer than 15\% of query tokens occur in the evidence, "
  r"the system abstains with a fixed message. This procedure is a heuristic filter, not a formal guarantee: a single "
  r"unsupported entity, a substituted lowercase word, or an unsupported claim phrased with supported vocabulary can pass.")

sub("Geometry-Constrained Multiple-Choice Question Generation")
p(r"The quiz generator mines fact anchors from chunks using copular and enumerative patterns (for example, "
  r"``$X$ is defined as $Y$'') and converts each anchor into a question stem whose key is the anchor concept. Candidate "
  r"distractors come from concepts in the folder and from curated lists of sibling concepts per domain. With "
  r"$\mathbf{e}_a$ and $\mathbf{e}_d$ the MiniLM embeddings of the key and a candidate, the eligible set is")
eq(r"\mathcal{D}(a)=\{d\in\mathcal{C}:\ 0.25\leq\cos(\mathbf{e}_a,\mathbf{e}_d)<0.92\}", "eq:band")
p(r"The upper bound excludes near-synonyms of the key and the lower bound excludes unrelated concepts. Eligible "
  r"candidates are ranked by decreasing similarity, and string-level rules remove options that are too similar to the "
  r"key or to each other. If fewer than three distractors remain, curated siblings are added; if the pool is still "
  r"insufficient, FLAN-T5 is prompted to list related concepts. The procedure is therefore primarily geometric but not "
  r"entirely non-generative.")

sub("Full-Stack Application Integration")
p(r"The client is a single-page application written in HTML, CSS, and JavaScript with four views (Fig.~\ref{fig:ui}): a "
  r"home dashboard; a conversational question-answering view with source citations and page previews; a notes view "
  r"that assembles multi-document study notes and exports them to PDF; and a quiz view that presents ten-question "
  r"multiple-choice quizzes with feedback and page references.")
fig(None, r"Nalanda AI user interface: home dashboard, conversational question answering with citations, study-notes "
    r"view with PDF export, and practice-quiz view.", "fig:ui", True, 2.8)

# V. METHODOLOGY ----------------------------------------------------------------
sec("Experimental Methodology")
sub("Data")
inn = R["env"]
p(r"The in-distribution corpus consists of the 253 stored chunk embeddings of one course folder containing three Big "
  r"Data Analytics lecture decks (65, 116, and 72 chunks, covering Hadoop and HDFS, the Hadoop ecosystem, and data "
  r"visualization). The OOD document set consists of the 62 stored chunk embeddings of a lecture deck from a different "
  r"course (APDD, Unit 5). In addition, the authors wrote 25 in-domain probe queries, derived from section titles of the "
  r"in-distribution decks, and 25 off-topic probe queries on general subjects; these were embedded with the same "
  r"encoder. Although the original README also mentions Operating Systems, Machine Learning, and Cloud Infrastructure "
  r"material, no experiment in this paper uses those corpora.")
sub("Protocol")
p(r"For E1 (RQ1), the in-distribution embeddings were split 80/20 (202 training and 51 held-out chunks) using five "
  r"seeds. PCA was fitted on the training split and the autoencoder was trained with the deployed hyperparameters "
  r"(Section~IV-D) for $k\in\{8,16,32,64,128\}$. Reconstruction quality was measured on the held-out split. A "
  r"sensitivity check trained the $k=32$ autoencoder for 35, 200, and 1{,}000 epochs. E2 recorded training and "
  r"held-out loss over 200 epochs for the autoencoder and the VAE ($\beta=1$). For E3 (RQ2), each seed's autoencoder was "
  r"used to score held-out in-distribution chunks, OOD chunks, and both probe-query sets; it was compared with the PCA "
  r"residual at $k=32$ and with the cosine distance to the training centroid. The deployed decision rule in "
  r"(\ref{eq:rule}) was applied without retuning. For E4 (RQ4), the 203 unique section titles of the in-distribution "
  r"decks were embedded, and all 20{,}503 pairwise cosine similarities were compared with the band in (\ref{eq:band}). "
  r"E5 (RQ3) reanalyzes the project's existing answer-refinement benchmark, which contains five questions about data "
  r"visualization (definition, procedure, applications, reasons, and one unanswerable question). Gold passages are "
  r"supplied with each question, so retrieval is held fixed and only answer construction is evaluated. The three "
  r"systems are direct extraction of the top passage, FLAN-T5 prompted with the passages (512-token input, 150 new "
  r"tokens, two beams) without structuring or validation, and the full pipeline.")
sub("Environment")
p(r"E1--E4 were executed on the CPU of an Apple M4 machine with 16\,GB of memory, using Python " + inn["python"] +
  r", PyTorch " + inn["torch"] + r", and NumPy " + inn["numpy"] + r". The hardware on which the E5 latencies were "
  r"recorded is not stored in the evaluation report and is therefore marked as [HARDWARE FOR E5 TO BE VERIFIED]. The "
  r"experiment script and all raw outputs are included in the project repository "
  r"(\texttt{paper\_experiments.py} and \texttt{paper\_results.json}).")

# VI. METRICS -------------------------------------------------------------------
sec("Evaluation Metrics")
p(r"\emph{Reconstruction.} Mean squared error is averaged over held-out vectors and dimensions. Angular fidelity is the "
  r"mean cosine similarity between inputs and reconstructions,")
eq(r"\bar{c}=\frac{1}{N}\sum_{i=1}^{N}\frac{\mathbf{x}_i^{\top}\hat{\mathbf{x}}_i}{\|\mathbf{x}_i\|\,\|\hat{\mathbf{x}}_i\|}", "eq:cos")
p(r"and the explained-variance ratio $\eta_k$ is defined in (\ref{eq:pca}). \emph{Detection.} With OOD scores "
  r"$\{s_i\}_{i\in\mathcal{O}}$ and in-distribution scores $\{s_j\}_{j\in\mathcal{I}}$, the area under the ROC curve "
  r"\cite{fawcett2006} is computed with the Mann--Whitney statistic,")
eq(r"\mathrm{AUROC}=\frac{1}{|\mathcal{O}||\mathcal{I}|}\sum_{i\in\mathcal{O}}\sum_{j\in\mathcal{I}}"
   r"\left(\mathbf{1}[s_i>s_j]+\frac{1}{2}\mathbf{1}[s_i=s_j]\right)", "eq:auroc")
p(r"and the deployed rule is summarized by its true-positive rate (TPR) on OOD data and false-positive rate (FPR) on "
  r"in-distribution data. \emph{Answer quality.} For prediction tokens $\hat{y}$ and reference tokens $y$, token F1 "
  r"\cite{rajpurkar2016} and ROUGE-L \cite{lin2004} are")
eq(r"F_1=\frac{2PR}{P+R},\quad P=\frac{|\hat{y}\cap y|}{|\hat{y}|},\quad R=\frac{|\hat{y}\cap y|}{|y|}", "eq:f1")
eq(r"\mathrm{RL}=\frac{2P_LR_L}{P_L+R_L},\quad P_L=\frac{\mathrm{LCS}(\hat{y},y)}{|\hat{y}|},\quad "
   r"R_L=\frac{\mathrm{LCS}(\hat{y},y)}{|y|}", "eq:rouge")
p(r"Factual consistency is the fraction of answer content tokens that occur in the evidence, i.e., $\rho$ in "
  r"(\ref{eq:ground}). The unsupported-entity rate (UER) is $u$ in (\ref{eq:ground}); because it uses the same entity "
  r"extractor as the grounding check, it is not an independent measure of faithfulness. Structural conformance is a "
  r"rule-based score in $[0,1]$ that penalizes preambles, a mismatch with the expected format, and residual OCR artifacts. "
  r"Latency is wall-clock time per answer. The software test pass rate is the fraction of automated tests that pass.")

# VII. RESULTS ------------------------------------------------------------------
sec("Results and Comparative Analysis")
sub("RQ1: PCA Versus the Undercomplete Autoencoder")
p(r"Table~\ref{tab:pca} and Fig.~\ref{fig:pcaae} report held-out reconstruction. PCA achieved lower error and higher "
  r"angular fidelity at every latent size. At the deployed size $k=32$, the PCA error was "
  + f"{k32['pca_mse'][0]*1e3:.3f}" + r"$\times10^{-3}$ with a mean cosine of " + f"{k32['pca_cos'][0]:.3f}" +
  r", whereas the autoencoder reached " + f"{k32['ae_mse'][0]*1e3:.3f}" + r"$\times10^{-3}$ and " +
  f"{k32['ae_cos'][0]:.3f}" + r". The autoencoder error is almost independent of $k$, which indicates that 35 epochs "
  r"leave it far from convergence; Fig.~\ref{fig:train} confirms that the held-out loss is still falling at that point. "
  r"Longer training narrowed the gap (held-out MSE " +
  f"{EPS['ae_mse'][1][0]*1e3:.3f}" + r" and " + f"{EPS['ae_mse'][2][0]*1e3:.3f}" + r"$\times10^{-3}$ after 200 and "
  r"1{,}000 epochs; cosine " + f"{EPS['ae_cos'][1][0]:.3f}" + r" and " + f"{EPS['ae_cos'][2][0]:.3f}" +
  r"), but even after 1{,}000 epochs the autoencoder did not reach the PCA error at the same $k$. PCA retained "
  + f"{k32['pca_evr'][0]*100:.1f}" + r"\% of the training variance at $k=32$ and " + f"{E1[64]['pca_evr'][0]*100:.1f}" +
  r"\% at $k=64$. Fitting PCA took about 4.6\,ms, compared with about 33\,ms for 35 autoencoder epochs on the CPU.")
tab(r"Held-out reconstruction of PCA and the undercomplete autoencoder (35 epochs) on 253 lecture-chunk embeddings "
    r"(mean $\pm$ s.d. over five 80/20 splits). MSE values are $\times10^{-3}$; $\eta_k$ is computed on the training split.",
    "tab:pca", ["$k$", "PCA MSE", "AE MSE", "PCA cosine", "AE cosine", "$\\eta_k$", "PCA (ms)", "AE (ms)"],
    T_PCA_ROWS, "cccccccc", True)
fig("figures/fig3_pca_vs_ae.png", r"Held-out reconstruction versus latent dimension $k$ for PCA and the deployed "
    r"undercomplete autoencoder: (a) mean squared error and (b) mean cosine similarity between inputs and "
    r"reconstructions. Error bars show one standard deviation over five splits.", "fig:pcaae", True)
fig("figures/fig4_training_dynamics.png", r"Training dynamics at $k=32$: (a) undercomplete autoencoder training and "
    r"held-out MSE, with the deployed 35-epoch budget marked (training loss is higher because dropout is active during "
    r"training); (b) VAE negative ELBO with its reconstruction and KL components.", "fig:train", True)
p(r"Table~\ref{tab:repro} compares these measurements with the figures reported in the project README. The direction of "
  r"the comparison is the same, but the absolute values differ because the README figures were computed in-sample on "
  r"48 vectors with an unrecorded device, whereas Table~\ref{tab:pca} uses held-out data. The README training time of "
  r"452\,ms was not reproduced on the CPU, and the README's explained-variance target of at least 85\% at $k=32$ is not "
  r"met on this corpus.")
tab(r"Project-reported values (README) versus values reproduced in this study at $k=32$.", "tab:repro",
    ["Quantity", "README", "This study"], T_REPRO_ROWS, "lcc")

sub("RQ2: Out-of-Distribution Detection")
p(r"Table~\ref{tab:ood} reports AUROC values. All three detectors separated off-topic from in-domain probe queries "
  r"almost perfectly. Detecting chunks from a different course was harder: the PCA residual achieved an AUROC of "
  + f"{E3['pca_auroc_chunks'][0]:.3f}" + r", clearly above the autoencoder (" + f"{E3['ae_auroc_chunks'][0]:.3f}" +
  r") and the centroid distance (" + f"{E3['centroid_auroc_chunks'][0]:.3f}" + r"). Fig.~\ref{fig:ood} shows the "
  r"autoencoder error distributions. The deployed $3\sigma$ rule is conservative: it flagged about 3\% of in-domain "
  r"chunks and queries, but only " + pct(E3['rule_tpr_apdd_chunks'][0]) + r" of the cross-course chunks and "
  + pct(E3['rule_tpr_out_queries'][0]) + r" of the off-topic queries. Because the service additionally requires a top "
  r"retrieval similarity below 0.28 before intercepting a query, the end-to-end interception rate can only be lower than "
  r"the rates in Table~\ref{tab:ood}; it was not measured separately. The autoencoder forward pass for one query took a "
  r"median of " + f"{E3['single_query_check_ms_median'][0]:.3f}" + r"\,ms, excluding query embedding, which retrieval "
  r"requires in any case.")
tab(r"OOD detection (mean $\pm$ s.d. over five seeds). Top: AUROC for cross-course chunks (51 held-out ID vs. 62 OOD) "
    r"and probe queries (25 ID vs. 25 OOD). Bottom: rates of the deployed $3\sigma$ rule.", "tab:ood",
    ["Detector / data", "Chunks", "Queries"], T_OOD_ROWS + [["\\midrule"]] + [[a, b, c] for a, b, c in T_RULE_ROWS], "lcc")
fig("figures/fig5_ood_detection.png", r"Autoencoder reconstruction error (seed 0) for (a) held-out in-domain chunks "
    r"versus chunks from another course and (b) in-domain versus off-topic probe queries. The dashed line is the "
    r"deployed $3\sigma$ threshold.", "fig:ood", True)

sub("RQ3: Grounded Answer Refinement")
p(r"Table~\ref{tab:ref} reproduces the summary of the five-question benchmark and adds averages over the four "
  r"answerable questions. Over all five questions, the pipeline had the highest token F1 and ROUGE-L and an UER of zero, "
  r"while the vanilla FLAN-T5 baseline had a mean UER of 0.20. The per-case breakdown in Fig.~\ref{fig:case} shows that "
  r"the aggregate advantage comes from the unanswerable question, where the pipeline's abstention message matched the "
  r"reference exactly (F1 = 1.0) and both baselines scored zero. On the four answerable questions, the pipeline had the "
  r"lowest token F1 (" + f"{E5A['Proposed_ATML_Pipeline']['f1']:.3f}" + r" vs. " + f"{E5A['Direct_Retrieval']['f1']:.3f}"
  + r" for direct extraction and " + f"{E5A['Transformer_Baseline']['f1']:.3f}" + r" for vanilla FLAN-T5) and the lowest "
  r"ROUGE-L. The baseline's UER of 0.20 corresponds to a single question. Moreover, the pipeline's answer to the "
  r"applications question begins with ``Aircraft'' where the source reads ``Airlines''; this substitution is not counted "
  r"by the UER because the entity extractor ignores single capitalized words. Mean latency was 1{,}211\,ms for the "
  r"pipeline and 1{,}065\,ms for the vanilla baseline.")
tab(r"Answer-refinement benchmark (project evaluation report). Columns 2--7 average all five questions; the last two "
    r"columns average the four answerable questions. Latency in milliseconds.", "tab:ref",
    ["System", "F1", "ROUGE-L", "Fact. cons.", "Struct.", "UER", "Latency", "F1 (4)", "ROUGE-L (4)"],
    T_REF_ROWS, "lcccccccc", True)
fig("figures/fig7_refinement_per_case.png", r"Token F1 per benchmark question. TC5 is unanswerable from the supplied "
    r"evidence; the reference answer is the system's abstention message.", "fig:case")

sub("RQ4: Distractor Similarity Band")
p(r"Fig.~\ref{fig:distr} shows the distribution of pairwise cosine similarities among the 203 section-title concepts "
  r"(mean " + f"{E4['mean']:.3f}" + r", s.d. " + f"{E4['std']:.3f}" + r"). Of all pairs, " + pct(E4['frac_below']) +
  r" fall below the lower bound of 0.25, " + pct(E4['frac_in_band']) + r" fall inside the band, and " +
  pct(E4['frac_above'], 2) + r" exceed 0.92. For title-level concepts the lower bound is therefore the binding "
  r"constraint, and the upper bound rarely applies. No human or psychometric evaluation of the generated items was "
  r"performed, so these results describe the geometry of the candidate space rather than the quality of the questions.")
fig("figures/fig6_distractor_similarity.png", r"Pairwise cosine similarity among 203 lecture concepts with the "
    r"implemented distractor band $[0.25, 0.92)$.", "fig:distr")

sub("Software Verification and Documentation Audit")
p(r"The automated test suite contains 104 tests in 13 modules (Table~\ref{tab:tests}); all 104 passed in 45.97\,s in "
  r"the environment described in Section~V-C. The tests verify functional behavior, such as latent dimensions, KL "
  r"non-negativity, distractor uniqueness, citation preservation, and user isolation; they do not measure learning "
  r"outcomes. Table~\ref{tab:audit} lists discrepancies between the project README and the code that were found while "
  r"preparing this paper; the paper describes the implemented behavior throughout.")
tab(r"Automated tests by module.", "tab:tests", ["Area", "Module", "Tests"], T_TEST_ROWS, "llc")
tab(r"Discrepancies between the project documentation and the implementation.", "tab:audit",
    ["Item", "README", "Implementation"], T_AUDIT_ROWS, "lll", True)

# VIII. DISCUSSION --------------------------------------------------------------
sec("Discussion")
p(r"\emph{Linear versus nonlinear compression.} The PCA result is consistent with theory: PCA is the optimal linear "
  r"reconstructor \cite{eckart1936}, and an autoencoder can surpass it only by exploiting nonlinear structure "
  r"\cite{baldi1989}. With 202 training vectors and a model of roughly $1.1\times10^5$ parameters trained for 35 "
  r"full-batch steps, no such advantage was observed. For a per-folder detector that must be retrained whenever "
  r"documents change, the closed-form PCA residual is cheaper, deterministic, and, in this study, more accurate. The "
  r"autoencoder remains a reasonable design if it is trained to convergence with a validation criterion, but its benefit "
  r"must be demonstrated rather than assumed.")
p(r"\emph{Scope detection.} Reconstruction error is a useful and inexpensive signal for off-topic queries, which is the "
  r"case that matters most for the user interface. The low TPR of the fixed $3\sigma$ rule on cross-course chunks "
  r"indicates that the threshold, rather than the score, limits performance; a threshold calibrated on a validation set "
  r"would allow an explicit trade-off between false alarms and missed detections.")
p(r"\emph{Grounding and refinement.} The measured benefit of the grounded pipeline is concentrated in abstention, which "
  r"is valuable for a study tool because it avoids answering from outside the course. The evidence does not show that "
  r"FLAN-T5-small refinement improves lexical agreement on answerable questions; in this benchmark it reduced it. The "
  r"grounding check is transparent and cheap, but it can reject legitimate paraphrases through the overlap threshold and "
  r"can accept substitutions such as the one observed in TC3. A zero UER on five questions should therefore be read as "
  r"an observation on this test set under this metric, not as a guarantee.")
p(r"\emph{Distractor selection.} The band encodes a sensible design rationale, plausible but non-equivalent options, and "
  r"its geometry is easy to inspect. Whether it produces items of appropriate difficulty and discrimination is a "
  r"hypothesis that requires evaluation with students.")

# IX. LIMITATIONS ---------------------------------------------------------------
sec("Limitations and Threats to Validity")
lst([r"\emph{Corpus size.} Embedding experiments use one folder of 253 chunks from a single subject and one OOD course; "
     r"results may not transfer to other subjects or larger collections.",
     r"\emph{Benchmark size.} The answer-refinement benchmark has five questions, one of which is unanswerable; no "
     r"statistical significance can be claimed, and single cases dominate the averages.",
     r"\emph{References and metrics.} Reference answers were written by the project authors, and token F1 and ROUGE-L "
     r"penalize valid paraphrases. The UER shares its entity extractor with the grounding check and misses single-word "
     r"and lowercase substitutions.",
     r"\emph{Strict grounding.} The overlap threshold may reject correct paraphrased answers; this effect was not "
     r"measured.",
     r"\emph{Embedding dependence.} Retrieval, OOD detection, and distractor selection all depend on the quality of "
     r"all-MiniLM-L6-v2 embeddings for technical vocabulary.",
     r"\emph{Probe queries.} The 50 probe queries were written by the authors and may be easier to separate than real "
     r"student queries.",
     r"\emph{Hardware.} Latencies depend on hardware and software versions; E5 latencies were recorded on unreported "
     r"hardware.",
     r"\emph{Educational effectiveness.} Passing software tests does not establish learning benefits; no user study "
     r"was conducted.",
     r"\emph{Replication.} All experiments were run by the authors on their own system; no independent replication has "
     r"been performed."])

# X. CONCLUSION -----------------------------------------------------------------
sec("Conclusion and Future Work")
p(r"This paper presented Nalanda AI, a locally hosted study workspace that combines dense retrieval, grounded FLAN-T5 "
  r"refinement with extractive fallback and abstention, autoencoder-based scope detection, and cosine-constrained "
  r"distractor selection behind a four-view user interface. On real lecture embeddings, PCA outperformed the deployed "
  r"autoencoder in held-out reconstruction and cross-course detection, while both separated off-topic queries reliably. "
  r"On a small answer-refinement benchmark, the grounded pipeline abstained correctly and produced no unsupported "
  r"entities under the project metric, but it did not improve lexical agreement on answerable questions. Future work "
  r"should (i) build a larger benchmark spanning several subjects with independently written references; (ii) add "
  r"human evaluation of answer faithfulness and of quiz item quality, including difficulty and discrimination; (iii) "
  r"run ablations that isolate the structurer, the generator, and the grounding check; (iv) calibrate OOD thresholds on "
  r"validation data and compare PCA-residual and converged autoencoder detectors; (v) test robustness to OCR noise; and "
  r"(vi) report latency on a documented set of edge devices.")

# ---------------------------------------------------------------------------
# Citation and reference numbering
# ---------------------------------------------------------------------------
def all_text():
    out = []
    for b in B:
        if b[0] in ("p",):
            out.append(b[1])
        elif b[0] == "list":
            out += b[1]
    return " ".join(out)

CITE_ORDER = []
for m in re.finditer(r"\\cite\{([^}]*)\}", all_text()):
    for k in m.group(1).split(","):
        k = k.strip()
        if k not in CITE_ORDER:
            CITE_ORDER.append(k)
missing = [k for k in CITE_ORDER if k not in BIB]
assert not missing, missing
unused = [k for k in BIB if k not in CITE_ORDER]
assert not unused, f"Uncited references: {unused}"
CNUM = {k: i + 1 for i, k in enumerate(CITE_ORDER)}

LABELS = {}
counters = {"fig": 0, "tab": 0, "eq": 0}
for b in B:
    if b[0] == "fig":
        counters["fig"] += 1; LABELS[b[3]] = str(counters["fig"])
    elif b[0] == "tab":
        counters["tab"] += 1; LABELS[b[2]] = str(counters["tab"])
    elif b[0] == "eq":
        counters["eq"] += 1; LABELS[b[2]] = str(counters["eq"])

ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

# ===========================================================================
# LaTeX writer
# ===========================================================================
def tex_escape_cell(s):
    return s

def write_tex():
    L = []
    L.append(r"\documentclass[conference]{IEEEtran}")
    L.append(r"\IEEEoverridecommandlockouts")
    L.append(r"\usepackage{cite}")
    L.append(r"\usepackage{amsmath,amssymb,amsfonts,bm}")
    L.append(r"\usepackage{graphicx}")
    L.append(r"\usepackage{booktabs}")
    L.append(r"\usepackage{xcolor}")
    L.append(r"\usepackage{url}")
    L.append(r"\renewcommand{\boldsymbol}{\bm}")
    L.append(r"% Framed placeholder for figures to be inserted later")
    L.append(r"\newcommand{\figplaceholder}[2]{\fbox{\parbox[c][#1][c]{0.97#2}{\centering\color{gray}\footnotesize "
             r"Reserved space for figure}}}")
    L.append(r"\begin{document}")
    L.append(r"\title{" + TITLE + "}")
    auth = []
    for name, email in AUTHORS:
        auth.append(r"\IEEEauthorblockN{" + name + r"}" + "\n" + r"\IEEEauthorblockA{" +
                    r"\textit{" + AFFIL[0] + r"}\\" + AFFIL[1] + r"\\" + AFFIL[2] + r"\\" + email + "}")
    L.append(r"\author{" + "\n\\and\n".join(auth) + "}")
    L.append(r"\maketitle")
    L.append(r"\begin{abstract}" + ABSTRACT + r"\end{abstract}")
    L.append(r"\begin{IEEEkeywords}" + KEYWORDS + r"\end{IEEEkeywords}")
    for b in B:
        t = b[0]
        if t == "sec":
            L.append(r"\section{" + b[1] + "}")
        elif t == "sub":
            L.append(r"\subsection{" + b[1] + "}")
        elif t == "p":
            L.append(b[1] + "\n")
        elif t == "eq":
            L.append(r"\begin{equation}" + "\n" + b[1] + "\n" + r"\label{" + b[2] + "}\n" + r"\end{equation}")
        elif t == "list":
            env = "enumerate" if b[2] else "itemize"
            L.append(r"\begin{" + env + "}")
            for it in b[1]:
                L.append(r"\item " + it)
            L.append(r"\end{" + env + "}")
        elif t == "fig":
            _, path, cap, lab, wide, h = b
            env = "figure*" if wide else "figure"
            width = r"\textwidth" if wide else r"\columnwidth"
            L.append(r"\begin{" + env + r"}[!t]")
            L.append(r"\centering")
            if path:
                L.append(r"\includegraphics[width=" + ("0.92" if wide else "") + width + "]{" + path + "}")
            else:
                L.append(r"\figplaceholder{" + f"{h}in" + "}{" + width + "}")
            L.append(r"\caption{" + cap + "}")
            L.append(r"\label{" + lab + "}")
            L.append(r"\end{" + env + "}")
        elif t == "tab":
            _, cap, lab, head, rows, spec, wide = b
            env = "table*" if wide else "table"
            L.append(r"\begin{" + env + r"}[!t]")
            L.append(r"\centering")
            L.append(r"\caption{" + cap + "}")
            L.append(r"\label{" + lab + "}")
            L.append(r"\footnotesize")
            L.append(r"\setlength{\tabcolsep}{4pt}")
            L.append(r"\begin{tabular}{" + spec + "}")
            L.append(r"\toprule")
            L.append(" & ".join(head) + r" \\")
            L.append(r"\midrule")
            for row in rows:
                if row == ["\\midrule"]:
                    L.append(r"\midrule")
                else:
                    L.append(" & ".join(row) + r" \\")
            L.append(r"\bottomrule")
            L.append(r"\end{tabular}")
            L.append(r"\end{" + env + "}")
    L.append(r"\begin{thebibliography}{" + str(len(CITE_ORDER)) + "}")
    for k in CITE_ORDER:
        L.append(r"\bibitem{" + k + "} " + BIB[k])
    L.append(r"\end{thebibliography}")
    L.append(r"\end{document}")
    (ROOT / "nalanda_ai_ieee.tex").write_text("\n".join(L), encoding="utf-8")


# ===========================================================================
# DOCX writer
# ===========================================================================
from docx import Document  # noqa: E402
from docx.enum.section import WD_SECTION  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ROW_HEIGHT_RULE  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Pt, Inches, RGBColor  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EQ_DIR = ROOT / "build_eq"
EQ_DIR.mkdir(exist_ok=True)
FONT = "Times New Roman"

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "eta": "η", "theta": "θ",
         "mu": "μ", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "Sigma": "Σ", "pi": "π", "lambda": "λ"}
SYMS = {"le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "times": "×", "pm": "±", "in": "∈", "notin": "∉",
        "to": "→", "rightarrow": "→", "approx": "≈", "cdot": "·", "cap": "∩", "cup": "∪", "odot": "⊙",
        "sim": "∼", "infty": "∞", "dots": "…", "ldots": "…", "neq": "≠", "vee": "∨", "wedge": "∧",
        "top": "ᵀ", "sqrt": "√", "partial": "∂", "setminus": "∖", "mid": "|", "lVert": "‖", "|": "‖"}
CAL = {"D": "𝒟", "E": "ℰ", "C": "𝒞", "M": "ℳ", "N": "𝒩", "L": "ℒ", "O": "𝒪", "I": "ℐ"}
BB = {"R": "ℝ", "S": "𝕊"}


def math_runs(s):
    """Convert a small LaTeX math subset to [(text, {'i':bool,'b':bool,'sub':bool,'sup':bool})]."""
    out = []
    i = 0

    def grab(j):
        if j < len(s) and s[j] == "{":
            depth, k = 0, j
            while k < len(s):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[j + 1:k], k + 1
                k += 1
            return s[j + 1:], len(s)
        if j < len(s) and s[j] == "\\":
            m = re.match(r"\\[A-Za-z]+", s[j:])
            if m:
                return s[j:j + m.end()], j + m.end()
        return s[j:j + 1], j + 1

    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            cmd = m.group(1); i += m.end()
            if cmd in ("mathbf", "bm", "boldsymbol"):
                arg, i = grab(i)
                for t, f in math_runs(arg):
                    out.append((t, {**f, "b": True, "i": False if len(t) > 1 or t.isupper() else f["i"]}))
            elif cmd in ("mathrm", "textrm", "text", "operatorname", "texttt"):
                arg, i = grab(i)
                out.append((arg.replace("\\", ""), {"i": False, "b": False, "sub": False, "sup": False}))
            elif cmd == "mathcal":
                arg, i = grab(i); out.append((CAL.get(arg, arg), {"i": False, "b": False, "sub": False, "sup": False}))
            elif cmd == "mathbb":
                arg, i = grab(i); out.append((BB.get(arg, arg), {"i": False, "b": False, "sub": False, "sup": False}))
            elif cmd in ("hat", "bar", "tilde"):
                arg, i = grab(i)
                mark = {"hat": "\u0302", "bar": "\u0304", "tilde": "\u0303"}[cmd]
                inner = math_runs(arg)
                if inner:
                    t, f = inner[-1]; inner[-1] = (t + mark, f)
                out += inner
            elif cmd == "frac":
                a, i = grab(i); b_, i = grab(i)
                out += [("(", {"i": False, "b": False, "sub": False, "sup": False})] + math_runs(a) + \
                       [(")/(", {"i": False, "b": False, "sub": False, "sup": False})] + math_runs(b_) + \
                       [(")", {"i": False, "b": False, "sub": False, "sup": False})]
            elif cmd in GREEK:
                out.append((GREEK[cmd], {"i": True, "b": False, "sub": False, "sup": False}))
            elif cmd in SYMS:
                out.append((" " + SYMS[cmd] + " " if cmd in ("le", "leq", "ge", "geq", "in", "notin", "to", "approx", "vee")
                            else SYMS[cmd], {"i": False, "b": False, "sub": False, "sup": False}))
            elif cmd in (",", ";", " ", "quad", "qquad", "!"):
                out.append((" ", {"i": False, "b": False, "sub": False, "sup": False}))
            elif cmd in ("{", "}", "%", "_", "&"):
                out.append((cmd, {"i": False, "b": False, "sub": False, "sup": False}))
            elif cmd in ("left", "right", "big", "Big"):
                pass
            elif cmd in ("log", "exp", "min", "max", "cos", "sin"):
                out.append((cmd, {"i": False, "b": False, "sub": False, "sup": False}))
            else:
                out.append((cmd, {"i": False, "b": False, "sub": False, "sup": False}))
        elif c in "_^":
            arg, i = grab(i + 1)
            for t, f in math_runs(arg):
                if t == "ᵀ":
                    out.append(("ᵀ", f)); continue
                out.append((t, {**f, "sub": c == "_", "sup": c == "^"}))
        elif c in "{}":
            i += 1
        else:
            out.append((c, {"i": c.isalpha(), "b": False, "sub": False, "sup": False}))
            i += 1
    return out


def text_runs(t):
    """Convert paragraph LaTeX-lite markup to styled runs."""
    t = t.replace("~", " ").replace("\\%", "%").replace("\\&", "&").replace("``", "\u201c").replace("''", "\u201d")
    t = t.replace("---", "\u2014").replace("--", "\u2013").replace("{,}", ",").replace("\\,", "\u2009")
    t = t.replace("\\_", "_")
    runs = []
    pos = 0
    pat = re.compile(r"\$([^$]+)\$|\\cite\{([^}]*)\}|\\ref\{([^}]*)\}|\\emph\{([^}]*)\}|\\textbf\{([^}]*)\}"
                     r"|\\textit\{([^}]*)\}|\\texttt\{([^}]*)\}|\\midrule")
    for m in pat.finditer(t):
        if m.start() > pos:
            runs.append((t[pos:m.start()], {}))
        if m.group(1) is not None:
            for s, f in math_runs(m.group(1)):
                runs.append((s, {"i": f["i"], "b": f["b"], "sub": f["sub"], "sup": f["sup"]}))
        elif m.group(2) is not None:
            nums = sorted(CNUM[k.strip()] for k in m.group(2).split(","))
            runs.append(("[" + "], [".join(str(n) for n in nums) + "]", {}))
        elif m.group(3) is not None:
            lab = m.group(3)
            runs.append((ROMAN[int(LABELS[lab])] if lab.startswith("tab:") else LABELS[lab], {}))
        elif m.group(4) is not None:
            for s, f in text_runs(m.group(4)):
                runs.append((s, {**f, "i": True}))
        elif m.group(5) is not None:
            for s, f in text_runs(m.group(5)):
                runs.append((s, {**f, "b": True}))
        elif m.group(6) is not None:
            for s, f in text_runs(m.group(6)):
                runs.append((s, {**f, "i": True}))
        elif m.group(7) is not None:
            runs.append((m.group(7).replace("\\_", "_"), {"mono": True}))
        pos = m.end()
    if pos < len(t):
        runs.append((t[pos:], {}))
    return runs


def add_runs(par, runs, size=10, base_bold=False, base_italic=False):
    for s, f in runs:
        r = par.add_run(s)
        r.font.name = "Courier New" if f.get("mono") else FONT
        r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        r.font.size = Pt(size)
        r.bold = base_bold or f.get("b", False)
        r.italic = base_italic or f.get("i", False)
        if f.get("sub"):
            r.font.subscript = True
        if f.get("sup"):
            r.font.superscript = True


def set_cols(section, n, space_twips=245):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols"); sectPr.append(cols)
    cols.set(qn("w:num"), str(n)); cols.set(qn("w:space"), str(space_twips))


def para(doc, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True, before=0, after=0):
    p_ = doc.add_paragraph()
    pf = p_.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(before); pf.space_after = Pt(after)
    pf.first_line_indent = Inches(0.14) if indent else None
    pf.line_spacing = 1.0
    return p_


def render_eq(tex, name):
    path = EQ_DIR / f"{name}.png"
    t = tex.replace(r"\boldsymbol", r"\mathbf").replace(r"\bm", r"\mathbf").replace(r"\ \ ", r"\quad ")
    fig_ = plt.figure(figsize=(0.01, 0.01))
    fig_.text(0, 0, f"${t}$", fontsize=11, math_fontfamily="cm")
    fig_.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=False)
    plt.close(fig_)
    return path


def cell_border(cell, sz=6):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(sz)); el.set(qn("w:color"), "808080")
        borders.append(el)
    tcPr.append(borders)


def no_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}"); el.set(qn("w:val"), "nil"); borders.append(el)
    tblPr.append(borders)


def booktabs(table, nrows, mid_after=None):
    """IEEE-like rules: top, below header, bottom."""
    no_borders(table)
    def rule(row_idx, edge, sz):
        for cell in table.rows[row_idx].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            b = tcPr.find(qn("w:tcBorders"))
            if b is None:
                b = OxmlElement("w:tcBorders"); tcPr.append(b)
            el = OxmlElement(f"w:{edge}"); el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), "000000"); b.append(el)
    rule(0, "top", 8); rule(0, "bottom", 4); rule(nrows - 1, "bottom", 8)
    for r_ in (mid_after or []):
        rule(r_, "bottom", 4)


def write_docx():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = FONT; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    sec0 = doc.sections[0]
    sec0.page_width = Inches(8.5); sec0.page_height = Inches(11)
    sec0.top_margin = Inches(0.75); sec0.bottom_margin = Inches(1.0)
    sec0.left_margin = Inches(0.625); sec0.right_margin = Inches(0.625)
    set_cols(sec0, 1)

    # Title
    tp = para(doc, WD_ALIGN_PARAGRAPH.CENTER, False, 0, 10)
    add_runs(tp, [(TITLE, {})], size=22)
    # Authors (borderless two-column table)
    at = doc.add_table(rows=1, cols=len(AUTHORS)); no_borders(at); at.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ci, (name, email) in enumerate(AUTHORS):
        c = at.rows[0].cells[ci]
        lines = [(name, 11, False), (AFFIL[0], 9, True), (AFFIL[1], 9, False), (AFFIL[2], 9, False), (email, 9, False)]
        c.paragraphs[0].text = ""
        for li, (txt, sz, it) in enumerate(lines):
            pp = c.paragraphs[0] if li == 0 else c.add_paragraph()
            pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pp.paragraph_format.space_after = Pt(0)
            add_runs(pp, [(txt, {})], size=sz, base_italic=it)
    para(doc, indent=False, after=6)

    # Two-column body
    body = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(body, 2)
    ap = para(doc, indent=False, after=4)
    add_runs(ap, [("Abstract", {})], size=9, base_bold=True, base_italic=True)
    add_runs(ap, [("\u2014", {})], size=9, base_bold=True)
    add_runs(ap, text_runs(ABSTRACT), size=9, base_bold=True)
    kp = para(doc, indent=False, after=8)
    add_runs(kp, [("Index Terms", {})], size=9, base_bold=True, base_italic=True)
    add_runs(kp, [("\u2014" + KEYWORDS, {})], size=9, base_bold=True)

    sec_i = 0; sub_i = 0; eq_i = 0
    for b in B:
        t = b[0]
        if t == "sec":
            sec_i += 1; sub_i = 0
            hp = para(doc, WD_ALIGN_PARAGRAPH.CENTER, False, 8, 4)
            r = hp.add_run(f"{ROMAN[sec_i]}. {b[1]}"); r.font.name = FONT; r.font.size = Pt(10); r.font.small_caps = True
        elif t == "sub":
            sub_i += 1
            hp = para(doc, WD_ALIGN_PARAGRAPH.LEFT, False, 4, 2)
            add_runs(hp, [(f"{chr(64 + sub_i)}. {b[1]}", {})], size=10, base_italic=True)
        elif t == "p":
            pp = para(doc)
            add_runs(pp, text_runs(b[1]))
        elif t == "list":
            for n, it in enumerate(b[1], 1):
                pp = para(doc, indent=False)
                pp.paragraph_format.left_indent = Inches(0.2)
                pp.paragraph_format.first_line_indent = Inches(-0.15)
                add_runs(pp, [(f"{n}) " if b[2] else "\u2022 ", {})] + text_runs(it))
        elif t == "eq":
            eq_i += 1
            img = render_eq(b[1], f"eq{eq_i}")
            tb = doc.add_table(rows=1, cols=2); no_borders(tb)
            tb.columns[0].width = Inches(2.95); tb.columns[1].width = Inches(0.4)
            c0, c1 = tb.rows[0].cells
            c0.width = Inches(2.95); c1.width = Inches(0.4)
            c0.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            from PIL import Image
            w_px, h_px = Image.open(img).size
            w_in = min(w_px / 300 * 0.85, 2.85)
            c0.paragraphs[0].add_run().add_picture(str(img), width=Inches(w_in))
            c1.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            add_runs(c1.paragraphs[0], [(f"({eq_i})", {})])
        elif t == "fig":
            _, path, cap, lab, wide, h = b
            if wide:
                s1 = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(s1, 1)
            width = 7.0 if wide else 3.35
            if path:
                fp = para(doc, WD_ALIGN_PARAGRAPH.CENTER, False, 4, 2)
                fp.add_run().add_picture(str(ROOT / path), width=Inches(width * (0.92 if wide else 1)))
            else:
                ph = doc.add_table(rows=1, cols=1); ph.alignment = WD_TABLE_ALIGNMENT.CENTER
                cell = ph.rows[0].cells[0]; cell.width = Inches(width)
                ph.rows[0].height = Inches(h); ph.rows[0].height_rule = WD_ROW_HEIGHT_RULE.EXACTLY
                cell_border(cell)
                cp_ = cell.paragraphs[0]; cp_.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = cp_.add_run("Reserved space for figure"); r.font.size = Pt(8); r.italic = True
                r.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
            cp = para(doc, WD_ALIGN_PARAGRAPH.JUSTIFY, False, 2, 6)
            add_runs(cp, [(f"Fig. {LABELS[lab]}. ", {})] + text_runs(cap), size=8)
            if wide:
                s2 = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(s2, 2)
        elif t == "tab":
            _, cap, lab, head, rows, spec, wide = b
            if wide:
                s1 = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(s1, 1)
            c1p = para(doc, WD_ALIGN_PARAGRAPH.CENTER, False, 6, 0)
            r = c1p.add_run(f"TABLE {ROMAN[int(LABELS[lab])]}"); r.font.size = Pt(8); r.font.name = FONT
            c2p = para(doc, WD_ALIGN_PARAGRAPH.CENTER, False, 0, 3)
            add_runs(c2p, text_runs(cap), size=8)
            real_rows = [rw for rw in rows if rw != ["\\midrule"]]
            mid_idx = [rows.index(["\\midrule"])] if ["\\midrule"] in rows else []
            tb = doc.add_table(rows=1 + len(real_rows), cols=len(head)); tb.alignment = WD_TABLE_ALIGNMENT.CENTER
            for ci, h_ in enumerate(head):
                cp_ = tb.rows[0].cells[ci].paragraphs[0]; cp_.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(cp_, text_runs(h_), size=7.5, base_bold=True)
            for ri, rw in enumerate(real_rows, 1):
                for ci, val in enumerate(rw):
                    cp_ = tb.rows[ri].cells[ci].paragraphs[0]
                    cp_.alignment = WD_ALIGN_PARAGRAPH.LEFT if (ci == 0 or spec[ci] == "l") else WD_ALIGN_PARAGRAPH.CENTER
                    add_runs(cp_, text_runs(val), size=7.5)
            booktabs(tb, 1 + len(real_rows), mid_after=mid_idx)
            para(doc, indent=False, after=4)
            if wide:
                s2 = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(s2, 2)

    hp = para(doc, WD_ALIGN_PARAGRAPH.CENTER, False, 8, 4)
    r = hp.add_run("References"); r.font.name = FONT; r.font.size = Pt(10); r.font.small_caps = True
    for k in CITE_ORDER:
        rp = para(doc, WD_ALIGN_PARAGRAPH.JUSTIFY, False, 0, 1)
        rp.paragraph_format.left_indent = Inches(0.28); rp.paragraph_format.first_line_indent = Inches(-0.28)
        add_runs(rp, [(f"[{CNUM[k]}]\u2003", {})] + text_runs(BIB[k]), size=8)
    doc.save(ROOT / "nalanda_ai_ieee.docx")


if __name__ == "__main__":
    write_tex()
    write_docx()
    print("Citations:", len(CITE_ORDER), "| Figures:", counters["fig"], "| Tables:", counters["tab"],
          "| Equations:", counters["eq"])
