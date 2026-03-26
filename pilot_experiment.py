"""
MOCO2 Pilot Experiment
======================
RQ1: 多模態事件能否在向量空間中依語用功能聚集？
RQ2: 不同模態組合的檢索效能差異？

Data: 30 video segments (10 Amis / 10 Paiwan / 10 Truku)
Text: OCR Chinese subtitles → XLM-RoBERTa mean pooling
Audio: WAV → librosa MFCC + spectral global stats

Usage:
    python pilot_experiment.py

Output:
    pilot_labels.json    -- auto-detected pragmatic labels (edit before re-run)
    pilot_results.png    -- t-SNE + retrieval heatmap
    pilot_retrieval.json -- P@1, P@3, MAP per modality
"""

import json, re, sys, os, pickle
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import torch
import librosa

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path("C:/Users/user/Documents/GitHub/MOCO/data")
OCR_DIR   = ROOT / "ocr_corrected"
WAV_DIR   = ROOT / "wav"
LABEL_FILE = Path("C:/Users/user/Documents/GitHub/MOCO/pilot_labels.json")
OUT_PNG   = Path("C:/Users/user/Documents/GitHub/MOCO/pilot_results.png")
OUT_JSON  = Path("C:/Users/user/Documents/GitHub/MOCO/pilot_retrieval.json")

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON    = sys.executable
LANGUAGES = ["Amis", "Paiwan", "Truku"]

# Keyword-based auto-labeling (Chinese OCR text)
PRAGMATIC_KEYWORDS = {
    "Comforting":  ["慰問", "關懷", "關心", "安慰", "照顧", "探視", "支持", "陪伴", "受災", "災後"],
    "Celebrating": ["慶祝", "慶典", "慶賀", "豐年祭", "頒獎", "歡迎", "祝賀", "歡慶", "活動", "儀式",
                    "交車", "交接", "典禮", "歡欣"],
    "Explaining":  ["介紹", "說明", "教學", "傳承", "學習", "課程", "認識", "了解", "文化", "知識",
                    "記錄", "分享", "示範", "語言", "族語", "部落"],
    "Urging":      ["呼籲", "訴求", "防災", "政策", "保護", "推動", "加強", "改善", "要求", "希望",
                    "促進", "落實", "強化", "升級"],
}

# ── Step 1: Enumerate segments ────────────────────────────────────────────────
def get_segments():
    wavs = sorted(WAV_DIR.glob("*.wav"))
    segs = []
    for w in wavs:
        name = w.stem  # e.g. news-Amis-20260302-01
        lang = name.split("-")[1]
        segs.append({"id": name, "lang": lang, "wav": str(w)})
    return segs

# ── Step 2: Extract OCR Chinese text ─────────────────────────────────────────
def extract_ocr_text(seg_id):
    ocr_path = OCR_DIR / f"{seg_id}.ocr.json"
    if not ocr_path.exists():
        return ""
    with open(ocr_path, encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    seen = set()
    for frame_ts, dets in data:
        for det in dets:
            box, text, conf, label = det
            # Keep Chinese chars only, skip TITV logo
            if conf < 0.8 or label == "logo":
                continue
            if not re.search(r"[\u4e00-\u9fff]", text):
                continue
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)
    return " ".join(texts)

# ── Step 3: Auto-label from keywords ─────────────────────────────────────────
def auto_label(text):
    scores = {k: 0 for k in PRAGMATIC_KEYWORDS}
    for label, keywords in PRAGMATIC_KEYWORDS.items():
        for kw in keywords:
            scores[label] += text.count(kw)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Explaining"  # default

# ── Step 4: Text embedding (XLM-RoBERTa) ─────────────────────────────────────
def get_text_embeddings(segs, texts):
    print("[Text] Loading XLM-RoBERTa...")
    from transformers import AutoTokenizer, AutoModel
    model_name = "FacebookAI/xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    embeddings = []
    for i, text in enumerate(texts):
        seg_id = segs[i]["id"]
        print(f"  [{i+1:02d}/30] {seg_id} ({len(text)} chars)")
        if not text.strip():
            embeddings.append(np.zeros(768))
            continue
        # Truncate to 512 tokens
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        # Mean pool last hidden state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        embeddings.append(emb.squeeze().cpu().numpy())
    return np.array(embeddings)  # (30, 768)

# ── Step 5: Audio embedding (MFCC global stats) ───────────────────────────────
def get_audio_embeddings(segs):
    print("[Audio] Extracting MFCC features...")
    embeddings = []
    for i, seg in enumerate(segs):
        print(f"  [{i+1:02d}/30] {seg['id']}")
        try:
            y, sr = librosa.load(seg["wav"], sr=16000, mono=True)
            mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            delta  = librosa.feature.delta(mfcc)
            rms    = librosa.feature.rms(y=y)
            zcr    = librosa.feature.zero_crossing_rate(y)
            spec_c = librosa.feature.spectral_centroid(y=y, sr=sr)
            # Global stats: mean + std for each feature
            vec = np.concatenate([
                mfcc.mean(1), mfcc.std(1),   # 40
                delta.mean(1), delta.std(1),  # 40
                [rms.mean(), rms.std()],       # 2
                [zcr.mean(), zcr.std()],       # 2
                [spec_c.mean(), spec_c.std()], # 2
            ])  # 86-dim
            embeddings.append(vec)
        except Exception as e:
            print(f"    ERROR: {e}")
            embeddings.append(np.zeros(86))
    return np.array(embeddings)  # (30, 86)

# ── Step 6: Retrieval evaluation ──────────────────────────────────────────────
def retrieval_eval(sim_matrix, labels, k_list=(1, 3)):
    """For each query, rank all others (excl. self), compute P@k and AP."""
    n = len(labels)
    results = {f"P@{k}": [] for k in k_list}
    aps = []

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        ranked = np.argsort(sims)[::-1]
        query_label = labels[i]

        # P@k
        for k in k_list:
            top_k = ranked[:k]
            hits = sum(1 for j in top_k if labels[j] == query_label)
            results[f"P@{k}"].append(hits / k)

        # AP
        rel = [1 if labels[j] == query_label else 0 for j in ranked]
        ap, n_rel = 0.0, 0
        for rank, r in enumerate(rel, 1):
            if r:
                n_rel += 1
                ap += n_rel / rank
        total_rel = sum(rel)
        aps.append(ap / total_rel if total_rel > 0 else 0.0)

    return {k: float(np.mean(v)) for k, v in results.items()} | {"MAP": float(np.mean(aps))}

# ── Step 7: Visualize ─────────────────────────────────────────────────────────
LANG_COLORS  = {"Amis": "#E74C3C", "Paiwan": "#3498DB", "Truku": "#2ECC71"}
LABEL_MARKERS = {"Comforting": "o", "Celebrating": "^", "Explaining": "s", "Urging": "D"}

def plot_tsne(ax, embs, segs, labels, title):
    if embs.shape[0] < 5:
        ax.set_title(title + " (not enough data)")
        return
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(8, embs.shape[0]-1))
    pts = tsne.fit_transform(embs)
    for i, (seg, label) in enumerate(zip(segs, labels)):
        lang = seg["lang"]
        ax.scatter(pts[i, 0], pts[i, 1],
                   c=LANG_COLORS.get(lang, "gray"),
                   marker=LABEL_MARKERS.get(label, "o"),
                   s=120, edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(seg["id"].split("-")[-1], pts[i], fontsize=6,
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

def plot_sim_heatmap(ax, sim_matrix, segs, title):
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ticks = list(range(len(segs)))
    labels_short = [s["id"].split("-")[1][:2] + s["id"].split("-")[-1] for s in segs]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(labels_short, rotation=90, fontsize=5)
    ax.set_yticklabels(labels_short, fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MOCO2 Pilot Experiment")
    print("=" * 60)

    # Step 1: Segments
    segs = get_segments()
    print(f"\nFound {len(segs)} segments: "
          + ", ".join(f"{l}×{sum(s['lang']==l for s in segs)}" for l in LANGUAGES))

    # Step 2: Extract OCR text
    print("\n[OCR] Extracting Chinese subtitle text...")
    texts = []
    for seg in segs:
        t = extract_ocr_text(seg["id"])
        texts.append(t)
        print(f"  {seg['id']}: {len(t)} chars")

    # Step 3: Labels — load existing or auto-detect
    if LABEL_FILE.exists():
        print(f"\n[Labels] Loading from {LABEL_FILE}")
        with open(LABEL_FILE, encoding="utf-8") as f:
            label_map = json.load(f)
        labels = [label_map.get(s["id"], auto_label(texts[i])) for i, s in enumerate(segs)]
    else:
        print("\n[Labels] Auto-detecting from OCR keywords...")
        labels = [auto_label(t) for t in texts]
        label_map = {s["id"]: l for s, l in zip(segs, labels)}
        with open(LABEL_FILE, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {LABEL_FILE}")
        print("  ** Review and correct labels before re-running! **")
        print("  Label distribution:", Counter(labels))

    print("\nLabel distribution:", Counter(labels))
    for seg, label, text in zip(segs, labels, texts):
        print(f"  {seg['id']:40s} [{label:12s}] {text[:60]}")

    # Step 4: Text embeddings
    print("\n" + "=" * 60)
    text_embs = get_text_embeddings(segs, texts)
    print(f"Text embeddings shape: {text_embs.shape}")

    # Step 5: Audio embeddings
    print("\n" + "=" * 60)
    audio_embs = get_audio_embeddings(segs)
    print(f"Audio embeddings shape: {audio_embs.shape}")

    # Step 6: Normalize + fuse
    text_n  = normalize(text_embs)
    audio_n = normalize(audio_embs)
    multi_n = normalize(np.hstack([text_n, audio_n]))

    # Step 7: Similarity matrices
    sim_text  = cosine_similarity(text_n)
    sim_audio = cosine_similarity(audio_n)
    sim_multi = cosine_similarity(multi_n)

    # Step 8: Retrieval evaluation
    print("\n[Retrieval] Evaluating...")
    results = {}
    for name, sim in [("text", sim_text), ("audio", sim_audio), ("multimodal", sim_multi)]:
        r = retrieval_eval(sim, labels)
        results[name] = r
        print(f"  {name:12s}: P@1={r['P@1']:.3f}  P@3={r['P@3']:.3f}  MAP={r['MAP']:.3f}")

    # Per-language breakdown
    print("\n[Retrieval] Per-language breakdown (text modality):")
    for lang in LANGUAGES:
        idx = [i for i, s in enumerate(segs) if s["lang"] == lang]
        if not idx:
            continue
        sub_sim = sim_text[np.ix_(idx, idx)]
        sub_labels = [labels[i] for i in idx]
        r = retrieval_eval(sub_sim, sub_labels)
        print(f"  {lang:8s}: P@1={r['P@1']:.3f}  P@3={r['P@3']:.3f}  MAP={r['MAP']:.3f}")

    # Cross-language retrieval (use full matrix, query = one language, retrieve = others)
    print("\n[Retrieval] Cross-language retrieval (text):")
    for query_lang in LANGUAGES:
        q_idx = [i for i, s in enumerate(segs) if s["lang"] == query_lang]
        r_idx = [i for i, s in enumerate(segs) if s["lang"] != query_lang]
        cross_hits1, cross_hits3 = [], []
        for qi in q_idx:
            sims = {ri: sim_text[qi, ri] for ri in r_idx}
            ranked = sorted(sims, key=sims.get, reverse=True)
            q_label = labels[qi]
            cross_hits1.append(1 if labels[ranked[0]] == q_label else 0)
            top3_hit = sum(1 for j in ranked[:3] if labels[j] == q_label) / 3
            cross_hits3.append(top3_hit)
        print(f"  Query={query_lang}: P@1={np.mean(cross_hits1):.3f}  P@3={np.mean(cross_hits3):.3f}")

    # Step 9: Save results
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "retrieval": results,
            "label_distribution": dict(Counter(labels)),
            "segments": [{"id": s["id"], "lang": s["lang"], "label": l}
                         for s, l in zip(segs, labels)],
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUT_JSON}")

    # Step 10: Plot
    print("[Plot] Generating t-SNE + heatmaps...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("MOCO2 Pilot: Multimodal Event Embedding Space", fontsize=14, y=1.01)

    plot_tsne(axes[0, 0], text_n,  segs, labels, "t-SNE: Text (XLM-RoBERTa)")
    plot_tsne(axes[0, 1], audio_n, segs, labels, "t-SNE: Audio (MFCC)")
    plot_tsne(axes[0, 2], multi_n, segs, labels, "t-SNE: Text + Audio (Late Fusion)")

    plot_sim_heatmap(axes[1, 0], sim_text,  segs, "Cosine Sim: Text")
    plot_sim_heatmap(axes[1, 1], sim_audio, segs, "Cosine Sim: Audio")
    plot_sim_heatmap(axes[1, 2], sim_multi, segs, "Cosine Sim: Multimodal")

    # Legends
    lang_patches  = [mpatches.Patch(color=c, label=l) for l, c in LANG_COLORS.items()]
    label_handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='gray',
                                markersize=9, label=lb)
                     for lb, m in LABEL_MARKERS.items()]
    fig.legend(handles=lang_patches + label_handles,
               loc="lower center", ncol=7, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    # Retrieval score table
    table_text = "Retrieval Scores\n" + "-" * 36 + "\n"
    table_text += f"{'Modality':12s} {'P@1':>6} {'P@3':>6} {'MAP':>6}\n"
    table_text += "-" * 36 + "\n"
    for name, r in results.items():
        table_text += f"{name:12s} {r['P@1']:6.3f} {r['P@3']:6.3f} {r['MAP']:6.3f}\n"
    fig.text(0.01, -0.07, table_text, fontsize=9, family="monospace",
             va="top", ha="left")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {OUT_PNG}")
    print("\nDone!")

if __name__ == "__main__":
    main()
