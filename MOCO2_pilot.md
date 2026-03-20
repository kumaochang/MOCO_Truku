# MoCo v2: Event-Driven Multimodal Corpus Query

## 從 Symbolic Filtering 到 Geometric Retrieval：多模態語料庫查詢的典範轉移

---

## 0. 核心論點

語料庫查詢的發展歷程：

```
CQP / Treebank Query         →  Dense Text Retrieval      →  ???
(symbolic, structural)           (geometric, vector-based)
                                                              ↑
"找所有 V-not-V 問句"           "找語意上像這句話的句子"        目標要做的：
                                                              
                                                           Multimodal Event Retrieval
                                                           (event-driven, cross-modal)
                                                              
                                                           "找所有跟這段『安慰』
                                                            互動相似的多模態事件"
```

**gap：** 文本語料庫已經從 symbolic 走向 geometric，但多模態語料庫的查詢仍停留在 metadata filtering 或 keyword search。沒有人提出過以**多模態事件**為查詢單位、以**跨模態向量相似度**為檢索機制的語料探索範式。

**為什麼重要：** 語用現象（勸告、安慰、拒絕、開玩笑）本質上是多模態的——語言內容可能含蓄，但語調、停頓、表情、手勢共同構成完整的交際行為。這些現象幾乎無法用傳統的形式化查詢條件描述，但人類可以直覺辨識「這兩個場景在做類似的事」。如果可以在向量空間中捕捉這種相似性，就打開了一扇全新的研究窗口。

---

## 1. 研究問題

**RQ1 (表示學習)：** 能否將影片中的多模態互動事件（語音 + 影像 + 文字）嵌入到統一的向量空間中，使得語用功能相近的事件在空間中彼此靠近，即使它們來自不同語言、不同文化？

**RQ2 (檢索效能)：** 以一段互動片段作為 query（query-by-example），系統檢索語用功能相似事件的準確率如何？不同模態組合（audio-only, video-only, text-only, multimodal）的檢索效能差異為何？

**RQ3 (語言學應用)：** 透過 event-driven 檢索，能否揭示不同語言/文化（如華語 vs. 太魯閣語 vs. 阿美語）在完成相似交際行為時，各模態的貢獻模式（modality profile）有何系統性差異？

---

## 2. 概念架構

### 2.1 多模態事件（Multimodal Event）的定義

一個多模態事件是一個時間界定的互動片段，包含以下層次：

```
Multimodal Event (e.g., "安慰", 15 秒)
│
├── Linguistic tier
│     ├── transcript (族語/華語)
│     ├── speech act label (comfort / advise / joke / ...)
│     └── code-switching markers
│
├── Prosodic tier
│     ├── pitch contour
│     ├── pause structure
│     ├── speech rate
│     └── voice quality
│
├── Visual tier
│     ├── facial expression
│     ├── gesture type + timing
│     ├── gaze direction
│     ├── body posture
│     └── proxemics (interpersonal distance)
│
└── Contextual tier
      ├── setting (indoor/outdoor, formal/informal)
      ├── participant roles (anchor/interviewee/elder/...)
      └── cultural context tags
```

### 2.2 查詢範式對照

| 面向 | 傳統語料庫查詢 | Event-Driven Multimodal Query |
|------|-------------|------|
| 查詢單位 | 詞、詞組、句法樹 | 多模態事件片段 |
| 查詢語言 | CQP, TGrep, XPath | Query-by-example (影片/語音片段) |
| 匹配機制 | Symbolic pattern matching | Vector similarity (cosine / hyperbolic distance) |
| 回傳結果 | Concordance lines (KWIC) | Ranked list of similar multimodal events |
| 適合研究 | 詞彙、語法、搭配 | 語用、互動、手勢、韻律、跨文化比較 |
| 使用者需要 | 精確的形式化條件 | 一個範例片段 + 直覺 |

### 2.3 系統架構

```
                    ┌─────────────────────────────┐
                    │    Multimodal Event Corpus   │
                    │  (segmented, multi-tier)     │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Modality-Specific Encoders │
                    │                              │
                    │  Audio: Whisper encoder       │
                    │         + prosody features    │
                    │                              │
                    │  Visual: SigLIP / CLIP       │
                    │          + pose estimation   │
                    │          + facial AU         │
                    │                              │
                    │  Text: multilingual LLM      │
                    │        sentence embedding    │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Cross-Modal Fusion Layer   │
                    │                              │
                    │  Option A: Late fusion       │
                    │    concat → projection       │
                    │                              │
                    │  Option B: Contrastive       │
                    │    learning (CLIP-style)     │
                    │    across modalities         │
                    │                              │
                    │  Option C: VLM embedding     │
                    │    直接用 Video LLM 的        │
                    │    internal representation   │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Unified Event Embedding    │
                    │   (Euclidean or Hyperbolic)  │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐   ┌──────────┐   ┌────────────┐
        │ Query-by-│   │ Cluster  │   │ Cross-     │
        │ Example  │   │ Analysis │   │ Cultural   │
        │ Retrieval│   │ (自動發現 │   │ Comparison │
        │          │   │  事件類型)│   │            │
        └──────────┘   └──────────┘   └────────────┘
```

---

## 3. Pilot Study 設計

### 3.1 範圍

| 面向 | 選擇 | 理由 |
|------|------|------|
| 語言 | 太魯閣語、阿美語、華語 | 2 族語 + 華語作對照；TITV 新聞中三者都有 |
| 語用事件類型 | 4 類（見下方）| 在新聞中可觀察到、跨文化可比較 |
| 事件數量 | 每語言 × 每類型 ~15–20 個 = ~200 個事件 | 足以訓練 retrieval 模型、做統計比較 |
| 資料來源 | TITV 族語新聞（含主播播報、訪談、現場報導）| 公開可取得、多語混用 |

### 3.2 四類語用事件

選擇標準：(a) 在 TITV 新聞中自然出現、(b) 跨語言可比較、(c) 多模態特徵明顯

| 事件類型 | 典型場景 | 多模態特徵預期 |
|---------|---------|--------------|
| **關懷 / 慰問 (Comforting)** | 災後訪問、部落探視報導 | 柔和語調、前傾身體、點頭、放慢語速 |
| **慶祝 / 歡迎 (Celebrating)** | 豐年祭報導、頒獎典禮 | 高亢語調、笑容、歡呼、rhythmic clapping |
| **解說 / 教導 (Explaining)** | 文化傳承報導、耆老訪談 | 指示性手勢、穩定語速、eye contact |
| **呼籲 / 勸告 (Urging)** | 政策訴求、環境議題報導 | 加重語氣、emphatic gesture、repeated patterns |


---

## 4. 實驗設計

### Experiment 1: Multimodal Event Embedding（表示學習）

**目標：** 比較不同方法將多模態事件映射到向量空間的效果。

**方法 A: Late Fusion Baseline**

各模態分別編碼，然後拼接 + 投影


**方法 B: Contrastive Learning（類 CLIP）**
- 正樣本對：同一事件的不同模態（audio ↔ visual ↔ text）
- 負樣本對：不同事件的跨模態配對
- 學習一個共享嵌入空間，使同一事件的各模態向量互相靠近

**方法 C: Video LLM as Encoder**
- 直接用 Qwen2.5-VL 或 InternVL3 處理完整影片片段
- 擷取最後一層的 [CLS] 或 pooled representation 作為 event embedding
- 最簡單，但可能最有效（利用了大量預訓練知識）

**方法 D: Hyperbolic Event Embedding**
- 將方法 A 或 C 的向量投射到 Poincaré ball
- 假說：語用事件具有層級結構（e.g., "Celebrating" 包含 "cheering" 和 "welcoming"），雙曲空間更適合
- 連結你在 hyperbolic embeddings 的研究背景

**評估：** 用事件類型標籤做 retrieval evaluation
- 給定一個 query 事件，檢索 top-k 最相似事件
- 計算 Precision@k, MAP, 是否正確檢索到同類型事件
- 分析：跨語言檢索是否可行（用太魯閣語事件 query，能否找到阿美語/華語中的類似事件）

### Experiment 2: Modality Ablation（模態貢獻分析）

| 條件 | 使用的模態 | 研究問題 |
|------|----------|---------|
| Text-only | 轉寫/翻譯文字 | 純語言內容能檢索到多少？ |
| Audio-only | 語音特徵（含韻律） | 韻律模式是否跨語言通用？ |
| Visual-only | 影像幀 + 骨架/表情 | 身體語言的跨文化可遷移性？ |
| Audio + Visual | 語音 + 影像（無文字） | 非語言模態的組合效果 |
| Full multimodal | 全部模態 | 完整模型的上界 |

**預期的有趣發現：**
- 「關懷/慰問」可能 visual 模態最重要（前傾、觸碰、柔和表情）
- 「解說/教導」可能 audio 模態最重要（穩定語調、指示性停頓）
- 「慶祝」可能 visual + audio 都很重要（歡呼聲 + 笑容 + 動作）
- 跨語言時，非語言模態的檢索效果可能優於語言模態

### Experiment 3: Cross-Cultural Pragmatic Comparison（跨文化語用比較）

可能是語言學貢獻最大的實驗。

**方法：**
1. 在 embedding 空間中，以事件類型為單位計算各語言的 cluster centroid
2. 比較華語 vs. 太魯閣語 vs. 阿美語在表達同一語用功能時的：
   - **Modality profile：** 各模態向量到 centroid 的距離分布（哪個模態貢獻最大？）
   - **Intra-class variance：** 同一語言內同類事件的向量分散程度（表達策略多樣性）
   - **Cross-lingual distance：** 不同語言表達同一語用功能的向量距離（文化差異量化）

**視覺化：**
```
           Comforting 事件在 embedding 空間中的分布

    ●  太魯閣語                    ○  華語
    ●  太魯閣語         △          ○  華語
      ●               △ △           ○
    ●                △              ○  ○
                    △

    （如果兩個語言的 cluster 分離 → 文化差異大）
    （如果重疊 → 該語用功能的表達方式跨文化相似）
```

**定性分析（Case Study）：**
- 從檢索結果中選取有代表性的 query-result pairs
- 比較太魯閣語和華語在「慰問」場景中的具體差異：
  - 太魯閣語是否更依賴 prosodic cues（語調變化）？
  - 華語是否更依賴 lexical cues（安慰性詞語）？
  - 身體語言的差異（如距離、觸碰的文化差異）？
