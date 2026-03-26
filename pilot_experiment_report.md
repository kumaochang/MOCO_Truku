# MOCO2 Pilot Experiment Report

**日期：** 2026-03-26
**目的：** 驗證 MOCO2_pilot.md 的 RQ1 與 RQ2 的可行性

---

## 1. 實驗設計

### 1.1 資料

30 支 TITV 族語新聞影片，分三個語言：

| 語言 | 數量 | 日期 |
|------|------|------|
| 阿美語 (Amis) | 10 | 2026-03-02 |
| 排灣語 (Paiwan) | 10 | 2026-03-07 |
| 太魯閣語 (Truku) | 10 | 2026-03-05 |

每支影片為完整新聞節目（約 5 分鐘）。分析單元為 document-level（整集新聞），尚未切割至 utterance-level 多模態事件。

### 1.2 模態與特徵

| 模態 | 來源 | 特徵 | 維度 |
|------|------|------|------|
| **Text** | OCR 中文字幕（`ocr_corrected/*.ocr.json`） | XLM-RoBERTa mean pooling | 768 |
| **Audio** | WAV 檔（`wav/*.wav`） | librosa MFCC (20 維) + delta + RMS + ZCR + spectral centroid，取 global mean+std | 86 |
| **Multimodal** | Text + Audio | L2-normalize 後 concatenate，再 normalize | 854 → normalized |

Visual 模態（`emit2/*.pkl`）因需要 `allosaurus` 套件才能讀取，未納入分析。

### 1.3 語用標籤

語用功能標籤由 **Claude Haiku（claude-haiku-4-5-20251001）** 擔任標注者，對每支影片進行分類，採用 LLM-as-annotator 策略。標注流程為：從每支影片均勻抽取 5 張 keyframe，結合 OCR 中文字幕摘要，送入 Claude API，回傳語用類別與判斷依據。全部 30 支均獲得 `confidence: high`。

標籤類別定義如下：

| 類型 | 定義 |
|------|------|
| **Explaining** | 解說、教導、文化傳承、分享知識 |
| **Urging** | 呼籲、勸告、政策訴求 |
| **Celebrating** | 慶祝、歡迎、典禮 |
| **Reporting** | 一般新聞播報，無明顯互動語用功能 |

標籤分佈：

| 類型 | 數量 | 語言分佈 |
|------|------|---------|
| Explaining | 16 | Amis×8, Paiwan×5, Truku×3 |
| Urging | 8 | Paiwan×3, Truku×5 |
| Celebrating | 4 | Amis×2, Paiwan×1, Truku×1 |
| Reporting | 2 | Paiwan×1, Truku×1 |

### 1.4 評估方法

**Query-by-example retrieval**：每個 segment 輪流作為 query，用 cosine similarity 排序其餘 29 個，計算命中同類型的比例。

- **P@1**：top-1 結果是否與 query 同類型
- **P@3**：top-3 結果中同類型的比例
- **MAP**：Mean Average Precision（考慮整體排序品質）
- **Random baseline P@1**：≈ 0.37（依標籤分佈估算：0.53² + 0.27² + 0.13² + 0.07²）

---

## 2. 結果

### 2.1 RQ2：不同模態的檢索效能

| Modality | P@1 | P@3 | MAP | vs. random |
|----------|-----|-----|-----|------------|
| **Text** (XLM-RoBERTa) | **0.667** | **0.567** | **0.533** | **+0.30** |
| Audio (MFCC) | 0.500 | 0.300 | 0.391 | +0.13 |
| Text + Audio (late fusion) | 0.400 | 0.333 | 0.407 | +0.03 |

**Per-language breakdown（text modality）：**

| 語言 | P@1 | P@3 | MAP |
|------|-----|-----|-----|
| Amis | **0.800** | **0.800** | **0.769** |
| Paiwan | 0.500 | 0.467 | 0.511 |
| Truku | 0.400 | 0.367 | 0.443 |

### 2.2 RQ1：跨語言語用事件的向量相似度（text modality）

用某語言的 segment 作為 query，只從其他兩個語言的 segments 中檢索：

| Query 語言 | P@1 (cross-lang) | P@3 (cross-lang) |
|-----------|-----------------|-----------------|
| Amis → Paiwan/Truku | 0.500 | 0.400 |
| **Paiwan → Amis/Truku** | **0.600** | **0.567** |
| Truku → Amis/Paiwan | 0.400 | 0.333 |

---

## 3. 解讀

### 3.1 Text 顯著優於 Audio；Multimodal 不如純 Text

```
Text (0.667) > Audio (0.500) > Multimodal (0.400) > Random (≈0.37)
```

Text P@1 = 0.667，顯著高於 random baseline（≈0.37），說明 XLM-RoBERTa 對 OCR 中文字幕的嵌入，能在語意空間中有效區分語用類別，效果不只是機率問題。

Audio P@1 = 0.500，接近 random，原因在於 MFCC global stats 捕捉的是整支影片的平均聲學特性，與語用功能的關聯過於間接。

Multimodal（late fusion）反而比純 Text 差：把 noise 較多的 audio 向量直接拼接，稀釋了 text 的語意信號。Late fusion 需要更細緻的加權設計，而非直接 concatenate。

### 3.2 Amis 語言內部一致性最高（P@1 = 0.800）

Amis 有 8/10 被標為 Explaining，且主題高度集中（族語傳承、文化工藝課程），在語意空間中形成緊密的 cluster。此結果部分反映**主題同質性**（topical homogeneity）的效應，不能直接解讀為語用功能的跨語言遷移能力。

### 3.3 跨語言檢索：Paiwan 表現最佳（P@1 = 0.600）

Paiwan 以 query 查詢其他語言，P@1 = 0.600，高於 random。排灣語的 Urging 影片（政策訴求、醫療推廣）在中文字幕的語意空間中，與其他語言的同類片段距離較近，初步支持 RQ1 的核心論點——語用功能相近的事件在向量空間中確實有跨語言聚集的傾向。但因 Paiwan Urging 僅 3 支，結論需保守。

### 3.4 Celebrating 與 Reporting 樣本數不足

Celebrating（4 支）和 Reporting（2 支）的 P@k 統計意義有限，特別是 Reporting 只有 2 支，無法做有意義的 retrieval 評估。四類語用功能的樣本需更均衡才能得出穩健結論。

---

## 4. 後續改進方向

| 問題 | 解法 |
|------|------|
| Audio 特徵太粗 | 用 Whisper encoder hidden states 取代 MFCC global stats |
| 缺少 Visual 模態 | 改用 torchvision 從 MP4 抽 keyframe + CLIP/SigLIP embedding |
| 分析單元太大 | 把影片切成 utterance-level events（利用 aligned_trees TextGrid） |
| Late fusion 效果差 | 嘗試加權融合，或訓練一個小 projection head |
| 標籤需驗證 | 對 Claude 標籤進行人工 spot-check |

---

## 5. 限制聲明

分析單元是完整的新聞節目（document-level），而非 MOCO2_pilot.md 設想的 utterance-level 多模態事件。因此，結果只能說明**整集新聞的語用定性是否可檢索**，不能直接回答「片段內的互動行為是否在向量空間中聚集」。

Visual 模態尚未納入。Audio 特徵僅為 MFCC 統計量，未使用 Whisper 語音語意特徵。語用標籤由 Claude VLM 自動生成（silver labels），尚未經人工系統性驗證。

在上述條件下，目前結果可支持的 claim 為：

> 在 document-level 的 pilot 規模下，XLM-RoBERTa 對 OCR 中文字幕的嵌入，能以顯著高於隨機基準（P@1 = 0.667 vs. baseline ≈ 0.37）的準確率，檢索到語用功能相近的新聞片段，且跨語言檢索初步可行（Paiwan cross-lang P@1 = 0.600）。
