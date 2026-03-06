# FormosanBench Pilot Study

**目標：** 驗證 benchmark 設計的可行性  
**時程：** 2026 年 3 月中 → 6 月中（約 12 週）  
---

## 1. 範圍界定

### 語言選擇：3 族語

| 族語 | 選擇理由 | TITV 新聞資料量 |
|------|---------|---------------|
| **太魯閣語 (Truku)** | 你最熟悉，有 MOCO_Truku 經驗，可做最深度標注 | 充足（每週固定時段）|
| **阿美語 (Amis)** | 使用人口最多的族語，資料量最豐富 | 充足 |
| **排灣語 (Paiwan)** | 與前兩者屬不同語群，增加類型學多樣性 | 充足 |

三個語言分屬不同分群（泰雅語群 / 東台灣南島 / 排灣語群），能初步涵蓋 Formosan 語言的多樣性。

### 資料量：每語 50 個片段，共 150 個

- 片段長度：15–45 秒（新聞單則或段落）
- 來源：TITV YouTube 族語新聞播放清單
- 選片策略：
  - 30 個主播播報片段（語音清晰、單一語言為主）
  - 10 個戶外採訪/現場報導片段（環境音、code-switching 多）
  - 10 個文化活動報導片段（祭典、傳統技藝、部落活動等）

### 任務：聚焦 3 個核心任務（從原本 5 個精簡）

| 任務 | Pilot 版本 | 每片段標注項目 |
|------|-----------|-------------|
| **T1: 語種辨識** | 多模態 LID：audio-only vs. video+audio | 語種標籤 + 信心度 |
| **T2: 文化接地 VQA** | 每片段 2 個 QA pair | 問題 + 答案 + 文化面向標籤 |
| **T3: 跨語言描述** | 華語 caption（1–3 句） | 編輯後的 gold caption |

Task 4 (CS 偵測) 和 Task 5 (對齊探測) 留待完整版再加入。

---

## 2. 技術 Pipeline 規格

### 2.1 影片蒐集與前處理

```bash
# Step 1: 下載 TITV 族語新聞
yt-dlp --playlist-items 1-50 \
  -f "bestvideo[height<=720]+bestaudio" \
  --write-sub --sub-langs zh-Hant \
  -o "raw/%(playlist_index)s_%(title)s.%(ext)s" \
  "https://www.youtube.com/playlist?list=PLtE4cgfvTnt5KsZHhyqnxibXuQE4Wi25z"

# Step 2: 切分為 15–45 秒片段
# 使用 PySceneDetect 偵測場景切換點，再人工微調
scenedetect -i video.mp4 detect-content -t 27 split-video

# Step 3: 擷取音訊
ffmpeg -i segment.mp4 -vn -acodec pcm_s16le -ar 16000 segment.wav

# Step 4: 擷取關鍵幀（每 2 秒一幀，供 VLM 使用）
ffmpeg -i segment.mp4 -vf "fps=0.5" frames/frame_%04d.jpg
```

### 2.2 自動化標注 Pipeline

**語種辨識（T1 候選標籤生成）**

```python
# 方案 A: Whisper language detection（對族語不準，但可偵測「非華語」）
import whisper
model = whisper.load_model("large-v3")
result = model.transcribe(audio, task="transcribe")
detected_lang = result["language"]  # 主要用來過濾華語段落

# 方案 B: Hsieh et al. (2024) 的 XLSR-53 classifier

# 方案 C: VLM 視覺輔助
# 將影片幀送入 VLM，問：「根據畫面中的文字、字幕、場景，
# 這段新聞最可能使用哪個台灣原住民族語言？」
```

**文化 VQA（T2 候選問答生成）**

```python
import openai, anthropic, google.generativeai

PROMPT = """你是台灣原住民族文化與語言學專家。
請觀看這段原住民族電視台（TITV）的新聞片段截圖，
並根據畫面中可見的文化元素，生成 3 組問答對。

要求：
1. 問題必須需要台灣原住民族文化知識才能回答（不能只靠一般視覺常識）
2. 答案要具體、可驗證
3. 為每題標注文化面向：服飾/祭儀/飲食/建築/地景/語言/社會組織/其他
4. 標注難度：easy / medium / hard

格式：
Q1: [問題]
A1: [答案]
文化面向: [標籤]
難度: [easy/medium/hard]
依據: [畫面中的哪些視覺線索支持此答案]
"""

# 對每個片段，送給 4 個模型
models = ["gpt-4o", "gemini-2.5-pro", "qwen2.5-vl-72b", "claude-sonnet-4-5"]
# 收集所有輸出，計算 cross-model agreement
```

**跨語言描述（T3 候選 caption 生成）**

```python
CAPTION_PROMPT = """請為這段台灣原住民族電視台的新聞片段撰寫華語描述（1-3句）。
描述應同時涵蓋：
1. 視覺內容（人物、場景、動作）
2. 文化脈絡（如果畫面涉及特定族群的文化活動，請指出）
3. 推測的語言使用情況（主播使用族語/華語/混用）

同時提供的參考資訊：
- OCR 擷取的華語字幕：{ocr_subtitle}
- Whisper 轉寫（可能不準確）：{whisper_output}
"""

# OCR 字幕作為 anchor，VLM 描述作為補充
# 你的工作：合併、修正、確認
```

### 2.3 Cross-Model Agreement 自動篩選

```python
def compute_agreement(responses: list[dict]) -> str:
    """
    比較多個 VLM 的回答，產出信心等級
    - HIGH: 3/4 模型答案語意一致 → auto-accept（你抽查 10%）
    - MEDIUM: 2/4 一致 → 你需要看一下，通常 30 秒可決定
    - LOW: 全部不同 → 你需要仔細審核，可能需要 2-3 分鐘
    """
    # 用 sentence embedding cosine similarity 比較答案
    # 或讓另一個 LLM 做 judge（判斷答案是否等價）
    ...
```

預期分布：HIGH ~50%, MEDIUM ~30%, LOW ~20%

你的時間因此主要花在 LOW agreement 的那 20% 上。

---

## 3. 標注工作流程

### Streamlit 介面

用 Streamlit 寫一個簡單的 review 頁面：

```
┌──────────────────────────────────────────────┐
│  [影片播放器]  片段 #037 / 150               │
│  族語: 太魯閣語 (auto-detected, HIGH conf.)  │
│  OCR 字幕: 「部落族人今天舉辦感恩祭典...」   │
├──────────────────────────────────────────────┤
│  ■ T1 語種辨識                               │
│  Auto: Truku (conf: 0.94)                    │
│  [✓ Accept] [✎ Correct: ___]                 │
├──────────────────────────────────────────────┤
│  ■ T2 文化 VQA（候選，按 agreement 排序）    │
│                                               │
│  Q1 [HIGH] 畫面中人們穿戴的紅色頭帶          │
│     屬於哪個族群的傳統服飾？                  │
│  A1: 太魯閣族 (GPT-4o ✓ Gemini ✓ Qwen ✓)   │
│  [✓] [✎] [✗]                                │
│                                               │
│  Q2 [MED] 這段影片記錄的是什麼祭典？         │
│  A2a: 感恩祭 (GPT-4o, Gemini)               │
│  A2b: 豐年祭 (Qwen, Claude)                 │
│  [選 A2a] [選 A2b] [✎ 自己寫: ___]          │
│                                               │
│  Q3 [LOW] ...                                │
│  [需要你仔細看影片後回答]                     │
├──────────────────────────────────────────────┤
│  ■ T3 跨語言描述                             │
│  OCR 字幕: 「部落族人今天舉辦感恩祭典...」   │
│  VLM draft: 「太魯閣族部落族人身著傳統        │
│  服飾，在戶外廣場舉行祭典活動。主播以         │
│  太魯閣語進行現場報導。」                     │
│  [✓ Accept] [✎ Edit ↓]                      │
│  [可編輯文字框]                               │
├──────────────────────────────────────────────┤
│  [← 上一題]  進度: 37/150  [下一題 →]        │
└──────────────────────────────────────────────┘
```

### 標注時間估算（150 片段 × 3 任務）

| 信心等級 | 預估佔比 | 片段數 | 每片段時間 | 小計 |
|---------|---------|-------|----------|------|
| HIGH（T1 auto-accept + T2 抽查 + T3 快速確認）| 50% | 75 | ~1 分鐘 | 75 分鐘 |
| MEDIUM（T1 ok + T2 需選擇 + T3 需小改）| 30% | 45 | ~3 分鐘 | 135 分鐘 |
| LOW（T2 需自己寫 + T3 需重寫）| 20% | 30 | ~6 分鐘 | 180 分鐘 |
| **合計** | | **150** | | **~6.5 小時** |

加上前後處理、品質回顧、少量重標，估計**總標注時間 8–10 小時**。可以分 3–4 個 session 完成。

---

## 4. 實驗設計

### 4.1 待評測模型

| 模型 | 類型 | 存取方式 |
|------|------|---------|
| GPT-4o | 商用 API | OpenAI API |
| Gemini 2.5 Pro | 商用 API | Google AI Studio |
| Claude Sonnet 4.5 | 商用 API | Anthropic API |
| Qwen2.5-VL-72B | 開源 | API 或本地（國網 GPU）|
| InternVL3-78B | 開源 | 本地部署 |
| LLaVA-OneVision-72B | 開源 | 本地部署 |

也可以加入小模型（Qwen2.5-VL-7B、InternVL3-8B）看 scaling 效果。

### 4.2 實驗條件

**T1 語種辨識實驗矩陣**

| 條件 | 輸入 | 測試重點 |
|------|------|---------|
| Audio-only | 音訊波形 | Whisper / XLSR baseline |
| Visual-only | 影片幀（無音訊） | VLM 能否從視覺猜語種？ |
| Subtitle-only | OCR 華語字幕 | 字幕是否洩漏族語資訊？ |
| Audio + Visual | 完整影片 | 多模態增益 |
| Audio + Visual + Subtitle | 完整影片 + OCR 字幕 | 最佳條件 |

**T2 文化 VQA 實驗矩陣**

| 條件 | Prompt 策略 | 測試重點 |
|------|-----------|---------|
| Zero-shot | 直接問 | Baseline |
| + Cultural context | 告知「這是台灣原住民族新聞」 | 文化提示效果 |
| + Language hint | 額外告知族語名稱 | 語言知識是否幫助文化推理 |
| + Few-shot | 提供 3 個同族語的範例 QA | In-context learning |

**T3 跨語言描述**

- 比較各模型的 caption 品質
- 評估指標：BERTScore, 人工評分（文化資訊覆蓋率 1–5 分）
- 對比條件：有/無 OCR 字幕輸入

### 4.3 分析維度

核心要回答的問題（也是論文的 findings）：

1. **語言維度**：三個族語之間的表現差異有多大？（假設阿美語因為使用人口最多，可能網路上有更多資料，VLM 表現會稍好）
2. **文化面向維度**：服飾 vs. 祭儀 vs. 飲食 vs. 地景，哪些面向 VLM 最弱？
3. **模態貢獻**：視覺模態對語種辨識的增益有多少？
4. **模型比較**：商用 vs. 開源、大 vs. 小模型的差距
5. **失敗模式分析**（最有趣的部分）：VLM 犯了什麼類型的錯誤？
   - 把原住民文化誤認為東南亞/太平洋島嶼文化？
   - 把族語誤認為其他語言？
   - 使用刻板印象式的描述？

---

## 5. 論文結構（Short Paper / Workshop Paper, 4–8 頁）

```
Title: FormosanBench: Probing Vision-Language Models' Understanding 
       of Taiwan's Indigenous Languages and Cultures

Abstract (150 words)

1. Introduction (1 page)
   - VLM 快速發展，但文化多樣性評測嚴重不足
   - 台灣南島語的獨特地位（南島語族發源地）
   - TITV 作為資料源的優勢
   - 本文貢獻：首個南島語多模態 benchmark (pilot)

2. Related Work (0.5–1 page)
   - Cultural VQA benchmarks (CulturalVQA, VideoNorms, VULCA-Bench)
   - Low-resource VLM evaluation (VLURes, Maya)
   - Formosan language resources (Hsieh et al. 2024, NTU Corpus)

3. FormosanBench Dataset (1.5 pages)
   - 3.1 Data source: TITV news
   - 3.2 Task design (T1, T2, T3)
   - 3.3 Human-AI collaborative annotation framework
        - Multi-VLM candidate generation
        - Cross-model agreement scoring
        - Expert verification protocol
   - 3.4 Dataset statistics

4. Experiments (1.5–2 pages)
   - 4.1 Models evaluated
   - 4.2 Results on T1 (language identification)
   - 4.3 Results on T2 (cultural VQA)
   - 4.4 Results on T3 (cross-lingual captioning)
   - 4.5 Error analysis and failure modes

5. Discussion (0.5–1 page)
   - 文化理解的系統性偏差
   - 視覺模態的貢獻與限制
   - 對 VLM 訓練資料多樣性的啟示
   - Limitations & 完整版展望

6. Conclusion

Appendix: 標注 guidelines, 完整 prompt templates, 
         per-language breakdown, 更多 error examples
```

---

## 6. 週次時程（12 週）

### Week 1–2：基礎建設
- [ ] 下載 TITV 影片（3 族語 × ~20 集 = ~60 小時原始影片）
- [ ] 場景切分 + 片段篩選（目標：每語 50 個，共 150 個）
- [ ] 架設前處理 pipeline（ffmpeg, PySceneDetect, PaddleOCR）
- [ ] 文獻閱讀：CulturalVQA, VideoNorms, wav2gloss

### Week 3–4：自動化候選標注
- [ ] 對 150 片段跑 Whisper（語種偵測 + 轉寫）
- [ ] 對 150 片段跑 OCR（華語字幕擷取）
- [ ] 對 150 片段送 4 個 VLM 生成候選 QA 和 caption
- [ ] 計算 cross-model agreement，排序片段優先順序
- [ ] 寫 Streamlit review 介面

### Week 5–6：人工審核標注
- [ ] 完成 150 片段 × 3 任務的標注（~8–10 小時）
- [ ] 品質回顧：抽樣 20% 重新檢查
- [ ] 最終 dataset 整理、格式化（JSON + 影片片段）

### Week 7–8：VLM 評測實驗
- [ ] T1 實驗：6 模型 × 5 條件 = 30 runs
- [ ] T2 實驗：6 模型 × 4 prompt 策略 = 24 runs
- [ ] T3 實驗：6 模型 × 2 條件 = 12 runs
- [ ] 自動評估指標計算
- [ ] 結果初步分析

### Week 9–10：深度分析 + 撰寫
- [ ] Error analysis：分類失敗模式
- [ ] 視覺化：confusion matrix, per-language radar chart
- [ ] 撰寫論文初稿

### Week 11–12：定稿 + 計畫書
- [ ] 論文修改、與指導教授討論
- [ ] 整合 pilot 結果到博士論文計畫書
- [ ] **6 月中提交計畫書**
- [ ] 論文投稿準備

---

## 7. 預算估算

| 項目 | 細項 | 預估費用 |
|------|------|---------|
| VLM API（標注用）| 150 片段 × 4 模型 × ~$0.05/片段 | ~$30 |
| VLM API（實驗用）| 150 片段 × 6 模型 × 5 條件 × ~$0.05 | ~$225 |
| Claude API | 同上 | 含在 Anthropic 額度內 |
| GPU（開源模型）| 國網中心或校內 cluster | 免費（學術帳號）|
| **合計** | | **~$250–300 USD** |

---

## 8. Pilot → 完整版的擴充路徑

| 面向 | Pilot (本次) | 完整版 (口試前) |
|------|-------------|----------------|
| 語言數 | 3 族語 | 8–10 族語 |
| 片段數 | 150 | 500–800 |
| 任務數 | 3 (T1–T3) | 5 (加 CS 偵測 + 對齊探測) |
| 標注深度 | VQA + caption | 加入 IGT tier（VideoLingua 方向）|
| 論文 | Short/workshop paper | ACL/EMNLP 長文 |
| 資料釋出 | 片段 metadata + 標注 | 完整 benchmark + leaderboard |

Pilot 的結果會直接回答兩個關鍵問題：
1. 這個 benchmark 能否有效區分不同 VLM 的文化理解能力？（如果所有模型表現都差不多，benchmark 區辨度不夠）
2. Human-AI collaborative annotation 流程是否夠高效？（為完整版的擴充提供可行性證據）
