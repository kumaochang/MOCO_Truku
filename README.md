# MOCO Truku (Truku MMOC Search)

A lightweight, client-side search UI for the Truku Multimodal Corpus. The page loads `corpus.json`, renders search results, and plays segments from a single shared video element with time-based seeking.

## Features

- Search by Truku, Chinese translation, phones, allophones (IPA), or gestures
- Inline segment playback with precise time ranges
- CSV export of filtered results
- Range-capable local server for smooth seeking

## Dataset Description

The corpus is a multimodal dataset aligned at the utterance level. Each utterance includes:

- Truku text and Chinese translation
- Time-aligned phone and allophone sequences
- Gesture annotations for different speakers
- Start/end timestamps used for segment playback

## corpus.json Structure

Top-level JSON object keys:

- `utterances`: Array of utterance entries
- `gestureInventory`: List of all gesture labels
- `phoneInventory`: List of phone symbols
- `allophoneInventory`: List of allophone symbols (IPA and variants)

Each utterance object has this structure:

```
{
	"start": 0,
	"end": 3.25,
	"truku": "...",
	"chinese": "...",
	"phones": [ { "s": 0.12, "e": 0.24, "l": "a" } ],
	"allophones": [ { "s": 0.12, "e": 0.24, "l": "[a]" } ],
	"gestures_sp1": [ { "start": 0.5, "end": 1.2, "labels": ["GAZE_DOWN"] } ],
	"gestures_sp2": [ { "start": 0.5, "end": 1.2, "labels": ["GAZE_DOWN"] } ]
}
```

Field notes:

- `start`/`end`: Utterance time range in seconds.
- `phones`/`allophones`: Time-aligned segment arrays with `s` (start), `e` (end), and `l` (label).
- `gestures_sp1`/`gestures_sp2`: Gesture intervals with `labels` containing one or more tags.

