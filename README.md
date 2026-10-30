# Mind The Gap (MTG-AD): Dataset and Fine-grained Evaluation for End-to-End Inline Audio Descriptions

This repository contains the code and dataset instructions for the paper:

> [**Mind The Gap: Dataset and Fine-grained Evaluation for End-to-End Inline Audio Descriptions**](https://arxiv.org/abs/2610.39955)
>
> [Subhashini Venugopalan](https://vsubhashini.github.io/), Jenny Tan, Taylor Roper, Jimmy Tobin,
> Anton Kast, Alicia Martin, Sam Sepah, [Amy Pavel](https://amypavel.com/)
>
> NeurIPS 2026 Evaluations & Datasets Track.

[Paper](https://arxiv.org/abs/2610.39955) ·
[Dataset](https://huggingface.co/datasets/google/mind-the-gap-inline-ad) ·
[Code](https://github.com/google/mtg-ad)

--------------------------------------------------------------------------------

## Overview

Audio descriptions (AD) are essential for making visual media accessible to
blind and low-vision (BLV) users. Inline ADs must fit within the **natural
silences** of a video — making this an end-to-end generation and timing
challenge, not just a captioning task.

This repository provides:

1.  **The *Mind the Gap* (MTG-AD) Dataset** — 433 YouTube videos (~36 hours)
    with professionally-annotated audio descriptions, created by paired blind
    and sighted AD professionals.
2.  **AD Generation Code** — scripts and prompt variants to generate inline
    audio descriptions using frontier multimodal LLMs (Gemini, GPT, Claude,
    Qwen).
3.  **AD Evaluation Code (MTG-AD-Eval)** — a suite of fine-grained metrics
    covering Content Accuracy, Temporal Precision, and Qualitative Alignment
    (reference-free Quality-RF by default, plus VideoA11y-Ref when ground
    truth exists).
4.  **TTS Rendering Code** — scripts to convert AD text to speech audio using
    Gemini TTS (default) or Google Cloud TTS at 1.2x (the paper's setting).
5.  **Transcript Generation Code** — scripts to obtain ASR transcripts for
    YouTube videos using Gemini.
6.  **End-to-End Pipeline** — `run_end_to_end.py` runs transcript
    generation, AD generation, visual event extraction, TTS synthesis, and
    evaluation for one video or a batch, caching every step.
7.  **Local MP4 Inputs** — every Gemini step (generation, transcripts, visual
    info, LLM judge) also accepts local `.mp4` files instead of YouTube video
    IDs, with the same output format and metrics (see
    [Evaluating Your Own MP4 Videos](#evaluating-your-own-mp4-videos)).

--------------------------------------------------------------------------------

## Dataset

The dataset is hosted on Hugging Face:
**[https://huggingface.co/datasets/google/mind-the-gap-inline-ad](https://huggingface.co/datasets/google/mind-the-gap-inline-ad)**

It contains 433 videos split into a **validation** set (332 videos) and a
**test** set (101 videos, the evaluation subset used for all multi-model
results in the paper). Two configurations are provided:

*   `annotations` (splits `validation`, `test`): one row per video with the
    ground-truth AD, dense visual information, ASR transcript, and metadata.
*   `model_outputs` (split `test`): one row per (video, model configuration)
    with the generated AD and per-video metric scores. Rows with
    `include_in_scoring = false` (the 25 unstandardized Qwen-3.5-Omni-Flash
    responses) are kept for reference but left out of aggregate metrics.

```python
from datasets import load_dataset

ann = load_dataset("google/mind-the-gap-inline-ad", "annotations")
preds = load_dataset(
    "google/mind-the-gap-inline-ad", "model_outputs", split="test"
)
```

### Raw files for this code

The same data is also available as plain-text files under `raw/`, in the
layout expected by the scripts in this repository. To download the full
dataset repository (including `raw/`):

```bash
huggingface-cli download google/mind-the-gap-inline-ad --repo-type dataset \
    --local-dir mind-the-gap-inline-ad
```

Paths such as `raw/test/...` and `test_video_ids.txt` in this README are
relative to that downloaded `mind-the-gap-inline-ad/` directory.

| File/Directory | Description |
| :--- | :--- |
| `raw/{split}/gt_ad/output_{VIDEO_ID}.txt` | Professionally annotated ground-truth AD |
| `raw/{split}/visual_info/output_{VIDEO_ID}.txt` | Dense time-stamped visual information |
| `raw/{split}/transcript/output_{VIDEO_ID}.txt` | Timestamped ASR transcript (empty file if the video has no speech) |
| `raw/test/model_ads/{CONFIG}/output_{VIDEO_ID}.txt` | Model-generated AD for each model configuration |
| `raw/test/model_metrics/{CONFIG}/output_{VIDEO_ID}.json` | Per-video metric scores for each model configuration |
| `raw/test/scoring_exclusions.json` | (config, video) pairs to leave out of aggregate metrics |
| `video_ids.txt`, `validation_video_ids.txt`, `test_video_ids.txt` | Video ID lists (433 / 332 / 101) |

`{split}` is `validation` or `test`. To evaluate on the test set, pass e.g.
`--input_gt_ad_dir raw/test/gt_ad/`, `--input_visual_info_dir
raw/test/visual_info/`, `--input_transcript_dir raw/test/transcript/` and
`--video_ids_file test_video_ids.txt` to `run_ad_eval.py`.

### Videos

For Gemini models, the scripts can stream public YouTube videos directly by URL
without downloading local video files, or upload local MP4 files (see
[Evaluating Your Own MP4 Videos](#evaluating-your-own-mp4-videos)). For
frame-based baselines (GPT, Claude, Qwen) or local video files, download them
using `yt-dlp`:

```bash
pip install yt-dlp
while IFS= read -r vid_id; do
  yt-dlp "https://www.youtube.com/watch?v=$vid_id" -o "videos/%(id)s.%(ext)s"
done < video_ids.txt
```

--------------------------------------------------------------------------------

## Repository Structure

```
mtg-ad/
├── common/                    # Shared helpers used by every script
│   ├── ad_format.py           #   Parse raw model output into timed AD lines
│   ├── file_utils.py          #   API key, video-ID and duration files, I/O
│   ├── gemini_utils.py        #   Video references, Gemini calls, retries
│   ├── local_video.py         #   Local MP4s: IDs from file names, durations
│   ├── transcripts.py         #   ASR transcript prompt + generation
│   ├── tts.py                 #   Gemini TTS / Cloud TTS, TTS metadata
│   └── usage.py               #   Token usage and cost (USD) accounting
├── generation/                # AD generation
│   ├── generation.py          #   Generation logic (Gemini, GPT, Claude, Qwen)
│   ├── gemini_prompt.py       #   Prompt variants (see --prompt_type)
│   └── README.md
├── evaluation/                # AD evaluation
│   ├── evaluation.py          #   compute_metrics() and the evaluation CLI
│   ├── metrics.py             #   Metric implementations
│   ├── reference_free.py      #   Quality-RF prompts and scoring
│   ├── parsing_lib.py         #   Timestamps, transcripts, intervals
│   ├── visual_info.py         #   Visual-information reference generation
│   └── README.md
├── analysis/                  # Paper tables and figures from per-video metrics
│   ├── aggregate.py           #   Filtering, video subsets, mean ± 95% CI
│   └── paper_tables_and_figures.ipynb
├── metadata/
│   ├── video_durations.json   # Durations of the 433 dataset videos
│   └── gemini_pricing.json    # Gemini prices per 1M tokens (cost estimates)
├── run_ad_generation.py       # CLI -> generation.generation.main()
├── run_ad_eval.py             # CLI -> evaluation.evaluation.main()
├── run_end_to_end.py          # End-to-end pipeline (single video or batch)
├── generate_transcripts.py    # CLI: Gemini ASR transcripts
├── generate_tts.py            # CLI: TTS rendering for temporal metrics
├── sample_video_ids.txt       # Five test-split video IDs used in the examples
├── tests/                     # Offline unit tests (no API calls)
└── requirements.txt           # Python dependencies
```

--------------------------------------------------------------------------------

## Installation

```bash
git clone https://github.com/google/mtg-ad.git
cd mtg-ad
pip install -r requirements.txt
# Optional: Cloud TTS at 1.2x, the paper's setting for temporal metrics.
pip install google-cloud-texttospeech
gcloud auth application-default login
# Optional, for local MP4 inputs: ffprobe (part of FFmpeg) to read video
# durations, e.g. on Debian/Ubuntu (OpenCV is used instead if installed):
sudo apt-get install ffmpeg
```

Requires Python 3.10+. Run all scripts from the repository root so that the
`common`, `generation`, and `evaluation` packages are importable.

Offline unit tests (AD parsing, metrics with a stub LLM judge, TTS and API
error handling, aggregation; no API key or network needed):

```bash
python -m unittest discover -s tests -t .
```

A single Gemini API key covers AD generation, Gemini TTS, and the LLM judge.

Default models:

| Step | Default |
| :--- | :--- |
| AD generation, transcripts, visual info | `gemini-3.8-flash` |
| LLM judge (all judged metrics, incl. VideoA11y) | `gemini-3.8-flash` |
| TTS | `gemini-2.5-flash-preview-tts` (Gemini TTS); Cloud TTS with `--cloud_tts` |

Model arguments of each script (any Gemini model name accepted by the API, e.g.
`gemini-3.5-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`):

| Script | Argument | Used for |
| :--- | :--- | :--- |
| `run_end_to_end.py` | `--generation_model` | Transcript, AD generation, visual info |
| | `--eval_model` | LLM judge |
| | `--model` | Shorthand: sets both of the above |
| | `--tts_model`, `--tts_voice`, `--cloud_tts`, `--speaking_rate` | TTS |
| `run_ad_generation.py` | `--gemini_model` (alias `--generation_model`) | AD generation and missing transcripts |
| | `--genai_model` | Model family: `gemini` (default), `gpt`, `claude`, `qwen` |
| `generate_transcripts.py` | `--model` | Transcripts |
| `generate_tts.py` | `--tts_model`, `--voice_name`, `--cloud_tts`, `--speaking_rate` | TTS |
| `run_ad_eval.py` | `--gemini_model` | LLM judge and missing transcripts / visual info |
| | `--genai_model` | Judge family: `gemini` (default), `gpt`, `claude`, `qwen` |

`run_end_to_end.py`, `run_ad_generation.py`, `generate_tts.py`, and
`run_ad_eval.py` also take `--pricing_file` (see
[Token usage and cost](#token-usage-and-cost)).

Gemini 3.5, 3.7, and 3.8 Flash have been tested for both generation and
evaluation. The defaults may differ from the configurations in the paper: to
reproduce a paper configuration, pass its model and prompt explicitly (e.g.
`--gemini_model gemini-3.5-flash --prompt_type timing_asr`).

> **Video IDs that start with `-`** (e.g. `--T4zFh82Ao`): `run_end_to_end.py
> --video_id --T4zFh82Ao` works as written. Quoting does not help in general,
> because the shell removes the quotes before Python sees them; if a tool
> rejects such an ID, use the `=` form (`--video_id=--T4zFh82Ao`). ID files are
> not affected.

--------------------------------------------------------------------------------

## Quickstart: End-to-End Pipeline

To run the full pipeline (transcript -> AD generation -> visual event extraction
-> TTS synthesis -> evaluation) on a single YouTube video or a list of video
IDs:

```bash
# Single video, generating everything (works for any YouTube video)
python run_end_to_end.py \
  --api_key_path /path/to/gemini_api_key.txt \
  --video_id --T4zFh82Ao \
  --output_dir results/vid_--T4zFh82Ao/

# Dataset test set, reusing the released transcripts / visual info / GT AD,
# with Cloud TTS at 1.2x as in the paper
python run_end_to_end.py \
  --api_key_path /path/to/gemini_api_key.txt \
  --video_ids_file test_video_ids.txt \
  --transcript_dir raw/test/transcript/ \
  --visual_info_dir raw/test/visual_info/ \
  --gt_ad_dir raw/test/gt_ad/ \
  --cloud_tts \
  --output_dir results/per_step/ \
  --layout per_step
```

Every step is cached, so re-running the same command resumes where it stopped.
AD generation uses the `timing_asr` prompt by default; choose another variant
with `--prompt_type` (e.g. `--prompt_type AGENTIC_GEMINI_3P7_UP` for lower token
costs with Gemini 3.7 Flash and later; see [Prompt variants](#prompt-variants)).
Token usage and estimated cost are tracked for every video; see
[Token usage and cost](#token-usage-and-cost). Run
`python run_end_to_end.py --help` for all options. The script prints a failure
summary and exits with status 1 if any video failed or has missing metrics.

### Token usage and cost

Every Gemini API response reports its token usage, and the scripts record it
per video and step (`transcript`, `ad_generation`, `visual_info`, `tts`,
`evaluation`):

*   input tokens, also split by modality (`input_video_tokens`,
    `input_audio_tokens`, `input_text_tokens`), and the part served from
    Gemini's implicit cache (`cached_input_tokens`, billed at the cache price);
*   output tokens and thinking tokens (both billed at the output price);
*   the estimated cost in USD, from the per-model prices in
    [metadata/gemini_pricing.json](metadata/gemini_pricing.json) (Standard
    paid tier, copied from the
    [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing) on
    2026-10-23; the price period in effect on the day of each call is used).

`run_end_to_end.py` prints a per-step table after each video, writes
`usage/output_{VIDEO_ID}.json`, and in batch mode writes
`{output_dir}/usage_summary.csv` (one row per video plus a TOTAL row) and
prints the total. `generate_transcripts.py`, `run_ad_generation.py`,
`generate_tts.py`, and `run_ad_eval.py` write the same per-video JSON under
`{output_dir}/usage/`.

Notes:

*   Usage files add up all runs of a video: steps served from cache on a
    re-run cost nothing, retried calls are counted, and calls that fail with
    an error (no response) are not. Delete the file to start over.
*   Prices change. Update `metadata/gemini_pricing.json` or pass
    `--pricing_file`. Models missing from the file (e.g. `gemini-3.7-flash`,
    which is not on the pricing page) still get full token counts, with
    `cost_usd: null`. Context-cache storage, batch/flex
    discounts, free-tier usage, and Cloud TTS charges are not included.
*   Transcripts or visual info passed in (e.g. the dataset's `raw/test/`
    files) are not generated, so they cost nothing.

--------------------------------------------------------------------------------

## Evaluating Your Own MP4 Videos

For Gemini models, a local `.mp4` file can be used instead of a YouTube video
ID. The file is uploaded to the Gemini Files API once and that upload is reused
for every step (transcript, AD generation, visual info, LLM judge). Outputs have
the same format as for YouTube videos and all metrics are computed the same way.

The video ID used in output file names (`output_{VIDEO_ID}.txt`,
`{VIDEO_ID}_tts_metadata.json`, ...) comes from the file name: characters
other than letters, digits, `_` and `-` become `_`, e.g. `How to build a catio
｜ DIY.mp4` -> `How_to_build_a_catio_DIY`. A file named `{youtube_id}.mp4`
keeps its YouTube ID, so dataset files (transcripts, visual info, GT AD) still
match. The video duration is read with `ffprobe` (or OpenCV, if installed)
when the video is not in `metadata/video_durations.json`.

```bash
# Single MP4 (--video_id is optional and overrides the derived ID)
python run_end_to_end.py \
  --api_key_path /path/to/gemini_api_key.txt \
  --mp4_path "videos/How to build a catio ｜ DIY.mp4" \
  --output_dir results/catio/

# Every MP4 in a folder, with the agentic prompt (Gemini 3.7 Flash and later)
python run_end_to_end.py \
  --api_key_path /path/to/gemini_api_key.txt \
  --video_dir videos/ \
  --generation_model gemini-3.8-flash \
  --prompt_type AGENTIC_GEMINI_3P7_UP \
  --output_dir results/my_videos/ \
  --layout per_step
```

With `--video_ids_file` and `--video_dir` together, each listed ID uses its
local MP4 when there is one and the YouTube URL otherwise. The modular scripts
take the same folder as `--input_video_dir` (`generate_transcripts.py`,
`run_ad_generation.py`, `run_ad_eval.py`); without `--video_ids_file` they
process every MP4 in the folder. `run_ad_generation.py` follows the generation
part of `run_end_to_end.py`: it also accepts `--mp4_path` / `--video_id` for a
single video and `--video_dir` for a folder, uses the same default prompt and
model, and (with Gemini) generates transcripts that the prompt needs but that
are not in `--input_transcript_dir`, caching them in
`{output_dir}/generated_transcript/` (`--no_generate_missing_transcripts` turns
this off):

```bash
python run_ad_generation.py --gemini_model gemini-3.8-flash \
  --prompt_type AGENTIC_GEMINI_3P7_UP \
  --api_key_path /path/to/gemini_api_key.txt \
  --input_video_dir videos/ --output_dir auto_ad/
python generate_tts.py --api_key_path /path/to/gemini_api_key.txt \
  --input_dir auto_ad/ --output_dir tts_gemini/
python run_ad_eval.py --api_key_path /path/to/gemini_api_key.txt \
  --input_video_dir videos/ --output_dir eval_results/ \
  --input_auto_ad_dir auto_ad/ --input_tts_dir tts_gemini/ \
  --input_transcript_dir transcript/ --input_visual_info_dir visual_info/
```

Missing transcripts and visual info are generated from the MP4s by
`run_ad_eval.py`. Reference-based metrics (VideoA11y-Ref, GT statistics) need a
ground-truth AD file and are skipped otherwise.

--------------------------------------------------------------------------------

## Modular Step-by-Step Usage

The examples below use [sample_video_ids.txt](sample_video_ids.txt), five videos
from the test split; swap in the dataset's `test_video_ids.txt` (or any list of
YouTube IDs) for full runs.

### Step 1: Generate Transcripts (if not using included transcripts)

```bash
python generate_transcripts.py \
  --api_key_path /path/to/gemini_api_key.txt \
  --video_ids_file sample_video_ids.txt \
  --output_dir transcript/
```

### Step 2: Generate Audio Descriptions

```bash
python run_ad_generation.py \
  --genai_model gemini \
  --gemini_model gemini-3.8-flash \
  --prompt_type timing_asr \
  --api_key_path /path/to/gemini_api_key.txt \
  --video_ids_file sample_video_ids.txt \
  --input_transcript_dir transcript/ \
  --output_dir auto_ad/
```

See [generation/README.md](generation/README.md) for full argument
documentation and prompt ablation flags (`--prompt_type`).

#### Prompt variants

| `--prompt_type` | Transcript | Notes |
| :--- | :--- | :--- |
| `timing_asr` | yes | Timing-ASR prompt of the paper (default) |
| `timing_asr_json_example` | yes | `timing_asr` with a worked JSON output example |
| `guideline_asr`, `guideline`, `simple` | `guideline_asr` only | Paper ablations |
| `agentic_gemini_3p7_up` | no | For Gemini 3.7 Flash and later (see below) |

Gemini 3.7 Flash and later models are agentic for videos: they watch and listen
to the video themselves and can apply the Timing-ASR instructions (describe only
in the silent gaps between dialogue, with word counts bounded by the gap length)
from the video's audio, without a speech transcript being extracted first and
added to the prompt. `AGENTIC_GEMINI_3P7_UP` takes no transcript, only the
video duration, which is filled in from the video IDs file,
`metadata/video_durations.json`, or the local MP4. Its response uses the same
`{"response": "h:mm:ss.ms-STANDARD ..."}` format as the other prompts, so TTS
and evaluation are unchanged. (`--prompt_type` is case-insensitive.) Use it
only with Gemini 3.7 Flash or later; older models should use the Timing-ASR
prompts with an explicit transcript.

**Which prompt to use.** Among the proposed prompts, Timing-ASR (`timing_asr`)
places ADs best: with an explicit transcript, its ADs overlap dialogue the
least. `AGENTIC_GEMINI_3P7_UP` is a lighter alternative. In our checks with
Gemini 3.7 and 3.8 Flash it scored on par with `timing_asr` on content and
quality (precision, redundancy, on-screen text, timeliness, VideoA11y), and
on uploaded MP4s its overlap with dialogue was also comparable. On the
`sample_video_ids.txt` YouTube videos, however, its ADs overlapped dialogue more
than `timing_asr` ADs when scored against the dataset's transcripts. Those were
small checks (five videos each, and three of the five MP4s had little speech),
so we recommend `timing_asr` when timing matters most, and
`AGENTIC_GEMINI_3P7_UP` when a single generation call without a transcript is
preferred (see the cost note below).

```bash
python run_ad_generation.py \
  --gemini_model gemini-3.8-flash \
  --prompt_type AGENTIC_GEMINI_3P7_UP \
  --api_key_path /path/to/gemini_api_key.txt \
  --video_ids_file sample_video_ids.txt \
  --output_dir auto_ad_agentic/
```

Evaluation still needs a transcript (for the overlap and silence metrics);
`run_end_to_end.py` generates it, and `run_ad_eval.py` generates missing ones.

**Lower token cost.** To spend fewer tokens on AD generation, use
`AGENTIC_GEMINI_3P7_UP`: the Timing-ASR prompts need a transcript first, which
costs a separate Gemini pass over the whole video, while the agentic prompt
needs a single call. On the five `sample_video_ids.txt` videos with Gemini 3.8
Flash (token counts from [Token usage and cost](#token-usage-and-cost), priced
without implicit-cache discounts), generating a transcript plus `timing_asr` ADs
came to about \$0.32, against about \$0.21 for `AGENTIC_GEMINI_3P7_UP` ADs.
The agentic prompt uses more thinking tokens per call, so the saving comes
only from skipping transcription: if a transcript is already available (e.g.
the dataset's `raw/*/transcript/`), `timing_asr` by itself was cheaper (about
\$0.18), and for full end-to-end runs the transcript is generated for
evaluation anyway. Costs vary from run to run with the number of thinking
tokens.

### Step 3: Render AD to Speech with TTS

The temporal metrics (overlap, coverage, silence) use the spoken duration of
each AD segment, so they need TTS audio; evaluation does not estimate durations
from word counts.

```bash
# Default: Gemini TTS (same API key)
python generate_tts.py \
  --api_key_path /path/to/gemini_api_key.txt \
  --input_dir auto_ad/ \
  --output_dir tts_gemini/

# Paper setting: Cloud TTS at 1.2x (optional dependency, see Installation)
python generate_tts.py \
  --cloud_tts \
  --input_dir auto_ad/ \
  --output_dir tts_cloud/
```

`generate_tts.py` saves per-segment WAV files (`{VIDEO_ID}_tts_{hash}.wav`) and
a metadata summary (`{VIDEO_ID}_tts_metadata.json` with `start`,
`duration_tts`, and `end_tts` in seconds). Use a separate output directory per
backend. If any segment fails, the video's metadata is not written and the
script exits with status 1; re-running retries only the missing segments.

> The paper's temporal numbers use Cloud TTS at 1.2x. Gemini TTS speaks more
> slowly, so it gives somewhat higher overlap and coverage values; compare
> systems with the same TTS backend.

### Step 4: Evaluate Audio Descriptions

```bash
python run_ad_eval.py \
  --api_key_path /path/to/gemini_api_key.txt \
  --video_ids_file sample_video_ids.txt \
  --output_dir eval_results/ \
  --input_auto_ad_dir auto_ad/ \
  --input_transcript_dir transcript/ \
  --input_visual_info_dir visual_info/ \
  --input_tts_dir tts_cloud/ \
  --input_gt_ad_dir gt_ad/          # optional: adds VideoA11y-Ref
```

Missing transcripts or visual info are generated on the fly with Gemini.
Video durations default to the bundled
[metadata/video_durations.json](metadata/video_durations.json). See
[evaluation/README.md](evaluation/README.md) for all arguments and the output
format.

### Evaluating existing model outputs (e.g. the released `model_outputs`)

Model responses can be evaluated as they are (raw responses, including any
preamble or JSON wrapper, are parsed). For the released outputs:

```bash
config=gemini_3p5_flash_timing_asr
python generate_tts.py --cloud_tts \
  --input_dir raw/test/model_ads/$config/ --output_dir tts_cloud/$config/
python run_ad_eval.py \
  --api_key_path /path/to/gemini_api_key.txt \
  --video_ids_file test_video_ids.txt \
  --output_dir eval_results/$config/ \
  --input_auto_ad_dir raw/test/model_ads/$config/ \
  --input_transcript_dir raw/test/transcript/ \
  --input_visual_info_dir raw/test/visual_info/ \
  --input_gt_ad_dir raw/test/gt_ad/ \
  --input_tts_dir tts_cloud/$config/
```

LLM-judge scores vary somewhat with the judge model and between runs, so
recomputed judged metrics will not match the released values exactly.

### Tables and figures (aggregation)

[analysis/paper_tables_and_figures.ipynb](analysis/paper_tables_and_figures.ipynb)
turns per-video metric JSONs into the paper's tables (LaTeX, mean ± 95% CI) and
figures. By default it reads the released metrics
(`raw/test/model_metrics/{config}/`) and drops the rows listed in
`raw/test/scoring_exclusions.json`; add your own `run_ad_eval.py` output folders
to `CONFIG_DIRS` to compare systems the same way. On the released metrics it
reproduces the paper's two model comparison tables (initial frontier models and
next-generation omni-modal models; each model is scored on all test videos it
has output for) and both model comparison figures exactly. The paper's prompt
comparison uses 372 validation + test videos, while the release has model
outputs for the 101 test videos only, so recomputing that table from the release
gives slightly different values. The paper's Human AD column comes from running
the ground-truth AD through the same TTS (Cloud TTS, 1.2x) and evaluation; to
reproduce it, run `generate_tts.py --cloud_tts` and `run_ad_eval.py` with
`raw/{split}/gt_ad/` as `--input_auto_ad_dir` and add the resulting metrics
folder to `CONFIG_DIRS`.

--------------------------------------------------------------------------------

## Metrics

| Dimension | Metric | JSON key(s) | Direction |
| :--- | :--- | :--- | :--- |
| Content | Prioritization (G1.1) | `metric_G1.1_precision_avg_ad_in_visual_info` | ↑ |
| Content | No Redundancy (G1.2) | `metric_2_1_does_not_restate` | ↑ |
| Content | On-Screen Text (G1.3) | `metric_1_2_narrates_onscreen_text` (videos with `onscreen_text_present` = 1) | ↑ |
| Temporal | Timeliness (G2) | `metric_g_2_avg_diff_time` (mean), `metric_g_2_max_diff_time` (max) | ↓ |
| Temporal | Audio Overlaps (G7) | `total_overlap` (s), `num_overlap_intervals` (count), `avg_overlap_ad_secs` (s per overlap) | ↓ |
| Temporal | Audio Coverage (G8) | `ad_total_speaking_time_per_silent_second` | ≤ 1 (higher is better up to 1) |
| Temporal | Avg. Silence Duration (G8) | `avg_silence_duration_w_ad_tts` | ↓ |
| Qualitative | Quality-RF (reference-free, default `video_asr`) | `video_asr_descriptive`, `video_asr_objective`, `video_asr_accurate`, `video_asr_clear` (1–5) | ↑ |
| Qualitative | VideoA11y-Ref (needs GT AD) | `videoa11y_descriptive`, `videoa11y_objective`, `videoa11y_accurate`, `videoa11y_clear` (1–5) | ↑ |

Metrics that cannot be computed for a video (e.g. no TTS output, no parseable
AD, failed judge call) are left out of its JSON rather than set to 0, so they
do not skew means; see [evaluation/README.md](evaluation/README.md).

### Metric names and mapping to the paper

`run_ad_eval.py` writes one JSON per video to
`{output_dir}/metrics/output_{VIDEO_ID}.json`. The released per-video scores in
the Hugging Face dataset use the same keys: as raw JSON files under
`raw/test/model_metrics/` and as columns of the `model_outputs` config. Column
names in that config replace `.` with `_`. The table below maps paper metrics
to these keys.

| Paper metric | Guideline | JSON key (code output / raw files) | HF `model_outputs` column | Direction |
| :--- | :--- | :--- | :--- | :--- |
| Prioritization | G1.1 | `metric_G1.1_precision_avg_ad_in_visual_info` | `metric_G1_1_precision_avg_ad_in_visual_info` | ↑ |
| No Redundancy | G1.2 | `metric_2_1_does_not_restate` | `metric_2_1_does_not_restate` | ↑ |
| On-Screen Text | G1.3 | `metric_1_2_narrates_onscreen_text` (aggregated over videos with `onscreen_text_present` = 1) | `metric_1_2_narrates_onscreen_text` | ↑ |
| Content Timeliness (s) | G2 | `metric_g_2_avg_diff_time` (max: `metric_g_2_max_diff_time`, code output only) | `metric_g_2_avg_diff_time` | ↓ |
| Overlaps w. TTS (s) | G7 | `total_overlap` (exact; `tts_ad_transcript_overlap_seconds` is the same quantity sampled every 0.1 s). Count K_overlap: `num_overlap_intervals` (code output only) | `total_overlap` | ↓ |
| Audio Coverage (ratio) | G8 | `ad_total_speaking_time_per_silent_second` | `ad_total_speaking_time_per_silent_second` | ≤ 1 (higher is better up to 1; values > 1 mean AD time exceeds available silence) |
| Average Silence Duration (s) | G8 | `avg_silence_duration_w_ad_tts` | `avg_silence_duration_w_ad_tts` | ↓ |
| VideoA11y-Ref Descriptive / Objective / Accurate / Clear | – | `videoa11y_descriptive`, `videoa11y_objective`, `videoa11y_accurate`, `videoa11y_clear` (1–5) | same | ↑ |
| Quality-RF Descriptive / Objective / Accurate / Clear | – | `{variant}_descriptive`, `_objective`, `_accurate`, `_clear` (1–5), default variant `video_asr` (code output only) | – | ↑ |

Other keys in the JSON give supporting statistics and diagnostics:

| JSON key | Meaning |
| :--- | :--- |
| `tts_rate` | TTS speaking-rate multiplier used when rendering (1.2 in the paper); present in the released files |
| `metric_G1.1_precision_avg_ad_in_visual_info_std` | Legacy field in the released metric files only (always 0; Prioritization uses a single LLM-judge call) |
| `onscreen_text_present` | 1 if the visual info lists on-screen text, else 0 |
| `metric_g_2_matched_visual_info_ratio` | Fraction of AD segments matched to a visual event when computing Timeliness |
| `metric_g_2_diffs_over_ten` | Number of matched AD segments more than 10 s from their visual event |
| `avg_overlap_ad_secs` | `total_overlap` divided by `num_overlap_intervals` (s) |
| `tts_ad_self_overlap_seconds` | Time AD segments overlap each other after TTS rendering (s) |
| `tts_ad_transcript_overlap_seconds` | Time AD is spoken during dialogue after TTS rendering (s) |
| `total_silence_duration_w_ad_tts` | Total silence remaining after inserting TTS AD (s) |
| `ad_total_time`, `ad_total_time_tts` | Total AD speaking time: sum of AD segment durations / after merging overlapping AD segments (s) |
| `ad_total_words`, `ad_total_words_per_silent_second` | AD word count and words per second of available silence |
| `gt_total_time`, `gt_total_words`, `gt_total_words_per_silent_second`, `gt_total_speaking_time_per_silent_second`, `total_overlap_gt` | Same statistics for the ground-truth AD (when provided) |
| `total_transcript_time`, `total_silence_time`, `dialog_density` | Speech time, silence time, and fraction of the video covered by speech |
| `speech_density_w_ad_tts_dialog` | Fraction of the video covered by dialogue or AD (overlap counted once) |
| `video_duration_seconds` | Video duration (s) |
| `status` | `ok`, or `no_parseable_ad` (no metric values written) — code output only |
| `skipped_metrics`, `failed_judge_calls` | Metric groups skipped for missing inputs / LLM-judge calls that failed — code output only |
| `{variant}_summary`, `{variant}_rationale` | Quality-RF judge's video summary and rationale |

Paper numbers are means over videos after the aggregation in
[analysis/aggregate.py](analysis/aggregate.py) (scoring exclusions, On-Screen
Text only on videos with on-screen text, timing values longer than the video
dropped), so plain column means differ from the reported values.

For a full description of the evaluation framework and guidelines G1–G10, see
the paper.

--------------------------------------------------------------------------------

## Citation

If you use this dataset or code, please cite:

```bibtex
@inproceedings{venugopalan2026mindthegap,
  title     = {Mind The Gap ({MTG-AD}): Dataset and Fine-grained Evaluation
               for End-to-End Inline Audio Descriptions},
  author    = {Venugopalan, Subhashini and Tan, Jenny and Roper, Taylor and
               Tobin, Jimmy and Kast, Anton and Martin, Alicia and
               Sepah, Sam and Pavel, Amy},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Evaluations and Datasets Track},
  year      = {2026}
}
```

--------------------------------------------------------------------------------

## License

The dataset annotations are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The code is released under the [Apache 2.0 License](LICENSE).

Copyright 2026 Google LLC

The underlying YouTube videos are subject to their respective creators' licenses
and YouTube's Terms of Service.

--------------------------------------------------------------------------------

## Notes

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).
