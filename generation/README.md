# AD Generation

This tool generates inline Audio Descriptions (AD) for videos using Multimodal
GenAI models. It processes videos and their corresponding ASR transcripts to
produce time-stamped audio descriptions tailored to fit within natural dialogue
pauses.

## Features

-   **Multi-Model Support**:
    -   Google Gemini (`gemini`) — supports both local `.mp4` files and direct
        YouTube URL streaming.
    -   OpenAI GPT (`gpt`)
    -   Anthropic Claude (`claude`)
    -   Alibaba Qwen (`qwen`)
-   **Prompt Ablations**: Includes all four prompt configurations from the paper
    in `gemini_prompt.py` (`timing_asr`, `guideline_asr`, `guideline`,
    `simple`), plus `timing_asr_json_example`, the `timing_asr` instructions
    with a worked JSON output example. `timing_asr` is the default.
-   **Agentic prompt for Gemini 3.7 Flash and later**
    (`agentic_gemini_3p7_up`, constant `AGENTIC_GEMINI_3P7_UP`): these models
    are agentic for videos and follow the Timing-ASR instructions directly from
    the video's audio, so the prompt takes no transcript (only the video
    duration). This works well on MP4 videos, and may not always work with a
    YouTube ID as input. We recommend the Timing-ASR prompt (`timing_asr`),
    which works more reliably across videos and models. The agentic prompt's
    output uses the same `{"response": ...}` format as the other prompts.
-   **Local MP4 files**: `--mp4_path` for a single file, or `--input_video_dir`
    (alias `--video_dir`) for a folder: Gemini uploads each video's local MP4
    (`{VIDEO_ID}.mp4`, or any MP4 whose file name maps to the ID, e.g.
    `How to build a catio ｜ DIY.mp4` -> `How_to_build_a_catio_DIY`). Without
    `--video_ids_file`, every MP4 in the folder is processed.
-   **Same as the generation part of `run_end_to_end.py`**: same default prompt
    (`timing_asr`) and model (`gemini-3.8-flash`); with Gemini, transcripts
    that the prompt needs but that are missing from `--input_transcript_dir`
    are generated (cached in `{output_dir}/generated_transcript/`); each video
    is uploaded once and reused for the transcript and AD calls; token usage and
    estimated cost are written to `{output_dir}/usage/output_{VIDEO_ID}.json`
    (steps `transcript` and `ad_generation`).
-   **Batch Processing**: Processes multiple videos defined in a list and skips
    already-generated outputs. Videos whose generation fails, or whose
    response has no timed AD lines, are listed at the end and the script exits
    with status 1. An unknown model name (`404 NOT_FOUND` from the Gemini API)
    or an invalid API key stops the run immediately.

The GPT, Claude, and Qwen baselines need optional packages:
`pip install anthropic dashscope openai opencv-python`. They are the scripts
used to produce the paper's baseline outputs; the current release was tested
end to end with Gemini (3.5, 3.7, and 3.8 Flash) only.

## Usage

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

### Arguments

| Argument                 | Description                   | Default            |
| :----------------------- | :---------------------------- | :----------------- |
| `--genai_model` | The model family to use (`gemini`, `qwen`, `gpt`, `claude`). | `gemini` |
| `--gemini_model` | Specific Gemini model name (alias `--generation_model`); used when `--genai_model=gemini`. | `gemini-3.8-flash` |
| `--prompt_type` | Prompt configuration (`timing_asr`, `timing_asr_json_example`, `guideline_asr`, `guideline`, `simple`, `agentic_gemini_3p7_up`); case-insensitive. | `timing_asr` |
| `--api_key_path` | Path to a file containing the API key for the selected model. | `""` |
| `--video_ids_file` | Path to a text file containing Video IDs to process (one per line, optional duration column). May be omitted with `--input_video_dir`. | `""` |
| `--input_video_dir` (alias `--video_dir`) | Directory containing local MP4 files (`{VIDEO_ID}.mp4` or a name mapping to the ID). Optional for `gemini` (falls back to streaming from YouTube URL if omitted or if local file is absent). | `""` |
| `--video_durations_file` | JSON `{video_id: duration}` for prompts that state the video length (`agentic_gemini_3p7_up`). Local MP4s not listed are measured with ffprobe. | bundled `metadata/video_durations.json` |
| `--input_transcript_dir` | Directory containing transcript files (`output_{VIDEO_ID}.txt` or `transcript_{VIDEO_ID}.txt`). With Gemini, transcripts the prompt needs but that are missing are generated and cached in `{output_dir}/generated_transcript/`. | `""` |
| `--no_generate_missing_transcripts` | Fail instead of generating missing transcripts. | off |
| `--video_id` | Single YouTube video ID; with `--mp4_path`, the ID used in output file names. | `""` |
| `--mp4_path` | Single local MP4 (Gemini). The ID defaults to the file name (`My video.mp4` -> `My_video`). | `""` |
| `--video_duration` | Duration of a single video (for `agentic_gemini_3p7_up`). | measured for MP4s |
| `--pricing_file` | Gemini prices for the token cost estimates written to `{output_dir}/usage/`. | bundled `metadata/gemini_pricing.json` |
| `--output_dir` | Directory where generated AD files (`output_{VIDEO_ID}.txt`) will be saved. | `""` |
