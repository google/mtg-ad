# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates inline audio descriptions with multimodal LLMs.

Library functions: fill_transcript() and generate_ad() (Gemini, client passed
explicitly) are reused by run_end_to_end.py. main() is the CLI behind
run_ad_generation.py and also supports GPT, Claude, and Qwen baselines.
"""

import argparse
import base64
import json
import math
import os
import sys

from common import ad_format
from common import file_utils
from common import gemini_utils
from common import local_video
from common import transcripts as transcripts_lib
from common import usage
from evaluation import parsing_lib
from generation import gemini_prompt
from google import genai
from google.genai import types
import numpy as np
import tqdm

# pylint: disable=g-import-not-at-top
try:
  import anthropic
except ImportError:
  anthropic = None

try:
  import cv2
except ImportError:
  cv2 = None

try:
  import dashscope
except ImportError:
  dashscope = None

try:
  import openai
except ImportError:
  openai = None
# pylint: enable=g-import-not-at-top

DEFAULT_GEMINI_MODEL = gemini_utils.DEFAULT_GEMINI_MODEL
TRANSCRIPT_PLACEHOLDER = "ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE"
DURATION_PLACEHOLDER = "VIDEO_DURATION_GOES_HERE_IF_AVAILABLE"

# Module state for the CLI's multi-provider inference functions below.
GEMINI_MODEL_NAME = DEFAULT_GEMINI_MODEL
client = None


def fill_transcript(prompt: str, transcript: str) -> str:
  """Substitutes the transcript into a prompt's placeholder (if any)."""
  return prompt.replace(TRANSCRIPT_PLACEHOLDER, transcript)


def fill_prompt(
    prompt: str,
    transcript: str = "",
    video_duration: float | str | None = None,
) -> str:
  """Substitutes the transcript and video duration placeholders (if any).

  Args:
    prompt: A prompt from gemini_prompt.PROMPTS.
    transcript: Timestamped transcript; "" for silent videos.
    video_duration: Duration in seconds or as a timestamp string (e.g. "4:47").
      Only used by prompts with a duration placeholder (agentic_gemini_3p7_up);
      "unknown" is substituted if it is not given.

  Returns:
    The filled-in prompt.
  """
  prompt = fill_transcript(prompt, transcript)
  if DURATION_PLACEHOLDER in prompt:
    if isinstance(video_duration, (int, float)):
      seconds = float(video_duration)
    else:
      seconds = parsing_lib.parse_duration_to_seconds(video_duration)
    prompt = prompt.replace(
        DURATION_PLACEHOLDER,
        f"{seconds:.1f}" if seconds and seconds > 0 else "unknown",
    )
  return prompt


def generate_ad(
    gemini_client: genai.Client,
    video_part: types.Part,
    prompt: str,
    transcript: str = "",
    model: str = DEFAULT_GEMINI_MODEL,
    video_duration: float | str | None = None,
) -> str | None:
  """Generates raw AD output for a video with Gemini.

  Args:
    gemini_client: Gemini client.
    video_part: Video reference from gemini_utils.make_video_part().
    prompt: A prompt from gemini_prompt.PROMPTS.
    transcript: Timestamped transcript; "" for silent videos.
    model: Gemini model name.
    video_duration: Optional duration (seconds or timestamp string), used by
      prompts that state the video length (agentic_gemini_3p7_up).

  Returns:
    The raw model response (normalize it with metrics.normalize_ad_output()),
    or None on failure.
  """
  return gemini_utils.generate_with_video(
      gemini_client,
      model,
      video_part,
      fill_prompt(prompt, transcript, video_duration),
  )


def _load_transcript(transcript_folder, video_id):
  """Loads transcript from output_{id}.txt or transcript_{id}.txt."""
  if not transcript_folder:
    return ""
  for filename in (f"output_{video_id}.txt", f"transcript_{video_id}.txt"):
    path = os.path.join(transcript_folder, filename)
    if os.path.exists(path):
      with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
  return None


def qwen_inference(
    prompt_, video_id, qwen_api_key, video_folder, transcript_folder
):
  """Generates audio descriptions for a video using Qwen-VL."""
  if cv2 is None:
    raise ImportError("Please install 'opencv-python' to process local videos.")
  dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
  video_path = local_video.find_local_video(video_folder, video_id)
  if not video_path:
    print(f"Error: no video file for {video_id} in {video_folder!r}")
    return None

  video = cv2.VideoCapture(video_path)
  base64_frames = 0
  while video.isOpened():
    success, frame = video.read()
    if not success:
      break
    _, _ = cv2.imencode(".jpg", frame)
    base64_frames += 1
  video.release()
  if base64_frames == 0:
    print(f"Error: No frames found in video file {video_path}")
    return None

  transcript = _load_transcript(transcript_folder, video_id)
  if transcript is None:
    print(
        f"Error: transcript file not found for {video_id} in"
        f" {transcript_folder}"
    )
    return None
  prompt_ = prompt_.replace("ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE", transcript)

  messages = [
      {"role": "system", "content": [{"text": "You are a helpful assistant."}]},
      {
          "role": "user",
          # The fps parameter controls the number of frames to extract from the
          # video (one frame every 1/fps seconds).
          "content": [{"video": video_path, "fps": 2}, {"text": prompt_}],
      },
  ]
  response = dashscope.MultiModalConversation.call(
      api_key=qwen_api_key, model="qwen-vl-max-latest", messages=messages
  )

  if response["status_code"] == 200:
    return response["output"]["choices"][0]["message"].content[0]["text"]
  else:
    print(
        f"Error_{video_id} ({response['status_code']}): {response['message']}"
    )
    return None


def claude_inference(
    prompt_, video_id, claude_api_key, video_folder, transcript_folder
):
  """Generates audio descriptions for a video using Claude."""
  del claude_api_key
  if cv2 is None:
    raise ImportError("Please install 'opencv-python' to process local videos.")

  def get_equal_interval_samples_numpy(video_list, num_samples=100):
    """Selects num_samples elements from video_list at equal intervals."""
    n = len(video_list)
    if num_samples <= 0 or n == 0:
      return []
    if n <= num_samples:
      return video_list[:]
    float_indices = np.linspace(start=0, stop=n - 1, num=num_samples)
    int_indices = np.round(float_indices).astype(int)
    return [video_list[i] for i in int_indices]

  video_path = local_video.find_local_video(video_folder, video_id)
  if not video_path:
    print(f"Error: no video file for {video_id} in {video_folder!r}")
    return None

  video = cv2.VideoCapture(video_path)
  fps = round(video.get(cv2.CAP_PROP_FPS))
  frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  sample_rate = 50
  target_frame = get_equal_interval_samples_numpy(
      range(0, frame_count), sample_rate
  )

  base64_frames = []
  count = 0
  while video.isOpened():
    success, frame = video.read()
    if not success:
      break
    if count not in target_frame:
      count += 1
      continue
    count += 1
    height, width, _ = frame.shape
    adjust_ratio = math.sqrt(1092 * 1092 / (height * width))
    frame = cv2.resize(
        frame, (int(width * adjust_ratio), int(height * adjust_ratio))
    )
    _, buffer = cv2.imencode(".jpg", frame)
    base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

  video.release()
  if not base64_frames:
    return None

  interval = frame_count / sample_rate / fps
  prompt_ = prompt_.replace("XXX", str(interval))

  transcript = _load_transcript(transcript_folder, video_id)
  if transcript is None:
    print(
        f"Error: transcript file not found for {video_id} in"
        f" {transcript_folder}"
    )
    return None
  prompt_ = prompt_.replace("ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE", transcript)

  with client.messages.stream(
      max_tokens=64000,
      messages=[
          {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": prompt_,
                  },
                  *[
                      {
                          "type": "image",
                          "source": {
                              "type": "base64",
                              "media_type": "image/jpeg",
                              "data": frame,
                          },
                      }
                      for frame in base64_frames
                  ],
              ],
          },
      ],
      model="claude-sonnet-4-20250514",
  ) as stream:
    final_message = stream.get_final_message()
    if final_message is None:
      return None
    return json.loads(final_message.to_json())["content"][0]["text"]


def gpt_inference(
    prompt_, video_id, gpt_api_key, video_folder, transcript_folder
):
  """Generates audio descriptions for a video using OpenAI GPT."""
  del gpt_api_key
  if cv2 is None:
    raise ImportError("Please install 'opencv-python' to process local videos.")
  video_path = local_video.find_local_video(video_folder, video_id)
  if not video_path:
    print(f"Error: no video file for {video_id} in {video_folder!r}")
    return None

  video = cv2.VideoCapture(video_path)
  base64_frames = []
  fps = round(video.get(cv2.CAP_PROP_FPS))
  count = 0
  while video.isOpened():
    success, frame = video.read()
    if count % fps != 0:
      count += 1
      continue
    count += 1
    if not success:
      break
    height, width, _ = frame.shape
    adjust_ratio = math.sqrt(640 * 360 / (height * width))
    frame = cv2.resize(
        frame, (int(width * adjust_ratio), int(height * adjust_ratio))
    )
    _, buffer = cv2.imencode(".jpg", frame)
    base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

  video.release()
  if not base64_frames:
    return None

  transcript = _load_transcript(transcript_folder, video_id)
  if transcript is None:
    print(
        f"Error: transcript file not found for {video_id} in"
        f" {transcript_folder}"
    )
    return None
  prompt_ = prompt_.replace("ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE", transcript)

  try:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_},
                *[
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{frame}",
                    }
                    for frame in base64_frames
                ],
            ],
        }],
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(e)
    return None
  return response.output_text


def gemini_inference(
    prompt_,
    video_id,
    gemini_api_key,
    video_folder,
    transcript_folder,
    video_duration=None,
    mp4_path=None,
    generated_transcript_dir=None,
):
  """Generates audio descriptions for a video using Gemini.

  Same steps as the generation part of run_end_to_end.py:
    * The video is the local MP4 `mp4_path`, else the MP4 for `video_id` in
      `video_folder` (see local_video.find_local_video()), else the YouTube
      URL. It is uploaded once and reused for the transcript and AD calls.
    * For prompts with a transcript (timing_asr, ...), the transcript is read
      from `transcript_folder`; if it is missing there and
      `generated_transcript_dir` is set, it is generated with Gemini (the
      transcript step of run_end_to_end.py) and cached in that directory.
    * For prompts that state the video length (agentic_gemini_3p7_up), the
      duration of a local MP4 is measured when `video_duration` is not given.

  Args:
    prompt_: Prompt from gemini_prompt.PROMPTS.
    video_id: Video ID (YouTube ID, or the ID derived from an MP4 file name).
    gemini_api_key: Unused (the module-level client is used).
    video_folder: Optional folder of local MP4 files.
    transcript_folder: Optional folder of transcripts (output_{id}.txt).
    video_duration: Optional duration (seconds or timestamp string).
    mp4_path: Optional local MP4 for this video.
    generated_transcript_dir: Where to cache generated transcripts; None
      disables transcript generation (a missing transcript is an error).

  Returns:
    The raw model response, or None on failure.
  """
  del gemini_api_key
  mp4_path = mp4_path or local_video.find_local_video(video_folder, video_id)
  if video_folder and not mp4_path:
    print(f"  No local MP4 for {video_id}; streaming the YouTube URL.")
  if not video_duration and mp4_path and DURATION_PLACEHOLDER in prompt_:
    video_duration = local_video.probe_duration_seconds(mp4_path)
  video_part = None

  transcript = ""
  if TRANSCRIPT_PLACEHOLDER in prompt_:
    transcript = (
        _load_transcript(transcript_folder, video_id)
        if transcript_folder
        else None
    )
    if transcript is None and generated_transcript_dir:
      transcript = _load_transcript(generated_transcript_dir, video_id)
    if transcript is None and generated_transcript_dir:
      video_part = gemini_utils.make_video_part(client, video_id, mp4_path)
      print(
          f"  Generating transcript for {video_id} with {GEMINI_MODEL_NAME}..."
      )
      with usage.step("transcript"):
        transcript = transcripts_lib.generate_transcript(
            client, video_part, model=GEMINI_MODEL_NAME
        )
      if transcript is None:
        print(f"Error: transcript generation failed for {video_id}.")
        return None
      file_utils.write_text(
          os.path.join(generated_transcript_dir, f"output_{video_id}.txt"),
          transcript,
      )
    if transcript is None:
      print(
          f"Error: transcript file not found for {video_id} in"
          f" {transcript_folder!r}"
      )
      return None

  if video_part is None:
    video_part = gemini_utils.make_video_part(client, video_id, mp4_path)
  return generate_ad(
      client,
      video_part,
      prompt_,
      transcript,
      model=GEMINI_MODEL_NAME,
      video_duration=video_duration,
  )


def main():
  """Parses CLI arguments and generates AD scripts for all listed videos."""
  parser = argparse.ArgumentParser(
      description="A script to call GenAI APIs to generate audio AD."
  )
  parser.add_argument(
      "--genai_model",
      type=str,
      default="gemini",
      help="genAI model family. Choose from gemini, qwen, gpt, claude.",
  )
  parser.add_argument(
      "--gemini_model",
      "--generation_model",
      dest="gemini_model",
      type=str,
      default=DEFAULT_GEMINI_MODEL,
      help=(
          "Specific Gemini model name when --genai_model=gemini"
          " (--generation_model as in run_end_to_end.py); also used to"
          f" generate missing transcripts. Default: {DEFAULT_GEMINI_MODEL}."
      ),
  )
  parser.add_argument(
      "--prompt_type",
      type=str.lower,
      default="timing_asr",
      choices=list(gemini_prompt.PROMPTS.keys()),
      help=(
          "Prompt variant to use: timing_asr (default),"
          " timing_asr_json_example, guideline_asr, guideline, simple, or"
          " agentic_gemini_3p7_up (Gemini 3.7 Flash and later; no transcript"
          " needed). Case-insensitive."
      ),
  )
  parser.add_argument(
      "--api_key_path", type=str, default="", help="API key path."
  )
  parser.add_argument(
      "--video_id",
      type=str,
      default="",
      help=(
          "Single YouTube video ID (IDs starting with '-' are supported). With"
          " --mp4_path, sets the ID used in output file names (default:"
          " derived from the file name)."
      ),
  )
  parser.add_argument(
      "--mp4_path",
      type=str,
      default="",
      help=(
          "Single local MP4 file (Gemini only), uploaded to the Gemini Files"
          " API instead of using a YouTube URL."
      ),
  )
  parser.add_argument(
      "--video_ids_file",
      type=str,
      default="",
      help=(
          "Video IDs file: one '<video_id> [<duration>]' per line. May be"
          " omitted with --input_video_dir to process every MP4 in it."
      ),
  )
  parser.add_argument(
      "--input_video_dir",
      "--video_dir",
      dest="input_video_dir",
      type=str,
      default="",
      help=(
          "Folder of local MP4 files (--video_dir as in run_end_to_end.py). A"
          " video's file is {video_id}.mp4 or any MP4 whose name maps to the"
          " ID (see common/local_video.py). Without --video_ids_file, every"
          " MP4 in the folder is processed."
      ),
  )
  parser.add_argument(
      "--video_duration",
      type=str,
      default="",
      help=(
          "Duration of a single video (seconds or [HH:]MM:SS[.mmm]), for"
          " prompts that state the video length. Local MP4s are measured"
          " when not given."
      ),
  )
  parser.add_argument(
      "--video_durations_file",
      type=str,
      default=file_utils.DEFAULT_VIDEO_DURATIONS_FILE,
      help=(
          "JSON map {video_id: duration}, used by prompts that state the"
          " video length (agentic_gemini_3p7_up). Local MP4s not listed are"
          " measured with ffprobe/OpenCV."
      ),
  )
  parser.add_argument(
      "--input_transcript_dir",
      type=str,
      default="",
      help=(
          "Folder of transcripts (output_{VIDEO_ID}.txt; empty file = no"
          " speech) for prompts that use one (timing_asr, ...). With Gemini,"
          " missing transcripts are generated (as in run_end_to_end.py) and"
          " cached in {output_dir}/generated_transcript/ unless"
          " --no_generate_missing_transcripts."
      ),
  )
  parser.add_argument(
      "--no_generate_missing_transcripts",
      action="store_true",
      help="Fail instead of generating missing transcripts with Gemini.",
  )
  parser.add_argument(
      "--pricing_file",
      type=str,
      default=usage.DEFAULT_PRICING_FILE,
      help=(
          "JSON of Gemini prices per 1M tokens for the cost estimates in"
          " {output_dir}/usage/. Default: metadata/gemini_pricing.json."
      ),
  )
  parser.add_argument("--output_dir", type=str, default="", help="output_dir")
  args = parser.parse_args(file_utils.fix_dash_prefixed_values(sys.argv[1:]))
  usage.set_pricing_file(args.pricing_file)

  single = bool(args.video_id or args.mp4_path)
  if single and args.video_ids_file:
    parser.error(
        "--video_id / --mp4_path cannot be used with --video_ids_file."
    )
  if args.mp4_path and not os.path.isfile(args.mp4_path):
    parser.error(f"--mp4_path not found: {args.mp4_path}")
  if args.mp4_path and args.genai_model != "gemini":
    parser.error("--mp4_path is supported for --genai_model=gemini only.")
  if args.video_duration and not single:
    parser.error("--video_duration can only be used for a single video.")
  mp4_paths = {}
  if single:
    vid = args.video_id or local_video.video_id_from_path(args.mp4_path)
    entries = [(vid, args.video_duration or None)]
    if args.mp4_path:
      mp4_paths[vid] = args.mp4_path
  elif args.video_ids_file:
    entries = file_utils.read_video_ids_file(args.video_ids_file)
  elif args.input_video_dir:
    entries = [
        (vid, None)
        for vid, _ in local_video.list_local_videos(args.input_video_dir)
    ]
    print(f"Found {len(entries)} MP4 files in {args.input_video_dir}.")
  else:
    parser.error(
        "pass --video_id and/or --mp4_path (single video), or"
        " --video_ids_file and/or --input_video_dir."
    )
  id_list = [vid for vid, _ in entries]
  durations = file_utils.load_video_durations(args.video_durations_file)
  durations.update({vid: d for vid, d in entries if d})
  api_key = file_utils.get_api_key(args.api_key_path)

  inference_functions = {
      "qwen": qwen_inference,
      "gpt": gpt_inference,
      "claude": claude_inference,
      "gemini": gemini_inference,
  }

  global client, GEMINI_MODEL_NAME
  GEMINI_MODEL_NAME = args.gemini_model
  prompt = gemini_prompt.PROMPTS[args.prompt_type]

  match args.genai_model:
    case "qwen":
      if openai is None or dashscope is None:
        raise ImportError(
            "Please install 'openai' and 'dashscope' to use --genai_model=qwen."
        )
      client = openai.OpenAI(
          api_key=api_key,
          base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
      )
    case "claude":
      if anthropic is None:
        raise ImportError(
            "Please install 'anthropic' to use --genai_model=claude."
        )
      client = anthropic.Anthropic(api_key=api_key)
    case "gpt":
      if openai is None:
        raise ImportError("Please install 'openai' to use --genai_model=gpt.")
      client = openai.OpenAI(api_key=api_key)
    case "gemini":
      client = genai.Client(api_key=api_key)
    case _:
      print(f"unsupported genai model {args.genai_model}.")
      return

  if args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)

  failures = {}
  for vid_id in tqdm.tqdm(id_list):
    output_path = os.path.join(args.output_dir, f"output_{vid_id}.txt")
    if os.path.exists(output_path) and os.stat(output_path).st_size != 0:
      continue
    inference_func = inference_functions.get(args.genai_model)
    extra = {}
    if args.genai_model == "gemini":
      extra["video_duration"] = durations.get(vid_id)
      extra["mp4_path"] = mp4_paths.get(vid_id)
      if not args.no_generate_missing_transcripts:
        extra["generated_transcript_dir"] = os.path.join(
            args.output_dir, "generated_transcript"
        )
    with usage.recording() as recorder, usage.step("ad_generation"):
      try:
        response = inference_func(
            prompt,
            vid_id,
            api_key,
            args.input_video_dir,
            args.input_transcript_dir,
            **extra,
        )
      except gemini_utils.FatalApiError as e:
        sys.exit(f"FATAL: {e}")
    if args.genai_model == "gemini":
      # Gemini token usage and estimated cost, summed over runs.
      usage.save(
          os.path.join(args.output_dir, "usage", f"output_{vid_id}.json"),
          recorder,
          extra={
              "video_id": vid_id,
              "prompt_type": args.prompt_type,
              "generation_model": args.gemini_model,
          },
      )

    if not response:
      print(f"ERROR: AD generation failed for {vid_id}.")
      failures[vid_id] = "no response"
      continue
    if not ad_format.parse_ad_lines(response):
      # Keep the raw response for inspection, but flag it.
      print(f"ERROR: no timed AD lines could be parsed for {vid_id}.")
      failures[vid_id] = "response has no parseable AD lines"

    try:
      with open(output_path, "w", encoding="utf-8") as f:
        f.write(response)
    except OSError as e:
      print(f"ERROR writing {output_path}: {e}")
      failures[vid_id] = str(e)

  file_utils.report_failures(failures, len(id_list), "AD generation")


if __name__ == "__main__":
  main()
