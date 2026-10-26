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

"""Aggregates per-video metric JSONs into the paper's tables and figures.

Works on the released per-video metrics (raw/test/model_metrics/{config}/) and
on the output of run_ad_eval.py ({output_dir}/metrics/). The aggregation is
the one used for the paper:

  * Per video (load_metrics_dir):
      - On-Screen Text is ignored for videos without on-screen text
        (onscreen_text_present == 0).
      - Timeliness, Overlap and Avg. Silence values larger than the video
        duration are ignored (treated as parse failures).
      - Unreadable / empty JSON files are skipped, as are (config, video) pairs
        listed in a scoring-exclusions file (raw/test/scoring_exclusions.json).
  * Per table (summarize): by default only videos that have a metrics file
    for *every* configuration in the table are used (common_videos=True).
    With common_videos=False, each configuration uses all of its own videos
    (the paper's model comparisons, where every model is scored on all test
    videos it has output for). Each cell is the mean over videos with a value,
    and ± is the 95% CI half-width, 1.96 * std(ddof=1)/sqrt(n).
  * Figures: same video selection; seaborn bars with bootstrapped 95% CIs. In
  the
    model-comparison figures, Timeliness values more than 3 std from the
    pooled mean are dropped before plotting (as for the paper figures); tables
    keep
    them.
"""

import glob
import json
import os
from typing import Callable, Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Metric key -> (table label, direction): "up" = higher is better, "down" =
# lower is better, "up_to_1" = higher is better up to 1 (Audio Coverage: values
# above 1 mean the AD is longer than all available silence).
METRICS = {
    "metric_G1.1_precision_avg_ad_in_visual_info": ("Prioritization", "up"),
    "metric_2_1_does_not_restate": ("No Redundancy", "up"),
    "metric_1_2_narrates_onscreen_text": ("On-Screen Text", "up"),
    "metric_g_2_avg_diff_time": ("Content Timeliness (s)", "down"),
    "total_overlap": ("Overlaps w. TTS (s)", "down"),
    "ad_total_speaking_time_per_silent_second": (
        "Audio Coverage (ratio)",
        "up_to_1",
    ),
    "avg_silence_duration_w_ad_tts": ("Average Silence Duration (s)", "down"),
    "videoa11y_descriptive": ("VideoA11y-Ref Descriptive", "up"),
    "videoa11y_objective": ("VideoA11y-Ref Objective", "up"),
    "videoa11y_accurate": ("VideoA11y-Ref Accurate", "up"),
    "videoa11y_clear": ("VideoA11y-Ref Clear", "up"),
    # Reference-free VideoA11y (default variant written by run_ad_eval.py).
    "video_asr_descriptive": ("Quality-RF Descriptive", "up"),
    "video_asr_objective": ("Quality-RF Objective", "up"),
    "video_asr_accurate": ("Quality-RF Accurate", "up"),
    "video_asr_clear": ("Quality-RF Clear", "up"),
}
GUIDELINE_METRICS = list(METRICS)[:7]
VIDEOA11Y_METRICS = list(METRICS)[7:11]
VIDEOA11Y_RF_METRICS = list(METRICS)[11:]
_CAPPED_BY_DURATION = (
    "metric_g_2_avg_diff_time",
    "total_overlap",
    "avg_silence_duration_w_ad_tts",
)


def _default_open(path: str):
  return open(path, "r", encoding="utf-8")


def load_video_metrics(path: str, open_fn: Callable = _default_open):
  """Returns {video_id, metric: value} for one metrics JSON, or None."""
  video_id = os.path.basename(path).split("output_")[1].split(".json")[0]
  try:
    with open_fn(path) as f:
      data = json.load(f)
  except (json.JSONDecodeError, OSError) as e:
    print(f"Skipping unreadable {path}: {e}")
    return None
  row = {"video_id": video_id}
  duration = data.get("video_duration_seconds")
  for key in METRICS:
    value = data.get(key)
    if value is None:
      continue
    if (
        key == "metric_1_2_narrates_onscreen_text"
        and "onscreen_text_present" in data
        and int(data["onscreen_text_present"]) == 0
    ):
      continue
    if (
        key in _CAPPED_BY_DURATION
        and duration is not None
        and float(value) > float(duration)
    ):
      continue
    row[key] = float(value)
  return row


def load_metrics_dir(
    metrics_dir: str,
    glob_fn: Callable = glob.glob,
    open_fn: Callable = _default_open,
) -> pd.DataFrame:
  """Loads every output_*.json in a directory into a DataFrame."""
  rows = [
      row
      for path in sorted(glob_fn(os.path.join(metrics_dir, "output_*.json")))
      if (row := load_video_metrics(path, open_fn)) is not None
  ]
  df = pd.DataFrame(rows, columns=["video_id", *METRICS])
  return df


def load_exclusions(path: str | None) -> dict[str, set[str]]:
  """Reads {"exclusions": {config: [video_id, ...]}}; {} if path is missing."""
  if not path or not os.path.exists(path):
    return {}
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
  return {k: set(v) for k, v in data.get("exclusions", {}).items()}


def load_configs(
    config_dirs: dict[str, str],
    exclusions: dict[str, set[str]] | None = None,
    **kwargs,
) -> pd.DataFrame:
  """Loads {config: metrics_dir} into one long DataFrame with a 'config' column.

  Args:
    config_dirs: Map of configuration name to its per-video metrics directory.
    exclusions: Optional {config: video_ids} to leave out of scoring.
    **kwargs: Passed to load_metrics_dir (e.g. glob_fn / open_fn for gfile).

  Returns:
    DataFrame with one row per (config, video).
  """
  exclusions = exclusions or {}
  frames = []
  for config, metrics_dir in config_dirs.items():
    df = load_metrics_dir(metrics_dir, **kwargs)
    excluded = exclusions.get(config, set())
    if excluded:
      df = df[~df.video_id.isin(excluded)]
    df["config"] = config
    note = f", {len(excluded)} excluded" if excluded else ""
    print(f"{config}: {len(df)} videos{note} ({metrics_dir})")
    frames.append(df)
  return pd.concat(frames, ignore_index=True)


def common_subset(df: pd.DataFrame, configs: Iterable[str]) -> pd.DataFrame:
  """Rows of `configs`, restricted to videos present for every config."""
  configs = list(configs)
  sub = df[df.config.isin(configs)]
  common = set.intersection(
      *[set(sub.loc[sub.config == c, "video_id"]) for c in configs]
  )
  return sub[sub.video_id.isin(common)]


def select_videos(
    df: pd.DataFrame, configs: Iterable[str], common_videos: bool = True
) -> pd.DataFrame:
  """Rows of `configs`; restricted to their common videos if requested."""
  configs = list(configs)
  if common_videos:
    return common_subset(df, configs)
  return df[df.config.isin(configs)]


def summarize(
    df: pd.DataFrame,
    configs: Iterable[str],
    metrics: Iterable[str] | None = None,
    common_videos: bool = True,
) -> pd.DataFrame:
  """Mean, 95% CI half-width and n per (metric, config).

  Args:
    df: Output of load_configs().
    configs: Configurations to compare.
    metrics: Metric keys; by default every key of METRICS that has at least one
      value among `configs` (e.g. Quality-RF only if it was computed).
    common_videos: If True, use only videos that every config has output for; if
      False, each config uses all of its own videos.

  Returns:
    Long DataFrame with columns metric, config, mean, ci95, n. attrs holds
    num_common_videos (distinct videos used) and num_videos_per_config.
  """
  configs = list(configs)
  sub = select_videos(df, configs, common_videos)
  if metrics is None:
    metrics = [m for m in METRICS if sub[m].notna().any()]
  records = []
  for metric in metrics:
    for config in configs:
      values = sub.loc[sub.config == config, metric].dropna()
      n = len(values)
      records.append({
          "metric": metric,
          "config": config,
          "mean": values.mean() if n else np.nan,
          "ci95": 1.96 * values.std() / np.sqrt(n) if n > 1 else np.nan,
          "n": n,
      })
  out = pd.DataFrame(records)
  out.attrs["num_common_videos"] = sub.video_id.nunique()
  out.attrs["num_videos_per_config"] = {
      c: sub.loc[sub.config == c, "video_id"].nunique() for c in configs
  }
  return out


def latex_table(
    summary: pd.DataFrame,
    labels: dict[str, str],
    caption: str = "",
    label: str = "",
    metric_labels: dict[str, str] | None = None,
) -> str:
  """LaTeX table: mean \\tiny{±CI}, best shaded E2E2E2 and second F2F2F2."""
  configs = list(labels)
  metrics = list(dict.fromkeys(summary.metric))
  lines = [
      "\\begin{tabular}{@{}l" + "c" * len(configs) + "@{}}",
      "\\toprule",
      " & ".join(
          ["\\textbf{Metric}"] + [f"\\textbf{{{labels[c]}}}" for c in configs]
      )
      + " \\\\",
      "\\midrule",
  ]
  for i, metric in enumerate(metrics):
    name, direction = METRICS[metric]
    if metric_labels and metric in metric_labels:
      name = metric_labels[metric]
    arrow = {
        "up": "$\\uparrow$",
        "down": "$\\downarrow$",
        "up_to_1": "$\\le$1",
    }[direction]
    rows = summary[summary.metric == metric].set_index("config")
    means = [rows.loc[c, "mean"] for c in configs]
    candidates = {round(m, 2) for m in means if not np.isnan(m)}
    if direction == "up_to_1":
      candidates = {m for m in candidates if m <= 1.0}
    ranked = sorted(candidates, reverse=(direction != "down"))
    cells = [f"{name} ({arrow})"]
    for c, m in zip(configs, means):
      if np.isnan(m):
        cells.append("N/A")
        continue
      cell = f"{m:.2f} \\tiny{{$\\pm${rows.loc[c, 'ci95']:.2f}}}"
      if ranked and round(m, 2) == ranked[0]:
        cell = "\\cellcolor[HTML]{E2E2E2} " + cell
      elif len(ranked) > 1 and round(m, 2) == ranked[1]:
        cell = "\\cellcolor[HTML]{F2F2F2} " + cell
      cells.append(cell)
    sep = (
        " \\\\ \\midrule"
        if (metric == GUIDELINE_METRICS[-1] and i < len(metrics) - 1)
        else " \\\\"
    )
    lines.append(" & ".join(cells) + sep)
  lines += ["\\bottomrule", "\\end{tabular}"]
  body = "\n".join(lines)
  if caption or label:
    body = (
        "\\begin{table*}[h]\n\\centering\n\\resizebox{\\columnwidth}{!}{\n"
        + body
        + f"\n}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table*}}"
    )
  return body


def _label_above_errorbars(ax, ymax: float, rotation: float):
  """Writes each bar's value just above the top of its error bar."""
  for patch in ax.patches:
    height = patch.get_height()
    if np.isnan(height) or patch.get_width() == 0:
      continue
    x = patch.get_x() + patch.get_width() / 2
    top = height
    for line in ax.lines:
      xs, ys = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
      xs = xs[~np.isnan(xs)]
      if len(xs) and abs(xs.mean() - x) < patch.get_width() / 4:
        top = max(top, np.nanmax(ys))
    ax.annotate(
        f"{height:.2f}",
        (x, min(top, 0.97 * ymax)),
        xytext=(0, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=11,
        rotation=rotation,
        rotation_mode="anchor",
    )


def bar_figure(
    df: pd.DataFrame,
    configs: list[str],
    labels: dict[str, str],
    metrics: list[str],
    titles: dict[str, str],
    ylims: dict[str, float],
    ylabel: str = "Mean Value (95% CI)",
    hatches: list[str] | None = None,
    palette: str = "Set2",
    drop_timeliness_outliers: bool = False,
    sharey: bool = False,
    common_videos: bool = True,
    label_rotation: float = 0,
    out_path: str | None = None,
):
  """One panel per metric; bars per config with 95% CIs."""
  sns.set_theme(style="whitegrid")
  sub = select_videos(df, configs, common_videos)
  long = sub.melt(
      id_vars=["config"],
      value_vars=metrics,
      var_name="Metric",
      value_name="Score",
  )
  colors = sns.color_palette(palette, n_colors=len(configs))
  fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=sharey)
  axes = np.atleast_1d(axes)
  for i, metric in enumerate(metrics):
    ax = axes[i]
    up = METRICS[metric][1] != "down"
    data = long[long.Metric == metric]
    if drop_timeliness_outliers and metric == "metric_g_2_avg_diff_time":
      m, s = data.Score.mean(), data.Score.std()
      data = data[(data.Score - m).abs() < 3 * s]
    sns.barplot(
        data=data,
        x="Metric",
        y="Score",
        hue="config",
        hue_order=configs,
        ax=ax,
        estimator=np.mean,
        errorbar=("ci", 95),
        capsize=0.1,
        palette=colors,
        legend=(i == 0),
    )
    if hatches:
      for j, patch in enumerate(ax.patches):
        patch.set_hatch(hatches[j % len(hatches)])
    ax.set_title(
        titles[metric] + (" (↑)" if up else " (↓)"),
        fontsize=14,
        weight="bold",
        color="#2c3e50" if up else "#c0392b",
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel if i == 0 else "")
    ax.set_xticklabels([])
    ax.set_ylim(0, ylims.get(metric, 1.0))
    if label_rotation:
      _label_above_errorbars(ax, ylims.get(metric, 1.0), label_rotation)
    else:
      for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=15, fontsize=11)
  handles, names = axes[0].get_legend_handles_labels()
  axes[0].get_legend().remove()
  if hatches:
    for h, hatch in zip(handles, hatches):
      h.set_hatch(hatch)
  fig.legend(
      handles,
      [labels.get(n, n) for n in names],
      loc="lower center",
      ncol=min(len(configs), 8),
      bbox_to_anchor=(0.5, -0.03),
      fontsize=16,
      frameon=True,
  )
  plt.tight_layout(rect=[0, 0.05, 1, 0.95])
  if out_path:
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
  return fig


def prompt_figure(df, configs, labels, metrics, titles, ylims, out_path=None):
  """Prompt-comparison figure (viridis palette, legend titled by strategy)."""
  fig = bar_figure(
      df,
      configs,
      labels,
      metrics,
      titles,
      ylims,
      palette="viridis",
  )
  legend = fig.legends[0]
  handles, texts = legend.legend_handles, [t.get_text() for t in legend.texts]
  legend.remove()
  invisible = mpatches.Rectangle((0, 0), 1, 1, fc="none", ec="none", alpha=0)
  fig.legend(
      [invisible] + list(handles),
      ["Prompt Strategy: "] + texts,
      loc="lower center",
      ncol=len(configs) + 1,
      bbox_to_anchor=(0.5, -0.03),
      fontsize=16,
  )
  if out_path:
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
  return fig
