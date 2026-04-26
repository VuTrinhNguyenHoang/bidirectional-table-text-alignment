import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_current_figure(path, dpi=200):
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def read_trainer_log_history(trainer_state_path):
    state = load_json(trainer_state_path)
    return pd.DataFrame(state.get("log_history", []))


def plot_metric_lines(df, metrics, title, output_path, dpi=200):
    if df.empty:
        return False

    if "step" not in df.columns:
        return False

    plotted = False

    plt.figure(figsize=(9, 5))

    for metric in metrics:
        if metric in df.columns:
            sub_df = df[["step", metric]].dropna()
            if len(sub_df) > 0:
                plt.plot(sub_df["step"], sub_df[metric], marker="o", label=metric)
                plotted = True

    if not plotted:
        plt.close()
        return False

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_current_figure(output_path, dpi=dpi)
    return True


def plot_bar_dict(metric_dict, title, output_path, dpi=200):
    if not metric_dict:
        return False

    labels = list(metric_dict.keys())
    values = [metric_dict[key] for key in labels]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Value")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(output_path, dpi=dpi)
    return True


def plot_hist(values, title, xlabel, output_path, bins=20, dpi=200):
    if not values:
        return False

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(output_path, dpi=dpi)
    return True


def plot_scatter(x_values, y_values, title, xlabel, ylabel, output_path, dpi=200):
    if not x_values or not y_values:
        return False

    plt.figure(figsize=(7, 5))
    plt.scatter(x_values, y_values, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    save_current_figure(output_path, dpi=dpi)
    return True