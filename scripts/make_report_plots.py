import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default="debug",
        choices=["debug", "small", "medium", "full"],
    )
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    path = Path(path)
    if not path.exists():
        print(f"[skip] Missing file: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        print(f"[skip] Missing file: {path}")
        return None

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def savefig(path, dpi=200):
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


def find_latest_trainer_state(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    direct = checkpoint_dir / "trainer_state.json"
    if direct.exists():
        return direct

    candidates = sorted(
        checkpoint_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[-1])
        if p.parent.name.split("-")[-1].isdigit()
        else -1,
    )

    if candidates:
        return candidates[-1]

    print(f"[skip] Cannot find trainer_state.json under: {checkpoint_dir}")
    return None


def read_log_history(trainer_state_path):
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    return pd.DataFrame(state.get("log_history", []))


def plot_lines(df, metrics, title, output_path, dpi=200):
    if df is None or df.empty or "step" not in df.columns:
        print(f"[skip] No log history for: {title}")
        return

    has_value = False
    plt.figure(figsize=(9, 5))

    for metric in metrics:
        if metric not in df.columns:
            continue

        sub = df[["step", metric]].dropna()
        if sub.empty:
            continue

        plt.plot(sub["step"], sub[metric], marker="o", label=metric)
        has_value = True

    if not has_value:
        plt.close()
        print(f"[skip] No requested metrics found for: {title}")
        return

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(output_path, dpi=dpi)


def plot_bar(metric_dict, title, output_path, dpi=200, rotate=True):
    if not metric_dict:
        print(f"[skip] Empty metrics for: {title}")
        return

    labels = list(metric_dict.keys())
    values = list(metric_dict.values())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Value")
    if rotate:
        plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    savefig(output_path, dpi=dpi)


def plot_hist(values, title, xlabel, output_path, bins=20, dpi=200):
    values = [v for v in values if v is not None]

    if not values:
        print(f"[skip] Empty values for: {title}")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    savefig(output_path, dpi=dpi)


def plot_scatter(x, y, title, xlabel, ylabel, output_path, dpi=200):
    if not x or not y or len(x) != len(y):
        print(f"[skip] Invalid scatter values for: {title}")
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    savefig(output_path, dpi=dpi)


def bleu_score(metrics):
    bleu = metrics.get("bleu", {})
    if isinstance(bleu, dict):
        return bleu.get("score")
    return bleu


def rouge_scores(metrics):
    rouge = metrics.get("rouge", {})
    return {
        "rouge1": rouge.get("rouge1"),
        "rouge2": rouge.get("rouge2"),
        "rougeL": rouge.get("rougeL"),
        "rougeLsum": rouge.get("rougeLsum"),
    }


def flatten_generation_metrics(prefix, metrics):
    if not metrics:
        return {}

    out = {}
    bleu = bleu_score(metrics)
    if bleu is not None:
        out[f"{prefix}_bleu"] = bleu

    for key, value in rouge_scores(metrics).items():
        if value is not None:
            out[f"{prefix}_{key}"] = value

    return out


def plot_training_history(config, mode, plot_dir, dpi):
    checkpoint_root = Path(config["paths"]["checkpoint_dir"])

    generator_dir = checkpoint_root / mode
    selector_dir = (
        checkpoint_root
        / config["training_cell_selector"]["checkpoint_subdir"]
        / mode
    )

    generator_state = find_latest_trainer_state(generator_dir)
    if generator_state is not None:
        print(f"[info] Generator trainer_state: {generator_state}")
        df = read_log_history(generator_state)

        plot_lines(
            df,
            metrics=["loss", "eval_loss"],
            title="Generator Training and Evaluation Loss",
            output_path=plot_dir / "generator_loss_history.png",
            dpi=dpi,
        )

        plot_lines(
            df,
            metrics=["learning_rate"],
            title="Generator Learning Rate",
            output_path=plot_dir / "generator_learning_rate.png",
            dpi=dpi,
        )

    selector_state = find_latest_trainer_state(selector_dir)
    if selector_state is not None:
        print(f"[info] Cell selector trainer_state: {selector_state}")
        df = read_log_history(selector_state)

        plot_lines(
            df,
            metrics=["loss", "eval_loss"],
            title="Cell Selector Training and Evaluation Loss",
            output_path=plot_dir / "cell_selector_loss_history.png",
            dpi=dpi,
        )

        plot_lines(
            df,
            metrics=["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"],
            title="Cell Selector Evaluation Metrics",
            output_path=plot_dir / "cell_selector_eval_metrics_history.png",
            dpi=dpi,
        )

        plot_lines(
            df,
            metrics=["learning_rate"],
            title="Cell Selector Learning Rate",
            output_path=plot_dir / "cell_selector_learning_rate.png",
            dpi=dpi,
        )


def find_topk_files(metric_dir, stem):
    metric_dir = Path(metric_dir)
    files = sorted(metric_dir.glob(f"{stem}_top*.json"))
    return files


def get_topk_from_name(path):
    name = Path(path).stem
    # e.g. cell_selector_metrics_top5 -> 5
    if "top" not in name:
        return None

    suffix = name.split("top")[-1]
    return int(suffix) if suffix.isdigit() else None


def plot_cell_selector_summary(config, mode, plot_dir, dpi):
    metric_dir = Path(config["paths"]["metric_dir"]) / mode

    files = find_topk_files(metric_dir, "cell_selector_metrics")

    if not files:
        fallback = metric_dir / "cell_selector_metrics.json"
        if fallback.exists():
            files = [fallback]

    if not files:
        print("[skip] No cell selector metric files found.")
        return

    topk_rows = []

    for path in files:
        metrics = load_json(path)
        if metrics is None:
            continue

        top_k = metrics.get("top_k", get_topk_from_name(path))
        row = {"top_k": top_k}
        row.update(
            {
                key: value
                for key, value in metrics.items()
                if key.startswith("cell_")
            }
        )
        topk_rows.append(row)

    if not topk_rows:
        return

    df = pd.DataFrame(topk_rows).sort_values("top_k")

    for metric in ["cell_precision", "cell_recall", "cell_f1", "cell_exact_match"]:
        if metric not in df.columns:
            continue

        plt.figure(figsize=(7, 5))
        plt.plot(df["top_k"], df[metric], marker="o")
        plt.title(f"Cell Selector {metric} by Top-k")
        plt.xlabel("Top-k")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        savefig(plot_dir / f"cell_selector_{metric}_by_topk.png", dpi=dpi)

    if len(df) == 1:
        row = df.iloc[0].to_dict()
        row.pop("top_k", None)
        plot_bar(
            row,
            title="Cell Selector Metrics",
            output_path=plot_dir / "cell_selector_metrics.png",
            dpi=dpi,
        )
    else:
        plt.figure(figsize=(9, 5))
        for metric in ["cell_precision", "cell_recall", "cell_f1", "cell_exact_match"]:
            if metric in df.columns:
                plt.plot(df["top_k"], df[metric], marker="o", label=metric)

        plt.title("Cell Selector Metrics by Top-k")
        plt.xlabel("Top-k")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(plot_dir / "cell_selector_metrics_by_topk.png", dpi=dpi)


def plot_cell_prediction_distributions(config, mode, plot_dir, dpi):
    pred_dir = Path(config["paths"]["prediction_dir"]) / mode

    files = sorted(pred_dir.glob("cell_selector_predictions_top*.jsonl"))
    if not files:
        fallback = pred_dir / "cell_selector_predictions.jsonl"
        if fallback.exists():
            files = [fallback]

    if not files:
        print("[skip] No cell selector prediction files found.")
        return

    # dùng file top-k đầu tiên để vẽ distribution chi tiết
    path = files[0]
    print(f"[info] Using cell prediction file for distributions: {path}")
    records = load_jsonl(path)

    if not records:
        return

    gold_counts = []
    pred_counts = []
    overlap_counts = []
    f1_scores = []

    for record in records:
        gold = {tuple(x) for x in record.get("gold_highlighted_cells", [])}
        pred = {tuple(x) for x in record.get("pred_highlighted_cells", [])}

        gold_counts.append(len(gold))
        pred_counts.append(len(pred))
        overlap_counts.append(len(gold & pred))

        metrics = record.get("cell_metrics", {})
        if "cell_f1" in metrics:
            f1_scores.append(metrics["cell_f1"])

    plot_hist(
        gold_counts,
        title="Gold Highlighted Cell Count per Example",
        xlabel="# gold cells",
        output_path=plot_dir / "gold_cell_count_distribution.png",
        bins=15,
        dpi=dpi,
    )

    plot_hist(
        pred_counts,
        title="Predicted Highlighted Cell Count per Example",
        xlabel="# predicted cells",
        output_path=plot_dir / "pred_cell_count_distribution.png",
        bins=15,
        dpi=dpi,
    )

    plot_hist(
        overlap_counts,
        title="Overlap Count between Predicted and Gold Cells",
        xlabel="|H_pred ∩ H_gold|",
        output_path=plot_dir / "cell_overlap_distribution.png",
        bins=15,
        dpi=dpi,
    )

    plot_hist(
        f1_scores,
        title="Per-example Cell F1 Distribution",
        xlabel="Cell F1",
        output_path=plot_dir / "cell_f1_distribution.png",
        bins=20,
        dpi=dpi,
    )


def plot_generation_and_e2e(config, mode, plot_dir, dpi):
    metric_dir = Path(config["paths"]["metric_dir"]) / mode

    generation_metrics = load_json(metric_dir / "generation_metrics.json")

    e2e_files = sorted(metric_dir.glob("e2e_metrics_top*.json"))
    if not e2e_files:
        fallback = metric_dir / "e2e_metrics.json"
        if fallback.exists():
            e2e_files = [fallback]

    if generation_metrics is None:
        print("[skip] Missing generation_metrics.json")
        return

    if not e2e_files:
        print("[skip] No E2E metrics files found.")
        return

    # Gold vs từng E2E top-k
    rows = []
    gold_flat = flatten_generation_metrics("gold", generation_metrics)

    for path in e2e_files:
        metrics = load_json(path)
        if metrics is None:
            continue

        top_k = metrics.get("top_k", get_topk_from_name(path))
        e2e_generation = metrics.get("generation", {})
        row = {"top_k": top_k}
        row.update(flatten_generation_metrics("e2e", e2e_generation))
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("top_k")

    for metric_name in ["bleu", "rouge1", "rouge2", "rougeL"]:
        gold_key = f"gold_{metric_name}"
        e2e_key = f"e2e_{metric_name}"

        gold_value = gold_flat.get(gold_key)
        if gold_value is None or e2e_key not in df.columns:
            continue

        plt.figure(figsize=(8, 5))
        plt.axhline(
            y=gold_value,
            linestyle="--",
            label=f"Gold evidence {metric_name}",
        )
        plt.plot(
            df["top_k"],
            df[e2e_key],
            marker="o",
            label=f"E2E {metric_name}",
        )
        plt.title(f"Gold Evidence vs E2E {metric_name}")
        plt.xlabel("Top-k")
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(plot_dir / f"gold_vs_e2e_{metric_name}.png", dpi=dpi)

    # Bar cho file e2e đầu tiên
    first = load_json(e2e_files[0])
    if first is not None:
        top_k = first.get("top_k", get_topk_from_name(e2e_files[0]))

        cell_metrics = first.get("cell_selection", {})
        faithfulness = first.get("faithfulness", {})

        plot_bar(
            cell_metrics,
            title=f"E2E Cell Selection Metrics Top-{top_k}",
            output_path=plot_dir / f"e2e_cell_selection_metrics_top{top_k}.png",
            dpi=dpi,
        )

        plot_bar(
            faithfulness,
            title=f"E2E Number Faithfulness Top-{top_k}",
            output_path=plot_dir / f"e2e_number_faithfulness_top{top_k}.png",
            dpi=dpi,
        )


def plot_e2e_prediction_distributions(config, mode, plot_dir, dpi):
    pred_dir = Path(config["paths"]["prediction_dir"]) / mode

    files = sorted(pred_dir.glob("e2e_predictions_top*.jsonl"))
    if not files:
        fallback = pred_dir / "e2e_predictions.jsonl"
        if fallback.exists():
            files = [fallback]

    if not files:
        print("[skip] No E2E prediction files found.")
        return

    path = files[0]
    print(f"[info] Using E2E prediction file for distributions: {path}")
    records = load_jsonl(path)

    if not records:
        return

    target_lengths = []
    prediction_lengths = []
    length_differences = []
    cell_f1_scores = []
    unsupported_counts = []

    for record in records:
        target = str(record.get("target", ""))
        prediction = str(record.get("prediction", ""))

        target_len = len(target.split())
        prediction_len = len(prediction.split())

        target_lengths.append(target_len)
        prediction_lengths.append(prediction_len)
        length_differences.append(prediction_len - target_len)

        metrics = record.get("cell_metrics", {})
        faithfulness = record.get("faithfulness", {})

        if "cell_f1" in metrics:
            cell_f1_scores.append(metrics["cell_f1"])
            unsupported_counts.append(
                faithfulness.get(
                    "unsupported_number_count",
                    faithfulness.get("n_unsupported_numbers", 0),
                )
            )

    plot_hist(
        target_lengths,
        title="Target Length Distribution",
        xlabel="# tokens",
        output_path=plot_dir / "target_length_distribution.png",
        bins=20,
        dpi=dpi,
    )

    plot_hist(
        prediction_lengths,
        title="Prediction Length Distribution",
        xlabel="# tokens",
        output_path=plot_dir / "prediction_length_distribution.png",
        bins=20,
        dpi=dpi,
    )

    plot_hist(
        length_differences,
        title="Prediction - Target Length Difference",
        xlabel="length difference",
        output_path=plot_dir / "length_difference_distribution.png",
        bins=20,
        dpi=dpi,
    )

    plot_scatter(
        cell_f1_scores,
        unsupported_counts,
        title="Cell F1 vs Unsupported Number Count",
        xlabel="Cell F1",
        ylabel="Unsupported number count",
        output_path=plot_dir / "cell_f1_vs_unsupported_numbers.png",
        dpi=dpi,
    )


def main():
    args = parse_args()
    config = load_yaml(args.config)

    dpi = config.get("plots", {}).get("dpi", 200)
    output_subdir = config.get("plots", {}).get("output_subdir", "plots")

    plot_dir = Path("outputs") / output_subdir / args.mode
    ensure_dir(plot_dir)

    plot_training_history(config, args.mode, plot_dir, dpi)
    plot_cell_selector_summary(config, args.mode, plot_dir, dpi)
    plot_cell_prediction_distributions(config, args.mode, plot_dir, dpi)
    plot_generation_and_e2e(config, args.mode, plot_dir, dpi)
    plot_e2e_prediction_distributions(config, args.mode, plot_dir, dpi)

    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()