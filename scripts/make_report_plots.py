import argparse
from pathlib import Path

from src.utils.io import load_yaml
from src.visualization.plot_utils import (
    ensure_dir,
    load_json,
    load_jsonl,
    plot_bar_dict,
    plot_hist,
    plot_metric_lines,
    plot_scatter,
    read_trainer_log_history,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default="small",
        choices=["debug", "small", "medium", "full"],
    )
    return parser.parse_args()


def safe_load_json(path):
    path = Path(path)

    if not path.exists():
        print(f"[skip] Missing file: {path}")
        return None

    return load_json(path)


def safe_load_jsonl(path):
    path = Path(path)

    if not path.exists():
        print(f"[skip] Missing file: {path}")
        return None

    return load_jsonl(path)


def get_bleu_score(metrics):
    bleu = metrics.get("bleu", {})
    if isinstance(bleu, dict):
        return bleu.get("score")
    return bleu


def get_rouge_scores(metrics):
    rouge = metrics.get("rouge", {})
    return {
        "rouge1": rouge.get("rouge1"),
        "rouge2": rouge.get("rouge2"),
        "rougeL": rouge.get("rougeL"),
    }


def plot_training_history(config, mode, plot_dir, dpi):
    generator_state = (
        Path(config["paths"]["checkpoint_dir"])
        / mode
        / "trainer_state.json"
    )

    selector_state = (
        Path(config["paths"]["checkpoint_dir"])
        / config["training_cell_selector"]["checkpoint_subdir"]
        / mode
        / "trainer_state.json"
    )

    if generator_state.exists():
        df = read_trainer_log_history(generator_state)

        plot_metric_lines(
            df=df,
            metrics=["loss", "eval_loss"],
            title="Generator Training / Evaluation Loss",
            output_path=plot_dir / "generator_loss_history.png",
            dpi=dpi,
        )

        plot_metric_lines(
            df=df,
            metrics=["learning_rate"],
            title="Generator Learning Rate Schedule",
            output_path=plot_dir / "generator_learning_rate.png",
            dpi=dpi,
        )
    else:
        print(f"[skip] Missing file: {generator_state}")

    if selector_state.exists():
        df = read_trainer_log_history(selector_state)

        plot_metric_lines(
            df=df,
            metrics=["loss", "eval_loss"],
            title="Cell Selector Training / Evaluation Loss",
            output_path=plot_dir / "cell_selector_loss_history.png",
            dpi=dpi,
        )

        plot_metric_lines(
            df=df,
            metrics=["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"],
            title="Cell Selector Evaluation Metrics",
            output_path=plot_dir / "cell_selector_eval_metrics_history.png",
            dpi=dpi,
        )

        plot_metric_lines(
            df=df,
            metrics=["learning_rate"],
            title="Cell Selector Learning Rate Schedule",
            output_path=plot_dir / "cell_selector_learning_rate.png",
            dpi=dpi,
        )
    else:
        print(f"[skip] Missing file: {selector_state}")


def plot_generation_vs_e2e(config, mode, plot_dir, dpi):
    metric_dir = Path(config["paths"]["metric_dir"]) / mode

    generation_metrics = safe_load_json(metric_dir / "generation_metrics.json")
    e2e_metrics = safe_load_json(metric_dir / "e2e_metrics.json")

    if generation_metrics is None or e2e_metrics is None:
        return

    e2e_generation = e2e_metrics.get("generation", {})

    scores = {
        "gold_bleu": get_bleu_score(generation_metrics),
        "e2e_bleu": get_bleu_score(e2e_generation),
        "gold_rouge1": get_rouge_scores(generation_metrics)["rouge1"],
        "e2e_rouge1": get_rouge_scores(e2e_generation)["rouge1"],
        "gold_rouge2": get_rouge_scores(generation_metrics)["rouge2"],
        "e2e_rouge2": get_rouge_scores(e2e_generation)["rouge2"],
        "gold_rougeL": get_rouge_scores(generation_metrics)["rougeL"],
        "e2e_rougeL": get_rouge_scores(e2e_generation)["rougeL"],
    }

    scores = {k: v for k, v in scores.items() if v is not None}

    plot_bar_dict(
        metric_dict=scores,
        title="Gold Evidence Generation vs E2E Generation",
        output_path=plot_dir / "gold_vs_e2e_generation_metrics.png",
        dpi=dpi,
    )


def plot_cell_selector_metrics(config, mode, plot_dir, dpi):
    metric_dir = Path(config["paths"]["metric_dir"]) / mode
    pred_dir = Path(config["paths"]["prediction_dir"]) / mode

    cell_metrics = safe_load_json(metric_dir / "cell_selector_metrics.json")
    cell_predictions = safe_load_jsonl(pred_dir / "cell_selector_predictions.jsonl")

    if cell_metrics is not None:
        selected = {
            key: value
            for key, value in cell_metrics.items()
            if key.startswith("cell_")
        }

        plot_bar_dict(
            metric_dict=selected,
            title="Cell Selector Metrics",
            output_path=plot_dir / "cell_selector_metrics.png",
            dpi=dpi,
        )

    if cell_predictions is None:
        return

    gold_counts = [
        len(record.get("gold_highlighted_cells", []))
        for record in cell_predictions
    ]

    pred_counts = [
        len(record.get("pred_highlighted_cells", []))
        for record in cell_predictions
    ]

    overlap_counts = []
    f1_scores = []

    for record in cell_predictions:
        gold = {
            tuple(item)
            for item in record.get("gold_highlighted_cells", [])
        }
        pred = {
            tuple(item)
            for item in record.get("pred_highlighted_cells", [])
        }

        overlap_counts.append(len(gold & pred))

        cell_metrics_record = record.get("cell_metrics", {})
        if "cell_f1" in cell_metrics_record:
            f1_scores.append(cell_metrics_record["cell_f1"])

    plot_hist(
        values=gold_counts,
        title="Gold Highlighted Cell Count per Example",
        xlabel="# Gold highlighted cells",
        output_path=plot_dir / "gold_highlighted_cell_count_distribution.png",
        bins=15,
        dpi=dpi,
    )

    plot_hist(
        values=pred_counts,
        title="Predicted Highlighted Cell Count per Example",
        xlabel="# Predicted highlighted cells",
        output_path=plot_dir / "pred_highlighted_cell_count_distribution.png",
        bins=15,
        dpi=dpi,
    )

    plot_hist(
        values=overlap_counts,
        title="Overlap Count between Predicted and Gold Cells",
        xlabel="|H_pred ∩ H_gold|",
        output_path=plot_dir / "cell_overlap_count_distribution.png",
        bins=15,
        dpi=dpi,
    )

    plot_hist(
        values=f1_scores,
        title="Per-example Cell F1 Distribution",
        xlabel="Cell F1",
        output_path=plot_dir / "per_example_cell_f1_distribution.png",
        bins=20,
        dpi=dpi,
    )


def plot_e2e_metrics(config, mode, plot_dir, dpi):
    metric_dir = Path(config["paths"]["metric_dir"]) / mode
    pred_dir = Path(config["paths"]["prediction_dir"]) / mode

    e2e_metrics = safe_load_json(metric_dir / "e2e_metrics.json")
    e2e_predictions = safe_load_jsonl(pred_dir / "e2e_predictions.jsonl")

    if e2e_metrics is not None:
        cell_metrics = e2e_metrics.get("cell_selection", {})
        faithfulness = e2e_metrics.get("faithfulness", {})

        plot_bar_dict(
            metric_dict=cell_metrics,
            title="E2E Cell Selection Metrics",
            output_path=plot_dir / "e2e_cell_selection_metrics.png",
            dpi=dpi,
        )

        plot_bar_dict(
            metric_dict=faithfulness,
            title="E2E Number Faithfulness Metrics",
            output_path=plot_dir / "e2e_number_faithfulness_metrics.png",
            dpi=dpi,
        )

    if e2e_predictions is None:
        return

    target_lengths = []
    prediction_lengths = []
    length_differences = []
    per_example_f1 = []
    unsupported_numbers = []

    for record in e2e_predictions:
        target = record.get("target", "")
        prediction = record.get("prediction", "")

        target_len = len(str(target).split())
        prediction_len = len(str(prediction).split())

        target_lengths.append(target_len)
        prediction_lengths.append(prediction_len)
        length_differences.append(prediction_len - target_len)

        metrics = record.get("cell_metrics", {})
        if "cell_f1" in metrics:
            per_example_f1.append(metrics["cell_f1"])

        faithfulness = record.get("faithfulness", {})
        unsupported_numbers.append(
            faithfulness.get("unsupported_number_count", 0)
        )

    plot_hist(
        values=target_lengths,
        title="Target Length Distribution",
        xlabel="# tokens",
        output_path=plot_dir / "target_length_distribution.png",
        bins=20,
        dpi=dpi,
    )

    plot_hist(
        values=prediction_lengths,
        title="Prediction Length Distribution",
        xlabel="# tokens",
        output_path=plot_dir / "prediction_length_distribution.png",
        bins=20,
        dpi=dpi,
    )

    plot_hist(
        values=length_differences,
        title="Prediction - Target Length Difference",
        xlabel="Length difference",
        output_path=plot_dir / "length_difference_distribution.png",
        bins=20,
        dpi=dpi,
    )

    if len(per_example_f1) == len(unsupported_numbers):
        plot_scatter(
            x_values=per_example_f1,
            y_values=unsupported_numbers,
            title="Cell F1 vs Unsupported Number Count",
            xlabel="Per-example Cell F1",
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
    plot_generation_vs_e2e(config, args.mode, plot_dir, dpi)
    plot_cell_selector_metrics(config, args.mode, plot_dir, dpi)
    plot_e2e_metrics(config, args.mode, plot_dir, dpi)

    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()