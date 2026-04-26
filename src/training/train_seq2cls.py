from transformers import TrainingArguments

def build_training_args(config, output_dir):
    common_kwargs = dict(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=config["training_cell_selector"]["save_steps"],
        logging_steps=config["training_cell_selector"]["logging_steps"],
        save_total_limit=config["training_cell_selector"]["save_total_limit"],
        learning_rate=config["training_cell_selector"]["learning_rate"],
        per_device_train_batch_size=config["training_cell_selector"]["batch_size"],
        per_device_eval_batch_size=config["training_cell_selector"]["batch_size"],
        gradient_accumulation_steps=config["training_cell_selector"]["gradient_accumulation_steps"],
        num_train_epochs=config["training_cell_selector"]["epochs"],
        weight_decay=config["training_cell_selector"]["weight_decay"],
        report_to="none",
        load_best_model_at_end=config["training_cell_selector"]["load_best_model_at_end"],
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    try:
        return TrainingArguments(
            eval_strategy="steps",
            eval_steps=config["training_cell_selector"]["eval_steps"],
            **common_kwargs,
        )
    except TypeError:
        return TrainingArguments(
            evaluation_strategy="steps",
            eval_steps=config["training_cell_selector"]["eval_steps"],
            **common_kwargs,
        )
