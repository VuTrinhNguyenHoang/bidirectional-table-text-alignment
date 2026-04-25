import torch
from transformers import Seq2SeqTrainingArguments

def build_training_args(config: dict, output_dir: str):
    common_kwargs = dict(
        output_dir=output_dir,

        save_strategy="steps",
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_total_limit=config["training"]["save_total_limit"],

        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["epochs"],
        weight_decay=config["training"]["weight_decay"],

        predict_with_generate=True,
        generation_max_length=config["generation"]["max_new_tokens"],
        generation_num_beams=config["generation"]["num_beams"],

        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    try:
        return Seq2SeqTrainingArguments(
            eval_strategy="steps",
            eval_steps=config["training"]["eval_steps"],
            **common_kwargs,
        )
    except TypeError:
        return Seq2SeqTrainingArguments(
            evaluation_strategy="steps",
            eval_steps=config["training"]["eval_steps"],
            **common_kwargs,
        )
