import evaluate

def compute_generation_metrics(predictions, references):
    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    bleu_result = sacrebleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )

    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
    )

    return {
        "bleu": bleu_result,
        "rouge": rouge_result,
    }
