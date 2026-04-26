import numpy as np
from sentence_transformers import SentenceTransformer

def load_sentence_embedding_model(model_name, device):
    return SentenceTransformer(model_name, device=device)

def compute_pairwise_cosine_scores(model, predictions, references):
    pred_embeddings = model.encode(
        predictions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    ref_embeddings = model.encode(
        references,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    scores = np.sum(pred_embeddings * ref_embeddings, axis=1)

    return [float(score) for score in scores]

def summarize_cosine_scores(scores):
    if not scores:
        return {
            "cosine_mean": 0.0,
            "cosine_std": 0.0,
            "cosine_min": 0.0,
            "cosine_max": 0.0,
        }

    return {
        "cosine_mean": float(np.mean(scores)),
        "cosine_std": float(np.std(scores)),
        "cosine_min": float(np.min(scores)),
        "cosine_max": float(np.max(scores)),
    }