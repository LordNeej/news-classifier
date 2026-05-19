# predict.py — Loads the fine-tuned model and classifies a news article.
# Returns the predicted category name and a confidence score for every class.

import os
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Where train.py saved the fine-tuned model
MODEL_DIR  = "./model"
MAX_LENGTH = 256  # must match the value used during training

# Labels in AG News order: index 0 → World, 1 → Sports, 2 → Business, 3 → Sci/Tech
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


class NewsClassifier:
    """
    Wraps the fine-tuned DistilBERT model for single-article inference.

    Usage:
        classifier = NewsClassifier()
        result = classifier.predict("SpaceX launches new rocket...")
        print(result["predicted_label"])   # e.g. "Sci/Tech"
        print(result["confidence_scores"]) # dict of label → probability
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        # Detect device — use GPU for speed if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from '{model_dir}' on {self.device}...")

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory '{model_dir}' not found. "
                "Run train.py first to generate the model."
            )

        # Load the tokenizer saved alongside the model weights
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

        # Load the classification head + weights
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()  # disable dropout layers during inference

        print("Model loaded successfully.")

    def predict(self, text: str) -> dict:
        """
        Classify a single news article.

        Args:
            text: Raw article text (headline + description works best).

        Returns:
            {
                "predicted_label":   str,          # e.g. "Sports"
                "predicted_index":   int,          # e.g. 1
                "confidence_scores": dict[str, float],  # all four classes
                "confidence":        float,        # score of the top class
            }
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        # Tokenise — same settings as during training
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",  # return PyTorch tensors
        )
        # Move input tensors to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass — no gradient tracking needed for inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert logits to probabilities with softmax
        logits = outputs.logits  # shape: (1, num_labels)
        probs  = torch.softmax(logits, dim=-1).squeeze(0)  # shape: (num_labels,)

        # Move to CPU numpy for easy handling
        probs_np = probs.cpu().numpy()

        predicted_index = int(np.argmax(probs_np))
        predicted_label = LABEL_NAMES[predicted_index]
        confidence      = float(probs_np[predicted_index])

        # Build a readable scores dict: {"World": 0.03, "Sports": 0.91, ...}
        confidence_scores = {
            label: float(score)
            for label, score in zip(LABEL_NAMES, probs_np)
        }

        return {
            "predicted_label":   predicted_label,
            "predicted_index":   predicted_index,
            "confidence":        confidence,
            "confidence_scores": confidence_scores,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Classify multiple articles at once (more efficient than looping predict())."""
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs_all = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        results = []
        for probs_np in probs_all:
            predicted_index = int(np.argmax(probs_np))
            results.append({
                "predicted_label":   LABEL_NAMES[predicted_index],
                "predicted_index":   predicted_index,
                "confidence":        float(probs_np[predicted_index]),
                "confidence_scores": {
                    label: float(score)
                    for label, score in zip(LABEL_NAMES, probs_np)
                },
            })
        return results


# ------------------------------------------------------------------ #
# CLI entry-point — python predict.py
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    classifier = NewsClassifier()

    # Accept text from a command-line argument or fall back to demo examples
    if len(sys.argv) > 1:
        article = " ".join(sys.argv[1:])
    else:
        # Demo articles — one from each category
        demo_articles = [
            "NASA's James Webb Space Telescope has captured stunning new images of distant galaxies, "
            "revealing unprecedented detail about the early universe.",
            "The Golden State Warriors defeated the Boston Celtics 112-98 in Game 5 of the NBA Finals, "
            "with Stephen Curry scoring 38 points.",
            "Federal Reserve raises interest rates by 25 basis points amid ongoing inflation concerns, "
            "signalling further hikes may come later this year.",
            "United Nations Security Council holds emergency session as tensions escalate on the "
            "border between the two neighbouring nations.",
        ]
        print("No article provided — running demo predictions:\n")
        for article in demo_articles:
            result = classifier.predict(article)
            print(f"Article : {article[:80]}...")
            print(f"Predicted: {result['predicted_label']} (confidence {result['confidence']:.2%})")
            print("All scores:", {k: f"{v:.2%}" for k, v in result["confidence_scores"].items()})
            print()
        sys.exit(0)

    result = classifier.predict(article)
    print(f"Predicted category : {result['predicted_label']}")
    print(f"Confidence         : {result['confidence']:.2%}")
    print("\nAll class scores:")
    for label, score in result["confidence_scores"].items():
        bar = "█" * int(score * 40)
        print(f"  {label:<12} {score:.2%}  {bar}")
