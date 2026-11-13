import argparse
import json
import sys
from src.inference import BlogClassifier
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Predict blog category using fine-tuned DistilBERT.")
    parser.add_argument("--text", type=str, help="Blog text snippet to classify.")
    parser.add_argument("--model_dir", type=str, default="saved_models/distilbert_blogs", help="Saved model directory.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top candidate categories to display.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    args = parser.parse_args()

    set_seed(42)
    classifier = BlogClassifier(model_dir=args.model_dir)

    if args.text:
        input_text = args.text
        res = classifier.predict(input_text, top_k=args.top_k)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("\n" + "=" * 60)
            print("BLOG CLASSIFICATION INFERENCE RESULT")
            print("=" * 60)
            print(f"Text Snippet:       {res['text_snippet']}")
            print(f"Predicted Category: {res['predicted_category']}")
            print(f"Confidence Score:   {res['confidence'] * 100:.2f}%")
            print("\nTop Candidates:")
            for rank, item in enumerate(res["top_k"], 1):
                print(f"  {rank}. {item['category']:<26} (Probability: {item['probability'] * 100:.2f}%)")
            print("=" * 60 + "\n")
    else:
        print("\nEntering interactive prediction mode (type 'exit' or 'quit' to stop):")
        while True:
            try:
                line = input("\nEnter blog text: ").strip()
                if not line or line.lower() in ("exit", "quit"):
                    break
                res = classifier.predict(line, top_k=args.top_k)
                print(f"--> Predicted: {res['predicted_category']} (Confidence: {res['confidence'] * 100:.2f}%)")
                print("    Top-3:")
                for rank, item in enumerate(res["top_k"], 1):
                    print(f"      {rank}. {item['category']} ({item['probability'] * 100:.1f}%)")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExited.")


if __name__ == "__main__":
    main()

