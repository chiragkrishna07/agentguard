"""
Fine-tune DistilBERT for prompt-injection detection.

Usage
-----
Google Colab (free tier) or a local GPU:

    pip install transformers datasets torch scikit-learn accelerate
    python training/train_injection_classifier.py

To publish to HuggingFace Hub (required for use_ml=True in PromptShield):
    export HF_TOKEN=hf_...
    python training/train_injection_classifier.py

The model will be pushed to: agentguard/prompt-injection-detector
"""
import os

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "distilbert-base-uncased"
HF_REPO = "agentguard/prompt-injection-detector"
LABELS = {"BENIGN": 0, "INJECTION": 1}
ID2LABEL = {0: "BENIGN", 1: "INJECTION"}


def load_data():
    """Load the deepset/prompt-injections dataset from HuggingFace."""
    print("Loading dataset: deepset/prompt-injections ...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    print(f"  Loaded {len(ds)} examples")

    # Normalise label column: the dataset uses 'label' as 0/1 already
    # 0 = benign, 1 = injection (verify with your dataset version)
    return ds.train_test_split(test_size=0.2, seed=42)


def tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    hf_token = os.environ.get("HF_TOKEN")
    push_to_hub = bool(hf_token)

    dataset = load_data()

    print("Loading tokenizer and model ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABELS,
    )

    print("Tokenising ...")
    tokenised = dataset.map(
        lambda x: tokenize_fn(x, tokenizer), batched=True, remove_columns=["text"]
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir="./checkpoints/injection-classifier",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        report_to="none",
        push_to_hub=push_to_hub,
        hub_model_id=HF_REPO if push_to_hub else None,
        hub_token=hf_token,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("\nTraining ...")
    trainer.train()

    print("\nFinal evaluation:")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    if push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {HF_REPO}")
        trainer.push_to_hub(commit_message="Initial DistilBERT injection classifier")
    else:
        save_path = "./checkpoints/injection-classifier-final"
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"\nModel saved to {save_path}")
        print("Set HF_TOKEN env var and re-run to publish to HuggingFace Hub.")


if __name__ == "__main__":
    main()
