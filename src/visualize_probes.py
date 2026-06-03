#!/usr/bin/env python3
"""Visualize saved interpretability probes for the full multimodal model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from transformers import BertTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "output" / "analysis" / "image_text_meta"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "figures"


def normalize(values: torch.Tensor) -> torch.Tensor:
    value_min = values.min()
    value_max = values.max()
    if torch.isclose(value_min, value_max):
        return torch.zeros_like(values)
    return (values - value_min) / (value_max - value_min)


def load_attention_scores(probe_path: Path, tokenizer: BertTokenizerFast) -> tuple[list[str], list[float], dict[str, object]]:
    payload = torch.load(probe_path, map_location="cpu", weights_only=False)
    attention = payload["bert_last_layer_attention"]
    text = payload["text"]

    if attention is None:
        raise ValueError(f"{probe_path} does not contain BERT attention data.")

    encoded = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    mask = encoded["attention_mask"][0].bool()

    # Average heads, then use [CLS] attention to estimate token importance.
    token_scores = attention.mean(dim=0)[0]
    valid_tokens = []
    valid_scores = []

    for token, score, keep in zip(tokens, token_scores, mask):
        if not keep or token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        valid_tokens.append(token.replace("##", ""))
        valid_scores.append(float(score))

    scores_tensor = normalize(torch.tensor(valid_scores, dtype=torch.float32))
    return valid_tokens, scores_tensor.tolist(), payload


def plot_text_attention(probe_path: Path, tokenizer: BertTokenizerFast, output_path: Path) -> None:
    tokens, scores, payload = load_attention_scores(probe_path, tokenizer)

    fig_width = max(10, min(18, len(tokens) * 0.55))
    plt.figure(figsize=(fig_width, 3.8))
    colors = plt.cm.Reds(scores)
    plt.bar(range(len(tokens)), scores, color=colors)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Normalized Attention")
    plt.title(
        f"BERT Attention Highlight | image_id={payload['image_id']} | "
        f"label={payload['label']} pred={payload['prediction']}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_cosine_similarity_distribution(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=df,
        x="cosine_similarity",
        hue="label_name",
        fill=True,
        common_norm=False,
        alpha=0.35,
    )
    plt.xlabel("Image-Text Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Image-Text Cosine Similarity Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualizations from saved probe files.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-index", type=int, default=1, help="Probe file index to visualize, e.g. 1 for sample_1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    probe_path = args.analysis_dir / f"sample_{args.probe_index}_probes.pt"
    cosine_csv = args.analysis_dir / "cosine_similarity_dist.csv"

    plot_text_attention(
        probe_path=probe_path,
        tokenizer=tokenizer,
        output_path=args.output_dir / f"text_attention_sample_{args.probe_index}.png",
    )
    plot_cosine_similarity_distribution(
        csv_path=cosine_csv,
        output_path=args.output_dir / "cosine_similarity_kde.png",
    )

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
