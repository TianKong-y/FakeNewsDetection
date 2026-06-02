#!/usr/bin/env python3
"""Train a ResNet50 + BERT + metadata tri-tower fake-news classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import BertModel, BertTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "processed" / "multimodal_training"
BEST_MODEL_DIR = PROJECT_ROOT / "best_models"
ANALYSIS_DIR = PROJECT_ROOT / "output" / "analysis"


def get_device() -> torch.device:
    """Pick CUDA first, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_save_path(
    save_path: Path | None,
    use_image: bool,
    use_text: bool,
    use_meta: bool,
) -> Path:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path

    enabled_modalities = []
    if use_image:
        enabled_modalities.append("image")
    if use_text:
        enabled_modalities.append("text")
    if use_meta:
        enabled_modalities.append("meta")

    model_name = f"best_{'_'.join(enabled_modalities)}.pth"
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return BEST_MODEL_DIR / model_name


class FakedditMultimodalDataset(Dataset):
    """Loads paired news text, image, metadata, and binary label samples from a split CSV."""

    def __init__(
        self,
        csv_path: Path,
        image_dir: Path,
        tokenizer: BertTokenizerFast,
        transform: transforms.Compose,
        max_length: int = 128,
        limit_samples: int | None = None,
        text_column: str = "title",
    ) -> None:
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.text_column = text_column

        required_columns = {"image_id", text_column, "2_way_label", "score", "num_comments", "upvote_ratio"}
        missing_columns = required_columns.difference(self.data.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")

        self.data = self._filter_valid_images(csv_path)
        if limit_samples is not None:
            self.data = self.data.head(limit_samples).reset_index(drop=True)

    def _filter_valid_images(self, csv_path: Path) -> pd.DataFrame:
        valid_rows = []
        skipped = []

        for _, row in self.data.iterrows():
            image_path = self.image_dir / f"{row['image_id']}.jpg"
            if not image_path.exists():
                skipped.append((image_path, "missing"))
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                skipped.append((image_path, type(error).__name__))
                continue

            valid_rows.append(row)

        if skipped:
            print(f"{csv_path.name}: skipped {len(skipped)} unreadable images")
            for image_path, reason in skipped[:10]:
                print(f"  - {image_path} ({reason})")

        return pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _numeric_value(value: object, default: float = 0.0) -> float:
        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric_value):
            return default
        return float(numeric_value)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[index]
        image_path = self.image_dir / f"{row['image_id']}.jpg"

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        encoded = self.tokenizer(
            str(row[self.text_column]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        score = max(self._numeric_value(row["score"]), 0.0)
        num_comments = max(self._numeric_value(row["num_comments"]), 0.0)
        upvote_ratio = self._numeric_value(row["upvote_ratio"])
        metadata = torch.tensor(
            [
                torch.log1p(torch.tensor(score)).item(),
                torch.log1p(torch.tensor(num_comments)).item(),
                upvote_ratio,
            ],
            dtype=torch.float32,
        )

        return {
            "images": image_tensor,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "metadata": metadata,
            "labels": torch.tensor(int(row["2_way_label"]), dtype=torch.long),
            "image_ids": str(row["image_id"]),
            "texts": str(row[self.text_column]),
        }


class TriTowerFakeNewsModel(nn.Module):
    """Tri-tower multimodal classifier using ResNet50, BERT, and metadata features."""

    def __init__(
        self,
        num_classes: int = 2,
        projection_dim: int = 512,
        dropout: float = 0.3,
        use_image: bool = True,
        use_text: bool = True,
        use_meta: bool = True,
    ) -> None:
        super().__init__()
        self.projection_dim = projection_dim
        self.metadata_dim = 128
        self.use_image = use_image
        self.use_text = use_text
        self.use_meta = use_meta
        self.latest_visual_feature_maps: torch.Tensor | None = None
        self.latest_bert_attention: torch.Tensor | None = None
        self.latest_cosine_similarity: torch.Tensor | None = None

        resnet = models.resnet50(pretrained=True)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_projection = nn.Linear(resnet.fc.in_features, projection_dim)
        self.visual_backbone[7].register_forward_hook(self._capture_visual_feature_maps)

        self.text_backbone = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        self.text_projection = nn.Linear(self.text_backbone.config.hidden_size, projection_dim)

        self.metadata_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(projection_dim * 2 + self.metadata_dim, num_classes),
        )

    def _capture_visual_feature_maps(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.latest_visual_feature_maps = output.detach()

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = images.size(0)

        if self.use_image:
            visual_features = self.visual_backbone(images)
            visual_features = torch.flatten(visual_features, start_dim=1)
            visual_features = self.visual_projection(visual_features)
        else:
            self.latest_visual_feature_maps = None
            visual_features = torch.zeros(
                batch_size,
                self.projection_dim,
                device=images.device,
                dtype=images.dtype,
            )

        if self.use_text:
            text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
            if text_outputs.attentions:
                self.latest_bert_attention = text_outputs.attentions[-1].detach()
            cls_features = text_outputs.last_hidden_state[:, 0, :]
            text_features = self.text_projection(cls_features)
        else:
            self.latest_bert_attention = None
            text_features = torch.zeros(
                batch_size,
                self.projection_dim,
                device=input_ids.device,
                dtype=self.text_projection.weight.dtype,
            )

        if self.use_meta:
            metadata_features = self.metadata_mlp(metadata)
        else:
            metadata_features = torch.zeros(
                batch_size,
                self.metadata_dim,
                device=metadata.device,
                dtype=metadata.dtype,
            )

        self.latest_cosine_similarity = F.cosine_similarity(visual_features, text_features, dim=1).detach()
        fused_features = torch.cat([visual_features, text_features, metadata_features], dim=1)
        return self.classifier(fused_features)


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_dataloaders(
    data_dir: Path,
    tokenizer: BertTokenizerFast,
    batch_size: int,
    num_workers: int,
    max_length: int,
    limit_train_samples: int | None = None,
    limit_val_samples: int | None = None,
    limit_test_samples: int | None = None,
    text_column: str = "title",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = build_transforms()
    train_dataset = FakedditMultimodalDataset(
        csv_path=data_dir / "train.csv",
        image_dir=data_dir / "images" / "train",
        tokenizer=tokenizer,
        transform=transform,
        max_length=max_length,
        limit_samples=limit_train_samples,
        text_column=text_column,
    )
    val_dataset = FakedditMultimodalDataset(
        csv_path=data_dir / "val.csv",
        image_dir=data_dir / "images" / "val",
        tokenizer=tokenizer,
        transform=transform,
        max_length=max_length,
        limit_samples=limit_val_samples,
        text_column=text_column,
    )
    test_dataset = FakedditMultimodalDataset(
        csv_path=data_dir / "test.csv",
        image_dir=data_dir / "images" / "test",
        tokenizer=tokenizer,
        transform=transform,
        max_length=max_length,
        limit_samples=limit_test_samples,
        text_column=text_column,
    )

    print(f"Text column: {text_column}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def compute_class_weights(dataset: FakedditMultimodalDataset, device: torch.device) -> torch.Tensor:
    label_counts = dataset.data["2_way_label"].value_counts().reindex([0, 1], fill_value=0)
    safe_counts = label_counts.clip(lower=1)
    total = int(label_counts.sum())
    weights = torch.tensor(
        [total / (2 * safe_counts.loc[0]), total / (2 * safe_counts.loc[1])],
        dtype=torch.float32,
        device=device,
    )
    print(f"Train label counts: {label_counts.to_dict()}")
    print(f"CrossEntropy class weights: {[round(float(weight), 4) for weight in weights.cpu()]}")
    return weights


def train_model(
    model: TriTowerFakeNewsModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 2e-5,
    save_path: Path | None = None,
    log_every: int = 50,
    early_stopping_patience: int = 2,
    early_stopping_min_delta: float = 0.0,
) -> None:
    if save_path is None:
        raise ValueError("save_path must be resolved before training.")

    model.to(device)
    class_weights = compute_class_weights(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                metadata=metadata,
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

            if log_every > 0 and batch_idx % log_every == 0:
                running_loss = train_loss / (batch_idx * train_loader.batch_size)
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Train Loss: {running_loss:.4f}",
                    flush=True,
                )

        avg_train_loss = train_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
        )
        print_metrics("Validation", val_metrics)

        current_val_f1 = float(val_metrics["macro_f1"])
        improved = current_val_f1 > best_val_f1 + early_stopping_min_delta

        if improved:
            best_val_f1 = current_val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path} with Val Macro F1: {best_val_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"No Val Macro F1 improvement for "
                f"{epochs_without_improvement}/{early_stopping_patience} epoch(s)."
            )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best Val Macro F1: {best_val_f1:.4f}"
            )
            break


def evaluate_model(
    model: TriTowerFakeNewsModel,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                metadata=metadata,
            )
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

    total = len(y_true)
    accuracy = sum(int(pred == true) for pred, true in zip(y_pred, y_true)) / total
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["fake", "real"],
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_metrics(name: str, metrics: dict[str, object]) -> None:
    print(f"{name} classification report:")
    print(metrics["classification_report"])
    print(f"{name} confusion matrix:")
    print(metrics["confusion_matrix"])


def plot_confusion_matrix(matrix: object, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["fake", "real"],
        yticklabels=["fake", "real"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_analysis(
    model: TriTowerFakeNewsModel,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path = ANALYSIS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []
    similarity_records = []
    probe_count = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                metadata=metadata,
            )
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(logits, dim=1)
            batch_labels = labels.detach().cpu().tolist()
            batch_predictions = predictions.detach().cpu().tolist()
            y_true.extend(batch_labels)
            y_pred.extend(batch_predictions)

            cosine_values = model.latest_cosine_similarity
            if cosine_values is not None:
                for image_id, text, label, prediction, cosine in zip(
                    batch["image_ids"],
                    batch["texts"],
                    batch_labels,
                    batch_predictions,
                    cosine_values.detach().cpu().tolist(),
                ):
                    similarity_records.append(
                        {
                            "image_id": image_id,
                            "text": text,
                            "label": label,
                            "label_name": "fake" if label == 0 else "real",
                            "prediction": prediction,
                            "cosine_similarity": cosine,
                        }
                    )

            batch_size = labels.size(0)
            for sample_idx in range(batch_size):
                if probe_count >= 5:
                    break

                feature_map = None
                if model.latest_visual_feature_maps is not None:
                    feature_map = model.latest_visual_feature_maps[sample_idx].detach().cpu()

                bert_attention = None
                if model.latest_bert_attention is not None:
                    bert_attention = model.latest_bert_attention[sample_idx].detach().cpu()

                cosine_similarity = None
                if cosine_values is not None:
                    cosine_similarity = float(cosine_values[sample_idx].detach().cpu().item())

                probe_payload = {
                    "image_id": batch["image_ids"][sample_idx],
                    "text": batch["texts"][sample_idx],
                    "label": batch_labels[sample_idx],
                    "prediction": batch_predictions[sample_idx],
                    "visual_feature_map": feature_map,
                    "bert_last_layer_attention": bert_attention,
                    "cosine_similarity": cosine_similarity,
                }
                torch.save(probe_payload, output_dir / f"sample_{probe_count + 1}_probes.pt")
                probe_count += 1

    total = len(y_true)
    test_metrics = {
        "loss": total_loss / total,
        "accuracy": sum(int(pred == true) for pred, true in zip(y_pred, y_true)) / total,
        "macro_f1": f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["fake", "real"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "y_true": y_true,
        "y_pred": y_pred,
    }

    print(
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Accuracy: {test_metrics['accuracy']:.4f} | "
        f"Test Macro F1: {test_metrics['macro_f1']:.4f}"
    )
    print_metrics("Test", test_metrics)

    plot_confusion_matrix(test_metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    pd.DataFrame(similarity_records).to_csv(output_dir / "cosine_similarity_dist.csv", index=False)
    print(f"Analysis artifacts saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal fake-news classifier.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--limit-test-samples", type=int, default=None)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop after this many epochs without Val Macro F1 improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum Val Macro F1 improvement required to reset early stopping.",
    )
    parser.add_argument("--use-image", dest="use_image", action="store_true", default=True)
    parser.add_argument("--no-use-image", dest="use_image", action="store_false")
    parser.add_argument("--use-text", dest="use_text", action="store_true", default=True)
    parser.add_argument("--no-use-text", dest="use_text", action="store_false")
    parser.add_argument("--use-meta", dest="use_meta", action="store_true", default=True)
    parser.add_argument("--no-use-meta", dest="use_meta", action="store_false")
    parser.add_argument(
        "--text-column",
        type=str,
        default="title",
        help="CSV text column used by BERT. Default avoids the leaked subreddit in the generated text column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.use_image or args.use_text or args.use_meta):
        raise ValueError("At least one modality must be enabled.")

    device = get_device()
    print(f"Using device: {device}")
    print(
        "Modality config: "
        f"use_image={args.use_image}, "
        f"use_text={args.use_text}, "
        f"use_meta={args.use_meta}"
    )

    train_df = pd.read_csv(args.data_dir / "train.csv")
    print(f"Train CSV shape: {train_df.shape}")
    print(f"Train CSV columns: {train_df.columns.tolist()}")
    save_path = resolve_save_path(
        save_path=args.save_path,
        use_image=args.use_image,
        use_text=args.use_text,
        use_meta=args.use_meta,
    )
    print(f"Best model path: {save_path}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        limit_train_samples=args.limit_train_samples,
        limit_val_samples=args.limit_val_samples,
        limit_test_samples=args.limit_test_samples,
        text_column=args.text_column,
    )

    model = TriTowerFakeNewsModel(
        use_image=args.use_image,
        use_text=args.use_text,
        use_meta=args.use_meta,
    )
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=save_path,
        log_every=args.log_every,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    print(f"Loading best checkpoint for test analysis: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    run_analysis(model=model, dataloader=test_loader, device=device, output_dir=args.analysis_dir)


if __name__ == "__main__":
    main()
