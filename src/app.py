#!/usr/bin/env python3
"""Gradio demo for the multimodal fake news detector."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizerFast

from train_multimodal import PROJECT_ROOT, TriTowerFakeNewsModel, get_device


MODEL_PATH = PROJECT_ROOT / "best_models" / "best_image_text_meta.pth"
MAX_LENGTH = 128


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


device = get_device()
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
image_transform = build_transform()
model = TriTowerFakeNewsModel(use_image=True, use_text=True, use_meta=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def build_metadata(score: float, comments: float, upvote_ratio: float) -> torch.Tensor:
    score = max(float(score), 0.0)
    comments = max(float(comments), 0.0)
    upvote_ratio = float(upvote_ratio)
    return torch.tensor(
        [[torch.log1p(torch.tensor(score)).item(), torch.log1p(torch.tensor(comments)).item(), upvote_ratio]],
        dtype=torch.float32,
    )


def predict(image: Image.Image, title: str, score: float, comments: float, upvote_ratio: float) -> dict[str, float]:
    if image is None:
        raise gr.Error("请先上传一张新闻配图。")
    if not title or not title.strip():
        raise gr.Error("请输入新闻标题。")

    image_tensor = image_transform(image.convert("RGB")).unsqueeze(0).to(device)
    encoded = tokenizer(
        title,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    metadata = build_metadata(score, comments, upvote_ratio).to(device)

    with torch.no_grad():
        logits = model(
            images=image_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            metadata=metadata,
        )
        probabilities = torch.softmax(logits, dim=1).detach().cpu().squeeze(0)

    return {
        "假新闻": float(probabilities[0]),
        "真新闻": float(probabilities[1]),
    }


def create_demo() -> gr.Interface:
    return gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="新闻配图"),
            gr.Textbox(label="新闻标题", lines=2, placeholder="输入新闻标题"),
            gr.Number(label="Score", value=0),
            gr.Number(label="Comments", value=0),
            gr.Slider(label="Upvote Ratio", minimum=0.0, maximum=1.0, step=0.01, value=0.5),
        ],
        outputs=gr.Label(label="预测概率"),
        title="多模态虚假新闻检测演示",
        description="上传图片并输入标题与互动元数据，模型将输出真新闻/假新闻概率。",
        allow_flagging="never",
    )


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    create_demo().launch()
