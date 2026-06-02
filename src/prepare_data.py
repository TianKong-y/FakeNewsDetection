#!/usr/bin/env python3
"""
准备多模态训练数据：图片 + 文本（修复版）
"""

import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
MAPPING_DIR = PROCESSED_DIR / "mapping"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw_data" / "image_set"
OUTPUT_DIR = PROCESSED_DIR / "multimodal_training"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "images", exist_ok=True)

def load_mapping():
    """加载映射表"""
    df = pd.read_csv(MAPPING_DIR / "complete_multimodal_mapping.csv")
    print(f"加载映射表: {len(df)} 条记录")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n标签分布:")
    print(df['2_way_label'].value_counts())
    return df

def verify_images(df):
    """验证图片是否存在"""
    print("\n验证图片文件...")
    
    existing_images = []
    missing_images = []
    
    for img_id in df['image_id']:
        img_path = IMAGE_DIR / f"{img_id}.jpg"
        if os.path.exists(img_path):
            existing_images.append(img_id)
        else:
            missing_images.append(img_id)
    
    print(f"存在的图片: {len(existing_images)}")
    print(f"缺失的图片: {len(missing_images)}")
    
    # 只保留有图片的样本
    df_valid = df[df['image_id'].isin(existing_images)].copy()
    print(f"有效样本: {len(df_valid)}")
    
    return df_valid

def prepare_training_data(df):
    """准备训练数据"""
    print("\n准备训练数据...")
    
    # 查看可用列
    print(f"可用列: {df.columns.tolist()}")
    
    # 创建文本特征（使用可用的列）
    text_parts = []
    
    if 'title' in df.columns:
        text_parts.append(df['title'].fillna(''))
    
    # 如果没有title，使用其他列
    if not text_parts:
        if 'clean_title' in df.columns:
            text_parts.append(df['clean_title'].fillna(''))
        elif 'id' in df.columns:
            text_parts.append(df['id'].fillna(''))
    
    # 合并文本
    if text_parts:
        df['text'] = text_parts[0]
        for part in text_parts[1:]:
            df['text'] = df['text'] + " " + part
    else:
        df['text'] = "unknown"
    
    # 清理文本
    df['text'] = df['text'].str.replace('nan', '').str.strip()
    
    # 如果文本为空，使用默认值
    df.loc[df['text'] == '', 'text'] = 'no_text'
    
    # 打印统计
    print(f"文本长度统计:")
    print(f"  平均长度: {df['text'].str.len().mean():.2f}")
    print(f"  空文本数: {(df['text'] == 'no_text').sum()}")
    
    # 标签
    y = df['2_way_label']
    
    # 分割数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        df, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n数据分割:")
    print(f"  训练集: {len(X_train)}")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)}")
    
    # 保存分割结果
    X_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    X_val.to_csv(OUTPUT_DIR / "val.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "test.csv", index=False)
    
    # 复制图片到训练文件夹
    print("\n复制图片文件...")
    for split_name, split_df in [('train', X_train), ('val', X_val), ('test', X_test)]:
        split_img_dir = OUTPUT_DIR / "images" / split_name
        os.makedirs(split_img_dir, exist_ok=True)
        
        copied = 0
        for img_id in split_df['image_id']:
            src = IMAGE_DIR / f"{img_id}.jpg"
            dst = split_img_dir / f"{img_id}.jpg"
            if os.path.exists(src):
                shutil.copy(src, dst)
                copied += 1
        
        print(f"  {split_name}: {copied}/{len(split_df)} 张图片")
    
    return X_train, X_val, X_test

def generate_report(df_train, df_val, df_test):
    """生成报告"""
    report = []
    report.append("# 多模态训练数据集准备报告\n")
    report.append(f"**生成时间**: {pd.Timestamp.now()}\n")
    report.append("## 数据统计\n")
    report.append(f"| 数据集 | 样本数 | 标签0 (假新闻) | 标签1 (真新闻) |")
    report.append(f"|--------|--------|----------------|----------------|")
    report.append(f"| 训练集 | {len(df_train)} | {(df_train['2_way_label']==0).sum()} | {(df_train['2_way_label']==1).sum()} |")
    report.append(f"| 验证集 | {len(df_val)} | {(df_val['2_way_label']==0).sum()} | {(df_val['2_way_label']==1).sum()} |")
    report.append(f"| 测试集 | {len(df_test)} | {(df_test['2_way_label']==0).sum()} | {(df_test['2_way_label']==1).sum()} |")
    
    report.append("\n## 文本示例\n")
    report.append("| 标签 | 文本预览 |")
    report.append("|------|----------|")
    for _, row in df_train.head(5).iterrows():
        label = "假新闻" if row['2_way_label'] == 0 else "真新闻"
        text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
        report.append(f"| {label} | {text_preview} |")
    
    report.append("\n## 输出文件\n")
    report.append("| 文件 | 说明 |")
    report.append("|------|------|")
    report.append("| train.csv | 训练集元数据（含文本和标签）|")
    report.append("| val.csv | 验证集元数据 |")
    report.append("| test.csv | 测试集元数据 |")
    report.append("| images/train/ | 训练集图片 |")
    report.append("| images/val/ | 验证集图片 |")
    report.append("| images/test/ | 测试集图片 |")
    
    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ 报告已保存: {report_path}")

def show_sample(df_train):
    """显示样本"""
    print("\n" + "=" * 60)
    print("训练样本示例")
    print("=" * 60)
    
    for _, row in df_train.head(5).iterrows():
        image_path = OUTPUT_DIR / "images" / "train" / f"{row['image_id']}.jpg"
        print(f"\n图片ID: {row['image_id']}")
        print(f"标签: {row['2_way_label']} ({'假新闻' if row['2_way_label']==0 else '真新闻'})")
        print(f"文本: {row['text'][:150]}...")
        print(f"图片路径: {image_path}")

def main():
    print("=" * 60)
    print("准备多模态训练数据")
    print("=" * 60)
    
    # 加载映射表
    df = load_mapping()
    
    # 验证图片
    df_valid = verify_images(df)
    
    # 准备训练数据
    train, val, test = prepare_training_data(df_valid)
    
    # 生成报告
    generate_report(train, val, test)
    
    # 显示样本
    show_sample(train)
    
    print("\n" + "=" * 60)
    print("✅ 多模态训练数据准备完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("\n下一步: 开始多模态模型训练！")
    print("=" * 60)

if __name__ == "__main__":
    main()
