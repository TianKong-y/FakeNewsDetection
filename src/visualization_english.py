#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fake News Detection - Data Visualization (English)
使用最终的多模态训练数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Paths
PROJECT_ROOT = "/Users/bessie/Desktop/fake_news_detection"
MAPPING_FILE = f"{PROJECT_ROOT}/processed/mapping/complete_multimodal_mapping.csv"
OUTPUT_DIR = f"{PROJECT_ROOT}/output/figures/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """加载最终映射数据"""
    df = pd.read_csv(MAPPING_FILE)
    print(f"加载数据: {len(df)} 条记录")
    return df

def print_statistics(df):
    """打印关键统计"""
    print("\n" + "=" * 60)
    print("KEY STATISTICS")
    print("=" * 60)
    
    print(f"\n[Dataset Overview]")
    print(f"  Total samples: {len(df)}")
    print(f"  Fake news (label=0): {(df['2_way_label']==0).sum()}")
    print(f"  Real news (label=1): {(df['2_way_label']==1).sum()}")
    
    print(f"\n[Text Features]")
    df['text'] = df['title'].fillna('') + " " + df['subreddit'].fillna('')
    df['text_len'] = df['text'].str.len()
    print(f"  Avg text length: {df['text_len'].mean():.2f} chars")
    
    print(f"\n[Article Features]")
    print(f"  Avg score: {df['score'].mean():.2f}")
    print(f"  Avg upvote_ratio: {df['upvote_ratio'].mean():.2f}")
    
    print(f"\n[Key Findings]")
    fake_avg_score = df[df['2_way_label']==0]['score'].mean()
    real_avg_score = df[df['2_way_label']==1]['score'].mean()
    if fake_avg_score > 0:
        print(f"  Real news have {real_avg_score/fake_avg_score:.1f}x higher average score than fake news")
    
    fake_avg_comments = df[df['2_way_label']==0]['num_comments'].mean()
    real_avg_comments = df[df['2_way_label']==1]['num_comments'].mean()
    if fake_avg_comments > 0:
        print(f"  Real news have {real_avg_comments/fake_avg_comments:.1f}x more comments than fake news")

def create_visualizations(df):
    """生成可视化图表"""
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # 准备文本数据
    df['text'] = df['title'].fillna('') + " " + df['subreddit'].fillna('')
    df['text_len'] = df['text'].str.len()
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    
    # Figure 1: Label Distribution (Pie Chart)
    ax1 = fig.add_subplot(2, 2, 1)
    label_counts = df['2_way_label'].value_counts()
    colors_pie = ['#ff6b6b', '#51cf66']
    labels_pie = ['Fake News', 'Real News']
    ax1.pie(label_counts.values, labels=labels_pie, colors=colors_pie, 
            autopct='%1.1f%%', startangle=90, explode=(0.05, 0), shadow=True)
    ax1.set_title('Figure 1: Label Distribution', fontsize=14, fontweight='bold')
    
    # Figure 2: Text Length Distribution (Histogram)
    ax2 = fig.add_subplot(2, 2, 2)
    for label, color, name in zip([0, 1], ['#ff6b6b', '#51cf66'], ['Fake News', 'Real News']):
        subset = df[df['2_way_label'] == label]
        ax2.hist(subset['text_len'], bins=30, alpha=0.6, color=color, label=name, edgecolor='black')
    ax2.set_xlabel('Text Length (characters)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Figure 2: Text Length Distribution by Label', fontsize=14, fontweight='bold')
    ax2.legend()
    
    # Figure 3: Score Comparison (Box Plot)
    ax3 = fig.add_subplot(2, 2, 3)
    score_data = [df[df['2_way_label'] == 0]['score'].dropna(),
                  df[df['2_way_label'] == 1]['score'].dropna()]
    bp = ax3.boxplot(score_data, labels=['Fake News', 'Real News'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#ff6b6b', '#51cf66']):
        patch.set_facecolor(color)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Figure 3: Score Comparison by Label', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    
    # Figure 4: Average Features Comparison (Bar Chart)
    ax4 = fig.add_subplot(2, 2, 4)
    stats = []
    for label in [0, 1]:
        subset = df[df['2_way_label'] == label]
        stats.append({
            'avg_score': subset['score'].mean(),
            'avg_comments': subset['num_comments'].mean()
        })
    
    x = np.arange(2)
    width = 0.35
    bars1 = ax4.bar(x - width/2, [s['avg_score'] for s in stats], width, label='Average Score', color='#3498db')
    bars2 = ax4.bar(x + width/2, [s['avg_comments'] for s in stats], width, label='Average Comments', color='#e74c3c')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Fake News', 'Real News'])
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Figure 4: Article Features Comparison', fontsize=14, fontweight='bold')
    ax4.legend()
    
    # Add value labels
    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Fake News Detection - Data Visualization Report', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = f"{OUTPUT_DIR}/visualization_report_english.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✅ Figure saved: {output_path}")

def create_additional_plots(df):
    """创建附加图表"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Score distribution by label
    ax1 = axes[0]
    for label, color, name in zip([0, 1], ['#ff6b6b', '#51cf66'], ['Fake News', 'Real News']):
        subset = df[df['2_way_label'] == label]['score'].dropna()
        ax1.hist(subset, bins=50, alpha=0.6, color=color, label=name, edgecolor='black')
    ax1.set_xlabel('Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Score Distribution by Label', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xscale('log')
    
    # Plot 2: Upvote ratio distribution
    ax2 = axes[1]
    for label, color, name in zip([0, 1], ['#ff6b6b', '#51cf66'], ['Fake News', 'Real News']):
        subset = df[df['2_way_label'] == label]['upvote_ratio'].dropna()
        ax2.hist(subset, bins=30, alpha=0.6, color=color, label=name, edgecolor='black', density=True)
    ax2.set_xlabel('Upvote Ratio', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Upvote Ratio Distribution by Label', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/additional_plots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✅ Additional plots saved: {output_path}")

def main():
    print("=" * 60)
    print("Fake News Detection - Data Visualization (English)")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Print statistics
    print_statistics(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Create additional plots
    create_additional_plots(df)
    
    print("\n" + "=" * 60)
    print("✅ All visualizations completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
