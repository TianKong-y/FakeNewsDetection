#!/usr/bin/env python3
"""
完整匹配：TSV索引 + 本地图片 + 评论
"""

import pandas as pd
import os
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TSV_DIR = PROJECT_ROOT / "data" / "raw_tsv"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw_data" / "image_set"
OUTPUT_DIR = PROJECT_ROOT / "processed" / "mapping"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_tsv_files():
    """查找所有TSV文件"""
    print("=" * 60)
    print("1. 查找TSV文件")
    print("=" * 60)
    
    # 查找所有TSV文件
    tsv_files = glob.glob(str(TSV_DIR / "**" / "*.tsv"), recursive=True)
    
    if not tsv_files:
        tsv_files = glob.glob(str(TSV_DIR / "*.tsv"))
    
    print(f"找到 {len(tsv_files)} 个TSV文件:")
    for f in tsv_files:
        print(f"  - {os.path.basename(f)}")
    
    return tsv_files

def load_all_tsv_files(tsv_files):
    """加载所有TSV文件"""
    print("\n" + "=" * 60)
    print("2. 加载TSV文件")
    print("=" * 60)
    
    dataframes = {}
    
    for tsv_file in tsv_files:
        name = os.path.basename(tsv_file).replace('.tsv', '')
        print(f"加载: {name}")
        
        try:
            df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
            print(f"  行数: {len(df)}")
            print(f"  列数: {len(df.columns)}")
            print(f"  列名: {df.columns.tolist()[:10]}...")
            dataframes[name] = df
        except Exception as e:
            print(f"  错误: {e}")
    
    return dataframes

def analyze_columns(dataframes):
    """分析所有TSV的列结构"""
    print("\n" + "=" * 60)
    print("3. 分析列结构")
    print("=" * 60)
    
    for name, df in dataframes.items():
        print(f"\n【{name}】")
        print(f"  总列数: {len(df.columns)}")
        
        # 查找关键字段
        key_fields = ['id', 'image_id', 'title', 'text', 'comment', 
                      'subreddit', '2_way_label', '3_way_label', 
                      'num_comments', 'author']
        
        for field in key_fields:
            if field in df.columns:
                print(f"  ✅ 包含: {field}")
        
        # 显示前几行
        print(f"\n  前2行预览:")
        print(df.head(2).to_string())
        print("-" * 40)

def match_with_local_images(dataframes):
    """匹配本地图片"""
    print("\n" + "=" * 60)
    print("4. 匹配本地图片")
    print("=" * 60)
    
    # 获取本地图片ID列表
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    image_ids = set([f.replace('.jpg', '') for f in image_files])
    print(f"本地图片总数: {len(image_ids)}")
    print(f"图片ID示例: {list(image_ids)[:10]}")
    
    # 尝试在每个数据框中匹配
    all_matches = []
    
    for name, df in dataframes.items():
        print(f"\n尝试匹配: {name}")
        
        # 找出可能的图片ID列
        possible_id_cols = [c for c in df.columns if 'image' in c.lower() or c == 'id']
        
        for id_col in possible_id_cols:
            # 检查该列的值是否与图片ID匹配
            if id_col not in df.columns:
                continue
            
            # 转换为字符串并匹配
            df_ids = set(df[id_col].dropna().astype(str).tolist())
            matches = image_ids.intersection(df_ids)
            
            if len(matches) > 0:
                print(f"   ✅ 通过列 '{id_col}' 匹配到 {len(matches)} 个图片")
                
                # 提取匹配的行
                matched_df = df[df[id_col].astype(str).isin(matches)].copy()
                matched_df['matched_image_id'] = matched_df[id_col].astype(str)
                matched_df['source_file'] = name
                matched_df['match_column'] = id_col
                
                all_matches.append(matched_df)
                break
        else:
            print(f"   ❌ 未找到匹配的图片ID")
    
    if all_matches:
        combined_matches = pd.concat(all_matches, ignore_index=True)
        print(f"\n总匹配数: {len(combined_matches)}")
        return combined_matches
    else:
        print("\n❌ 没有找到任何匹配的图片")
        return None

def analyze_image_id_pattern(image_ids, dataframes):
    """分析图片ID的规律"""
    print("\n" + "=" * 60)
    print("5. 分析图片ID规律")
    print("=" * 60)
    
    print(f"本地图片ID长度分布:")
    lengths = [len(x) for x in image_ids]
    print(f"  最小: {min(lengths)}, 最大: {max(lengths)}, 平均: {sum(lengths)/len(lengths):.1f}")
    
    # 检查各个数据框中的ID格式
    for name, df in dataframes.items():
        if 'id' in df.columns:
            sample_ids = df['id'].dropna().astype(str).head(20).tolist()
            sample_lengths = [len(x) for x in sample_ids]
            print(f"\n{name} 中的id列长度: 平均 {sum(sample_lengths)/len(sample_lengths):.1f}")
            print(f"  示例: {sample_ids[:5]}")

def create_final_mapping(matched_df):
    """创建最终映射表"""
    print("\n" + "=" * 60)
    print("6. 创建最终映射表")
    print("=" * 60)
    
    if matched_df is None or len(matched_df) == 0:
        print("没有匹配数据，无法创建映射表")
        return
    
    # 选择关键列
    important_cols = ['matched_image_id', 'id', 'title', 'text', 
                      'subreddit', '2_way_label', '3_way_label', 
                      'num_comments', 'score', 'upvote_ratio', 
                      'source_file', 'match_column']
    
    existing_cols = [c for c in important_cols if c in matched_df.columns]
    final_df = matched_df[existing_cols].copy()
    
    # 重命名列
    final_df = final_df.rename(columns={
        'matched_image_id': 'image_id',
        'id': 'article_id'
    })
    
    # 保存
    output_path = OUTPUT_DIR / "complete_multimodal_mapping.csv"
    final_df.to_csv(output_path, index=False)
    print(f"✅ 完整映射表已保存: {output_path}")
    print(f"   总样本数: {len(final_df)}")
    
    if '2_way_label' in final_df.columns:
        print(f"\n标签分布:")
        print(final_df['2_way_label'].value_counts())
    
    print(f"\n示例映射:")
    print(final_df.head(10).to_string())
    
    return final_df

def main():
    print("\n" + "=" * 60)
    print("  完整图片-文章-评论匹配系统")
    print("=" * 60)
    
    # 1. 查找TSV文件
    tsv_files = find_tsv_files()
    
    if not tsv_files:
        print("\n❌ 未找到TSV文件，请先解压文件到 data/raw_tsv/")
        print("解压命令:")
        print("  unzip ~/Desktop/raw_data/your_file.zip -d data/raw_tsv/")
        return
    
    # 2. 加载TSV文件
    dataframes = load_all_tsv_files(tsv_files)
    
    # 3. 分析列结构
    analyze_columns(dataframes)
    
    # 4. 获取本地图片ID
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    image_ids = set([f.replace('.jpg', '') for f in image_files])
    
    # 5. 分析ID规律
    analyze_image_id_pattern(image_ids, dataframes)
    
    # 6. 匹配图片
    matched_df = match_with_local_images(dataframes)
    
    # 7. 创建最终映射
    final_mapping = create_final_mapping(matched_df)
    
    print("\n" + "=" * 60)
    print("✅ 匹配完成！")
    print(f"输出位置: {OUTPUT_DIR / 'complete_multimodal_mapping.csv'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
