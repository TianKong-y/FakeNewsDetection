# 多模态训练数据集准备报告

**生成时间**: 2026-06-01 18:05:30.262809

## 数据统计

| 数据集 | 样本数 | 标签0 (假新闻) | 标签1 (真新闻) |
|--------|--------|----------------|----------------|
| 训练集 | 7000 | 4567 | 2433 |
| 验证集 | 1500 | 978 | 522 |
| 测试集 | 1500 | 979 | 521 |

## 文本示例

| 标签 | 文本预览 |
|------|----------|
| 假新闻 | Jar Jar is a cute little fellow psbattle_artwork |
| 假新闻 | ...And all liars, thieves, burglars, murderers, slanderers, and evil-doers will ... |
| 真新闻 | PsBattle: Jonathan Stewart enters the field photoshopbattles |
| 真新闻 | Stephen Miller Is an Immigration Hypocrite. I Know Because I’m His Uncle. usnews |
| 真新闻 | The text wasn’t backwards on the other side, also notice that the Warner Bros lo... |

## 输出文件

| 文件 | 说明 |
|------|------|
| train.csv | 训练集元数据（含文本和标签）|
| val.csv | 验证集元数据 |
| test.csv | 测试集元数据 |
| images/train/ | 训练集图片 |
| images/val/ | 验证集图片 |
| images/test/ | 测试集图片 |