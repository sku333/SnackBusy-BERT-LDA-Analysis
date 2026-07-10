# 硬折扣零食店消费者评论深度学习分析

**Deep Learning Analysis of Consumer Reviews for Hard-Discount Snack Stores**

> 论文《"食"分心动：硬折扣解锁上海青年的零食新选择》第三章方法升级实现  
> Methodology upgrade for Chapter 3: from traditional NLP to deep learning pipeline

---

## 项目概述

将传统 NLP 分析管道（Jieba + SnowNLP + LDA）升级为深度学习方法，在同一数据集上形成**新旧方法对比**。

| 分析模块 | 原方法 | 升级方法 |
|---------|--------|---------|
| 情感分析 | SnowNLP (朴素贝叶斯) | RoBERTa (电商领域微调) |
| 主题挖掘 | Jieba + LDA | BERT 语义聚类 + KMeans + LDA |
| 词云切词 | Jieba | Jieba × BERT 词表筛选 |

## 技术栈

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

- **情感分析模型**: [`uer/roberta-base-finetuned-jd-binary-chinese`](https://huggingface.co/uer/roberta-base-finetuned-jd-binary-chinese)
- **语义向量模型**: [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **主题建模**: scikit-learn LDA + KMeans
- **可视化**: matplotlib + WordCloud

## 数据集

- 来源：大众点评、美团（上海零食很忙、好想来门店）
- 规模：1,377 条用户评论
- 清洗后：~1,375 条有效评论

## 结果对比

### 情感分析

| 指标 | 原方法 (SnowNLP) | 新方法 (RoBERTa) |
|------|----------------|----------------|
| 正向 | 73.8% | **89.8%** |
| 中性 | 3.9% | N/A（二分类）|
| 负向 | 22.3% | **10.2%** |
| 平均置信度 | 无 | **96.93%** |

> RoBERTa 识别出更高比例的正向评论，SnowNLP 对口语化模糊表达存在误判，将部分正向评论归为负向。

### 主题挖掘对比

**好评主题：**

| 主题 | 原方法 (LDA) | 两次LDA均发现 |
|------|------------|------------|
| 主题1 | 价格、便宜、性价比、方便、不错 | 价格优势是核心驱动 |
| 主题2 | 环境、服务、品种、实惠、整洁 | 门店体验影响满意度 |
| 主题3 | 好吃、口味、品类、丰富、味道 | 产品口味决定复购 |

**差评新发现（BERT独有）：**

BERT 语义聚类识别出原 LDA 未发现的细粒度主题：

- **包装开封困难**（关键词：剪刀、撕开、果冻、勺子）— 部分产品需借助工具才能开封
- **排队秩序问题**（关键词：插队、队伍、核销）— 高峰期收银区秩序管理不足

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行分析（自动从 HF 镜像下载模型）
python main.py

# 3. 使用本地模型（无网络环境）
python main.py --local
```

详细说明见 [升级方案说明.md](升级方案说明.md)

## 输出文件

```
output/
├── sentiment_comparison.png        # 情感分布对比图 (新旧方法)
├── sentiment_results.csv           # 每条评论情感标签 + 置信度
├── topic_comparison.txt            # BERT+KMeans+LDA 主题词
├── topic_comparison_pure_lda.txt   # 纯 LDA 主题词（对比基准）
├── wordcloud_bert_tokenizer.png    # 词云图（BERT词表筛选）
├── word_frequency_bert_tokenizer.txt
└── summary_report.txt              # 综合分析报告
```

## 运行环境

- Python 3.8+
- CPU 推理（无需 GPU），建议 8GB+ RAM
- 首次运行需下载模型约 520MB
