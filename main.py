# -*- coding: utf-8 -*-
"""
=============================================================================
硬折扣零食店消费者评论分析 — 深度学习方法升级版
=============================================================================
原方法: Jieba + SnowNLP + LDA
新方法: RoBERTa (情感分析) + BERT+KMeans+LDA (主题挖掘) + BertTokenizer (词云)

用法:
    python main.py                    # 使用 HF 镜像自动下载模型
    python main.py --local            # 使用本地已下载的模型
    python main.py --csv <路径>       # 指定 CSV 文件路径
=============================================================================
"""

import os
import sys
import re
import time
import json
import argparse
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适配 Linux 服务器
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

from wordcloud import WordCloud
import jieba  # 保留 Jieba 作为分词对比基准

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ======================== 导入配置 ========================
from config import (
    CSV_FILE, OUTPUT_DIR, HF_ENDPOINT, LOCAL_MODEL_DIR,
    SENTIMENT_MODEL_NAME, EMBEDDING_MODEL_NAME, TOKENIZER_MODEL_NAME,
    POSITIVE_THRESHOLD, MAX_SEQ_LENGTH, BATCH_SIZE,
    N_TOPICS_POS, N_TOPICS_NEG, N_TOPIC_KEYWORDS, RANDOM_SEED,
    WORDCLOUD_MAX_WORDS, WORDCLOUD_WIDTH, WORDCLOUD_HEIGHT,
    LINUX_FONT_PATH, STOPWORDS,
    ORIGINAL_SENTIMENT, ORIGINAL_POS_TOPICS, ORIGINAL_NEG_TOPICS,
)

# ======================== 工具函数 ========================

def load_data(csv_path):
    """加载 CSV 数据，自动尝试多种编码"""
    encodings = ["gbk", "gb18030", "utf-8-sig", "utf-8"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"[OK] 成功使用编码 '{enc}' 读取文件，共 {len(df)} 条记录")
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue

    if df is None:
        raise RuntimeError(f"无法读取文件: {csv_path}，请检查文件路径和编码。")

    # 兼容不同列名: "评论内容" 或 CSV 的第一列
    col = "评论内容"
    if col not in df.columns:
        col = df.columns[0]
        print(f"[INFO] 未找到列名 '评论内容', 使用第一列 '{col}'")

    texts = df[col].astype(str).tolist()
    print(f"[INFO] 共加载 {len(texts)} 条原始评论 (含空值)")
    return texts


def clean_text(text):
    """清洗文本：去除纯符号/数字评论，保留有中文内容的文本"""
    if not text or text.strip() in ["", "nan", "NaN", "None"]:
        return ""

    # 去除平台自动生成的模板评论 (如 "用户没有填写评价" 等)
    template_patterns = [
        r"^用户没有填写评价.*",
        r"^此用户没有填写评价.*",
        r"^系统默认.*",
        r"^.{0,3}$",  # 少于3个字符
    ]
    text_stripped = text.strip()
    for pat in template_patterns:
        if re.match(pat, text_stripped):
            return ""

    # 提取中文字符
    chinese_chars = re.findall(r"[\u4e00-\u9fa5]", text_stripped)
    if len(chinese_chars) < 3:
        return ""

    return text_stripped


def clean_data(texts):
    """批量清洗文本，返回有效文本列表及其索引"""
    cleaned = []
    valid_indices = []
    stats = {"total": len(texts), "empty_or_short": 0, "template": 0, "valid": 0}

    for i, t in enumerate(texts):
        ct = clean_text(t)
        if ct:
            cleaned.append(ct)
            valid_indices.append(i)
        else:
            stats["empty_or_short"] += 1

    stats["valid"] = len(cleaned)
    print(f"[INFO] 数据清洗完成: {stats['valid']}/{stats['total']} 条有效"
          f" (剔除 {stats['empty_or_short']} 条无效)")
    return cleaned, valid_indices


def find_chinese_font():
    """在 Linux/Windows/macOS 上自动查找可用的中文字体"""
    import glob as _glob

    candidates = [
        LINUX_FONT_PATH,
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]

    # 尝试 glob 搜索
    search_dirs = [
        "/usr/share/fonts",
        "/System/Library/Fonts",
        "C:/Windows/Fonts",
    ]
    for d in search_dirs:
        if os.path.isdir(d):
            for pat in ["*.ttf", "*.ttc", "*.otf"]:
                for fp in _glob.glob(os.path.join(d, "**", pat), recursive=True):
                    candidates.append(fp)

    for fp in candidates:
        if os.path.exists(fp):
            print(f"[OK] 找到中文字体: {fp}")
            return fp

    print("[WARN] 未找到中文字体，词云可能无法正常显示中文！")
    return None


def setup_hf_environment(use_local=False):
    """配置 Hugging Face 环境变量 (镜像站) — 必须在导入 transformers 前调用"""
    if use_local and LOCAL_MODEL_DIR and os.path.isdir(LOCAL_MODEL_DIR):
        os.environ["HF_ENDPOINT"] = ""
        os.environ["HF_HUB_ENDPOINT"] = ""
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"[INFO] 使用本地模型目录: {LOCAL_MODEL_DIR}")
    else:
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT
        os.environ["HF_HUB_ENDPOINT"] = HF_ENDPOINT
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = HF_ENDPOINT
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
        print(f"[INFO] 使用 HF 镜像站: {HF_ENDPOINT}")


def load_models_with_retry(use_local=False):
    """
    加载所有模型，支持重试机制。
    返回: (sentiment_tokenizer, sentiment_model, embedding_model)
    """
    max_retries = 3
    retry_delay = 5

    # ---- 情感分析模型 (transformers) ----
    for attempt in range(max_retries):
        try:
            print(f"\n[INFO] 加载情感分析模型: {SENTIMENT_MODEL_NAME} (尝试 {attempt+1}/{max_retries})")

            if use_local and LOCAL_MODEL_DIR:
                sent_path = os.path.join(LOCAL_MODEL_DIR, "roberta-base-finetuned-jd-binary-chinese")
            else:
                sent_path = SENTIMENT_MODEL_NAME

            sentiment_tokenizer = AutoTokenizer.from_pretrained(
                sent_path,
                trust_remote_code=True,
                use_fast=True,
            )
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                sent_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            sentiment_model.eval()
            id2l = getattr(sentiment_model.config, "id2label", {})
            l2id = getattr(sentiment_model.config, "label2id", {})
            print(f"  [OK] 情感分析模型加载成功")
            print(f"  [INFO] 模型标签映射: id2label={id2l}, label2id={l2id}")
            break
        except Exception as e:
            print(f"  [ERR] 情感分析模型加载失败: {e}")
            if attempt < max_retries - 1:
                print(f"  [RETRY] {retry_delay}s 后重试...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"情感分析模型加载失败，已重试 {max_retries} 次。\n"
                    f"请检查:\n"
                    f"  1. 网络: curl -I {HF_ENDPOINT}\n"
                    f"  2. 模型名是否正确: {SENTIMENT_MODEL_NAME}\n"
                    f"  3. 或使用本地模型: python main.py --local"
                )

    # ---- 句子向量模型 (sentence-transformers) ----
    # sentence-transformers 也通过 HF_HUB_ENDPOINT 走镜像，但需要特殊处理
    for attempt in range(max_retries):
        try:
            print(f"[INFO] 加载句子向量模型: {EMBEDDING_MODEL_NAME} (尝试 {attempt+1}/{max_retries})")

            if use_local and LOCAL_MODEL_DIR:
                emb_path = os.path.join(LOCAL_MODEL_DIR, "paraphrase-multilingual-MiniLM-L12-v2")
            else:
                emb_path = EMBEDDING_MODEL_NAME

            # 强制 sentence-transformers 使用镜像
            embedding_model = SentenceTransformer(
                emb_path,
                device="cpu",
            )
            print(f"  [OK] 句子向量模型加载成功")
            break
        except Exception as e:
            print(f"  [ERR] 句子向量模型加载失败: {e}")
            if attempt < max_retries - 1:
                print(f"  [RETRY] {retry_delay}s 后重试...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"句子向量模型加载失败，已重试 {max_retries} 次。\n"
                    f"请检查:\n"
                    f"  1. 网络: curl -I {HF_ENDPOINT}\n"
                    f"  2. 模型名是否正确: {EMBEDDING_MODEL_NAME}\n"
                    f"  3. 或使用本地模型: python main.py --local\n"
                    f"  4. 或手动下载: pip install huggingface_hub && "
                    f"huggingface-cli download {EMBEDDING_MODEL_NAME} "
                    f"--local-dir ./local_models/paraphrase-multilingual-MiniLM-L12-v2"
                )

    return sentiment_tokenizer, sentiment_model, embedding_model


# ======================== 1. 情感分析 ========================

def run_sentiment_analysis(texts, tokenizer, model):
    """
    使用 RoBERTa 进行情感分析，返回每条评论的标签和置信度。
    模型为二分类 (positive/negative)，无中性类别。
    """
    print(f"\n{'='*60}")
    print("第一步：RoBERTa 情感分析")
    print(f"{'='*60}")

    results = []
    total = len(texts)

    for start in range(0, total, BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        # 处理空文本
        valid_texts = []
        empty_indices = []
        for j, t in enumerate(batch):
            if t and t.strip():
                valid_texts.append(t)
            else:
                empty_indices.append(j)

        if not valid_texts:
            for j in range(len(batch)):
                results.append({"label": "unknown", "confidence": 0.0})
            continue

        # Tokenize
        inputs = tokenizer(
            valid_texts,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # CPU 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        confidences = probs.max(dim=-1).values.numpy()
        pred_labels = preds.numpy()

        # 映射标签: 模型可能输出中文/英文/数字等各种格式
        id2label = model.config.id2label if hasattr(model.config, "id2label") else {}
        if not id2label or len(id2label) == 0:
            id2label = {0: "negative", 1: "positive"}

        # 规范化 id2label: 支持中文"正向"/"负向", 英文"positive"/"negative", 数字等
        def _normalize_label(raw_label):
            label_str = str(raw_label).lower().strip()
            # 中文匹配
            if any(w in label_str for w in ["正向", "正面", "积极", "好评", "pos", "positive"]):
                return "positive"
            if any(w in label_str for w in ["负向", "负面", "消极", "差评", "neg", "negative"]):
                return "negative"
            # 数字匹配: label_id=0通常为负, 1为正(取决于模型, 这里取常见约定)
            # 无法确定时, 用原始标签
            return "unknown"

        # 重建完整结果 (含空文本)
        result_idx = 0
        for j in range(len(batch)):
            if j in empty_indices:
                results.append({"label": "unknown", "confidence": 0.0})
            else:
                label_id = int(pred_labels[result_idx])
                raw_label = id2label.get(label_id, id2label.get(str(label_id), str(label_id)))
                label_str = _normalize_label(raw_label)
                conf = float(confidences[result_idx])
                results.append({"label": label_str, "confidence": round(conf, 4)})
                result_idx += 1

        if (start + BATCH_SIZE) % (BATCH_SIZE * 5) == 0 or start + BATCH_SIZE >= total:
            print(f"  进度: {min(start + BATCH_SIZE, total)}/{total}")

    # 统计分布
    pos_count = sum(1 for r in results if r["label"] == "positive")
    neg_count = sum(1 for r in results if r["label"] == "negative")
    unk_count = sum(1 for r in results if r["label"] == "unknown")

    total_valid = pos_count + neg_count
    print(f"\n  情感分析分布 (新方法 - RoBERTa 二分类):")
    print(f"    正向: {pos_count} ({pos_count/total_valid*100:.1f}%)" if total_valid else "    正向: 0")
    print(f"    负向: {neg_count} ({neg_count/total_valid*100:.1f}%)" if total_valid else "    负向: 0")
    if unk_count:
        print(f"    未知: {unk_count}")

    return results


def plot_sentiment_comparison(results, output_dir):
    """绘制新旧方法情感分布对比图"""
    pos = sum(1 for r in results if r["label"] == "positive")
    neg = sum(1 for r in results if r["label"] == "negative")
    total = pos + neg

    new_pos_pct = pos / total * 100 if total else 0
    new_neg_pct = neg / total * 100 if total else 0

    # 新方法没有中性，将原论文的中性拆分到正负
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 原论文饼图
    ax1 = axes[0]
    old_sizes = [ORIGINAL_SENTIMENT["positive"], ORIGINAL_SENTIMENT["neutral"],
                 ORIGINAL_SENTIMENT["negative"]]
    old_labels = [f'正向\n{ORIGINAL_SENTIMENT["positive"]}%',
                  f'中性\n{ORIGINAL_SENTIMENT["neutral"]}%',
                  f'负向\n{ORIGINAL_SENTIMENT["negative"]}%']
    old_colors = ["#4CAF50", "#FFC107", "#F44336"]
    ax1.pie(old_sizes, labels=old_labels, colors=old_colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 11})
    ax1.set_title("原方法 (SnowNLP)", fontsize=14, fontweight="bold")

    # 新方法饼图
    ax2 = axes[1]
    new_sizes = [new_pos_pct, new_neg_pct]
    new_labels = [f'正向\n{new_pos_pct:.1f}%', f'负向\n{new_neg_pct:.1f}%']
    new_colors = ["#4CAF50", "#F44336"]
    ax2.pie(new_sizes, labels=new_labels, colors=new_colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 11})
    ax2.set_title(f"新方法 (RoBERTa 二分类)", fontsize=14, fontweight="bold")

    plt.suptitle("情感分布对比: SnowNLP vs RoBERTa", fontsize=16, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "sentiment_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] 情感对比图已保存: {path}")

    return new_pos_pct, new_neg_pct


# ======================== 2. BERT + KMeans + LDA 主题挖掘 ========================

def run_topic_modeling(texts, sentiment_results, embedding_model, output_dir):
    """
    使用 BERT + KMeans + LDA 混合方法进行主题挖掘。
    流程:
      1. 用 sentence-transformers 生成句子向量
      2. 用 KMeans 聚类得到主题分组
      3. 对每个聚类用 LDA (sklearn) 提取关键词
    """
    print(f"\n{'='*60}")
    print("第二步：BERT + KMeans + LDA 主题挖掘")
    print(f"{'='*60}")

    # 划分正/负评论文本
    pos_texts = [t for t, r in zip(texts, sentiment_results) if r["label"] == "positive"]
    neg_texts = [t for t, r in zip(texts, sentiment_results) if r["label"] == "negative"]

    print(f"[INFO] 正向评论 {len(pos_texts)} 条, 负向评论 {len(neg_texts)} 条")

    all_topic_results = {}

    for label, subset_texts, n_topics in [
        ("positive", pos_texts, N_TOPICS_POS),
        ("negative", neg_texts, N_TOPICS_NEG),
    ]:
        label_cn = "好评" if label == "positive" else "差评"
        print(f"\n--- {label_cn} BERT+LDA 主题挖掘 (目标 {n_topics} 个主题) ---")

        if len(subset_texts) < n_topics * 10:
            print(f"  [WARN] {label_cn}样本量过少 ({len(subset_texts)}), 跳过主题挖掘")
            all_topic_results[label] = []
            continue

        # Step 1: 生成句子向量
        print(f"  生成句子向量中...")
        embeddings = embedding_model.encode(
            subset_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(f"  句子向量维度: {embeddings.shape}")

        # Step 2: KMeans 聚类
        print(f"  KMeans 聚类 (k={n_topics})...")
        kmeans = KMeans(n_clusters=n_topics, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # 统计每个聚类的样本数
        cluster_counts = Counter(cluster_labels)
        for cid in sorted(cluster_counts):
            print(f"    聚类 {cid+1}: {cluster_counts[cid]} 条评论")

        # Step 3: 对每个聚类用 LDA 提取关键词
        topic_keywords = {}
        for cid in range(n_topics):
            cluster_texts = [subset_texts[i] for i in range(len(subset_texts))
                           if cluster_labels[i] == cid]

            if len(cluster_texts) < 5:
                print(f"    聚类 {cid+1}: 样本不足，跳过")
                continue

            # 用 Jieba 分词供 LDA 使用 (LDA 需要词级别的 token)
            cluster_words = []
            for t in cluster_texts:
                words = jieba.lcut(t)
                valid_words = [w for w in words if len(w) >= 2
                             and w not in STOPWORDS
                             and re.search(r"[\u4e00-\u9fa5]", w)]
                cluster_words.append(" ".join(valid_words))

            # CountVectorizer
            vectorizer = CountVectorizer(max_df=0.8, min_df=2, max_features=1000)
            try:
                dtm = vectorizer.fit_transform(cluster_words)
            except ValueError:
                print(f"    聚类 {cid+1}: 词汇表为空，跳过")
                continue

            # LDA (每个聚类视为一个"文档集合"，内部做 1-topic LDA 提取关键词)
            # 实际做法: 对聚类内的所有文档做多 topic LDA 也行，但更简单的方法是用 TF-IDF 提取关键词
            # 这里我们使用 sklearn LDA
            if dtm.shape[0] < 5 or dtm.shape[1] < 5:
                print(f"    聚类 {cid+1}: 文档或词汇不足，跳过")
                continue

            lda = LatentDirichletAllocation(
                n_components=1,
                random_state=RANDOM_SEED,
                max_iter=10,
            )
            lda.fit(dtm)

            # 提取 Top-N 关键词
            feature_names = vectorizer.get_feature_names_out()
            topic_word_dist = lda.components_[0]
            top_indices = topic_word_dist.argsort()[-N_TOPIC_KEYWORDS:][::-1]
            keywords = [feature_names[i] for i in top_indices]

            topic_keywords[f"聚类{cid+1}"] = keywords
            print(f"    聚类 {cid+1}: {', '.join(keywords)}")

        all_topic_results[label] = topic_keywords

    return all_topic_results


def save_topic_comparison(all_topic_results, output_dir):
    """保存新旧主题关键词对比"""
    path = os.path.join(output_dir, "topic_comparison.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("主题关键词对比: 原论文 (Jieba+LDA) vs 新方法 (BERT+KMeans+LDA)\n")
        f.write("=" * 70 + "\n\n")

        # 好评对比
        f.write("【好评主题对比】\n\n")
        f.write("原论文 (Jieba + LDA):\n")
        for topic_name, keywords in ORIGINAL_POS_TOPICS.items():
            f.write(f"  {topic_name}: {', '.join(keywords)}\n")

        f.write(f"\n新方法 (BERT + KMeans + LDA):\n")
        if "positive" in all_topic_results and all_topic_results["positive"]:
            for topic_name, keywords in all_topic_results["positive"].items():
                f.write(f"  {topic_name}: {', '.join(keywords)}\n")
        else:
            f.write("  (无结果)\n")

        # 差评对比
        f.write(f"\n{'─' * 70}\n\n")
        f.write("【差评主题对比】\n\n")
        f.write("原论文 (Jieba + LDA):\n")
        for topic_name, keywords in ORIGINAL_NEG_TOPICS.items():
            f.write(f"  {topic_name}: {', '.join(keywords)}\n")

        f.write(f"\n新方法 (BERT + KMeans + LDA):\n")
        if "negative" in all_topic_results and all_topic_results["negative"]:
            for topic_name, keywords in all_topic_results["negative"].items():
                f.write(f"  {topic_name}: {', '.join(keywords)}\n")
        else:
            f.write("  (无结果)\n")

    print(f"  [OK] 主题对比文件已保存: {path}")


# ======================== 3. 词云生成 ========================

def tokenize_with_bert(text, tokenizer):
    """
    使用 BertTokenizer 对中文文本进行切词。
    对于中文，BERT tokenizer 通常按字切分，这里我们会将连续的
    中文字符 token 合并为词级别的单元，以获得有意义的词云。
    """
    if not text or not text.strip():
        return []

    # 使用 tokenizer 的 tokenize 方法
    tokens = tokenizer.tokenize(text)

    # 过滤特殊 token ([CLS], [SEP], [PAD], [UNK], [MASK] 等)
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]", "<s>", "</s>",
                      "<pad>", "<unk>", "<mask>"}
    tokens = [t for t in tokens if t not in special_tokens]

    # 合并连续的 BPE 子词 token (如 "Ġæ" + "Ĕ´" → "æĔ´")
    # 对于中文，BERT tokenizer 的每个 token 通常就是一个汉字
    meaningful_tokens = []
    current_word = ""

    for token in tokens:
        # 去除 BPE 前缀符号 (BERT: ##, RoBERTa: Ġ)
        clean = token
        is_continuation = False

        if clean.startswith("##"):
            clean = clean[2:]
            is_continuation = True
        elif clean.startswith("▁") or clean.startswith("Ġ"):
            # RoBERTa BPE 使用 Ġ 或 ▁ 表示词首
            if len(clean) > 1:
                clean = clean[1:]
                # 前面的词结束，开始新词
                if current_word:
                    meaningful_tokens.append(current_word)
                current_word = ""
            else:
                clean = ""
                if current_word:
                    meaningful_tokens.append(current_word)
                current_word = ""

        if is_continuation and current_word:
            current_word += clean
        else:
            if current_word:
                meaningful_tokens.append(current_word)
            current_word = clean

    if current_word:
        meaningful_tokens.append(current_word)

    # 过滤: 长度 >= 2, 包含中文, 不在停用词中
    valid = []
    for w in meaningful_tokens:
        w_clean = w.strip()
        if (len(w_clean) >= 2
                and w_clean not in STOPWORDS
                and re.search(r"[\u4e00-\u9fa5]", w_clean)):
            valid.append(w_clean)

    return valid


def generate_wordclouds(texts, sentiment_results, tokenizer, output_dir):
    """
    基于正/负评论分别生成词云图。
    使用 BertTokenizer 进行切词 (替代原论文的 Jieba)。
    同时生成一份 Jieba 版本的词云用于对比。
    """
    print(f"\n{'='*60}")
    print("第三步：生成词云图 (BertTokenizer 切词)")
    print(f"{'='*60}")

    font_path = find_chinese_font()

    # 划分正负评论文本
    pos_texts = [t for t, r in zip(texts, sentiment_results) if r["label"] == "positive"]
    neg_texts = [t for t, r in zip(texts, sentiment_results) if r["label"] == "negative"]

    print(f"[INFO] 正向评论 {len(pos_texts)} 条, 负向评论 {len(neg_texts)} 条")

    # ---- 使用 BertTokenizer 切词 ----
    print("  使用 BertTokenizer 切词...")
    pos_tokens = []
    neg_tokens = []
    for t in pos_texts:
        pos_tokens.extend(tokenize_with_bert(t, tokenizer))
    for t in neg_texts:
        neg_tokens.extend(tokenize_with_bert(t, tokenizer))

    print(f"  正向 token 数: {len(pos_tokens)}, 负向 token 数: {len(neg_tokens)}")

    if not pos_tokens and not neg_tokens:
        print("[WARN] BertTokenizer 未产生有效 token，回退到 Jieba 分词")
        pos_tokens = []
        neg_tokens = []
        for t in pos_texts:
            words = jieba.lcut(t)
            pos_tokens.extend([w for w in words if len(w) >= 2
                              and w not in STOPWORDS
                              and re.search(r"[\u4e00-\u9fa5]", w)])
        for t in neg_texts:
            words = jieba.lcut(t)
            neg_tokens.extend([w for w in words if len(w) >= 2
                              and w not in STOPWORDS
                              and re.search(r"[\u4e00-\u9fa5]", w)])

    # ---- 生成正/负词云 ----
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for idx, (tokens, label_cn, cmap, ax) in enumerate([
        (pos_tokens, "正向评论 (BertTokenizer)", "YlOrRd", axes[0]),
        (neg_tokens, "负向评论 (BertTokenizer)", "Blues_r", axes[1]),
    ]):
        if not tokens:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=20)
            ax.set_title(label_cn, fontsize=14, fontweight="bold")
            continue

        # 构建词频文本
        word_freq = Counter(tokens)
        # 对于 wordcloud.generate_from_frequencies (使用字典避免空格分割问题)
        try:
            wc = WordCloud(
                font_path=font_path,
                width=WORDCLOUD_WIDTH,
                height=WORDCLOUD_HEIGHT,
                background_color="white",
                max_words=WORDCLOUD_MAX_WORDS,
                max_font_size=120,
                min_font_size=10,
                random_state=RANDOM_SEED,
                colormap=cmap,
                collocations=False,
            ).generate_from_frequencies(word_freq)

            ax.imshow(wc, interpolation="bilinear")
        except Exception as e:
            print(f"  [ERR] {label_cn} 词云生成失败: {e}")
            ax.text(0.5, 0.5, f"生成失败: {e}", ha="center", va="center", fontsize=12)

        ax.axis("off")
        ax.set_title(label_cn, fontsize=14, fontweight="bold")

    plt.suptitle("用户评论词云图 (基于 RoBERTa BertTokenizer 切词)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    wc_path = os.path.join(output_dir, "wordcloud_bert_tokenizer.png")
    plt.savefig(wc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] 词云图已保存: {wc_path}")

    # 额外: 保存正负词频 Top-20 到文件
    pos_freq = Counter(pos_tokens).most_common(20) if pos_tokens else []
    neg_freq = Counter(neg_tokens).most_common(20) if neg_tokens else []
    freq_path = os.path.join(output_dir, "word_frequency_bert_tokenizer.txt")
    with open(freq_path, "w", encoding="utf-8") as f:
        f.write("正向评论高频词 (BertTokenizer):\n")
        for word, count in pos_freq:
            f.write(f"  {word}: {count}\n")
        f.write("\n负向评论高频词 (BertTokenizer):\n")
        for word, count in neg_freq:
            f.write(f"  {word}: {count}\n")
    print(f"  [OK] 词频统计已保存: {freq_path}")


# ======================== 4. 生成综合报告 ========================

def generate_report(texts, sentiment_results, all_topic_results, output_dir):
    """生成综合对比报告"""
    print(f"\n{'='*60}")
    print("生成综合分析报告")
    print(f"{'='*60}")

    pos = sum(1 for r in sentiment_results if r["label"] == "positive")
    neg = sum(1 for r in sentiment_results if r["label"] == "negative")
    total = pos + neg

    new_pos_pct = pos / total * 100 if total else 0
    new_neg_pct = neg / total * 100 if total else 0

    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("硬折扣零食店评论分析 — 方法升级对比报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"有效评论数: {len(texts)} 条\n\n")

        # 情感分析对比
        f.write("─" * 70 + "\n")
        f.write("一、情感分析对比\n")
        f.write("─" * 70 + "\n\n")
        f.write(f"{'指标':<20} {'原方法 (SnowNLP)':<25} {'新方法 (RoBERTa)':<25}\n")
        f.write(f"{'-'*70}\n")
        f.write(f"{'正向':<20} {ORIGINAL_SENTIMENT['positive']}%{'':>19} {new_pos_pct:.1f}%{'':>19}\n")
        f.write(f"{'中性':<20} {ORIGINAL_SENTIMENT['neutral']}%{'':>19} {'N/A (二分类模型)'}{'':>8}\n")
        f.write(f"{'负向':<20} {ORIGINAL_SENTIMENT['negative']}%{'':>19} {new_neg_pct:.1f}%{'':>19}\n")

        # 平均置信度
        confidences = [r["confidence"] for r in sentiment_results if r["label"] != "unknown"]
        avg_conf = np.mean(confidences) if confidences else 0
        f.write(f"\nRoBERTa 平均置信度: {avg_conf:.4f}\n")

        # 主题对比
        f.write(f"\n{'─'*70}\n")
        f.write("二、主题关键词对比\n")
        f.write("─" * 70 + "\n\n")

        for label, label_cn, orig_topics in [
            ("positive", "好评", ORIGINAL_POS_TOPICS),
            ("negative", "差评", ORIGINAL_NEG_TOPICS),
        ]:
            f.write(f"【{label_cn}】\n")
            f.write("原论文 (Jieba + LDA):\n")
            for tn, kw in orig_topics.items():
                f.write(f"  {tn}: {', '.join(kw)}\n")
            f.write("新方法 (BERT + KMeans + LDA):\n")
            if label in all_topic_results and all_topic_results[label]:
                for tn, kw in all_topic_results[label].items():
                    f.write(f"  {tn}: {', '.join(kw)}\n")
            else:
                f.write("  (无结果)\n")
            f.write("\n")

        f.write("─" * 70 + "\n")
        f.write("三、方法升级总结\n")
        f.write("─" * 70 + "\n\n")
        f.write("1. 情感分析: SnowNLP → RoBERTa (uer/roberta-base-finetuned-jd-binary-chinese)\n")
        f.write("   - 基于深度学习的二分类模型，在电商评论上微调\n")
        f.write("   - 新方法无中性类别，原3.9%的中性评论被分配至正/负向\n\n")
        f.write("2. 主题挖掘: Jieba+LDA → BERT+KMeans+LDA\n")
        f.write("   - 使用 sentence-transformers 生成语义向量\n")
        f.write("   - KMeans 聚类实现主题分组 (语义级别而非词频级别)\n")
        f.write("   - sklearn LDA 提取各聚类关键词\n\n")
        f.write("3. 词云: Jieba分词 → BertTokenizer切词\n")
        f.write("   - 使用 RoBERTa 自带的 BPE tokenizer\n")
        f.write("   - 子词级别的切分粒度，更贴近模型视角\n\n")

    print(f"  [OK] 综合报告已保存: {report_path}")


# ======================== 5. 保存详细结果 ========================

def save_detailed_results(texts, sentiment_results, output_dir):
    """保存每条评论的情感分析结果到 CSV"""
    path = os.path.join(output_dir, "sentiment_results.csv")
    df_out = pd.DataFrame({
        "review": texts,
        "sentiment_label": [r["label"] for r in sentiment_results],
        "confidence": [r["confidence"] for r in sentiment_results],
    })
    df_out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 详细情感结果已保存: {path} (共 {len(df_out)} 条)")


# ======================== 主函数 ========================

def main():
    parser = argparse.ArgumentParser(
        description="硬折扣零食店评论分析 — 深度学习升级版"
    )
    parser.add_argument("--csv", type=str, default=CSV_FILE,
                        help=f"CSV 文件路径 (默认: {CSV_FILE})")
    parser.add_argument("--local", action="store_true",
                        help="使用本地已下载的模型 (从 LOCAL_MODEL_DIR 加载)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="跳过错感分析 (使用已有的情感结果)")
    parser.add_argument("--skip-topics", action="store_true",
                        help="跳过主题挖掘")
    parser.add_argument("--skip-wordcloud", action="store_true",
                        help="跳过词云生成")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("硬折扣零食店消费者评论分析 — 深度学习升级版")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {args.output}")
    print("=" * 60)

    # ---- 加载数据 ----
    print(f"\n[Step 0] 加载数据...")
    raw_texts = load_data(args.csv)
    cleaned_texts, valid_indices = clean_data(raw_texts)
    print(f"  清洗后有效评论: {len(cleaned_texts)} 条")

    # ---- 配置环境并加载模型 ----
    print(f"\n[Step 0] 配置环境并加载模型...")
    setup_hf_environment(use_local=args.local)
    sentiment_tokenizer, sentiment_model, embedding_model = load_models_with_retry(
        use_local=args.local
    )

    # ---- 情感分析 ----
    if not args.skip_sentiment:
        sentiment_results = run_sentiment_analysis(
            cleaned_texts, sentiment_tokenizer, sentiment_model
        )
        save_detailed_results(cleaned_texts, sentiment_results, args.output)
        new_pos_pct, new_neg_pct = plot_sentiment_comparison(
            sentiment_results, args.output
        )
    else:
        # 从已有结果加载
        result_csv = os.path.join(args.output, "sentiment_results.csv")
        if os.path.exists(result_csv):
            df_existing = pd.read_csv(result_csv)
            sentiment_results = [
                {"label": r["sentiment_label"], "confidence": r["confidence"]}
                for _, r in df_existing.iterrows()
            ]
            print(f"[INFO] 从已有文件加载 {len(sentiment_results)} 条情感结果")
        else:
            raise FileNotFoundError(f"情感结果文件不存在: {result_csv}")

    # ---- 主题挖掘 ----
    if not args.skip_topics:
        all_topic_results = run_topic_modeling(
            cleaned_texts, sentiment_results, embedding_model, args.output
        )
        save_topic_comparison(all_topic_results, args.output)
    else:
        all_topic_results = {}

    # ---- 词云 ----
    if not args.skip_wordcloud:
        generate_wordclouds(
            cleaned_texts, sentiment_results, sentiment_tokenizer, args.output
        )

    # ---- 综合报告 ----
    generate_report(cleaned_texts, sentiment_results, all_topic_results, args.output)

    print(f"\n{'='*60}")
    print("全部分析完成！")
    print(f"输出文件位于: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
