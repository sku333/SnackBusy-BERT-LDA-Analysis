# -*- coding: utf-8 -*-
"""
=============================================================================
硬折扣零食店评论分析 — 深度学习升级版 配置文件
=============================================================================
说明：
  - 所有模型默认从 Hugging Face 镜像站 hf-mirror.com 下载
  - 也可指定本地模型路径（LOCAL_MODEL_DIR）加载已下载好的模型
  - 数据仅 1377 条，所有推理在 CPU 上完成
=============================================================================
"""

import os

# ======================== 路径配置 ========================

# 原始数据 CSV 文件路径
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "大众点评评论合集.csv")

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ======================== Hugging Face 镜像配置 ========================

# 方式一：从镜像站自动下载（服务器无法访问 huggingface.co 时使用）
HF_ENDPOINT = "https://hf-mirror.com"

# 方式二：本地模型目录（如果已提前下载好，设置此路径，留空则自动从镜像下载）
# 目录结构示例：
#   LOCAL_MODEL_DIR/
#     ├── roberta-base-finetuned-jd-binary-chinese/   (情感分析模型)
#     └── paraphrase-multilingual-MiniLM-L12-v2/       (句子向量模型)
LOCAL_MODEL_DIR = ""  # 例如: "/home/user/models" 或 "E:/models"

# ======================== 模型名称配置 ========================

# 情感分析模型 (uer/roberta-base-finetuned-jd-binary-chinese)
SENTIMENT_MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"

# 句子向量模型 (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 用于词云切词的 BERT Tokenizer (使用情感分析模型的 tokenizer)
TOKENIZER_MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"

# ======================== 情感分析配置 ========================

# 情感分类阈值
POSITIVE_THRESHOLD = 0.5   # 置信度 >= 0.5 视为正向
NEGATIVE_THRESHOLD = 0.5   # 置信度 < 0.5 视为负向 (二分类模型无中性)

# 最大文本长度 (RoBERTa 最大 512)
MAX_SEQ_LENGTH = 256

# 批处理大小 (CPU 推理用较小的 batch)
BATCH_SIZE = 16

# ======================== 主题挖掘配置 ========================

# KMeans 聚类的主题数量 (与原论文保持一致: 好评3个主题, 差评3个主题)
N_TOPICS_POS = 3
N_TOPICS_NEG = 3

# 每个主题提取的关键词数量 (与原论文保持一致: 5个)
N_TOPIC_KEYWORDS = 5

# KMeans 随机种子
RANDOM_SEED = 42

# ======================== 词云配置 ========================

# 词云最大词数
WORDCLOUD_MAX_WORDS = 200

# 词云图片尺寸
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 600

# 中文字体路径 (Linux 服务器上需要指定中文字体)
# 常见路径: "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
#           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
#           "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
LINUX_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

# 停用词 (与原论文基本一致)
STOPWORDS = set([
    "的", "了", "在", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这个", "那个", "有", "和", "也", "就", "都", "而", "及", "与", "或",
    "一个", "没有", "很", "非常", "有点", "一些", "这种", "因为", "所以",
    "但是", "不过", "还是", "还有", "就是", "真的", "可以", "买",
    "零食", "店", "里面", "东西", "价格", "性价比", "环境", "服务", "排队",
    "收银", "品种", "种类", "购物", "小朋友", "时候", "感觉", "知道",
    "特别", "一直", "每次", "经常", "已经", "如果", "一般", "比较", "然后",
    "这样", "觉得", "现在", "开", "新",
    "上海", "四川北路", "周杰伦", "代言", "很忙",
    "看到", "还是", "来", "去", "说", "人", "上", "下",
    "这里", "那里", "哦", "呢", "啊", "吗", "啦",
    "让", "这家", "只", "又", "才", "再", "而", "且",
    "很多", "真的", "每次", "一点", "这边", "外面", "今天",
])

# ======================== 原论文结果 (用于对比) ========================

ORIGINAL_SENTIMENT = {
    "positive": 73.8,   # 正面 73.8%
    "neutral": 3.9,     # 中性 3.9%
    "negative": 22.3,   # 负面 22.3%
}

ORIGINAL_POS_TOPICS = {
    "主题1: 性价比与消费体验": ["价格", "便宜", "性价比", "方便", "不错"],
    "主题2: 环境与服务体验": ["环境", "服务", "品种", "实惠", "整洁"],
    "主题3: 产品口味与品类": ["好吃", "口味", "品类", "丰富", "味道"],
}

ORIGINAL_NEG_TOPICS = {
    "主题1: 收银与排队体验": ["收银", "结账", "排队", "收银员", "袋子"],
    "主题2: 服务与商品问题": ["服务", "态度", "商品", "品牌", "营业员"],
    "主题3: 会员与价格感知": ["会员", "价格", "便宜", "货架", "品种"],
}
