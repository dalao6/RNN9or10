import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载预训练模型和分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # 正面和负面两类
    ignore_mismatched_sizes=True
)

# 影评和外卖评价数据
film_reviews = {
    "0": "这部电影太精彩了，节奏紧凑毫无冷场，完全沉浸其中！",
    "1": "剧情设定新颖不落俗套，每个转折都让人惊喜。",
    "2": "导演功力深厚，镜头语言非常有张力，每一帧都值得回味。",
    "3": "美术、服装、布景细节丰富，完全是视觉盛宴！",
    "4": "是近年来最值得一看的国产佳作，强烈推荐！",
    "5": "剧情拖沓冗长，中途几次差点睡着。",
    "6": "演员表演浮夸，完全无法让人产生代入感。",
    "7": "剧情老套，充满套路和硬凹的感动。",
    "8": "对白尴尬，像是AI自动生成的剧本。",
    "9": "看完只觉得浪费了两个小时，再也不想看第二遍。"
}

food_reviews = {
    "0": "食物完全凉了，吃起来像隔夜饭，体验极差。",
    "1": "汤汁洒得到处都是，包装太随便了。",
    "2": "味道非常一般，跟评论区说的完全不一样。",
    "3": "分量太少了，照片看着满满的，实际就几口。",
    "4": "食材不新鲜，有异味，感觉不太卫生。",
    "5": "食物份量十足，性价比超高，吃得很满足！",
    "6": "味道超级赞，和店里堂食一样好吃，五星好评！",
    "7": "这家店口味稳定，已经回购好几次了，值得信赖！",
    "8": "点单备注有按要求做，服务意识很棒。",
    "9": "包装环保、整洁美观，整体体验非常好。"
}

# 学号
student_id = "202210179038"
last_two_digits = student_id[-2:]
film_review_index = last_two_digits[-1]
food_review_index = last_two_digits[-2]

# 获取对应的影评和外卖评价
film_review = film_reviews[film_review_index]
food_review = food_reviews[food_review_index]


# 情感分类函数
def classify_sentiment(text):
    # 对输入文本进行分词
    inputs = tokenizer(text, return_tensors="pt")

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测结果
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    # 返回分类结果
    return "正面" if prediction == 1 else "负面"


# 对影评和外卖评价进行情感分类
film_sentiment = classify_sentiment(film_review)
food_sentiment = classify_sentiment(food_review)

# 输出结果
print(f"学号: {student_id}")
print(f"影评索引: {film_review_index}")
print(f"影评内容: {film_review}")
print(f"影评情感分类: {film_sentiment}")
print()
print(f"外卖评价索引: {food_review_index}")
print(f"外卖评价内容: {food_review}")
print(f"外卖评价情感分类: {food_sentiment}")