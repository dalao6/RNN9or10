# 词向量
import pandas as pd
import jieba
from gensim.models.word2vec import Word2Vec

# 读入训练集文件
data = pd.read_csv('train.csv')
# 转字符串数组
corpus = data['comment'].values.astype(str)
# 分词，再重组为字符串数组
corpus = [jieba.lcut(corpus[index]
                     .replace("，", "")
                     .replace("!", "")
                     .replace("！", "")
                     .replace("。", "")
                     .replace("~", "")
                     .replace("；", "")
                     .replace("？", "")
                     .replace("?", "")
                     .replace("【", "")
                     .replace("】", "")
                     .replace("#", "")
                     ) for index in range(len(corpus))]

# 1) 使用 Skip - Gram 训练 Word2Vec 模型
# sg=1 表示使用 Skip - Gram 算法
model = Word2Vec(corpus, sg=1, vector_size=300, window=5, min_count=3, workers=4)

# 模型显示
print('模型参数：', model, '\n')

# 2) 输出“环境”的词向量及其形状
try:
    env_vector = model.wv['环境']
    print("'环境'的词向量：", env_vector)
    print("'环境'词向量的形状：", env_vector.shape)
except KeyError:
    print("词 '环境' 不在词汇表中。")

# 3) 输出与“好吃”语义最接近的 3 个词
try:
    similar_words = model.wv.most_similar('好吃', topn=3)
    print("与 '好吃' 语义最接近的 3 个词：", similar_words)
except KeyError:
    print("词 '好吃' 不在词汇表中。")

# 4) 计算“好吃”和“美味”的相似度、“好吃”和“蟑螂”的相似度
try:
    similarity1 = model.wv.similarity('好吃', '美味')
    similarity2 = model.wv.similarity('好吃', '蟑螂')
    print("'好吃' 和 '美味' 的相似度：", similarity1)
    print("'好吃' 和 '蟑螂' 的相似度：", similarity2)
except KeyError as e:
    print(f"计算相似度时出错，{e} 不在词汇表中。")

# 执行向量运算“餐厅+聚会-安静=？”，输出 1 个最相关结果
try:
    result = model.wv.most_similar(positive=['餐厅', '聚会'], negative=['安静'], topn=1)
    print("向量运算 '餐厅+聚会-安静' 的最相关结果：", result)
except KeyError as e:
    print(f"向量运算时出错，{e} 不在词汇表中。")
