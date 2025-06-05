from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练模型和分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 为避免警告，设置填充标记为结束标记
tokenizer.pad_token = tokenizer.eos_token

# 学号
student_id = "202210179038"
last_digit = student_id[-1]

# 文本开头集合
text_starts = {
    "0": "如果我拥有一台时间机器",
    "1": "当人类第一次踏上火星",
    "2": "如果动物会说话，它们最想告诉人类的是",
    "3": "有一天，城市突然停电了",
    "4": "当我醒来，发现自己变成了一本书",
    "5": "假如我能隐身一天，我会",
    "6": "我走进了那扇从未打开过的门",
    "7": "在一个没有网络的世界里",
    "8": "如果世界上只剩下我一个人",
    "9": "当第一缕阳光照进这个废弃的城市"
}

# 获取对应的文本开头
text_start = text_starts[last_digit]

# 生成文本
input_ids = tokenizer.encode(text_start, return_tensors="pt")
output = model.generate(
    input_ids,
    max_length=300,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 输出结果
print(f"学号: {student_id}")
print(f"文本开头索引: {last_digit}")
print(f"文本开头内容: {text_start}")
print("\n生成的文本:")
print(generated_text)
