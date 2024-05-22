from transformers import BertTokenizer, BertModel
import torch

# 加载模型和分词器
def load_model(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir)
    model.eval()  # 设置为评估模式
    return tokenizer, model

# 将文本转换为向量，使用嵌入层输出
def text_to_embedding(text, tokenizer, model, device):
    tokens = tokenizer([text], padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    tokens = tokens.to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        embedding = outputs.last_hidden_state[:, 0, :]  # 使用CLS标记的输出
    return embedding.cpu().numpy()
