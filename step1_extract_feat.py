import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# ==========================================
# 1. 配置路径 
# ==========================================
# 如果你的服务器断网了，请把这里替换成你本地存大模型（比如 Vicuna 或 Qwen）的绝对路径
# 比如: model_path = "/root/autodl-tmp/vicuna-7b"
model_path = "/root/autodl-tmp/weight/" 

# ==========================================
# 2. 从 pkl 文件直接获取完美对齐的文本字典 (最精妙的一步)
# ==========================================
print("正在读取 pkl 文件获取完美对齐的电影文本...")
# 注意：确认这里的路径是你存 pkl 的路径
train_df = pd.read_pickle('/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/train_ood2.pkl')
valid_df = pd.read_pickle('/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/valid_ood2.pkl')
test_df = pd.read_pickle('/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/test_ood2.pkl')

# 把三个文件拼起来，提取所有的 iid 和 title
all_data = pd.concat([train_df, valid_df, test_df])

# 去重，得到极其纯粹的 {iid: 电影名称} 的映射字典！
item_text_df = all_data[['iid', 'title']].drop_duplicates().set_index('iid')
item_dict = item_text_df['title'].to_dict()

# 获取最大的 iid 编号。由于 0 号是 padding，矩阵总大小必须是 max_iid + 1
max_iid = max(item_dict.keys())
num_items = max_iid + 1
print(f"总共有 {num_items} 个物品 (包含 0 号占位符)")

# ==========================================
# 3. 加载大模型
# ==========================================
print(f"正在加载大模型: {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
model.eval()

llm_dim = model.config.hidden_size
print(f"✅ 模型加载成功，特征维度为: {llm_dim}维")

# 初始化全 0 矩阵 (这样 0 号 padding 位置自然就是全 0 向量)
item_llm_embs = np.zeros((num_items, llm_dim), dtype=np.float32)

# ==========================================
# 4. 提取特征并保存
# ==========================================
print("开始提取电影文本特征...")
with torch.no_grad():
    for idx in range(1, num_items): # 从 1 开始遍历，避开 0 号
        # 从字典中直接取出绝对正确的电影名称
        movie_title = item_dict.get(idx, "Unknown Movie")
        text = f"Movie Title: {movie_title}"
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to('cuda')
        outputs = model(**inputs)
        
        # 取大模型最后一层的平均池化作为这句文本的向量表示
        text_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        item_llm_embs[idx] = text_embedding
        
        if idx % 500 == 0:
            print(f"进度: {idx} / {max_iid}")

# 保存为终极弹药 .npy
save_path = "/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/item_llm_features.npy"
np.save(save_path, item_llm_embs)
print(f"🎉 成功！高维特征矩阵已保存至: {save_path}，形状为 {item_llm_embs.shape}")