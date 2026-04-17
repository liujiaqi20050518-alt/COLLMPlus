import numpy as np

# 1. 设置你刚刚保存的文件路径
file_path = "/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/item_llm_features.npy"

print("正在读取特征文件...")
item_llm_embs = np.load(file_path)

# 2. 打印最基础的统计信息
print(f"✅ 成功读取！")
print(f"👉 矩阵形状: {item_llm_embs.shape}")
print(f"👉 数据类型: {item_llm_embs.dtype}")

# 3. 查看具体内容（比如看第 1 号电影的前 10 个特征值）
print("\n🔍 来看一眼第 1 号电影（通常是 Toy Story）的前 10 个维度的脑电波：")
print(item_llm_embs[1])

# 4. 做一个简单的健康度体检
print("\n🏥 矩阵健康度检查：")
print(f"最大值: {np.max(item_llm_embs):.4f}")
print(f"最小值: {np.min(item_llm_embs):.4f}")
print(f"是否包含 NaN (坏数据): {np.isnan(item_llm_embs).any()}")