from datasets import load_dataset

# 尝试加载 WMT19 中英翻译数据集
try:
    wmt_zh_en = load_dataset("wmt19", "zh-en")
    print("WMT19 中英数据集信息:")
    print(wmt_zh_en)
except Exception as e:
    print(f"加载 WMT19 中英数据集失败: {e}")

print(wmt_zh_en['validation']['translation'][:5])  # 打印前5条训练数据的翻译对