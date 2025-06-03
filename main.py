import torch
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
from evaluate import load

# 1. 模型选择
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 2. 数据准备
#   这里需要替换成你的低资源领域数据
#   假设你有一个包含 'translation' 字段的数据集，其中包含 'en' 和 'zh'
#   例如： dataset = load_dataset("your_dataset_name", split='train')
#   为了演示，我们使用一个小的虚拟数据集
data = {
    "translation": [
        {"en": "This is a small example.", "zh": "这是一个小例子。"},
        {"en": "The cat is on the mat.", "zh": "猫在垫子上。"},
        {"en": "Please pass the salt.", "zh": "请递一下盐。"},
        {"en": "Where is the library?", "zh": "图书馆在哪里？"},
    ]
}
from datasets import Dataset
dataset = Dataset.from_dict(data)

def tokenize_function(examples):
    inputs = tokenizer(examples["translation"]["en"], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples["translation"]["zh"], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": targets.input_ids}

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets
eval_dataset = tokenized_datasets  # 使用相同的数据集进行演示，实际中你需要分开

# 3. 定义评估指标
metric = load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# 4. 完全微调
training_args_full = TrainingArguments(
    output_dir="./full-tuned-marian-en-zh",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=20,  # 可以调整
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    predict_with_generate=True,
    report_to="none" #  避免与 LoRA 的回调冲突
)

trainer_full = Trainer(
    model=model,
    args=training_args_full,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer_full.train()
full_tuned_results = trainer_full.evaluate()

# 5. PEFT (LoRA) 微调
# config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="SEQ_2_SEQ_LM" #  适用于翻译任务
# )

# model_lora = get_peft_model(model, config)
# model_lora.print_trainable_parameters() #  打印可训练参数

# training_args_lora = TrainingArguments(
#     output_dir="./lora-tuned-marian-en-zh",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     learning_rate=2e-4, #  LoRA 通常可以使用更高的学习率
#     num_train_epochs=20,
#     weight_decay=0.01,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="bleu",
#     predict_with_generate=True,
# )

# trainer_lora = Trainer(
#     model=model_lora,
#     args=training_args_lora,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# trainer_lora.train()
# lora_tuned_results = trainer_lora.evaluate()

# # 6. 结果比较
# print("Full Fine-tuning Results:", full_tuned_results)
# print("LoRA Fine-tuning Results:", lora_tuned_results)

#  灾难性遗忘的评估需要一个通用的数据集，并在微调前后进行比较。
#  这里为了简化，省略了这部分代码。
#  你需要自己添加加载通用数据集，并在微调前后评估BLEU等指标的代码。