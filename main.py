from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

# 选择一个合适的预训练模型
model_name = "Helsinki-NLP/opus-mt-en-zh" # 示例：一个小型NMT模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 配置LoRA（如果使用PEFT）
peft_config = LoraConfig(
    r=8, # LoRA 秩
    lora_alpha=16, # LoRA 缩放因子
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM, # 序列到序列语言模型任务
    target_modules=["q_proj", "v_proj"] # 目标模块，通常是注意力机制中的查询和值投影层
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # 打印可训练参数

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
)

# 定义评估指标计算函数
from evaluate import load
bleu_metric = load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[label] for label in decoded_labels] # 调整references格式
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": bleu_result["bleu"]}

# 创建Trainer实例并进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()