import os
from datasets import load_dataset
import evaluate
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer, TrainerCallback
import torch
import numpy as np

# Đường dẫn tới folder chứa dữ liệu
data_dir = "/home/ubuntu/1/BTL_CSDLDPT/fruits-360_100x100/fruits-360/Training/merged_results"

# Tải dataset từ folder và chia train/test
dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")
dataset = dataset.train_test_split(test_size=0.2, seed=42)  # Chia 80% train, 20% test

# Tải image processor (thay ViTFeatureExtractor vì đã deprecated)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Hàm tiền xử lý ảnh
def transform(example_batch):
    inputs = processor([x.convert("RGB") for x in example_batch['image']], return_tensors="pt")
    example_batch['pixel_values'] = inputs['pixel_values']
    return example_batch

# Áp dụng transform
prepared_ds = dataset.with_transform(transform)

# Tạo model
labels = prepared_ds['train'].features['label'].names
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True  # Thêm để xử lý kích thước khác biệt
)

# Hàm tính metrics
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Cấu hình training
training_args = TrainingArguments(
    output_dir="./vit-fruits",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",  # Đánh giá sau mỗi epoch
    save_strategy="epoch",        # Lưu model sau mỗi epoch
    num_train_epochs=3,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    load_best_model_at_end=True,  # Tải model tốt nhất sau khi train
)

# Khởi tạo trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    tokenizer=processor,  # Dùng processor thay vì feature_extractor
    compute_metrics=compute_metrics,
)

# Callback để in kết quả đánh giá
class PrintEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            print(f"\nEpoch {int(state.epoch)} - Eval metrics: {metrics}")

trainer.add_callback(PrintEvalCallback())

# Huấn luyện
trainer.train()