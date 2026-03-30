from transformers import BertTokenizer, BertForMaskedLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from torch.nn.functional import cross_entropy
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from transformers import TrainerCallback
import os

os.environ["WANDB_MODE"] = "offline"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'roundup_power2_divisions:[256:1,512:2,1024:4,>:8],max_split_size_mb:128'


class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()


tokenizer = BertTokenizer.from_pretrained('roberta')
model = BertForMaskedLM.from_pretrained('roberta')

df2 = pd.read_csv(r'emr.csv')
df2.drop(columns=['id_num', 'admission_date_std', 'admission_date_ori', 'org_code'], inplace=True)
line_names = df2.columns.tolist()

empty_df = pd.DataFrame()
empty_df['newline'] = df2[line_names].apply(
    lambda row: ','.join([str(x) for x in row if pd.notna(x) and str(x).strip() != '']),
    axis=1
)
print(empty_df.head(3))
dataset = Dataset.from_pandas(empty_df)


def preprocess_function(examples):
    return tokenizer(
        examples['newline'],
        truncation=True,
        padding='max_length',
        max_length=128,
        return_special_tokens_mask=True
    )


encoded_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=1000,
    remove_columns=dataset.column_names
)

encoded_dataset.set_format(
    type="torch",
    columns=["input_ids", "token_type_ids", "attention_mask"]
)

datasets_split = encoded_dataset.train_test_split(test_size=0.2)
print(
    f"Dataset split complete: Train set has {len(datasets_split['train'])} samples, Test set has {len(datasets_split['test'])} samples")

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    return_tensors="pt"
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    mask = labels != -100
    labels_filtered = labels[mask]
    predictions_filtered = predictions[mask]

    if len(labels_filtered) == 0:
        return {"accuracy": 0.0, "perplexity": 0.0, "f1": 0.0}

    accuracy = accuracy_score(labels_filtered.flatten(), predictions_filtered.flatten())

    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    loss = cross_entropy(
        logits_tensor.reshape(-1, logits_tensor.shape[-1]),
        labels_tensor.reshape(-1),
        ignore_index=-100
    )
    perplexity = torch.exp(loss).item()

    f1 = f1_score(labels_filtered.flatten(), predictions_filtered.flatten(), average='weighted')

    return {
        'accuracy': round(accuracy, 4),
        'perplexity': round(perplexity, 4),
        'f1': round(f1, 4)
    }


training_args = TrainingArguments(
    output_dir='./results',
    run_name="my_custom_run_name",
    learning_rate=5e-5,
    num_train_epochs=8,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    save_strategy="epoch",
    save_total_limit=3,
    eval_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
    logging_nan_inf_filter=True,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    load_best_model_at_end=False,
    eval_accumulation_steps=1,
    gradient_accumulation_steps=2,
    fp16=True,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=datasets_split['train'],
    compute_metrics=compute_metrics,
    callbacks=[ClearCacheCallback()]
)

print("Starting training============================")
trainer.train()

save_path = 'ssl'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

loaded_tokenizer = BertTokenizer.from_pretrained(save_path)
loaded_model = BertForMaskedLM.from_pretrained(save_path)
print("Model and tokenizer loaded successfully.")