import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import joblib
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

total_emr = pd.read_excel("top_10_diseases_data_cleaned.xlsx", engine='openpyxl')
labels = total_emr['standard_name']
total_emr.drop(['id_num', 'tdate'], axis=1, inplace=True)
total_emr.drop("standard_name", axis=1, inplace=True)


def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

line_names = total_emr.columns.tolist()

empty_df = pd.DataFrame()
empty_df['newline'] = total_emr[line_names].apply(
    lambda row: ','.join([str(x) for x in row if pd.notna(x) and str(x).strip() != '']),
    axis=1
)

label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
print(f"Detected {num_classes} classes")
joblib.dump(label_encoder, 'label_encoder.pkl')

empty_df['labels'] = numeric_labels

PRE_TRAINED_MODEL_NAME = 'roberta'

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=num_classes).to(device)

seed = 42
train_df, test_df = train_test_split(empty_df, test_size=0.2, random_state=seed)

print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")


class MedicalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['newline']
        label = self.dataframe.iloc[idx]['labels']
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = MedicalDataset(train_df, tokenizer)
test_dataset = MedicalDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

optimizer = AdamW(model.parameters(), lr=5e-5)

train_losses = []
test_losses = []

num_epochs = 4
best_test_loss = float('inf')
save_path = './best_bert_model'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    epoch_test_loss = 0
    for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1} [Test]"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        epoch_test_loss += loss.item()

    avg_test_loss = epoch_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        print(f"Found better model, saving to {save_path} ...")

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        joblib.dump(label_encoder, os.path.join(save_path, 'label_encoder.pkl'))

print("Loading best model for final evaluation...")
model = AutoModelForSequenceClassification.from_pretrained(save_path).to(device)

os.makedirs('./no_val_npy/', exist_ok=True)
np.save('./no_val_npy/train_losses.npy', np.array(train_losses))
np.save('./no_val_npy/test_losses.npy', np.array(test_losses))
print("Loss values saved as numpy files.")

model.eval()
all_preds = []
all_labels = []
all_probs = []

for batch in tqdm(test_loader, desc="Final Evaluating"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())
    all_probs.extend(probs.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc_score:.4f}")
print(f"Test F1-score: {f1:.4f}")

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

n_classes = num_classes
labels_binarized = label_binarize(all_labels, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(labels_binarized[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(10, 8))

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                'purple', 'brown', 'pink', 'gray', 'olive'])
for i, color in zip(range(n_classes), colors):
    if i < len(label_encoder.classes_):
        label_name = label_encoder.classes_[i]
    else:
        label_name = str(i)

    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(label_name, roc_auc[i]))

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random guessing')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")

plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("ROC curve saved as roc_curve.png")