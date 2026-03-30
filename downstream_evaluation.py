'''
 **Note: Make a copy of this code and execute it on your own. Do not make changes to this file.

 Execute each section of the code one by one. Do not make changes to the data processing or evaluation.
 Only change the section "model" as per your needs.
 If Some model may require data to be in a specific formate, then you can write the code seperatly.
'''

import random
import numpy as np

np.set_printoptions(threshold=np.inf)
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, \
    TrainingArguments
import pandas as pd
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, \
    davies_bouldin_score
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.svm import SVC
import ast
import os

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

config = AutoConfig.from_pretrained("/home/yuanfuchong/dianzi/pre_trained_models/ehealth/ehealth/config.json")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"Random seed set to: {RANDOM_SEED}")

"""
    Helping Functions    
"""


def preprocess_data(json_file, columns_to_use, cut_occurrences, train_size_percent=0.8, save_labels_count=True):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    texts, labels = [], []
    ages, sexes = [], []

    total_patients = len(data)
    for i, (patient_id, patient_data) in enumerate(tqdm(data.items(), total=total_patients)):
        text_data = []
        icd10_code = None

        if not patient_data: continue

        sorted_visit_dates = sorted(patient_data.keys())
        first_visit_date = sorted_visit_dates[0]
        first_visit_details = patient_data[first_visit_date]

        for col in columns_to_use:
            val = first_visit_details.get(col, "")
            if val is None: val = ""
            text_data.append(str(val))

        diag_entries = first_visit_details.get('diag', [])
        if diag_entries:
            for diag_item in diag_entries:
                if isinstance(diag_item, list) and len(diag_item) > 1:
                    icd10_code = diag_item[1]

        sex_raw = str(first_visit_details.get('sex_name', 'Unknown'))
        if sex_raw in ['1', '男']:
            sex = 'Male'
        elif sex_raw in ['2', '女']:
            sex = 'Female'
        else:
            sex = 'Unknown'

        age = -1
        try:
            born = str(first_visit_details.get('born_date', ''))
            visit = str(first_visit_details.get('admission_date_std', ''))
            if len(born) >= 4 and len(visit) >= 4:
                age = int(visit[:4]) - int(born[:4])
        except:
            age = -1

        record = ' '.join(text_data).strip()

        if record:
            texts.append(record)
            labels.append(icd10_code)
            ages.append(age)
            sexes.append(sex)

        if i >= total_patients - 1: break

    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'age': ages,
        'sex': sexes
    })

    df.dropna(subset=['label'], inplace=True)
    print(f"Number of samples in df after filtering NaN labels ---> {len(df)}")

    label_counts = df['label'].value_counts()
    if save_labels_count:
        label_counts.reset_index().to_csv("label_counts.csv", index=False)

    labels_to_keep = label_counts[label_counts >= cut_occurrences].index
    filtered_df = df[df['label'].isin(labels_to_keep)].copy()

    def get_age_group(a):
        if a < 0: return 'Unknown'
        if a < 18:
            return '0-17'
        elif a <= 45:
            return '18-45'
        elif a <= 65:
            return '46-65'
        else:
            return '>65'

    filtered_df['age_group'] = filtered_df['age'].apply(get_age_group)

    print(f"Number of samples after filtering ---> {len(filtered_df)}")

    train_size = int(train_size_percent * len(filtered_df))
    train_df = filtered_df[:train_size]
    test_df = filtered_df[train_size:]

    unique_label_count = filtered_df['label'].nunique()
    print("Number of unique labels ---> ", unique_label_count)

    return filtered_df, train_df, test_df, unique_label_count


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    input_ids = inputs['input_ids'][0]
    chunks = [input_ids[i:i + 512] for i in range(0, len(input_ids), 512)]

    chunk_embeddings = []

    for chunk in chunks:
        chunk_input = {'input_ids': chunk.unsqueeze(0).to(device),
                       'attention_mask': (chunk != tokenizer.pad_token_id).unsqueeze(0).to(device)}
        with torch.no_grad():
            outputs = model(**chunk_input)

        chunk_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        chunk_embeddings.append(chunk_embedding)

    if chunk_embeddings:
        embedding = np.mean(chunk_embeddings, axis=0)
    else:
        embedding = np.zeros(model.config.hidden_size)
    return embedding


def process_batch(batch, device):
    return [get_embedding(text) for text in batch]


def evaluate_fairness(y_true, y_pred, demographic_df, model_name="Model"):
    print(f"\n====== {model_name} Fairness Analysis (Bias Check) ======")

    res_df = demographic_df.copy()
    res_df['y_true'] = y_true
    res_df['y_pred'] = y_pred

    attributes = ['sex', 'age_group']
    metrics = []

    for attr in attributes:
        groups = res_df[attr].unique()
        for group in groups:
            if group == 'Unknown': continue

            sub_df = res_df[res_df[attr] == group]
            if len(sub_df) < 10: continue

            acc = accuracy_score(sub_df['y_true'], sub_df['y_pred'])
            f1 = f1_score(sub_df['y_true'], sub_df['y_pred'], average='weighted', zero_division=0)

            metrics.append({
                'Attribute': attr,
                'Group': group,
                'Sample_Size': len(sub_df),
                'Accuracy': acc,
                'F1_Weighted': f1
            })

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)
    metrics_df.to_csv(f"{model_name}_fairness_metrics.csv", index=False)

    return metrics_df


'''
    Vectors Functions
    Do not change unless needed. 
    Some model may need slightly modeified "Vectors Functions" if you are not using the models from 
    the hugging face. In that case you change it. 
'''


def gen_vectors(filtered_df, batch_size=32, vector_name="Vectors_A.xlsx"):
    max_len = 512
    overlap = 128

    train_embeddings = []

    for i in tqdm(range(0, len(filtered_df), batch_size), desc="Processing batches"):
        batch_records = filtered_df['text'].iloc[i:i + batch_size].tolist()
        embeddings_batch = process_batch(batch_records, device)

        for idx, embedding in enumerate(embeddings_batch):
            train_embeddings.append({"patient_id": filtered_df.index[i + idx], "embedding": embedding})

    train_embeddings_df = pd.DataFrame(train_embeddings)
    train_embeddings_df.to_excel(vector_name, index=False)
    print("Train embeddings have been saved to train_patient_embeddings_CBertA.xlsx")


'''
    Evaluation Functions
'''


###### ---->  K-Mean Clustering
def k_mean_simple(filtered_df, unique_label_count, vector_name="Vectors_A.xlsx"):
    train_embeddings_df = pd.read_excel(vector_name)

    embeddings = np.array(
        [np.fromstring(embedding_str.strip('[]'), sep=' ') for embedding_str in train_embeddings_df['embedding']])

    n_clusters = unique_label_count
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)

    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    cluster_to_label = {}

    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_disease_labels = filtered_df['label'].iloc[cluster_indices]
        most_common_label = cluster_disease_labels.mode()[0]
        cluster_to_label[cluster] = most_common_label

    predicted_labels = [cluster_to_label[cluster] for cluster in cluster_labels]
    true_labels = filtered_df['label'].values

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    print("K-Mean performance ----")
    print(f"  - Accuracy : {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall   : {recall:.4f}")
    print(f"  - F1 Score : {f1:.4f}")

    binarized_true_labels = label_binarize(true_labels, classes=np.unique(true_labels))
    predicted_binarized = label_binarize(predicted_labels, classes=np.unique(true_labels))

    auc_scores = []
    for i in range(n_clusters):
        auc = roc_auc_score(binarized_true_labels[:, i], predicted_binarized[:, i])
        auc_scores.append(auc)

    average_auc = np.mean(auc_scores)
    print(f" Average AUC across all diseases: {average_auc:.4f}")


###### ---->  K-Mean Clustering (weighted)
def k_mean_weighted(filtered_df, unique_label_count, vector_name="Vectors_A.xlsx",
                    label_counts_csv="filtered_label_counts.csv"):
    label_counts_df = pd.read_csv(label_counts_csv)

    max_count = label_counts_df['count'].max()
    label_counts_df['weight'] = max_count / label_counts_df['count']

    train_embeddings_df = pd.read_excel(vector_name)

    embeddings = np.array(
        [np.fromstring(embedding_str.strip('[]'), sep=' ') for embedding_str in train_embeddings_df['embedding']])

    data_labels = filtered_df['label'].values
    label_weights = label_counts_df.set_index('label')['weight'].to_dict()
    weights = np.array([label_weights.get(label, 1) for label in data_labels])

    n_clusters = len(np.unique(data_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)

    kmeans.fit(embeddings, sample_weight=weights)
    cluster_labels = kmeans.labels_

    cluster_to_label = {}

    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_disease_labels = filtered_df['label'].iloc[cluster_indices]
        most_common_label = cluster_disease_labels.mode()[0]
        cluster_to_label[cluster] = most_common_label

    predicted_labels = [cluster_to_label[cluster] for cluster in cluster_labels]
    true_labels = filtered_df['label'].values

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    print("K-Mean (weighted) performance ----")
    print(f"  - Accuracy : {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall   : {recall:.4f}")
    print(f"  - F1 Score : {f1:.4f}")

    binarized_true_labels = label_binarize(true_labels, classes=np.unique(true_labels))
    predicted_binarized = label_binarize(predicted_labels, classes=np.unique(true_labels))

    auc_scores = []
    for i in range(n_clusters):
        auc = roc_auc_score(binarized_true_labels[:, i], predicted_binarized[:, i])
        auc_scores.append(auc)

    average_auc = np.mean(auc_scores)
    print(f" Average AUC across all diseases: {average_auc:.4f}")


###### ----> agglomerative clustering
def agglomerative_clusters(filtered_df, unique_label_count, vector_name="Vectors_A.xlsx",
                           label_counts_csv="filtered_label_counts.csv"):
    train_embeddings_df = pd.read_excel(vector_name)

    embeddings = np.array(
        [np.fromstring(embedding_str.strip('[]'), sep=' ') for embedding_str in train_embeddings_df['embedding']])

    agglomerative = AgglomerativeClustering(n_clusters=unique_label_count, linkage='ward')
    agglomerative_labels = agglomerative.fit_predict(embeddings)

    cluster_to_label_agglomerative = {}

    for cluster in set(agglomerative_labels):
        cluster_indices = np.where(agglomerative_labels == cluster)[0]
        cluster_disease_labels = filtered_df['label'].iloc[cluster_indices]
        most_common_label = cluster_disease_labels.mode()[0]
        cluster_to_label_agglomerative[cluster] = most_common_label

    predicted_labels_agglomerative = [cluster_to_label_agglomerative[cluster] for cluster in agglomerative_labels]

    accuracy_agglomerative = accuracy_score(filtered_df['label'], predicted_labels_agglomerative)
    precision_agglomerative = precision_score(filtered_df['label'], predicted_labels_agglomerative, average='weighted',
                                              zero_division=1)
    recall_agglomerative = recall_score(filtered_df['label'], predicted_labels_agglomerative, average='weighted',
                                        zero_division=1)
    f1_agglomerative = f1_score(filtered_df['label'], predicted_labels_agglomerative, average='weighted',
                                zero_division=1)

    print("Agglomerative performance ----")
    print(f"  - Accuracy : {accuracy_agglomerative:.4f}")
    print(f"  - Precision: {precision_agglomerative:.4f}")
    print(f"  - Recall   : {recall_agglomerative:.4f}")
    print(f"  - F1 Score : {f1_agglomerative:.4f}")

    true_labels = filtered_df['label'].values
    binarized_true_labels = label_binarize(true_labels, classes=np.unique(true_labels))
    predicted_binarized = label_binarize(predicted_labels_agglomerative, classes=np.unique(true_labels))

    auc_scores = []
    for i in range(unique_label_count):
        auc = roc_auc_score(binarized_true_labels[:, i], predicted_binarized[:, i])
        auc_scores.append(auc)

    average_auc = np.mean(auc_scores)
    print(f" Average AUC across all diseases : {average_auc:.4f}")


###### ----> SVM Classifier
def svm_classifier(filtered_df, unique_label_count, vector_name="Vectors_A.xlsx",
                   label_counts_csv="filtered_label_counts.csv", kernel='rbf'):
    embeddings_df = pd.read_excel(vector_name)

    def convert_embedding(embedding_str):
        embedding_str = embedding_str.strip("[]")
        return [float(x) for x in embedding_str.split()]

    embeddings_df['embedding'] = embeddings_df['embedding'].apply(convert_embedding)

    X = pd.DataFrame(embeddings_df['embedding'].tolist())
    y = filtered_df['label'].values

    indices = np.arange(len(filtered_df))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.25, random_state=RANDOM_SEED
    )

    print("Training SVM...")
    svm_model = SVC(kernel=kernel, gamma='scale', random_state=RANDOM_SEED)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    print("SVM (rbf) performance ----")
    print(classification_report(y_test, y_pred))

    test_demographics = filtered_df.iloc[idx_test][['age_group', 'sex']].copy()
    evaluate_fairness(y_test, y_pred, test_demographics, model_name="SVM")


###### ----> Neural Networks (Weighted, for umbalance data)
def nn_classifier(filtered_df, unique_label_count, vector_name="Vectors_A.xlsx",
                  label_counts_csv="filtered_label_counts.csv"):
    embeddings_df = pd.read_excel(vector_name)

    def convert_embedding(embedding_str):
        embedding_str = embedding_str.strip("[]")
        return [float(x) for x in embedding_str.split()]

    embeddings_df['embedding'] = embeddings_df['embedding'].apply(convert_embedding)

    X = np.array(embeddings_df['embedding'].tolist())
    y = filtered_df['label'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=RANDOM_SEED)

    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / class_sample_count
    samples_weights = weights[y_train]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    sampler = WeightedRandomSampler(samples_weights, num_samples=len(y_train), replacement=True, generator=generator)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    class NeuralNetwork(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(NeuralNetwork, self).__init__()

            self.fc1 = nn.Linear(input_dim, 4096)
            self.bn1 = nn.BatchNorm1d(4096)
            self.dropout1 = nn.Dropout(0.1)

            self.fc2 = nn.Linear(4096, 2048)
            self.bn2 = nn.BatchNorm1d(2048)

            self.fc3 = nn.Linear(2048, 512)
            self.bn3 = nn.BatchNorm1d(512)

            self.fc4 = nn.Linear(512, 256)
            self.bn4 = nn.BatchNorm1d(256)

            self.fc5 = nn.Linear(256, 128)
            self.bn5 = nn.BatchNorm1d(128)

            self.fc6 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = torch.nn.functional.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
            x = self.dropout1(x)
            x = torch.nn.functional.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
            x = torch.nn.functional.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
            x = torch.nn.functional.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01)
            x = torch.nn.functional.leaky_relu(self.bn5(self.fc5(x)), negative_slope=0.01)
            x = self.fc6(x)
            return x

    model = NeuralNetwork(input_dim=X_train.shape[1], num_classes=len(np.unique(y_encoded)))

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    best_val_loss = float('inf')
    best_model_wts = None
    epochs = 150
    batch_size = 1024
    verbose = False
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        train_loader = DataLoader(dataset=torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
                                  batch_size=batch_size, sampler=sampler)

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(X_test_tensor)
            val_loss = criterion(val_output, y_test_tensor)
            val_losses.append(val_loss.item())

            _, predicted = torch.max(val_output, 1)
            correct = (predicted == y_test_tensor).sum().item()
            val_accuracy = correct / y_test_tensor.size(0)
            val_accuracies.append(val_accuracy)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

        if patience_counter >= patience:
            print("Early stopping: No improvement in validation loss for {} epochs.".format(patience))
            break

    model.load_state_dict(best_model_wts)

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        _, y_pred = torch.max(y_pred_tensor, 1)

    print("\n Classification Report : ")
    print(classification_report(y_test_tensor, y_pred))

    accuracy = accuracy_score(y_test_tensor, y_pred)
    print(f"- Accuracy : {accuracy:.4f}")
    precision = precision_score(y_test_tensor, y_pred, average='macro')
    print(f"- Precision : {precision:.4f}")
    recall = recall_score(y_test_tensor, y_pred, average='macro')
    print(f"- Recall    : {recall:.4f}")
    f1 = f1_score(y_test_tensor, y_pred, average='macro')
    print(f"- F1        : {f1:.4f}")


""" 
    Load your model here
    You can change it according to the model requirement
    if the model have some special requirements, you can code here as well.
    I used 2 models from the huggingface. So if you want to use the model from the hugging face
    just upload it to the new folder and provide the path. 
"""
model_name = "ehealth"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)

"""
    Main
    (Do not make changes, unless necessary)
"""

###### ---->  Data Pre-processing
json_file = 'final_emr.json'
columns_to_use = [
    'chief_complaint', 'present_illness', 'past_history', 'personal_history', 'marital_history', 'family_history',
    'infection_history', 'vaccination_history', 'surgery_history', 'bt_history', 'allergy_history',
    'menstruation_history', 'therapeutic_treatment', 'four_diagnosis', 'special_exam', 'physical_check'
]

filtered_df, train_df, test_df, unique_label_count = preprocess_data(json_file,
                                                                     columns_to_use=columns_to_use,
                                                                     cut_occurrences=471,
                                                                     save_labels_count=True,
                                                                     train_size_percent=0.8
                                                                     )

###### ---->  Vectorization
vector_name = "Vectors_A.xlsx"
gen_vectors(filtered_df=filtered_df, batch_size=32, vector_name=vector_name)

'''
    Evaluating the vectors
'''
vector_name = "Vectors_A.xlsx"

k_mean_simple(filtered_df=filtered_df, vector_name=vector_name, unique_label_count=unique_label_count)
print("\n - - - - - - - - - - - - - - - - - - - - - - - - - - \n ")

k_mean_weighted(filtered_df=filtered_df, vector_name=vector_name, label_counts_csv="filtered_label_counts.csv",
                unique_label_count=unique_label_count)
print("\n - - - - - - - - - - - - - - - - - - - - -  - - - - - \n ")

agglomerative_clusters(filtered_df, unique_label_count, vector_name=vector_name,
                       label_counts_csv="filtered_label_counts.csv")
print("\n - - - - - - - - - - - - - - - - - - - - - - - - - - \n ")

svm_classifier(filtered_df, unique_label_count, vector_name=vector_name, label_counts_csv="filtered_label_counts.csv")
print("\n - - - - - - - - - - - - - - - - - - - - - - - - - - \n ")

nn_classifier(filtered_df, unique_label_count, vector_name=vector_name, label_counts_csv="filtered_label_counts.csv")
print("\n - - - - - - - - - - - - - - - - - - - - - - - - - - \n ")