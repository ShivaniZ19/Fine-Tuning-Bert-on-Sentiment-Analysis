import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm
import seaborn as sns
from IPython.display import Markdown
import matplotlib.pyplot as plt
from pandas.plotting import table

# Function to calculate class weights
def calculate_class_weights(class_counts):
    total = sum(class_counts)
    weights = [total / class_count for class_count in class_counts]
    return torch.tensor(weights, dtype=torch.float32)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=2)

# Print the model's architecture
print(model)
# Print named parameters with shapes
for name, param in model.named_parameters():
    print(f"{name}: {param.size()}")
# Calculate and print the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")


# Load a small subset of training and testing data (10 rows each)
train_df = pd.read_csv('NTC_SV_train.csv', sep=None, header=0, encoding='utf-8', engine='python')
test_df = pd.read_csv('NTC_SV_test.csv', sep=None, header=0, encoding='utf-8', engine='python')

# Function to clean and prepare data
def prepare_data(df, column_name='review'):
    print(f"Initial null values in '{column_name}':", df[column_name].isnull().sum())
    print(f"Non-string values in '{column_name}':", df[column_name].apply(lambda x: isinstance(x, str)).sum())

    # Drop rows where the column is NaN and convert all to strings
    df = df.dropna(subset=[column_name])
    df[column_name] = df[column_name].astype(str)
    return df

# Prepare training and testing data
train_df = prepare_data(train_df)
test_df = prepare_data(test_df)

# Tokenize the data with padding and create attention masks
train_encodings = tokenizer(train_df['review'].tolist(), truncation=True, padding=True, max_length=256, return_tensors="pt")
test_encodings = tokenizer(test_df['review'].tolist(), truncation=True, padding=True, max_length=256, return_tensors="pt")

# Define a custom dataset
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        # Pass the labels directly if they are already tensors
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Convert labels to a tensor
train_labels = torch.tensor(train_df['label'].tolist())
test_labels = torch.tensor(test_df['label'].tolist())

# Calculate class weights
class_weights = calculate_class_weights([sum(train_labels == 0), sum(train_labels == 1)])

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# DataLoader
def collate_fn(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Loss function with class weights
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*3)

# Metrics tracking
epoch_metrics = {
    'loss': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

# Train and evaluate the model
model.train()
for epoch in range(5):  # More epochs for better training
    running_loss = 0.0
    running_corrects = 0
    total_examples = 0
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        optimizer.zero_grad()

        # Include attention mask in the model's forward pass
        input_ids = batch['input_ids'].to(torch.long)
        attention_mask = batch['attention_mask'].to(torch.long)
        labels = batch['labels'].to(torch.long)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_function(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()  # Update the learning rate

        # Calculate running loss and accuracy
        running_loss += loss.item() * input_ids.size(0)
        running_corrects += torch.sum(torch.argmax(outputs.logits, axis=1) == labels)
        total_examples += input_ids.size(0)

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_examples
    epoch_acc = running_corrects.double() / total_examples

    epoch_metrics['loss'].append(epoch_loss)
    epoch_metrics['accuracy'].append(epoch_acc.item())

    # Evaluate model performance after each epoch
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(torch.long)
            attention_mask = batch['attention_mask'].to(torch.long)
            labels = batch['labels'].to(torch.long)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, axis=1).tolist())
            true_labels.extend(labels.tolist())

    # Calculate performance metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    epoch_metrics['precision'].append(precision)
    epoch_metrics['recall'].append(recall)
    epoch_metrics['f1'].append(f1)

    print(f'Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

# Plot confusion matrix
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
plt.close()

# Line chart for each metric over epochs
plt.figure(figsize=(10, 5))
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    # epoch_metrics[metric] is the 'y' array
    y_length = len(epoch_metrics[metric])
    plt.plot(range(1, y_length + 1), epoch_metrics[metric], label=metric.capitalize())
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Metric Scores Over Epochs')
plt.legend()
plt.savefig('metrics_over_epochs.png')
plt.show()
plt.close()

# Bar chart summarizing overall performance
overall_metrics = {metric: sum(epoch_metrics[metric]) / len(epoch_metrics[metric]) for metric in epoch_metrics}
plt.bar(overall_metrics.keys(), overall_metrics.values(), color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Overall Model Performance')
plt.ylim([0, 1])
plt.savefig('overall_performance.png')
plt.show()
plt.close()

# Generate the headers and data for the table
metrics_headers = ['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1']
data = {metric: epoch_metrics[metric] for metric in metrics_headers[1:]}
data['Epoch'] = list(range(1, len(epoch_metrics['loss']) + 1))

# Convert to DataFrame
df_metrics = pd.DataFrame(data)
df_metrics.set_index('Epoch', inplace=True)

# Adding overall metrics to the DataFrame
overall_metrics = {metric: np.mean(epoch_metrics[metric]) for metric in metrics_headers[1:]}
overall_metrics['Epoch'] = "Overall"
df_metrics = df_metrics.append(overall_metrics, ignore_index=True)

# Setting 'Epoch' as the last row for 'Overall'
df_metrics.at[len(df_metrics) - 1, 'Epoch'] = "Overall"
df_metrics.set_index('Epoch', inplace=True)

# Creating the table as a figure
fig, ax = plt.subplots(figsize=(10, 2))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
table_data = table(ax, df_metrics, loc='center', colWidths=[0.1] * len(df_metrics.columns))
plt.savefig('training_metrics_table.png', bbox_inches='tight')
plt.show()
plt.close(fig)

