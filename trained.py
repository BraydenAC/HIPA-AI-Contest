import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
csv_file = '/content/Compiled Annotations Distribution.csv'
data = pd.read_csv(csv_file, encoding='ISO-8859-1')

# Import libraries
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

# Load the csv with proper encoding
df = pd.read_csv('/content/Compiled Annotations Distribution.csv', encoding='ISO-8859-1')

# Extract texts and labels
texts = df['Features'].tolist()
labels = df['Label'].tolist()

# Load BERT
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Tokenize and encode the text inputs
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Pass the inputs through the BERT model
with torch.no_grad():
    outputs = model(**inputs)

# Extract the sentence embeddings
sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # shape: [batch_size, hidden_state_size]

# Save embeddings to avoid recomputation
pd.DataFrame(sentence_embeddings).to_csv('sentence_embeddings.csv', index=False)
pd.DataFrame(labels, columns=['Label']).to_csv('labels.csv', index=False)


# Load embeddings and labels
embeddings = pd.read_csv('sentence_embeddings.csv').values
labels = pd.read_csv('labels.csv').values.ravel()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42)

# Train the logistic regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate f1 score

print(f1_score(y_test, y_pred, pos_label='Yes'))
