import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

train_csv = 'train.csv'
train_data = pd.read_csv(train_csv, encoding='ISO-8859-1')
dev_csv = 'dev.csv'
dev_data = pd.read_csv(dev_csv, encoding='ISO-8859-1')
test_csv = 'x_test.csv'
X_test = pd.read_csv(test_csv, encoding='ISO-8859-1')

X_train = train_data.drop(columns='Label')
y_train = train_data['Label']

X_dev = dev_data.drop(columns='Label')
y_dev = dev_data['Label']


dummy_model = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_model.fit(X_train, y_train)

# #Make Predictions
Model_1_Predictions = dummy_model.predict(X_dev)

# Model_1_Predictions = Model_1.predict(X_test)


# # #Display Results
print(f"dummy_model: {f1_score(y_dev, Model_1_Predictions)}")

# print(f"Model 1: {f1_score(y_test, dummy_model_Predictions)}")

print("dummy model")
print(classification_report(y_dev, Model_1_Predictions))