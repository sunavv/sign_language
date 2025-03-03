# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()

# 
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: data.pickle file not found")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Validate data
if data.size == 0 or labels.size == 0:
    raise ValueError("Empty data or labels")
if np.isnan(data).any() or np.isinf(data).any():
    raise ValueError("Data contains NaN or infinite values")
if data.shape[0] != labels.shape[0]:
    raise ValueError("Mismatch between number of samples and labels")

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

unique_classes = np.unique(labels)
print(f"Unique classes: {unique_classes}")
if len(unique_classes) < 2:
    raise ValueError("Not enough classes in the dataset for stratified splitting.")

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))
print("Detailed metrics:\n", classification_report(y_test, y_predict))

# Save model
try:
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")