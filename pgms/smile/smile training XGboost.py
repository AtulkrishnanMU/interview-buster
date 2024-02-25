import csv
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

RANDOM_SEED = 42

dataset = r'E:\website files\bodylang\myapp\pgms\smile\model\keypoint_classifier\keypoint.csv'
model_save_path = r'E:\website files\bodylang\myapp\pgms\smile\model\keypoint_classifier\xgboost_model.pkl'

NUM_CLASSES = 4

X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (478 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

# Split the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.2, random_state=RANDOM_SEED)

# Create an XGBoost model
xgb_model = XGBClassifier(random_state=RANDOM_SEED)

# Fit the XGBoost model on training data
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = xgb_model.predict(X_test)

# Calculate and print validation accuracy
val_accuracy = accuracy_score(y_test, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")

# Calculate precision, recall, and F-measure for each class
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_val_pred, average=None)

# Print precision, recall, and F-measure for each class
for i in range(NUM_CLASSES):
    print(f"Class {i + 1} - Precision: {precision[i]}, Recall: {recall[i]}, F-measure: {f1[i]}")

# Save the XGBoost model
joblib.dump(xgb_model, model_save_path)
