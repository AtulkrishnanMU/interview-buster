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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import joblib

# Paths to the saved models
model_paths = [
    r'E:\project demo\pgms\smile\model\keypoint_classifier\xgboost_model.pkl',
    r'E:\project demo\pgms\smile\model\keypoint_classifier\adaboost_model.pkl',
    r'E:\project demo\pgms\smile\model\keypoint_classifier\decision_tree_model.pkl',
    r'E:\project demo\pgms\smile\model\keypoint_classifier\knn_model.pkl',
    r'E:\project demo\pgms\smile\model\keypoint_classifier\logistic_regression_model.pkl',
    r'E:\project demo\pgms\smile\model\keypoint_classifier\nb_model.pkl',
    r'E:\project demo\pgms\smile\model\keypoint_classifier\random_forest_model.joblib',
    r'E:\project demo\pgms\smile\model\keypoint_classifier\svm_model.pkl'
]

# Load the test data
X_test, y_test

# Binarize the output
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3])

# Plot ROC curves for each class
plt.figure(figsize=(10, 8))

for model_path in model_paths:
    # Load the model
    model = joblib.load(model_path)

    # Make predictions on the test set
    y_pred_proba = OneVsRestClassifier(model).fit(X_test, y_test_binarized).predict_proba(X_test)

    # Compute ROC curve and area under the curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.plot(fpr[i], tpr[i], label=f'{model_path} (Class {i} AUC = {roc_auc[i]:.2f})')

# Plot the random guess line
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')

# Show the plot
plt.show()

