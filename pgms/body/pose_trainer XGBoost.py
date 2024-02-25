import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

RANDOM_SEED = 42

dataset = r'E:\project demo\pgms\body\model\keypoint_classifier\keypoint.csv'

NUM_CLASSES = 5

X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (25 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.8, random_state=RANDOM_SEED)

# XGBoost model initialization
xgb_model = XGBClassifier(n_estimators=100, random_state=RANDOM_SEED)

# Model training
xgb_model.fit(X_train, y_train)

# Model evaluation
accuracy = xgb_model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Classification report
y_pred = xgb_model.predict(X_test)
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Inference test
predict_result = xgb_model.predict([X_test[0]])
print(predict_result)

model_save_path = r'E:\project demo\pgms\body\model\keypoint_classifier\XGB_model.pkl'
joblib.dump(xgb_model, model_save_path)
