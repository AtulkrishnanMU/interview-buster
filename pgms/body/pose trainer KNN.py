import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
import joblib  # Import joblib for model saving

RANDOM_SEED = 42

dataset = r'E:\project demo\pgms\body\model\keypoint_classifier\keypoint.csv'

NUM_CLASSES = 5

X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (25 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.8, random_state=RANDOM_SEED)

# KNN model initialization
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Model training
knn_model.fit(X_train, y_train)

# Save the trained model
model_save_path = r'E:\project demo\pgms\body\model\keypoint_classifier\knn_model.joblib'
joblib.dump(knn_model, model_save_path)
print(f"Model saved to: {model_save_path}")

# Model evaluation
accuracy = knn_model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Inference test
predict_result = knn_model.predict([X_test[0]])
print(predict_result)
