# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import recall_score
# import numpy as np
# import matplotlib.pyplot as plt
# from ForestDiffusion import ForestDiffusionModel

# # Load the Iris dataset
# my_data = load_iris()
# X, y = my_data['data'], my_data['target']

# # Create an imbalanced dataset with class ratio 50:1 (minority class 1)
# class_0_indices = np.where(y == 0)[0]
# class_1_indices = np.where(y == 1)[0]
# class_2_indices = np.where(y == 2)[0]

# # Keep all samples of class 0 and 2, and reduce class 1 to just 1 sample
# X_class_0 = X[class_0_indices]
# X_class_2 = X[class_2_indices]
# X_class_1 = X[class_1_indices[:len(class_0_indices) // 50]]  # Reduce class 1 to ~50:1

# y_class_0 = y[class_0_indices]
# y_class_2 = y[class_2_indices]
# y_class_1 = y[class_1_indices[:len(class_0_indices) // 50]]

# # Combine back into an imbalanced dataset
# X_imbalanced = np.concatenate((X_class_0, X_class_1, X_class_2), axis=0)
# y_imbalanced = np.concatenate((y_class_0, y_class_1, y_class_2), axis=0)

# # Plot the original imbalanced data
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(X_imbalanced[:, 0], X_imbalanced[:, 1], c=y_imbalanced, cmap='viridis', label='Original Data')
# plt.title('Original Imbalanced Data')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

# # Separate the minority class (class 1)
# X_minority = X[y == 1]

# # Upsample the minority class using ForestDiffusionModel
# forest_model = ForestDiffusionModel(X_minority, label_y=None, n_t=50, duplicate_K=100, bin_indexes=[], cat_indexes=[], int_indexes=[], diffusion_type='flow', n_jobs=-1)
# X_minority_fake = forest_model.generate(batch_size=len(class_0_indices) // 2)  # Generate samples to reduce imbalance

# # Add generated samples to the main imbalanced dataset
# X_balanced = np.concatenate((X_imbalanced, X_minority_fake), axis=0)
# y_balanced = np.concatenate((y_imbalanced, np.ones(X_minority_fake.shape[0])), axis=0)

# # Plot the generated data
# plt.subplot(1, 2, 2)
# plt.scatter(X_balanced[:, 0], X_balanced[:, 1], c=y_balanced, cmap='viridis', label='Generated Data')
# plt.title('Data After Generation')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()

# # Split the datasets into train and test sets
# X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_imbalanced, y_imbalanced, test_size=0.3, random_state=42)
# X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# # Train a simple classifier on both the original and generated datasets
# from sklearn.linear_model import LogisticRegression

# clf_orig = LogisticRegression(max_iter=1000)
# clf_orig.fit(X_train_orig, y_train_orig)

# clf_bal = LogisticRegression(max_iter=1000)
# clf_bal.fit(X_train_bal, y_train_bal)

# # Predict and calculate recall score
# y_pred_orig = clf_orig.predict(X_test_orig)
# y_pred_bal = clf_bal.predict(X_test_bal)

# recall_orig = recall_score(y_test_orig, y_pred_orig, average='macro')
# recall_bal = recall_score(y_test_bal, y_pred_bal, average='macro')

# print(f"Recall score (original data): {recall_orig:.4f}")
# print(f"Recall score (generated data): {recall_bal:.4f}")
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from ForestDiffusion import ForestDiffusionModel
from sklearn.linear_model import LogisticRegression

# Create a binary imbalanced dataset with a 50:1 class ratio
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.98, 0.02],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=10000, random_state=10)

# Plot the original imbalanced data (only first two features for visualization)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Original Data', s=1)
plt.title('Original Imbalanced Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Separate the minority class (class 1)
X_minority = X[y == 1]

# Upsample the minority class using ForestDiffusionModel
forest_model = ForestDiffusionModel(X_minority, label_y=None, n_t=50, duplicate_K=100, bin_indexes=[], cat_indexes=[], int_indexes=[], diffusion_type='flow', n_jobs=-1)
X_minority_fake = forest_model.generate(batch_size=len(X) // 50)  # Generate samples to reduce imbalance

# Add generated samples to the main imbalanced dataset
X_balanced = np.concatenate((X, X_minority_fake), axis=0)
y_balanced = np.concatenate((y, np.ones(X_minority_fake.shape[0])), axis=0)

# Plot the generated data (only first two features for visualization)
plt.subplot(1, 2, 2)
plt.scatter(X_balanced[:, 0], X_balanced[:, 1], c=y_balanced, cmap='viridis', label='Generated Data', s=1)
plt.title('Data After Generation')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Split the datasets into train and test sets
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Train a simple classifier on both the original and generated datasets
clf_orig = LogisticRegression(max_iter=1000)
clf_orig.fit(X_train_orig, y_train_orig)

clf_bal = LogisticRegression(max_iter=1000)
clf_bal.fit(X_train_bal, y_train_bal)

# Predict and calculate recall and F1 scores
y_pred_orig = clf_orig.predict(X_test_orig)
y_pred_bal = clf_bal.predict(X_test_bal)

recall_orig = recall_score(y_test_orig, y_pred_orig)
recall_bal = recall_score(y_test_bal, y_pred_bal)

f1_orig = f1_score(y_test_orig, y_pred_orig)
f1_bal = f1_score(y_test_bal, y_pred_bal)

print(f"Recall score (original data): {recall_orig:.4f}")
print(f"Recall score (generated data): {recall_bal:.4f}")
print(f"F1 score (original data): {f1_orig:.4f}")
print(f"F1 score (generated data): {f1_bal:.4f}")
