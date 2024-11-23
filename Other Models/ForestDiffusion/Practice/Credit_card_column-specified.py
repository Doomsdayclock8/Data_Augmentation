# In this file we have specoified column data type 
# inside forrest diffusion function

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from ForestDiffusion import ForestDiffusionModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from openTSNE import TSNE

# Step 1: Load the CSV file
file_path = 'creditcard.csv'
data = pd.read_csv(file_path)

# Step 2: Inspect the data and check for class imbalance
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Labels (binary classification)

# Check and print the original class distribution
unique, counts = np.unique(y, return_counts=True)
class_dist_before = dict(zip(unique, counts))
print(f"Class distribution before augmentation: {class_dist_before}")

# Step 3: Plot the original imbalanced data (first two features for visualization)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Original Data', s=1)
plt.title('Original Imbalanced Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Separate the minority class
X_minority = X[y == 1]

# Step 4: Upsample the minority class using ForestDiffusionModel
int_indexes = [0]  # Make the first column integer
cat_indexes = [29]  # Treat the 30th column as a mixed type (initially categorical)

forest_model = ForestDiffusionModel(X_minority, label_y=None, n_t=50, duplicate_K=100, 
                                   bin_indexes=[], 
                                   cat_indexes=cat_indexes, 
                                   int_indexes=int_indexes, 
                                   diffusion_type='flow', n_jobs=-1)

X_minority_fake = forest_model.generate(batch_size=len(X) // 50)

# Step 4.1: Calculate the proportion of integers in the 30th column of the original minority class
num_integers_in_original = np.sum(X_minority[:, 29] == X_minority[:, 29].astype(int))
proportion_integers = num_integers_in_original / X_minority.shape[0]

# Step 4.2: Apply the same proportion of integers to the 30th column of the generated data
num_integers_to_convert = int(proportion_integers * X_minority_fake.shape[0])
indices_to_convert = np.random.choice(X_minority_fake.shape[0], num_integers_to_convert, replace=False)

# Convert the selected values in the 30th column to integers
X_minority_fake[indices_to_convert, 29] = X_minority_fake[indices_to_convert, 29].astype(int)

# Add generated samples to the main imbalanced dataset
X_balanced = np.concatenate((X, X_minority_fake), axis=0)
y_balanced = np.concatenate((y, np.ones(X_minority_fake.shape[0])), axis=0)

# Save the generated data to a CSV file
# Combine X_balanced and y_balanced into a single DataFrame
data_balanced = pd.DataFrame(X_balanced)
data_balanced['target'] = y_balanced  # Add the target variable as the last column

# Save to CSV file
data_balanced.to_csv('generated_data.csv', index=False)
print("Generated data saved to 'generated_data.csv'.")

# Step 5: Plot the generated data (first two features for visualization)
plt.subplot(1, 2, 2)
plt.scatter(X_balanced[:, 0], X_balanced[:, 1], c=y_balanced, cmap='viridis', label='Generated Data', s=1)
plt.title('Data After Generation')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Check and print the class distribution after augmentation
unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
class_dist_after = dict(zip(unique_bal, counts_bal))
print(f"Class distribution after augmentation: {class_dist_after}")