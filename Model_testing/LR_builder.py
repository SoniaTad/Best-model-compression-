import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('balanced_data.csv')
print(df)
AVONA_XX = df.drop(['result'], axis=1)
print(AVONA_XX)
y_balanced = df['result']
print(y_balanced)

scaler = StandardScaler()
X_balanced_scaled = scaler.fit_transform(AVONA_XX)

X_train, X_test, y_train, y_test = train_test_split(X_balanced_scaled, y_balanced, test_size=0.2, random_state=42)

# Create a new KNN classifier with the best hyperparameters
best_logreg = LogisticRegression(multi_class='ovr',C=10,penalty='l1',solver= 'liblinear')

best_logreg.fit(X_train, y_train)

shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

# Calculate the training data accuracy for each fold using cross-validation
train_accuracy_scores = cross_val_score(best_logreg, X_train, y_train, cv=shuffle_split)

# Calculate the testing data accuracy
test_accuracy = best_logreg.score(X_test, y_test)

# Print the results
print("Training Data Accuracy for each fold:")
for fold, accuracy in enumerate(train_accuracy_scores):
    print(f"Fold {fold+1}: {accuracy}")
print("Testing Data Accuracy:", test_accuracy)

# Save the fitted scaler
joblib.dump(scaler, 'LR_scaler.pkl')

# Save the trained model
joblib.dump(best_logreg, 'LR_model.pkl')