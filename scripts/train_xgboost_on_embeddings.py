import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the input files
embeddings_file = '/pscratch/sd/n/nberk/grass/balanced_shuffled_dataset.tsv'
labels_file = '/pscratch/sd/n/nberk/grass/balanced_shuffled_labels.tsv'

# Read embeddings and labels with Pandas
embeddings_df = pd.read_csv(embeddings_file, sep='\t', header=None)
labels_df = pd.read_csv(labels_file, sep='\t', usecols=[2], names=['label'], header=None)

# Combine embeddings and labels into a single DataFrame
combined_df = pd.concat([embeddings_df, labels_df], axis=1)

# Encode the labels
label_encoder = LabelEncoder()
combined_df['label'] = label_encoder.fit_transform(combined_df['label'])

# Split the data into features and labels
X = combined_df.iloc[:, :-1]
y = combined_df['label']

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=10)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=10)

# Train the XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),
    'eval_metric': 'mlogloss'
}

evals = [(dtrain, 'train'), (dval, 'eval')]

bst = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)

# Evaluate the model
y_pred = bst.predict(dtest)
predictions = pd.DataFrame(y_pred, columns=[f'class_{i}' for i in range(y_pred.shape[1])])
predictions['true_label'] = y_test.values
predictions['predicted_label'] = y_pred.argmax(axis=1)

# Decode the labels back to their original string representation
predictions['true_label'] = label_encoder.inverse_transform(predictions['true_label'])
predictions['predicted_label'] = label_encoder.inverse_transform(predictions['predicted_label'])

# Plot and save the histogram
plt.figure(figsize=(10, 6))
sns.histplot(predictions['predicted_label'], multiple="stack", bins=30)
plt.title('Prediction Distribution')
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.savefig('prediction_distribution_bal.png')
plt.close()

# Create and save the confusion matrix plot
conf_matrix = confusion_matrix(predictions['true_label'], predictions['predicted_label'], labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(20, 20))  # Adjust the figure size for long labels
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax, values_format='')  # Remove values from the heatmap
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_bal.png')
plt.close()

# Write the numerical data of the confusion matrix to a file
conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
conf_matrix_df.to_csv('confusion_matrix_bal.csv')

print("Plots and confusion matrix have been saved to files.")