import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys

labels_file = "/clusterfs/jgi/scratch/science/wcplant/nberkowitz/labels.tsv"
embeddings_file = "/clusterfs/jgi/scratch/science/wcplant/nberkowitz/embeddings.tsv"

# Load the embeddings and labels
print("loading embeddings")
embeddings = np.loadtxt(embeddings_file, delimiter='\t', skiprows=0)

print("loading labels")
labels = []
with open(labels_file) as lbf:
    lbf.readline() #skip header
    for n in lbf:
        idx, pos, region_label = n.split("\t")
        labels.append(region_label)

# Convert labels to numerical values
le = LabelEncoder()
labels = le.fit_transform(labels)


print("splitting test and train")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='multi:softmax', num_class=len(le.classes_))

# Fit the model
print("fitting model")
bst.fit(X_train, y_train)

print("saving")
bst.save_model('/clusterfs/jgi/scratch/science/wcplant/nberkowitz/xg_model.json')

print("creating figures")

# Make predictions
preds = bst.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, preds)

# Save the confusion matrix to a file
np.save('confusion_matrix.npy', cm)

# Plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.xlabel('Predicted Classes')
plt.ylabel('Actual Classes')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close()

# Create a bar chart of the predicted classes
plt.bar(range(len(le.classes_)), [sum(preds == i) for i in range(len(le.classes_))])
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(range(len(le.classes_)), le.classes_, rotation=90)
plt.tight_layout()
plt.savefig('predicted_classes.png', bbox_inches='tight')
plt.close()

# Save the predicted classes to a file
np.save('predicted_classes.npy', [sum(preds == i) for i in range(len(le.classes_))])

# Get the unique labels from both the training and test data
all_labels = np.unique(np.concatenate((y_train, y_test)))

# Create a classification report
from sklearn.metrics import classification_report
with open('classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, preds, labels=all_labels))
