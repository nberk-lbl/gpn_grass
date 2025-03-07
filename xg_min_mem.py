import dask.dataframe as dd
import dask.array as da
from dask_ml.model_selection import train_test_split
from xgboost import dask as dxgb
from dask.distributed import Client
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

def main():
    # Parameters
    test_size = 0.2
    random_state = 10

    # Load your actual data
    #embeddings_file = 'minimal_embeddings.tsv'
    #labels_file = 'minimal_labels.tsv'
    labels_file = "/pscratch/sd/n/nberk/grass/labels_subset.tsv"
    embeddings_file = "/pscratch/sd/n/nberk/grass/embeddings.tsv"

    # Load embeddings
    embeddings_df = dd.read_csv(embeddings_file, sep='\t', header=None)
    
    # Load labels and extract the labels from the third column
    labels_df = dd.read_csv(labels_file, sep='\t', usecols=[2], names=['label'], header=None)
    
    # Check the first few rows of the embeddings and labels
    print("Embeddings DataFrame:")
    print(embeddings_df.head())
    print("Labels DataFrame:")
    print(labels_df.head())
    
    # Combine embeddings and labels into a single DataFrame
    df = dd.concat([embeddings_df, labels_df], axis=1)

    # Ensure the column names are appropriate
    df.columns = [f'feature_{i}' for i in range(df.shape[1] - 1)] + ['label']

    # Check the combined DataFrame
    print("Combined DataFrame:")
    print(df.head())

    # Encode the string labels to numeric values
    label_encoder = LabelEncoder()
    df['label'] = df['label'].map_partitions(lambda part: label_encoder.fit_transform(part))

    # Split into features and labels
    X = df.drop(columns=['label'])
    y = df['label']

    # Split into train and test sets using Dask's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

    # Convert to Dask arrays
    X_train = X_train.to_dask_array(lengths=True)
    X_test = X_test.to_dask_array(lengths=True)
    y_train = y_train.to_dask_array(lengths=True)
    y_test = y_test.to_dask_array(lengths=True)

    # Initialize Dask distributed client
    client = Client()

    # Initialize Dask XGBoost model
    params = {
        'objective': 'multi:softmax',
        'num_class': len(label_encoder.classes_),  # Dynamically set the number of classes
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss'
    }

    # Train the model
    dtrain = dxgb.DaskDMatrix(client, X_train, y_train)
    dtest = dxgb.DaskDMatrix(client, X_test, y_test)
    output = dxgb.train(client, params=params, dtrain=dtrain, num_boost_round=100, evals=[(dtest, 'test')])

    # Save the model
    output['booster'].save_model('xgboost_dask_model.json')

    # Make predictions
    preds = dxgb.predict(client, model=output, data=dtest)

    # Convert predictions to numpy array
    preds = preds.compute()

    # Decode the numeric predictions back to string labels
    preds = label_encoder.inverse_transform(preds.astype(int))

    # Evaluate the model
    y_test_np = y_test.compute()
    y_test_np = label_encoder.inverse_transform(y_test_np.astype(int))
    cm = confusion_matrix(y_test_np, preds)
    print("Confusion Matrix:")
    print(cm)

    report = classification_report(y_test_np, preds)
    print("Classification Report:")
    print(report)

    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.xlabel('Predicted Classes')
    plt.ylabel('Actual Classes')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # Create a bar chart of the predicted classes
    plt.bar(range(len(label_encoder.classes_)), [sum(preds == cls) for cls in label_encoder.classes_])
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=90)
    plt.tight_layout()
    plt.savefig('predicted_classes.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()