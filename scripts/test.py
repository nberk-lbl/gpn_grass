import dask.dataframe as dd
import dask.array as da
from dask_ml.model_selection import train_test_split
from xgboost import dask as dxgb
from dask.distributed import Client
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def main():
    # Parameters
    test_size = 0.2
    random_state = 10

    # Load your actual data
    # Replace the following lines with the code to load your actual dataset
    # Make sure your data is in the form of a DataFrame with features and a label column

    # Example: Loading data from a CSV file
    # df = pd.read_csv('your_data.csv')
    # Assuming your data has columns 'feature_0', 'feature_1', ..., 'feature_n', and 'label'

    # For demonstration, we'll assume df is already loaded as a pandas DataFrame
    # df = pd.read_csv('path_to_your_csv_file.csv')

    # Convert to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=100)

    # Split into features and labels
    X = ddf.drop(columns=['label'])
    y = ddf['label']

    # Split into train and test sets
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
        'num_class': len(y.unique().compute()),  # Dynamically set the number of classes
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

    # Evaluate the model
    y_test_np = y_test.compute()
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
    plt.bar(range(len(y.unique().compute())), [sum(preds == i) for i in range(len(y.unique().compute()))])
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(range(len(y.unique().compute())), range(len(y.unique().compute())), rotation=90)
    plt.tight_layout()
    plt.savefig('predicted_classes.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()