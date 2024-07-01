import pandas as pd
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import shap
import lightgbm as lgb
import os
import joblib
import plotly.express as px


from bs4 import BeautifulSoup
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, learning_curve
from tabulate import tabulate
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from IPython.display import display, HTML

def parse_credit_data(url):
    """
    Fetches and parses HTML content from a URL,
    extracts credit attribute data blocks,
    and converts them into a pandas DataFrame.

    Parameters:
    - url (str): URL of the web page containing the credit data.

    Returns:
    - attributes_df (pandas.DataFrame): DataFrame with parsed credit attributes and values.
      Returns None if no relevant data block is found.
    """
    
    # Fetch HTML content from the web page
    response = requests.get(url)
    html_content = response.content

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract all data blocks with the specific class
    data_blocks = soup.find_all('p', class_='whitespace-pre-wrap svelte-17wf9gp')

    # Check and select the required block
    for block in data_blocks:
        # Get the text of the data block
        text_data = block.get_text(separator='\n')

        # Check if the block contains the required information (e.g., presence of 'Attribute 1')
        if 'Attribute 1' in text_data:
            # Split the text into lines
            lines = text_data.split('\n')

            # Initialize variables to store attributes and their values
            attributes = []
            current_attribute = None

            # Parse lines and extract attributes and their values
            for line in lines:
                line = line.strip()
                if line.startswith('Attribute'):
                    # Start of a new attribute
                    if current_attribute:
                        attributes.append(current_attribute)
                    current_attribute = {'name': line, 'values': []}
                elif line and current_attribute:
                    # Add value to the current attribute
                    current_attribute['values'].append(line)

            # Add the last attribute to the list
            if current_attribute:
                attributes.append(current_attribute)

            # Create a list to store data in a format suitable for DataFrame
            data_for_df = []

            # Form data for DataFrame
            for attribute in attributes:
                name = attribute['name']
                for value in attribute['values']:
                    data_for_df.append({'Attribute': name, 'Value': value})

            # Create DataFrame from the data
            attributes_df = pd.DataFrame(data_for_df)

            # Return the DataFrame
            return attributes_df

    # Return None if the required data block is not found
    return None

def calculate_cost(conf_matrix, cost_fp, cost_fn):
    """
    Calculates the total cost based on a confusion matrix,
    false positive (FP) cost, and false negative (FN) cost.

    Parameters:
    - conf_matrix (numpy.ndarray): Confusion matrix array.
    - cost_fp (float): Cost per false positive prediction.
    - cost_fn (float): Cost per false negative prediction.

    Returns:
    - cost (float): Total cost computed from the confusion matrix and costs.
    """
    
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    cost = (FP * cost_fp) + (FN * cost_fn)
    return cost

def evaluate_models(X_train, X_test, y_train, y_test, description, features_type, models, cost_fp, cost_fn):
    """
    Evaluate multiple models by training on `X_train` and `y_train`, 
    testing on `X_test` and `y_test`, and collecting evaluation metrics including accuracy, 
    confusion matrix, precision, recall, F1-score, and custom cost calculation.

    Args:
    - X_train (DataFrame): Training data features.
    - X_test (DataFrame): Test data features.
    - y_train (Series): Training data target.
    - y_test (Series): Test data target.
    - description (str): Description or identifier for the evaluation.
    - features_type (str): Type or description of the features used.

    Returns:
    - results (list of dicts): List containing dictionaries with evaluation results for each model.
    Each dictionary includes metrics such as accuracy, precision, recall, F1-score, and custom cost.
    """
    
    # class weights
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    # training and testing for each model
    results = []

    for model_name, model in models.items():

        if model_name == "LightGBM":
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                'max_depth': 10,  
                'min_child_samples': 20  
            }
            model = lgb.LGBMClassifier(**params)

        # training
        model.fit(X_train, y_train)

        # testing
        y_pred = model.predict(X_test)
        
        # evaluation
        accuracy = model.score(X_test, y_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        # class_report = classification_report(y_test, y_pred, output_dict=True)
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        cost = calculate_cost(conf_matrix, cost_fp, cost_fn)
        
        print(model_name)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", conf_matrix)
        print("Cost:", cost)
        
        # results to dataframe
        results.append({
            "Description": description,
            "Features": features_type,
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision Class 1": class_report['0']['precision'],
            "Precision Class 2": class_report['1']['precision'],
            "Recall Class 1": class_report['0']['recall'],
            "Recall Class 2": class_report['1']['recall'],
            "F1-Score Class 1": class_report['0']['f1-score'],
            "F1-Score Class 2": class_report['1']['f1-score'],
            "Cost": cost
        })

    return results

def evaluate_GB(X, y, model):
    """
    Train a Gradient Boosting model on data X and target y, and evaluate its performance.

    Args:
    - X (DataFrame): Features dataset.
    - y (Series): Target variable.
    - model: Gradient Boosting model instance.

    Returns:
    - mean_accuracy (float): Mean cross-validated accuracy score.
    - selected_features (array): Selected top features based on model's feature importance.
    """
    # Train the model and get the feature importance
    model.fit(X, y)
    
    # Get the feature importance or selected feature indices depending on the model
    if isinstance(model, GradientBoostingClassifier):
        feature_importances = model.feature_importances_
        top_features_indices = np.argsort(feature_importances)[::-1][:10]  # Example: Select top 10 features
        if isinstance(X, pd.DataFrame):
            selected_features = X.columns[top_features_indices]
        else:
            selected_features = top_features_indices
    else:
        selected_features = np.arange(X.shape[1])  # For other models, return all feature indices
    
    # Calculate cross-validated accuracy
    scores = cross_val_score(model, X[:, top_features_indices] if isinstance(X, np.ndarray) else X.iloc[:, top_features_indices], y, cv=5)
    mean_accuracy = np.mean(scores)
    
    return mean_accuracy, selected_features

def evaluate_models_new( models, X_train, X_test, y_train, y_test, description, cost_fp, cost_fn):
    """
    Evaluate multiple models based on various metrics including accuracy, precision, recall,
    F1-score, and cost, using confusion matrix analysis and class weights.

    Args:
    - models (dict): Dictionary of models where keys are model names and values are model objects.
    - X_train (DataFrame): Training data features.
    - X_test (DataFrame): Test data features.
    - y_train (Series): Training data target.
    - y_test (Series): Test data target.
    - description (str): Description or identifier for the evaluation session.

    Returns:
    - results (list of dict): List of dictionaries containing evaluation results for each model.
    Each dictionary includes metrics such as Accuracy, Precision, Recall, F1-score, and Cost.
    """
    
    # Calculating class weights
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    
    # Training and testing for each model
    results = []

    for model_name, model in models.items():
        if model_name == "LightGBM":
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                'max_depth': 10,  
                'min_child_samples': 20  
            }
            model = lgb.LGBMClassifier(**params)

        # Training
        model.fit(X_train, y_train)

        # Testing
        y_pred = model.predict(X_test)
        
        # Evaluation
        accuracy = model.score(X_test, y_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        cost = calculate_cost(conf_matrix, cost_fp, cost_fn)
        
        # Storing results in DataFrame
        results.append({
            "Description": description,
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision Class 0": class_report['0']['precision'],
            "Precision Class 1": class_report['1']['precision'],
            "Recall Class 0": class_report['0']['recall'],
            "Recall Class 1": class_report['1']['recall'],
            "F1-Score Class 0": class_report['0']['f1-score'],
            "F1-Score Class 1": class_report['1']['f1-score'],
            "Cost": cost
        })

    return results

def evaluate_models_gr(X_train, X_test, y_train, y_test, description, features_type, models, cost_fp, cost_fn, param_grid=None):
    """
    Evaluate multiple models using GridSearchCV for parameter tuning (if specified) 
    and compute evaluation metrics including accuracy, precision, recall, F1-score, and cost.

    Args:
    - X_train (DataFrame): Training data features.
    - X_test (DataFrame): Test data features.
    - y_train (Series): Training data target.
    - y_test (Series): Test data target.
    - description (str): Description of the evaluation scenario.
    - features_type (str): Type of features used in the evaluation.
    - param_grid (dict, optional): Grid of hyperparameters for GridSearchCV (default=None).

    Returns:
    - results (list of dicts): List containing evaluation results for each model, 
    including metrics such as accuracy, precision, recall, F1-score, and cost.
    """
    
    # Compute class weights
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    
    # training and testing for each model
    results = []
    # Iterate over each model
    for model_name, model in models.items():
        if param_grid:
            # Initialize GridSearchCV for parameter tuning
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            
            # Perform grid search on training data
            grid_search.fit(X_train, y_train)
            
            # Best parameters found
            best_params = grid_search.best_params_
            print(f"Best parameters for {model_name}: {best_params}")
            
            # Use the best model found
            model = grid_search.best_estimator_
        
        # Train the model (either with or without parameter tuning)
        model.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        cost = calculate_cost(conf_matrix, cost_fp, cost_fn)
        
        # Save results for the model
        results.append({
            "Description": description,
            "Features": features_type,
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision Class 1": class_report['0']['precision'],
            "Precision Class 2": class_report['1']['precision'],
            "Recall Class 1": class_report['0']['recall'],
            "Recall Class 2": class_report['1']['recall'],
            "F1-Score Class 1": class_report['0']['f1-score'],
            "F1-Score Class 2": class_report['1']['f1-score'],
            "Cost": cost
        })

    return results

def evaluate_performance_with_features(model, X_train, y_train, X_test, y_test, selected_features, pos_label):
    """
    Evaluates the performance of a given model on selected features using precision score.

    Parameters:
    - model (sklearn estimator): The model to evaluate.
    - X_train (DataFrame): Training data.
    - y_train (Series): Training labels.
    - X_test (DataFrame): Test data.
    - y_test (Series): Test labels.
    - selected_features (list): List of selected features for evaluation.
    - pos_label (int): The positive class label for precision calculation.

    Returns:
    - precision (float): Precision score of the model on the selected features.
    """

    model.fit(X_train[selected_features], y_train)
    y_pred = model.predict(X_test[selected_features])

    precision = precision_score(y_test, y_pred, pos_label=pos_label)

    return precision

def calculate_shap_feature_importance(X_train, y_train, models):
    """
    Calculate SHAP feature importance for each model in `models` dictionary.

    Args:
    - X_train (DataFrame): Training data features.
    - y_train (Series): Training data target.
    - models (dict): Dictionary of models where keys are model names and values are model objects.

    Returns:
    - shap_results (dict): Dictionary containing SHAP feature importance results for each model.
    """
    
    shap_results = {}

    for name, model in models.items():
        if isinstance(model, LogisticRegression):
            print(f"Applying SHAP for LogisticRegression model: {name}")
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer.shap_values(X_train)
        elif isinstance(model, RandomForestClassifier):
            print(f"Applying SHAP for RandomForestClassifier model: {name}")
            explainer = shap.TreeExplainer(model, X_train)
            shap_values = explainer.shap_values(X_train)
        else:
            print(f"Model {name} is not supported for SHAP analysis.")
            continue

        # Create a DataFrame for feature importance
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'shap_mean': np.zeros(X_train.shape[1]),
            'shap_std': np.zeros(X_train.shape[1])
        })

        # Calculate mean SHAP values and standard deviations
        shap_mean = np.mean(np.abs(shap_values), axis=0)
        shap_std = np.std(np.abs(shap_values), axis=0)

        # Add calculated values to DataFrame
        feature_importance_df['shap_mean'] = shap_mean
        feature_importance_df['shap_std'] = shap_std

        # Sort by descending mean SHAP value
        feature_importance_df = feature_importance_df.sort_values(by='shap_mean', ascending=False)

        # Save SHAP results for the current model
        shap_results[name] = feature_importance_df

    return shap_results

def calculate_shap_feature_importance_classes(X_train, y_train, models, target_class):
    """
    Calculate SHAP feature importance for each model in `models` dictionary 
    focusing on a specific target class.

    Args:
    - X_train (DataFrame): Training data features.
    - y_train (Series): Training data target.
    - models (dict): Dictionary of models where keys are model names and values are model objects.
    - target_class (int): Index of the target class to analyze.

    Returns:
    - shap_results (dict): Dictionary containing SHAP feature importance results for each model.
    """
    
    shap_results = {}

    for name, model in models.items():
        # Train the model on X_train and y_train
        model.fit(X_train, y_train)

        if isinstance(model, LogisticRegression):
            print(f"Applying SHAP for LogisticRegression model: {name}")
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer.shap_values(X_train)
        elif isinstance(model, RandomForestClassifier):
            print(f"Applying SHAP for RandomForestClassifier model: {name}")
            explainer = shap.TreeExplainer(model, X_train)
            shap_values = explainer.shap_values(X_train)
        else:
            print(f"Model {name} is not supported for SHAP analysis.")
            continue

        # Create a DataFrame for feature importance
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'shap_mean': np.zeros(X_train.shape[1]),
            'shap_std': np.zeros(X_train.shape[1])
        })

        # Calculate mean SHAP values and standard deviations for the selected class
        shap_mean = np.mean(np.abs(shap_values[target_class]), axis=0)
        shap_std = np.std(np.abs(shap_values[target_class]), axis=0)

        # Add calculated values to DataFrame
        feature_importance_df['shap_mean'] = shap_mean
        feature_importance_df['shap_std'] = shap_std

        # Sort by descending mean SHAP value
        feature_importance_df = feature_importance_df.sort_values(by='shap_mean', ascending=False)

        # Save SHAP results for the current model
        shap_results[name] = feature_importance_df

    return shap_results

def calculate_shap_feature_importance_classes_bal(X_train, y_train, models, target_class):
    """
    Calculate SHAP feature importance for each model in `models` dictionary 
    focusing on a specific target class with class balancing applied for logistic regression.

    Args:
    - X_train (DataFrame): Training data features.
    - y_train (Series): Training data target.
    - models (dict): Dictionary of models where keys are model names and values are model objects.
    - target_class (int): Index of the target class to analyze.

    Returns:
    - shap_results (dict): Dictionary containing SHAP feature importance results for each model.
    """
    
    shap_results = {}

    for name, model in models.items():
        # Class balancing for LogisticRegression (may not be supported for other models)
        if isinstance(model, LogisticRegression):
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            model = LogisticRegression(class_weight=dict(zip(np.unique(y_train), class_weights)))
        
        # Train the model on X_train and y_train
        model.fit(X_train, y_train)

        if isinstance(model, LogisticRegression):
            print(f"Applying SHAP for LogisticRegression model: {name}")
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer.shap_values(X_train)
        elif isinstance(model, RandomForestClassifier):
            print(f"Applying SHAP for RandomForestClassifier model: {name}")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        else:
            print(f"Model {name} is not supported for SHAP analysis.")
            continue

        # Create a DataFrame for feature importance
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,  # Use original feature names
            'shap_mean': np.zeros(X_train.shape[1]),
            'shap_std': np.zeros(X_train.shape[1])
        })

        # Calculate mean SHAP values and standard deviations for the selected class
        shap_mean = np.mean(np.abs(shap_values[target_class]), axis=0)
        shap_std = np.std(np.abs(shap_values[target_class]), axis=0)

        # Add calculated values to DataFrame
        feature_importance_df['shap_mean'] = shap_mean
        feature_importance_df['shap_std'] = shap_std

        # Sort by descending mean SHAP value
        feature_importance_df = feature_importance_df.sort_values(by='shap_mean', ascending=False)

        # Save SHAP results for the current model
        shap_results[name] = feature_importance_df

    return shap_results

def evaluate_model_overfitting(model, X, y, feature_names, test_size=0.2, random_state=42, cv=5):
    """
    Evaluate a machine learning model for overfitting by training it on a subset 
    of features, then assessing its performance on training, testing data, 
    and using cross-validation. Also plots the learning curve to visualize 
    training/testing scores and variances.

    Args:
    - model (estimator): Machine learning model to evaluate.
    - X (DataFrame): Features dataset.
    - y (Series): Target variable.
    - feature_names (list): List of feature names to select from X.
    - test_size (float, optional): Proportion of the dataset to include in the test split (default=0.2).
    - random_state (int, optional): Random state for reproducibility (default=42).
    - cv (int, optional): Number of folds in cross-validation (default=5).

    Returns:
    - dict: Dictionary containing evaluation metrics including train/test accuracies, 
    cross-validation mean and standard deviation accuracies, and learning curve data.
    """
    
    # Select the required features
    X_selected = X[feature_names]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on training and testing data
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_selected, y, cv=cv)
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Plotting the learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X_selected, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    # Calculate the mean and standard deviation for training and testing data
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std(),
        "learning_curve": {
            "train_sizes": train_sizes,
            "train_scores_mean": train_scores_mean,
            "train_scores_std": train_scores_std,
            "test_scores_mean": test_scores_mean,
            "test_scores_std": test_scores_std
        }
    }

def evaluate_threshold(threshold, predictions, y_true, cost_fp, cost_fn):
    """
    Evaluate model performance and cost at a given threshold.

    Parameters:
    threshold (float): The probability threshold for classifying positive class.
    predictions (array-like): The predicted probabilities from the model.
    y_true (array-like): The true class labels.
    cost_fp (float): The cost associated with false positives.
    cost_fn (float): The cost associated with false negatives.

    Returns:
    dict: A dictionary containing the threshold, total cost, precision, recall,
          f1-score, and accuracy for the given threshold.
    """
    
    predicted_class = (predictions > threshold).astype(int)
    report = classification_report(y_true, predicted_class, output_dict=True)
    cm = confusion_matrix(y_true, predicted_class)
    
    # Calculate cost based on confusion matrix and costs
    tn, fp, fn, tp = cm.ravel()
    total_cost = fp * cost_fp + fn * cost_fn
    
    return {
        'threshold': threshold,
        'cost': total_cost,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1-score': report['1']['f1-score'],
        'accuracy': report['accuracy']
    }

def display_model_description(description):
    """
    Display model description in a formatted HTML output.

    Args:
    - description (dict): Dictionary containing model description information.

    Returns:
    - None
    """
    for key, value in description.items():
        if isinstance(value, (list, dict)):
            try:
                # Convert value to DataFrame if it's a list or dictionary
                value_df = pd.DataFrame(value)
                display(HTML(f'<h3>{key}</h3>'))
                display(value_df)
            except ValueError as e:
                # If conversion to DataFrame fails, display an error message and show the raw value
                display(HTML(f'<h3>{key}</h3><p>Error converting to DataFrame: {str(e)}</p>'))
                display(HTML(f'<pre>{value}</pre>'))
        else:
            # Display key-value pair as a header and paragraph if value is not list or dictionary
            display(HTML(f'<h3>{key}</h3><p>{value}</p>'))
