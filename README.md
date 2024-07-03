# **Credit Scoring Model Development**

![Illustration](bank.jpg)


**Objective:**

The objective of this project is to develop a credit scoring model using machine learning techniques to predict the likelihood of a borrower defaulting on a loan. Key steps include:

- **Data Preprocessing** Preprocessed the dataset by handling missing values, encoding categorical variables, and scaling numerical features.

- **Feature Selection** Selected relevant features based on their correlation and importance using a Gradient Boosting model.

- **Model Training and Evaluation** Trained and evaluated the model using cross-validation techniques to optimize its performance.

- **Model Application** Implemented the trained model for making predictions on new data.

- **Threshold Optimization** Determined optimal thresholds for classification to enhance model performance.

- **Results** Stored and visualized the outcomes of the model evaluation, focusing on its effectiveness with the selected features.

---

**Outcomes**

After executing this code, the following files are saved in the repository:

| Name | Format | Contents |
|-----------------|-----------------|-----------------|
| german_credit.csv |  CSV | DataFrame german_credit data saved without indices |
| gradient_boosting_model.pkl | Pickle (.pkl) | Trained Gradient Boosting model used for credit scoring |
| model_description.txt | Text file (.txt) | Detailed description of the credit scoring model, encompassing model parameters, selected features, feature importances, and example predictions with class probabilities|
| selected_features.pkl | Pickle (.pkl) | List of selected features used in model training |
| standard_scaler.pkl | Trained Scaler object for data normalization |


**Note**: All these files are essential for further use of the model and analysis of results within the credit scoring project.

---

**Technologies**: Python (pandas, numpy, matplotlib.pyplot, seaborn, scikit-learn, shap, lightgbm, joblib, plotly.express, pickle, BeautifulSoup, ucimlrepo, tabulate, imbalanced-learn, xgboost, catboost, lime)

Developed a credit scoring model using machine learning techniques to predict the likelihood of a borrower defaulting on a loan. Leveraged a comprehensive stack of tools and libraries, including:

- **Data Manipulation and Analysis**: pandas, numpy

- **Data Visualization**: matplotlib.pyplot, seaborn, plotly.express

- **Machine Learning and Evaluation**: scikit-learn (Logistic Regression, Lasso, Ridge, ElasticNet, RandomForestClassifier, GradientBoostingClassifier, confusion_matrix, classification_report, accuracy_score, precision_score, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score, StandardScaler, train_test_split, GridSearchCV, learning_curve, cross_val_score, SVC, KNeighborsClassifier, GaussianNB, MLPClassifier, KMeans), xgboost (XGBClassifier), lightgbm (LGBMClassifier), catboost (CatBoostClassifier)

- **Feature Importance and Model Interpretation**: shap, lime (lime.lime_tabular)

- **Handling Imbalanced Data**: imbalanced-learn (SMOTE)

- **Data Scraping**: BeautifulSoup, requests

- **Model Persistence**: joblib, pickle

- **Data Retrieval**: ucimlrepo

- **Tabular Data Display**: tabulate

- **Custom Scoring Functions**: custom functions for parsing data, calculating costs, evaluating models, and assessing model performance and feature importance

This extensive toolkit ensured the robust development and thorough evaluation of the credit scoring model.

---

**Folder File Description**

| Folder | File | Description |
|-----------------|-----------------|-----------------|
| Functions | scoring_functions.py |Python script containing scoring functions for evaluating model performance and analyzing data     |
|Python | SLEBID_Scoring_Model_Development_June_2024.ipynb |Jupyter notebook - the process of cleaning, transforming, and analyzing the original data with visualizations, including data downloaded via web scraping.|
|PPT | SLEBID_Scoring_Model_Development_June_2024.pptx  |Project Presentation in PowerPoint Format [Presentation Slides](https://docs.google.com/presentation/d/178v7TiIdxXEeY77qPhUNx4hxJFgdtN40MWnh_xNDoOU/edit?usp=sharing)|

---

**Future Work**
Further feature engineering and selection to improve model performance.
Exploration of different machine learning algorithms.
Implementation of more advanced techniques for handling imbalanced data.

---

**Contact**
If you have any questions or need further information, please feel free to contact me.
E-mail: **Sergii.Lebid@yahoo.com**
