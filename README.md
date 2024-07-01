# Credit_scoring_development
![Illustration](https://github.com/jon/coolproject/raw/master/image/image.png)
# **Credit Scoring Model Development**
**Overview**
This project involves developing a credit scoring model using machine learning techniques to predict the likelihood of a borrower defaulting on a loan. The process includes data preprocessing, feature selection, model training, evaluation, and optimization to achieve the best performance.

Project Structure

README.md: This file, providing an overview of the project.

**Data Preprocessing**
The dataset is preprocessed to handle missing values, encode categorical variables, and scale numerical features. This is done using the data_preprocessing.py script in the scripts directory.

**Feature Selection**
We select features based on their correlation and importance as determined by a Gradient Boosting model. The feature selection process is detailed in the Jupyter notebooks within the notebooks directory. Specifically, the top 27 features are used based on their combined correlation and importance scores.

**Model Training and Evaluation**
The model is trained and evaluated using cross-validation techniques. We perform hyperparameter tuning using GridSearchCV to find the best parameters for the Gradient Boosting model. The process is implemented in the model_training.py script and the corresponding notebook.

**Example Model Parameters:**
Best Parameters: {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
Best Cross-Validation Accuracy: 0.7557
Test Set Accuracy: 0.7433
Model Application
We demonstrate how to load the saved model and apply it to new data. The steps include data normalization and predictions using the trained model. The example is provided in the model_application.py script.

**Threshold Optimization**
We determine the optimal threshold for classification based on accuracy and cost considerations. The code for this is in the threshold_optimization.py script and associated notebook.

Example Threshold Optimization:
Cross-Validation Accuracy Scores: [0.78, 0.75, 0.775, 0.77, 0.74]
Mean Cross-Validation Accuracy: 0.763
Optimal Threshold Results:
Precision (Class 1): 0.8167
Recall (Class 1): 0.9614
F1-Score (Class 1): 0.8832
Cost: 69
Results
The results of the model evaluation, including accuracy, precision, recall, and cost, are stored in the results directory. Visualizations of the model's performance are also provided. For instance, the model using 27 features has shown high accuracy and low cost, making it an optimal choice.

**Conclusion**
The project demonstrates the development and application of a machine learning-based credit scoring model. The selected model, utilizing the top 27 features, shows high accuracy and low cost, making it a viable option for credit scoring purposes.

**Future Work**
Further feature engineering and selection to improve model performance.
Exploration of different machine learning algorithms.
Implementation of more advanced techniques for handling imbalanced data.

**Tools and Technologies Used**
Python: Primary programming language for the project.
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
Scikit-learn: For machine learning algorithms and model evaluation.
Matplotlib and Seaborn: For data visualization.
Jupyter Notebook: For interactive development and analysis.
Pickle: For saving and loading models and other data objects.

**PPP**
https://docs.google.com/presentation/d/178v7TiIdxXEeY77qPhUNx4hxJFgdtN40MWnh_xNDoOU/edit?usp=sharing

**Contact**
If you have any questions or need further information, please feel free to contact me.
E-mail: Sergii.Lebid@yahoo.com
