🎯 Objectives
Clean and preprocess a mental health survey dataset
Encode categorical variables and normalize features
Train and compare multiple classification models
Evaluate with metrics such as Accuracy, ROC-AUC, Precision, Recall, and Confusion Matrix
⚙️ Methods & Models
The following models were implemented and compared:
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Support Vector Classifier (SVC)
Gaussian Naive Bayes
AdaBoost
XGBoost (Best performing model)
(Optional) Neural Networks with TensorFlow/Keras
Feature engineering techniques:
Label/One-Hot Encoding for categorical data
MinMax Scaling for numerical features
Handling outliers and invalid data (e.g., unrealistic ages)
📊 Results
Indicative results from the notebook:
Logistic Regression: ~70% accuracy, ROC-AUC ~0.75
KNN / Decision Tree: ~74–75% accuracy
XGBoost: ~77% accuracy, ROC-AUC ~0.80 (best performance)
⚠️ Results may vary slightly depending on environment and random seeds.
🚀 How to Run
1. Clone the repository
git clone https://github.com/<your-username>/mental-health-ml.git
cd mental-health-ml
2. Create environment & install dependencies
# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate mental-health
3. Run the notebook
jupyter notebook "Mental Health.ipynb"
📑 Dataset
Source: Mental_Survey.csv
Includes demographic and workplace survey responses
Target variable: Mental Health Status (Yes/No)
(Note: dataset is not included in the repo for privacy reasons — update path in the notebook if needed.)
📦 Requirements
Python 3.10+
pandas, numpy, matplotlib, seaborn
scikit-learn
xgboost
statsmodels, scipy, mlxtend
tensorflow (for neural networks)
jupyter
