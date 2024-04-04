# Titanic Survival Prediction(Kaggle Competition)

This code is a Python script for predicting the survival status of passengers aboard the Titanic using machine learning techniques. It utilizes the Random Forest Classifier algorithm to make predictions based on features such as passenger class and sex.

## Prerequisites
- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Installation
1. Install Python from [python.org](https://www.python.org/downloads/).
2. Install required libraries using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

## Usage
1. Download the dataset files `train.csv` and `test.csv`.
2. Place the dataset files in the same directory as the script.
3. Run the script using Python:
   ```bash
   python titanic_survival_prediction.py
   ```

## Description
- The script starts by importing necessary libraries including Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.
- It loads the training and testing datasets (`train.csv` and `test.csv`) using Pandas.
- The missing values in the `Cabin` column are filled with "Unknown".
- Missing values in the `Age` column are filled with the mean age of passengers.
- The data types of `Age` and `PassengerId` columns are transformed.
- Missing values in the `Embarked` column are filled with the most common value.
- A heatmap is plotted to visualize the correlation between different features.
- Features `Pclass` and `Sex` are selected for prediction.
- One-hot encoding is applied to categorical features.
- A Random Forest Classifier model is trained on the training data.
- Predictions are made on the testing data.
- Results are saved to a CSV file named `submission.csv`.
- A success message is displayed upon successful submission.
