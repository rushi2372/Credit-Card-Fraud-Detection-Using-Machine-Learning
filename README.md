## Credit-Card-Fraud-Detection-Using-Machine-Learning

## Project Overview
This project is aimed at identifying potentially fraudulent credit card transactions by using data analysis, visualization, and machine learning techniques. The objective is to uncover hidden patterns that separate legitimate and fraudulent activity, and build a model that can predict suspicious transactions accurately.

## Dataset
The dataset used for this project is sourced from Kaggle's Credit Card Fraud Detection dataset(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It includes transaction records with features like amount, time, and anonymized attributes that provide patterns for fraud detection.

## Project Structure

1. **Data Reading & Preprocessing**:
   - Load the dataset and clean data, including handling missing values and duplicates.
   - Normalize and prepare the data for analysis and model input.

2. **Data Analysis**:
   - Explore the data to understand the proportion of fraud vs. non-fraud transactions.
   - Calculate and interpret key statistics, such as the percentage of fraudulent transactions and average transaction amount.

3. **Data Visualization**:
   - Visualize the frequency of fraudulent and non-fraudulent transactions.
   - Display the distribution of transaction amounts for both categories to identify patterns in fraud detection.

4. **Model Development**:
   - Split the dataset into training and testing sets.
   - Train and evaluate machine learning models to classify transactions as fraudulent or non-fraudulent.
   - Assess model performance and identify key insights for improving fraud detection accuracy.

## Requirements
Install the necessary libraries using:
```bash
pip install -r requirements.txt
```
where `requirements.txt` includes:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`

## Usage
1. **Load Data**: Import the dataset into the project directory and load it using `pandas`.
2. **Preprocess and Analyze**: Run the analysis scripts to understand data structure and fraud patterns.
3. **Visualize**: Generate visuals to compare fraudulent and non-fraudulent transactions.
4. **Train Model**: Use the scripts provided to train and evaluate models for fraud detection.

## Results
- Found that fraud cases are highly imbalanced, with very few fraudulent transactions.
- The best-performing model effectively identifies fraud with high precision and recall.
