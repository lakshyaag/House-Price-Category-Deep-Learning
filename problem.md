# MGSC 673: Final Project

## Objective

Your objective in this project is to build a multi-task learning model that predicts both house prices (a regression task) and house category (a classification task). This  involves understanding and implementing multi-task learning models, and using PyTorch Lightning's advanced features for managing such complex projects.

## Dataset

[House Prices - Advanced Regression Techniques Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

You will create a new variable 'House Category' based on 'House Style', 'Bldg Type', 'Year Built', and 'Year Remod/Add' for the classification task.

## Steps

- Data Exploration and Preprocessing: Understand your data. Handle missing values, encode categorical variables, and normalize numerical variables.
- Multi-task Model Building: Use PyTorch Lightning to build a feed-forward neural network model that predicts both house prices (a regression task) and house category (a classification task). This requires a shared bottom model and task-specific top layers.
- Activation Functions and Optimizers: Experiment with various activation functions and optimizers. Compare their effects on the performance of your model.
- Loss Functions: Implement and use appropriate loss functions for both tasks. Combine these into a single loss function for training your multi-task model.
- Model Evaluation: Use suitable metrics to evaluate the performance of your model on both tasks.
Advanced PyTorch Lightning Features: Use PyTorch Lightning's features like logging, callback system, and Trainer API to effectively manage your project.
- Hyperparameter Tuning: Use PyTorch Lightning's integration with Optuna for hyperparameter optimization.
- Report: Write a comprehensive report detailing your approach, experiments, and results.

## Deliverables

1. Python scripts for data preprocessing, model building, training, and evaluation.
2. Trained model files.
3. A comprehensive report on your findings.

## Learning Outcomes

- Understanding and implementing multi-task learning models with PyTorch and PyTorch Lightning.
- Applying different activation functions, optimizers, and loss functions in a multi-task learning context.
- Using advanced features of PyTorch Lightning.
- Performing hyperparameter tuning with PyTorch Lightning's integration with Optuna or Ray Tune.
- Handling real-world data and predicting outcomes for multiple tasks using a shared model architecture.
