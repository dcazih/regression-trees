# Regression Tree Implementation

This repository contains an implementation of a **binary regression tree** based on the **Classification And Regression Tree (CART) algorithm**. The goal is to understand and build **regression trees** to approximate continuous functions and dynamical systems. It supports flexible constraints like maximum depth and minimum leaf size while providing predictions, decision paths, and visualization of results.

## Features
- **Custom Regression Tree**: Implements a Python class `RegressionTree` for training a regression tree with user-defined constraints (max height or leaf size).
- **Prediction & Decision Path**: Supports making predictions and displaying the decision path for any input sample.
- **Tree Construction Without Pruning**: Uses **sum of squared errors** to determine the best split feature and value.
- **Function Approximation**: Tests the model on `y = 0.8 sin(x − 1)` over `x ∈ [-3,3]`, evaluating tree complexity vs. accuracy.
- **Dynamical System Approximation**: Implements regression trees for predicting **system state evolution** in multidimensional cases.
- **Hyperparameter Tuning**: Evaluates the impact of tree height and leaf size constraints on performance.

## Implementation Overview
1. **Tree Construction**:
   - Selects the best feature and split value based on sum of squared errors reduction.
   - Allows limiting tree growth by **height** or **leaf size**.
2. **Prediction & Decision Path**:
   - `predict()`: Returns the estimated value for a given input.
   - `decision_path()`: Displays the path taken to reach the prediction.
3. **Testing & Evaluation**:
   - Compares model accuracy under different tree constraints.
   - Tests tree-based approximations on **discrete-time dynamical systems**.
   - Evaluates performance in predicting system trajectories over time.

## Usage
- Generates training and test data using `numpy.random.uniform`.
- Splits data using `sklearn.model_selection.train_test_split`.
- Visualizes results to analyze approximation accuracy.

## Report
Includes:
- Explanation of tree structure and split selection.
- Experimental results on function approximation.
- Analysis of regression tree-based system modeling.
- Hyperparameter tuning results and performance metrics.
