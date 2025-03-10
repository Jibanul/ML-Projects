# Statistical Computing - Project 1
Implementation of Least Squares Support Vector Machines (LSSVM) and Support Vector Regression (SVR) Algorithms

## Overview
This project implements various machine learning algorithms focusing on Least Squares Support Vector Machines (LS-SVM) and Support Vector Regression (SVR). The implementation is available in both R and Python.

## Goal
The goal of this project is to implement and analyze **Least Squares Support Vector Machines (LS-SVM)** and **Support Vector Regression (SVR)** using different numerical approaches. These methods are widely used for **regression tasks and large-scale optimization problems**. By implementing these models in **both R and Python**, we aim to:
- Understand the mathematical foundation behind LS-SVM and SVR.
- Compare different numerical methods (e.g., **Conjugate Gradient, QR Decomposition, Incremental Learning**).


## Project Structure
```
Codes/
├── LSSVM_SVR.R            # Main code file 
├── LSSVM_SVR.py           # Python implementation
├── pb2.txt               # Dataset for Problem 1
├── Problem.pdf           # Assignment questions
└── Report.pdf            # Solution report
```

## Problem Description

### Problem 1: LS-SVM Implementation
Implements LS-SVM using three different approaches:
1. **Large Scale Algorithm**: Uses Hestenes-Stiefel conjugate gradient method
2. **QR Update**: Implements row addition to training set using Quotient-Remainder (QR) decomposition
3. **Incremental LSSVM**: Matrix inversion approach with incremental learning

### Problem 2: Support Vector Regression
Implements SVR for the Boston Housing Dataset, including:
- Training and prediction functions
- Data preprocessing
- Model evaluation

## Implementation Details

### R Implementation (`LSSVM_SVR.R`)
- Complete implementation of both problems
- Includes:
  - LSSVM with conjugate gradient method
  - QR decomposition updates
  - Incremental method
  - SVR implementation
  - Model evaluation metrics

### Python Implementation (`LSSVM_SVR.py`)
- Focuses on numerical methods
- Implements:
  - Conjugate Gradient Method
  - Incremental solver for linear systems

## Usage

### Prerequisites
R packages required:
```r
install.packages(c("caTools", "matrixcalc", "quadprog", "MASS"))
```

Python packages required:
```python
pip install numpy scipy
```

### Running the Code
1. For R implementation:
```r
source("LSSVM_SVR.R")
```

2. For Python implementation:
```python
python LSSVM_SVR.py
```

## Datasets
- **Problem 1**: Uses `pb2.txt` dataset
- **Problem 2**: Uses Boston Housing Dataset from MASS package in R

## References
- Suykens et al. (1999) paper on LS-SVM
- Boston Housing Dataset: [UCI Machine Learning Repository](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/Boston.html)