# Simple Linear Regression Overview

## What It Is
Simple Linear Regression predicts a continuous target variable ($y$) using a single explanatory feature ($x$) by fitting a straight line to the data points.

## The Mathematical Equation
$$y = \beta_0 + \beta_1x + \epsilon$$

*   **$y$**: The dependent target variable you want to predict (e.g., wine quality, Salary).
*   **$x$**: The independent predictor feature you use as input (e.g.,  alcohol content, Years of Experience).
*   **$\beta_0$**: The intercept value where the regression line crosses the y-axis.
*   **$\beta_1$**: The slope coefficient indicating the change in $y$ per unit change in $x$.
*   **$\epsilon$**: The residual error representing unexplained variations in the data.

## Core Assumptions
1.  **Linearity**: The relationship between predictor $x$ and target $y$ is a straight line.
2.  **Independence**: The data observations are completely independent of each other.
3.  **Homoscedasticity**: Prediction error variance remains constant across all levels of $x$.
4.  **Normality**: The residual errors follow a standard normal distribution.
