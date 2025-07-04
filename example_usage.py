#!/usr/bin/env python3
"""
Simple example showing how to use the LeastSquaresFit class
"""

import numpy as np
import matplotlib.pyplot as plt
from least_squares_fit import LeastSquaresFit

def example_linear_fit():
    """Example of linear least squares fitting"""
    print("=" * 50)
    print("LINEAR LEAST SQUARES EXAMPLE")
    print("=" * 50)
    
    # Generate sample data with some noise
    np.random.seed(42)  # For reproducible results
    x = np.linspace(0, 10, 20)
    y_true = 2.5 * x + 1.0  # True relationship
    y = y_true + np.random.normal(0, 1, len(x))  # Add noise
    
    # Perform linear fit
    lsf = LeastSquaresFit()
    slope, intercept = lsf.linear_fit(x, y)
    
    print(f"True equation: y = 2.5x + 1.0")
    print(f"Fitted equation: y = {slope:.3f}x + {intercept:.3f}")
    print(f"R-squared: {lsf.r_squared:.6f}")
    
    # Plot results
    lsf.plot_fit(x, y, 
                title="Linear Least Squares Fit Example",
                xlabel="x", ylabel="y")

def example_polynomial_fit():
    """Example of polynomial least squares fitting"""
    print("\n" + "=" * 50)
    print("POLYNOMIAL LEAST SQUARES EXAMPLE")
    print("=" * 50)
    
    # Generate sample data with quadratic relationship
    np.random.seed(42)
    x = np.linspace(-3, 3, 30)
    y_true = 0.5 * x**2 - 2 * x + 1  # True quadratic relationship
    y = y_true + np.random.normal(0, 0.3, len(x))  # Add noise
    
    # Try different polynomial degrees
    degrees = [1, 2, 3]
    
    for degree in degrees:
        lsf = LeastSquaresFit()
        coeffs = lsf.polynomial_fit(x, y, degree=degree)
        
        print(f"\nDegree {degree} fit:")
        print(f"Equation: {lsf.get_fit_equation()}")
        print(f"R-squared: {lsf.r_squared:.6f}")
        
        # Plot for degree 2 (should be best fit)
        if degree == 2:
            lsf.plot_fit(x, y,
                        title=f"Polynomial Fit (degree {degree})",
                        xlabel="x", ylabel="y")

def example_custom_data():
    """Example with custom data points"""
    print("\n" + "=" * 50)
    print("CUSTOM DATA EXAMPLE")
    print("=" * 50)
    
    # Custom data points
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.0, 19.9])
    
    print("Data points:")
    for i in range(len(x)):
        print(f"({x[i]}, {y[i]})")
    
    # Perform linear fit
    lsf = LeastSquaresFit()
    slope, intercept = lsf.linear_fit(x, y)
    
    lsf.print_statistics()
    
    # Make predictions
    x_new = np.array([11, 12, 13])
    y_pred = lsf.predict(x_new)
    
    print("\nPredictions:")
    for i in range(len(x_new)):
        print(f"x = {x_new[i]} -> y = {y_pred[i]:.3f}")
    
    # Plot results
    lsf.plot_fit(x, y,
                title="Custom Data Linear Fit",
                xlabel="x", ylabel="y")

if __name__ == "__main__":
    # Run all examples
    example_linear_fit()
    example_polynomial_fit()
    example_custom_data()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Check the plots to see the fitted curves.")
    print("=" * 50)