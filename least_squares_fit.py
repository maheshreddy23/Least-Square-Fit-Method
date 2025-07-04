import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings

class LeastSquaresFit:
    """
    A comprehensive implementation of the Least Squares method for curve fitting.
    
    This class provides methods for:
    - Linear least squares fitting
    - Polynomial least squares fitting
    - Statistical analysis of fit quality
    - Visualization of results
    """
    
    def __init__(self):
        self.coefficients = None
        self.r_squared = None
        self.residuals = None
        self.fitted_values = None
        self.degree = None
        
    def linear_fit(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Performs linear least squares fitting: y = a*x + b
        
        Mathematical formulation:
        For n data points (xi, yi), we minimize: Σ(yi - (a*xi + b))²
        
        Solution:
        a = (n*Σ(xi*yi) - Σ(xi)*Σ(yi)) / (n*Σ(xi²) - (Σ(xi))²)
        b = (Σ(yi) - a*Σ(xi)) / n
        
        Args:
            x (np.ndarray): Independent variable data
            y (np.ndarray): Dependent variable data
            
        Returns:
            Tuple[float, float]: (slope, intercept) coefficients
        """
        if len(x) != len(y):
            raise ValueError("x and y arrays must have the same length")
        
        n = len(x)
        
        # Calculate sums needed for linear least squares
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        # Calculate coefficients
        denominator = n * sum_x2 - sum_x * sum_x
        
        if abs(denominator) < 1e-12:
            raise ValueError("Cannot fit line: x values are too similar")
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        self.coefficients = [slope, intercept]
        self.degree = 1
        
        # Calculate fitted values and statistics
        self.fitted_values = slope * x + intercept
        self._calculate_statistics(y)
        
        return slope, intercept
    
    def polynomial_fit(self, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        """
        Performs polynomial least squares fitting: y = a_n*x^n + ... + a_1*x + a_0
        
        Uses matrix formulation: (X^T * X) * a = X^T * y
        where X is the Vandermonde matrix
        
        Args:
            x (np.ndarray): Independent variable data
            y (np.ndarray): Dependent variable data
            degree (int): Degree of polynomial (1=linear, 2=quadratic, etc.)
            
        Returns:
            np.ndarray: Polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
        """
        if len(x) != len(y):
            raise ValueError("x and y arrays must have the same length")
        
        if degree < 1:
            raise ValueError("Degree must be at least 1")
        
        if len(x) <= degree:
            warnings.warn("Number of data points should be greater than polynomial degree")
        
        # Create Vandermonde matrix
        X = np.vander(x, degree + 1, increasing=False)
        
        # Solve normal equations: (X^T * X) * a = X^T * y
        try:
            coefficients = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if normal equations are singular
            coefficients = np.linalg.pinv(X) @ y
        
        self.coefficients = coefficients
        self.degree = degree
        
        # Calculate fitted values and statistics
        self.fitted_values = np.polyval(coefficients, x)
        self._calculate_statistics(y)
        
        return coefficients
    
    def _calculate_statistics(self, y: np.ndarray) -> None:
        """Calculate R² and residuals for the fit"""
        self.residuals = y - self.fitted_values
        
        # Calculate R-squared
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            self.r_squared = 1.0  # Perfect fit when all y values are the same
        else:
            self.r_squared = 1 - (ss_res / ss_tot)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model
        
        Args:
            x (np.ndarray): Points where to make predictions
            
        Returns:
            np.ndarray: Predicted y values
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet")
        
        if self.degree == 1:
            return self.coefficients[0] * x + self.coefficients[1]
        else:
            return np.polyval(self.coefficients, x)
    
    def plot_fit(self, x: np.ndarray, y: np.ndarray, title: str = "Least Squares Fit", 
                 xlabel: str = "x", ylabel: str = "y") -> None:
        """
        Plot the original data and fitted curve
        
        Args:
            x (np.ndarray): Original x data
            y (np.ndarray): Original y data
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet")
        
        plt.figure(figsize=(10, 6))
        
        # Plot original data
        plt.scatter(x, y, alpha=0.6, color='blue', label='Original Data')
        
        # Plot fitted curve
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = self.predict(x_smooth)
        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'Fitted Curve (R² = {self.r_squared:.4f})')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_fit_equation(self) -> str:
        """
        Returns a string representation of the fitted equation
        
        Returns:
            str: Mathematical equation of the fit
        """
        if self.coefficients is None:
            return "No model fitted"
        
        if self.degree == 1:
            return f"y = {self.coefficients[0]:.6f}x + {self.coefficients[1]:.6f}"
        else:
            terms = []
            for i, coef in enumerate(self.coefficients):
                power = self.degree - i
                if power == 0:
                    terms.append(f"{coef:.6f}")
                elif power == 1:
                    terms.append(f"{coef:.6f}x")
                else:
                    terms.append(f"{coef:.6f}x^{power}")
            return "y = " + " + ".join(terms).replace("+ -", "- ")
    
    def print_statistics(self) -> None:
        """Print fit statistics"""
        if self.coefficients is None:
            print("No model fitted")
            return
        
        print("=" * 50)
        print("LEAST SQUARES FIT RESULTS")
        print("=" * 50)
        print(f"Equation: {self.get_fit_equation()}")
        print(f"Degree: {self.degree}")
        print(f"R-squared: {self.r_squared:.6f}")
        print(f"Root Mean Square Error: {np.sqrt(np.mean(self.residuals**2)):.6f}")
        print(f"Mean Absolute Error: {np.mean(np.abs(self.residuals)):.6f}")
        print("=" * 50)


def load_aerodynamic_data(filename: str = "data .csv") -> pd.DataFrame:
    """
    Load the aerodynamic data from CSV file
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    # Read the CSV file, skipping the header lines and handling the formatting
    data = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    # Skip header lines and process data
    for line in lines[2:]:  # Skip first two header lines
        if line.strip() and not line.startswith('-'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    alpha = float(parts[0])
                    cl = float(parts[1])
                    cd = float(parts[2])
                    data.append([alpha, cl, cd])
                except ValueError:
                    continue
    
    df = pd.DataFrame(data, columns=['alpha', 'CL', 'CD'])
    return df


def demonstrate_fitting():
    """
    Demonstrate the least squares fitting with aerodynamic data
    """
    print("Loading aerodynamic data...")
    
    try:
        data = load_aerodynamic_data()
        print(f"Loaded {len(data)} data points")
        print("\nFirst 5 rows of data:")
        print(data.head())
        
        # Create least squares fitter
        lsf = LeastSquaresFit()
        
        # Example 1: Linear fit between alpha (angle of attack) and CL (coefficient of lift)
        print("\n" + "="*60)
        print("EXAMPLE 1: LINEAR FIT - Alpha vs CL")
        print("="*60)
        
        alpha = data['alpha'].values
        cl = data['CL'].values
        
        slope, intercept = lsf.linear_fit(alpha, cl)
        lsf.print_statistics()
        
        # Plot the results
        lsf.plot_fit(alpha, cl, 
                    title="Linear Least Squares Fit: Coefficient of Lift vs Angle of Attack",
                    xlabel="Angle of Attack (degrees)", 
                    ylabel="Coefficient of Lift (CL)")
        
        # Example 2: Quadratic fit between CL and CD
        print("\n" + "="*60)
        print("EXAMPLE 2: QUADRATIC FIT - CL vs CD")
        print("="*60)
        
        lsf2 = LeastSquaresFit()
        cd = data['CD'].values
        
        # Sort by CL for better visualization
        sort_idx = np.argsort(cl)
        cl_sorted = cl[sort_idx]
        cd_sorted = cd[sort_idx]
        
        coeffs = lsf2.polynomial_fit(cl_sorted, cd_sorted, degree=2)
        lsf2.print_statistics()
        
        # Plot the results
        lsf2.plot_fit(cl_sorted, cd_sorted,
                     title="Quadratic Least Squares Fit: Drag Polar (CD vs CL)",
                     xlabel="Coefficient of Lift (CL)",
                     ylabel="Coefficient of Drag (CD)")
        
        # Example 3: Higher-order polynomial fit
        print("\n" + "="*60)
        print("EXAMPLE 3: CUBIC FIT - Alpha vs CL")
        print("="*60)
        
        lsf3 = LeastSquaresFit()
        coeffs = lsf3.polynomial_fit(alpha, cl, degree=3)
        lsf3.print_statistics()
        
        # Plot the results
        lsf3.plot_fit(alpha, cl,
                     title="Cubic Least Squares Fit: Coefficient of Lift vs Angle of Attack",
                     xlabel="Angle of Attack (degrees)",
                     ylabel="Coefficient of Lift (CL)")
        
        # Compare different polynomial degrees
        print("\n" + "="*60)
        print("COMPARISON: Different polynomial degrees for Alpha vs CL")
        print("="*60)
        
        degrees = [1, 2, 3, 4, 5]
        r_squared_values = []
        
        for degree in degrees:
            lsf_temp = LeastSquaresFit()
            lsf_temp.polynomial_fit(alpha, cl, degree=degree)
            r_squared_values.append(lsf_temp.r_squared)
            print(f"Degree {degree}: R² = {lsf_temp.r_squared:.6f}")
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(degrees, r_squared_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Polynomial Degree')
        plt.ylabel('R² Value')
        plt.title('Model Performance vs Polynomial Degree')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print("Error: Could not find data file 'data .csv'")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data
        x = np.linspace(-15, 15, 50)
        y_true = 0.1 * x + 0.05 * x**2 + np.random.normal(0, 0.1, len(x))
        
        lsf = LeastSquaresFit()
        
        # Linear fit
        slope, intercept = lsf.linear_fit(x, y_true)
        print("Linear fit on synthetic data:")
        lsf.print_statistics()
        
        # Quadratic fit
        lsf2 = LeastSquaresFit()
        coeffs = lsf2.polynomial_fit(x, y_true, degree=2)
        print("\nQuadratic fit on synthetic data:")
        lsf2.print_statistics()


if __name__ == "__main__":
    demonstrate_fitting()