# Credit Risk Simulation

This repository contains two approaches for **Credit Risk Modeling**:

1. **Classic Monte Carlo Simulation** – estimates credit losses based on predefined probabilities.
2. **ML-Enhanced Simulation** – predicts default probabilities using a machine learning model (Logistic Regression) to improve portfolio risk insights.

---

## 1. Classic Credit Risk Simulation

This project performs a **Credit Risk Simulation** to estimate credit losses for a portfolio of loans.  
Each loan is assigned a random Exposure at Default (EaD), Probability of Default (PD), and Loss Given Default (LGD).  
Based on these parameters, the simulation calculates expected and realized losses and runs repeated simulations to generate a loss distribution.

### Features

- Generate a synthetic credit portfolio.
- Calculate Expected Loss and Realized Loss.
- Simulate defaults for 10,000 Monte Carlo iterations.
- Compute summary statistics (mean, min, max, standard deviation).
- Visualize the distribution of simulated Real Loss values.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Peter-Krug/credit-risk-simulation
cd credit-risk-simulation
```
2. Install dependencies (Python 3.9+ recommended):

```bash
pip install pandas numpy matplotlib
````
### How it works

#### 1. Portfolio Creation

- A portfolio of 1,000 loans is generated with random:
- EaD (Exposure at Default)
- PD (Probability of Default)
- LGD (Loss Given Default)

#### 2. Expected vs Realized Loss

- Expected Loss formula:
    Expected Loss = PD × LGD × EaD
- Realized Loss:
    Calculated only when a loan defaults during simulation.

#### 3. Monte Carlo Simulation
- Random defaults are triggered based on PD.
- Losses are calculated using LGD and EaD.
- The simulation is repeated 10,000 times.
- All portfolio losses are saved and analyzed.

## 2. ML-Enhanced Credit Risk Simulation
This version adds a Machine Learning approach to predict whether a loan will default, using the features EaD, PD, and LGD.

### Features

- Train a Logistic Regression model to predict defaults.
- Compute predicted default probabilities for each loan.
- Evaluate model performance using ROC Curve and AUC score.
- Analyze feature importance to understand which factors impact defaults the most.
- Compare ML predictions against the classic Monte Carlo simulation.

### Installation

Install additional dependencies:

```bash
pip install scikit-learn matplotlib pandas numpy
````

### How it works

#### 1. Dataset Generation

- Similar to the classic version, but introduces a correlation between loan size and default probability.

#### 2. Train/Test Split

- Split the portfolio into training and testing sets for ML evaluation.

#### 3. Logistic Regression Model

- Trains on the training set.
- Predicts probability of default on the test set.

#### 4. Evaluation

- Uses AUC score and ROC curve to assess model performance.
- Computes feature importance for EaD, PD, and LGD.

#### 5. Comparison

- The ML model can identify high-risk loans more accurately than the classic simulation.
- Provides actionable insights for portfolio risk management.
