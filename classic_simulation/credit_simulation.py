import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(7)

# number of loans
n = 1000

# Table with n rows, 3 columns, random values
df = pd.DataFrame({
    "EaD": np.random.randint(1000, 50000, n),          # Exposure at Default
    "PD": np.random.uniform(0.01, 0.10, n),            # 1–10% Probability of Default
    "LGD": np.random.uniform(0.2, 0.6, n)              # 20–60% Loss Given Default
})

# Expected Loss
df["Expected_Loss"] = df["PD"] * df["LGD"] * df["EaD"]

# Defaults: 0 = Not defaulted; 1 = Defaulted
df["Default"] = (np.random.rand(len(df)) < df["PD"]).astype(int)

# Calculate Real Loss based on defaults
df["Real_Loss"] = df["Default"] * df["LGD"] * df["EaD"]
print(df.head())
total_real_loss = df["Real_Loss"].sum()
print("Simulated Real Loss:", round(total_real_loss, 2))

# Loss simulation loop
n_simulations = 10000
sim_losses = []
for i in range(n_simulations):
    defaults = (np.random.rand(len(df)) < df["PD"]).astype(int)
    real_loss_sum = (defaults * df["LGD"] * df["EaD"]).sum()
    sim_losses.append(real_loss_sum)

sim_losses = np.array(sim_losses)
print("Average Real Loss:", round(sim_losses.mean(), 2))
print("Minimum Real Loss:", round(sim_losses.min(), 2))
print("Maximum Real Loss:", round(sim_losses.max(), 2))
print("Standard Deviation of Real Losses:", round(sim_losses.std(), 2))

# Visualization
plt.hist(sim_losses, bins=30, color="skyblue", edgecolor="black")
plt.title("Simulation of Real Losses")
plt.xlabel("Real Loss")
plt.ylabel("Number of Simulations")
plt.show()