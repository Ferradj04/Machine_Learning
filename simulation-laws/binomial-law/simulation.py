import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


# Set the parameters for the binomial distribution
n_trials = 50  # Number of trials
probability_of_success = 0.5  # Probability of success in each trial

# Simulate a binomial distribution
random_sample = np.random.binomial(n_trials, probability_of_success, size=2000)

# Plot the histogram of the random sample
plt.hist( 
    random_sample, 
    bins = np.arange( 0 , n_trials + 2 ) - 0.5, 
    density = True , 
    alpha=0.5 , 
    label="Simulated Binomial Distribution",
    color="#FFFFFF" , 
    edgecolor="#000000" , 
    linewidth=1.0
)

# Plot the theoretical probability mass function (PMF)
x = np.arange(
    0 , 
    n_trials + 1
)
pmf_values = binom.pmf( 
    x , 
    n_trials , 
    probability_of_success 
)
plt.plot(
    x , 
    pmf_values , 
    'r-', 
    label="Theoretical Binomial PMF" , 
    color="#000000" ,
    linewidth=1.0
)
# Add labels and legend
plt.title("Simulated Binomial Distribution")
plt.xlabel("Number of Successes")
plt.ylabel("Probability Mass Function (PMF)")
plt.legend()

# Show the plot
plt.show()
