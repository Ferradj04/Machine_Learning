import numpy as np
import matplotlib.pyplot as plt

# Generate a random data
np.random.seed(42)  # For reproductibilty results
#Sample initialization
sample_50 = np.random.normal(loc=0, scale=2, size=50)
sample_100 = np.random.normal(loc=0, scale=2, size=100)
sample_1000 = np.random.normal(loc=0, scale=2, size=1000)
#Set colors
background = "#1d1d1d"
stroke = "#00CFE0"
#Definition of a the simulation model function
def simulation_normal_law_non_biased_estimator(data):
    # Calculate mean process
    mean_value = np.mean(data)
    variance_estimate = np.var(data, ddof=1)  # Utilisez ddof=1 pour obtenir la variance ajustée
    # Plot a histogram diagram
    plt.axes().set_facecolor(background)
    plt.hist(data, bins=20, density=True, alpha=0.7, color=background, label='Data',edgecolor=stroke, linewidth=1.0)

    # Tracer la distribution normale avec la moyenne et la variance estimée
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin,xmax,100)
    p = (1 / np.sqrt(2 * np.pi * variance_estimate)) * np.exp(-0.5 * ((x - mean_value) / np.sqrt(variance_estimate))**2)
    plt.plot(x,p,'k',linewidth=1.0,label='Normal estimated Law',color=stroke)

    plt.title('Estimated non biased variance of a normal law')
    plt.legend()
    plt.show()

    # Afficher la moyenne et la variance estimée
    print(f'Emprical mean : {mean_value}')
    print(f'Non-biased estimated variance : {variance_estimate}')

#test the simulation model 
simulation_normal_law_non_biased_estimator(sample_50)
simulation_normal_law_non_biased_estimator(sample_100)
simulation_normal_law_non_biased_estimator(sample_1000)