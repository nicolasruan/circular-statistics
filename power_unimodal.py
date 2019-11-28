# Test performance program

# Om testen toe te voegen 

# 1. schrijf uw functie testnaam(sample) onder 'Test definitions'. De functie moet een p-waarde 
# teruggeven. Als de functie een teststatistiek teruggeeft, bereken dan de 
# p-waarde met de monte carlo functie get_pval(sample, statistic), zie pycke test als voorbeeld

# 2. voeg de functie toe aan de list 'tests'

# 3. voeg de naam (string) van de functie toe aan de list 'test_names'


# Om verdelingen toe te voegen

# 1. schrijf een functie verdelingnaam(n) die een sample van grootte n geeft
# uit de verdeling

# 2. voeg in de functie 'samples()' de sample toe aan de list

# 3. voeg de naam (string) van de verdeling toe aan de list 'distribution_names'

# peace Nicolas


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

from tests import *
from sampling import *


N = 500 # number of populations for each sample size
max_size = 200
steps = 10
def samples():
    data = list()
    for i in range(0, steps):
        sample_size = (i+1)*round(max_size/steps)
        row = list()
        for j in range(0, N):
#           row.append([2*np.pi*np.random.rand(sample_size),
#                          np.random.vonmises(0, 1/2, sample_size),
#                        np.random.vonmises(0, 1, sample_size),
#                        np.random.vonmises(0, 2, sample_size),
#                          wrapped_cauchy(sample_size, 0, 1/2),
#                        wrapped_cauchy(sample_size, 0, 1),
#                        wrapped_cauchy(sample_size, 0, 2),
#                          wrapped_normal(sample_size, 0, 1/2),
#                        wrapped_normal(sample_size, 0, 1),
#                        wrapped_normal(sample_size, 0, 2),
#                          cardioid(sample_size, 0, 1/10),
#                        ,
#                        cardioid(sample_size, 0, 1/2),
#                          double_vonmises(sample_size, 3), semicircle(sample_size, 0.6), semicircle(sample_size, 0.8)] 
#                        )

#           row.append([ double_vonmises(sample_size, 3),  
#                         semicircle(sample_size, 0.6),
#                         semicircle(sample_size, 0.8)])
            row.append([
                          np.random.vonmises(0, 1, sample_size),
                        np.random.vonmises(0, 1/2, sample_size),
                        np.random.vonmises(0, 1/4, sample_size),
                          wrapped_cauchy(sample_size, 0, 1/2),
                        wrapped_cauchy(sample_size, 0, 1),
                        wrapped_cauchy(sample_size, 0, 2),
                          wrapped_normal(sample_size, 0, 1),
                        wrapped_normal(sample_size, 0, 1.5),
                        wrapped_normal(sample_size, 0, 2),
                          cardioid(sample_size, 0, 1/4),
                          cardioid(sample_size, 0, 1/8),
                          cardioid(sample_size, 0, 1/16),
                ])
            

        data.append(row)
    return data


## Test performance ##

tests = [watson, kuiper]
test_names = ["watson", "kuiper"]

distribution_names = ["vonmises", "wrapped cauchy", "wrapped normal", "cardioid"]

labels = ["$\\kappa=1$", "$\\kappa=1/2$", "$\\kappa=1/4$",
          "$\\gamma=1/2$", "$\\gamma=1$", "$\\gamma=2$",
          "$\\sigma=1$", "$\\sigma=3/2$", "$\\sigma=2$",
          "$\\rho=1/4$", "$\\rho=1/8$", "$\\rho=1/16$"]

#distribution_names = ["uniform", "vonmises (0, 1/2)", "vonmises (0, 1)", "vonmises (0, 2)",
#                      "cauchy (0, 1/2)", "cauchy (0, 1)", "cauchy (0, 2)",
#                      "normal (0, 1/2)", "normal (0, 1)", "normal (0, 2)",
#                      "cardioid (0, 1/10)", "cardioid (0, 1/4)", "cardioid (0, 1/2)", "double vonmises", "semicircle 0.6", "semicircle 0.8"]



def performance(data, alpha):
    rejections = [[[0 for y in range(0, len(data[0][0]))] for x in range(0, len(data))] for test in tests]

    a = [0 for b in range(0, len(tests))]
    for i in range(0, len(data)):
        print(i, " / ", len(data))
        for j in range(0, len(data[i])):
            for t in range(0, len(tests)):
                test = tests[t]
                start = time.time()
                for k in range(0, len(data[i][j])):

                    p = test(data[i][j][k])
                    
                    if p<alpha:
                        rejections[t][i][k] += 1
                end = time.time()
                a[t] += end-start
        print([round(x, 3) for x in a])
    print([round(x, 3) for x in a])
    return (1/(N))*np.array(rejections)


def plot_performance(table):

    fig, axs = plt.subplots(len(tests), 4, sharex = True, sharey = True)
    plt.suptitle("Power")
    X = np.linspace(20, max_size, steps)
    X_ticks = [20, 100, 200]
    if len(table[0][0])>1:
        for t in range(0, len(tests)):
            for i in range(0, 4):
                for q in range(0, 3):
                    Y_i = []
                    for j in range(0, steps):
                        Y_i.append(table[t][j][3*i+q])
                    if t == 1:
                        axs[t][i].plot(X, Y_i, label=labels[3*i+q])
                        axs[t][i].legend()
                    else:
                        axs[t][i].plot(X, Y_i)
                    axs[t][i].set_ylim(ymin = -0.05, ymax = 1.05)
                    axs[t][i].label_outer()
                    axs[t][i].set_xticks(X, True)
                    axs[t][i].set_xticks(X_ticks, False)
                #if i == 0:
                #    axs[t][i].hlines(0.05,0, max_size, colors="r")

        for i in range(0, len(axs)):
            axs[i][0].set(ylabel = test_names[i])
        for j in range(0, len(axs[0])):
            axs[-1][j].set(xlabel = distribution_names[j])
            axs[-1][j].xaxis.label.set_size(12)
    elif len(table[0][0])==1:
        for t in range(0, len(tests)):
            Y_i = []
            for j in range(0, steps):
                Y_i.append(table[t][j][0])
            axs[t].plot(X, Y_i)
            axs[t].set_ylim(ymin=0, ymax = 0.2)
            axs[t].hlines(0.05, 0, max_size, colors="r")
            axs[t].label_outer()
            axs[t].set(ylabel = test_names[t])
        axs[-1].set(xlabel = distribution_names[0])
        
    
            
    plt.show()



## Run ##

print("generating samples")
S =  samples()

print("testing performance")

perf = performance(S, 0.05)

print(perf)

plot_performance(perf)


                        
                            
                            
        


