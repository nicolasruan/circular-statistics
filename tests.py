import numpy as np
from scipy.stats import norm


## Tests

# Rayleigh test 
def rayleigh(sample): #sample is a list of angles in [0, 2*pi[
    n = len(sample)
    mrv = np.sum(np.exp([np.complex(0, s) for s in sample]))/n
    angle = np.angle(mrv)
    radius = np.absolute(mrv)
    z = n * (radius**2)
    p = 0
    if n<50:
        p = np.exp(-z)*(1+ (2*z - z**2)/(4*n) - (24*z-132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
    else:
        p = np.exp(-z)
    return p

# Rayleigh test with specified mean

def rayleigh2(sample): #sample is a list of angles in [0, 2*pi[
    n = len(sample)
    mu = 0
    C = (1/n) * np.sum([np.cos(s - mu) for s in sample])
    z = np.sqrt(2*n) * C
    p = 1 - norm.cdf(z) + norm.pdf(z) * ((3*z-z**3)/(16*n) + (15*z+305*z**3-125*z**5+9*z**7)/(4608*n**2))
    return p

# Watson test 
def watson(sample): #sample is a list of angles in [0, 2*pi[
    n = len(sample)
    sample = list(sample)
    sample.sort()
    sample = np.array(sample)
    sample = (1/(2*np.pi))*sample

    mean = np.mean(sample)
    sample = list(sample)
    sample.insert(0, 0)
    sample.append(1)
    sample = np.array(sample)
    u2 = 1/(12*n)
    
    for i in range(1, n+1):
        u2 += ((sample[i-1] - mean - (i-(1/2))/n + 1/2)**2)
    #u2_mod = (u2 - 0.1/n + 0.1/(n**2))*(1+0.8/n)
    p = 0
    for m in range(1, 100):
        p += np.exp( (-2) * (m**2) * (np.pi**2) * u2) * ((-1)**(m-1))
    p *= 2
    return p

# Kuiper test
def kuiper(sample): #sample is a list of angles in [0, 2*pi[
    n = len(sample)
    sample = list(sample)
    sample.sort()
    sample.insert(0, 0)
    sample.append(2* np.pi)
    sample = np.array(sample)
    sample = (1/(2*np.pi))*sample

    d1 = max([sample[i]- i/n for i in range(0, n)])
    d2 = min([sample[i]-i/n for i in range(0, n)])

    v = d1 - d2 + 1/n

    t = np.sqrt(n)*v

    k1 = 0
    k2 = 0
    
    for i in range(1, 100):
        k1 += (4*(i**2)*(t**2)-1)*np.exp((-2)*(i**2)*(t**2))
        k2 += (i**2)*(4*(i**2)*(t**2)-3) * np.exp((-2)*(i**2)*(t**2))


    p = 2 * k1 - (8*t)/(3*np.sqrt(n))*k2
    
    return p




### Hodges Ajnes test
##
##def ha_test(sample):
##  size = len(sample)*12 # want 4|12 zodat we makkelijk kwartcirkels kunnen maken
##  checkpoints = np.linspace(0,2*np.pi,size)
##  amounts = []
##  for i in range(size-1):
##    lower = checkpoints[(int(i - size/4)) % size]
##    upper = checkpoints[(int(i + size/4)) % size]
##
##    if lower < upper:
##      count = 0
##      for k in range(len(sample)):
##        if sample[k] <= upper and sample[k] >= lower:
##          count += 1
##      amounts.append(count)
##
##    else:
##      copy_sample = sample[:]
##      #print(copy_sample)
##      for k in range(len(copy_sample)):
##        if copy_sample[k] >= lower:
##          copy_sample[k] = copy_sample[k] - 2*np.pi
##      lower = lower - 2*np.pi
##      count = 0
##      for k in range(len(copy_sample)):
##        if copy_sample[k] <= upper and copy_sample[k] >= lower:
##          count += 1
##      amounts.append(count)
##  n = max(amounts)
##  m = min(amounts)      
##  return(n-m)
##
##def ha(sample):
##    n = len(sample)
##    return get_pval(sample, ha_test)
##
### Ajnes test
##
##def An(sample):
##    n = len(sample)
##    cmd = 0
##    for i in range(0, n):
##        for j in range(0, n):
##            cmd += np.pi - abs(np.pi - abs(sample[i] - sample[j]))
##    cmd = cmd/(n**2)
##    an = n*(1/4 - cmd/(2*np.pi))
##    return an
##
##def ajnes(sample):
##    n = len(sample)
##    an = An(sample)
##    k = 0
##    for i in range(1, 101):
##        k += (((-1)**(i-1))/(2*i-1))*np.exp(-((2*i-1)**2)*(np.pi**2)*an/2)
##    p = (4/np.pi)*k
##    return p
##
### Hermans-Rasson test
##
##def hermansrasson_stat(sample):
##    n = len(sample)
##    an = An(sample)
##    k = 0
##    for j in range(1, n):
##        for i in range(0, j):
##            k += abs(np.sin(sample[i]-sample[j]) - 2/np.pi)
##    hn = 2*np.pi*an - 2.895*k
##    return hn
##
##def get_pval(sample, statistic): # Monte Carlo method
##    n = len(sample)
##    stat = statistic(sample)
##    a = 0
##    b = 250
##    for i in range(0, 1000):
##        stat2 = statistic(2*np.pi*np.random.rand(n))
##        if stat2 > stat:
##            a += 1
##    return (a+1)/(b+1)
##
##def hermansrasson(sample):
##    n = len(sample)
##    return get_pval(sample, hermansrasson_stat)
##
### Pycke test
##
##def pycke_stat(sample):
##    n = len(sample)
##    k = 0
##    a = np.sqrt(0.5)
##    for i in range(0, n):
##        for j in range(0, n):
##            C = np.cos(sample[i] - sample[j])
##            k += (2*C - a)/(1.5 - 2*a*C)
##    k = k/n
##    return k
##
##def pycke(sample):
##    return get_pval(sample, pycke_stat)
##    


