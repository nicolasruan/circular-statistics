import numpy as np

## Sampling

def wrap_sample(sample):
    return [theta % (2*np.pi) for theta in sample]

def cauchy(sample_size, mu, gamma):
    x = np.random.rand(sample_size)
    y = gamma*np.tan(np.pi*(x-1/2))+mu
    return y

def wrapped_cauchy(sample_size, mu, gamma):
    return wrap_sample(cauchy(sample_size, mu, gamma))

def wrapped_normal(sample_size, mu, var):
    return wrap_sample(np.random.normal(mu, var, sample_size))

def cardioid(sample_size, mu, rho):
    x = 2*np.pi*np.random.rand(sample_size)
    y = inverse_card(x, mu, rho)
    return y


def card_f(t, mu , rho):
    return (1/(2*np.pi))*(1 + 2*rho*np.cos(t-mu))

def card_F(t, mu, rho):
    return (1/(2*np.pi))*(t + 2*rho*np.sin(t-mu) + 2*rho*np.sin(mu))


def inverse_card(y, mu, rho):
    t = 0
    error = 1
    steps = 0
    while np.linalg.norm(error) > 0.01 and steps<200:
        error = card_F(t, mu, rho) - y
        t -= (error/card_f(t, mu, rho))
        steps += 1
    return t


def double_vonmises(sample_size, kappa):
    x = []
    x = np.random.vonmises(0, kappa, sample_size)
    return x+k*np.pi


def final(sample_size, b):
    x = []
    for s in range(sample_size):
        k = np.random.rand()
        if k < 1/b:
            x.append(np.pi)
        else:
            x.append(2*np.pi*np.random.rand())
    return x


def vonmises_mix(sample_size, mu1, kappa1, mu2, kappa2, p):
    x = []
    for i in range(0, sample_size):
        k = (np.random.rand()>p)
        if k:
            x.append(np.random.vonmises(mu1, kappa1))
        else:
            x.append(np.random.vonmises(mu2, kappa2))

    return x


def semicircle(sample_size, p):
    C = []
    for i in range(0, sample_size):
      A = np.random.rand()
      if A < p:
        C.append(np.pi*np.random.rand())
      else:
        C.append(np.pi + np.pi*np.random.rand())
    return C
