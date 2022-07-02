README: Documentation of 'Circular statistics: Goodness of Fit'

This library contains the code used for my bachelor thesis 'Circular Statistics: Goodness of Fit'.

Four classical tests of uniformity for circular data are implemented in 'tests.py'. All test functions take as argument a list of angles in [0, 2pi[ and output the p-value of the hypothesis test
HO: the data is drawn from the circular uniform distribution
H1: the data is not drawn from the circular uniform distribution

These tests are
- rayleigh: Rayleigh test
uses the length of the mean resultant vector

- rayleigh2: Rayleigh test with specified mean
uses the mean component in the mean direction.

- watson: Watson test
uses the variance of the differences between the empirical cdf and the cdf of the uniform

- kuiper: Kuiper test
uses the sum of suprema of the differences between the empirical cdf and the cdf of the uniform


The 'sampling' module provides code for generating data from some important distributions and also a selection of
less well known distributions. The commands to draw a sample of size n from these distributions are

- wrapped_cauchy(n, mu, gamma): wrapped Cauchy distribution with parameters mu and gamma.

- wrapped_normal(n, mu, var): wrapped normal distribution with parameters mu and sigma

- cardioid(n, mu, rho): cardioid distribution with parameters mu and rho

- vonmises_mix(n, mu1, kappa1, mu2, kappa2, p): Von Mises mixture with two components
with parameters mu_i, kappa_i corresponding to the i-th component and p giving the weight
of the first component. The weight of the second component is 1-p.

- final(n, b): distribution which draws the observation pi with a probability of 1/b and
an observation from the uniform with a probability of 1 - (1/b)

- semicircle(n, p): semicircle distribution with parameter p giving the weight of the upper
half of the circle such that the upper half has a probability density of p/pi and the lower
half has a probability density of (1-p)/pi

The files 'calibration', 'power_unimodal', 'power_alternative' and 'power_final' are used to
measure the performance of the tests against data from the uniform, from the unimodal alternatives
and from the non-unimodal alternatives.

28/11/2019

For further information, you can contact me at
NICOLAS RUAN 
nicolasruan@hotmail.com
