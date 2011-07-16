import numpy as np
import matplotlib.pyplot as plt
from scikits.learn import mixture


mu, sigma = 0, 0.2
mu2, sigma2 = 1, 0.3

s = np.random.normal(mu, sigma, (100,1))
s2 = np.random.normal(mu2, sigma2, (500,1))
total = np.concatenate((s,s2))


count, bins, ignored = plt.hist(s, 30, normed=True)
count2, bins2, ignored2 = plt.hist(s2, 30, normed=True)
#plt.plot(bins, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins - mu)**2/(2*sigma**2)), linewidth=2, color= 'r')


g = mixture.GMM(n_states=2)
g.fit(total)

print g.weights
print g.means
print g.covars

w=g.weights[0]
w2=g.weights[1]
m=g.means[0][0]
m2=g.means[1][0]
sig=g.covars[0][0][0]
sig2=g.covars[1][0][0]

plt.plot(bins, 10*np.exp(-(bins - m)**2/(2*sig**2)),linewidth=2,color='r')




plt.show()
