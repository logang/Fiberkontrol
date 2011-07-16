import numpy as np
import pylab as p
from scikits.learn import mixture

mu1, sigma1 = 5, 4
mu2, sigma2 = 7, 1
mu3, sigma3 = 12, 1.5

three = False

print "mus", mu1, mu2
print "sigs", sigma1, sigma2

x1 = mu1 + sigma1*p.randn(7000)
x2 = mu2 + sigma2*p.randn(20000)

if three:
    x3 = mu3 + sigma3*p.randn(50000)
    xTotal = np.hstack((x1, x2, x3))
else:
    xTotal = np.hstack((x1, x2))

print "size(x1)", len(x1)
print "size(x2)", len(x2)

weight1 = .3
weight2 = .7
sum = weight1*x1.sum() + weight2*x2.sum()

#x1 = np.multiply(x1.sum()/xTotal.sum(), x1)
#x2 = np.multiply(x2.sum()/xTotal.sum(), x2)



##--plot actual curves--#


n1, bins1, patches1 = p.hist(x1, 50, normed=1, weights = np.multiply(weight1, np.ones(len(x1))))
y1 = p.normpdf(bins1, mu1, sigma1)
w1 = (1.0)*len(x1)/len(xTotal)
y1 = np.multiply(w1,y1)


#p.plot(bins1[:-1], n1)

print "x2", len(x2)
print "xTotal", len(xTotal)

n2, bins2, patches2 = p.hist(x2, 50, normed=1)
y2 = p.normpdf(bins2, mu2, sigma2)
w2 = (1.0)*len(x2)/len(xTotal)
y2 = np.multiply(w2,y2)

if three:
    n3, bins3, patches3 = p.hist(x3, 50, normed=1)
    y3 = p.normpdf(bins3, mu3, sigma3)
    w3 = (1.0)*len(x3)/len(xTotal)
    y3 = np.multiply(w3,y3)
    p.plot(bins3, y3, 'bo')


nT, binsT, patchesT = p.hist(xTotal, 50, normed=1)
p.plot(bins1, y1, 'go')
p.plot(bins2, y2, 'ro')



##----Now try fitting a curve--#
total = []

total = xTotal.reshape(len(xTotal),1)

for n in range(1,4):
    g = mixture.GMM(n_states=n)
    g.fit(total, n_iter=100, thresh=0.00001)

    print "weights", g.weights
    print "means", g.means
    print "covars", g.covars


    w = []
    m = []
    sig = []

    color = ['k', 'b', 'g', 'y']
    
    for i in range(n):
        w.append(g.weights[i])
        m.append(g.means[i][0])
        sig.append(np.sqrt(g.covars[i][0][0]))
        
        yfit = w[i]*p.normpdf(binsT, m[i], sig[i])
        
        p.plot(binsT, yfit, color[n%4], lw=2)

        score = np.sum(g.score(total))

        print "score ", n, ": ", score
        
        
p.show()
        
