import numpy as np
from scipy import stats

"""
This is simple code to produce a bar chart given a list of numbers.
Not currently in use, but a good reference.
"""

N = 11
s = [.45, .34, .60, .39, .80, 1.0, .43, .17, .45, .30, .40, .37]
n = [.30, .60, 1.0, .29, .35, .75, .20, .32, .59, .66, .79, .37]

snorm = [.65, .65, .65, .53, .93, .65, .68, .53, .65, .56, .66, .48]
nnorm = [.51, .55, .74, .45, .54, .62, .48, .73, .54, .57, .60, .47]

snorm = [.65, .65, .65, .53, .93, .65, .68, .65, .56, .58, .48]
nnorm = [.51, .55, .74, .45, .54, .62, .48, .54, .57, .60, .47]

print "s-n: ", (np.round(100*(np.array(s)-np.array(n)))/100).tolist()
print "snorm-nnorm: ",(np.round(100* (np.array(snorm)-np.array(nnorm)))/100).tolist()

print 'social mean: ', np.mean(s)
print 'novel mean: ', np.mean(n)
print 'social 95%: ', 1.96*np.sqrt(np.var(s)/N)
print 'novel 95%: ', 1.96*np.sqrt(np.var(n)/N)

print 'social normalized area  mean: ', np.mean(snorm)
print 'novel normalized area mean: ', np.mean(nnorm)
print 'social normalized area 95%: ', 1.96*np.sqrt(np.var(snorm)/N)
print 'novel normalized area 95%: ', 1.96*np.sqrt(np.var(nnorm)/N)
print 'sum intervals: ', 1.96*np.sqrt(np.var(snorm)/N) + 1.96*np.sqrt(np.var(nnorm)/N)
print 'difference means: ', np.mean(snorm) - np.mean(nnorm)

norm_diff = np.array(snorm) - np.array(nnorm)

[tvalue, pvalue] = stats.ttest_rel(s, n)
print "fluor tvalue: ", tvalue, " fluor pvalue: ", pvalue

[tvalue, pvalue] = stats.ttest_rel(snorm, nnorm)
print "normalized area tvalue: ", tvalue, " normalized area pvalue: ", pvalue


