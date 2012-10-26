import numpy as np
import os
from scipy.optimize import leastsq


def unpackArduino(data):
    data = data.tolist()[0][2]
    return data

def unpackLabjack(data):
    data = data.tolist()
    return data[0:-1000]

def filter(data):

    if data.count([]) > 0:
        data.remove([])

    return data


def FlattenData(flattenedData, givenData, givenX, fitCurves):
    fp = lambda a, x: a[0]*np.exp(a[1]*x)+a[2]*np.exp(a[3]*x)
    error = lambda a, x, y: (fp(a,x)-y)

    counter = 0
    y = np.array(givenData)

    x = np.array(givenX[0:len(y)])

#    print len(x)
#    print len(y)

    a0 = np.array([y[1], -1.0, y[1], -1.0])


#    print 'a', a0
    a, success = leastsq(error, a0, args=(x, y), maxfev=50)

    yfit = fp(a, x)

    dIntensity = fp(a, x[0])-fp(a,x)
    yflat = y+dIntensity

    flattenedData.append(yflat)
    fitCurves.append(yfit)
    counter += 1


def ConvertStringToIntList(phrase):
    list = []
    curr_value = ''
    for i in range(len(phrase)):
        if phrase[i] != ',':
            curr_value += phrase[i]
        else:
            list.append(int(curr_value))
            curr_value = ''

        if i == (len(phrase) - 1):
            list.append(int(curr_value))
            curr_value = ''

    return list


def ConvertStringToFloatList(phrase):
    list = []
    curr_value = ''
    for i in range(len(phrase)):
        if phrase[i] != ',':
            curr_value += phrase[i]
        else:
            list.append(float(curr_value))
            curr_value = ''

        if i == (len(phrase) - 1):
            list.append(float(curr_value))
            curr_value = ''

    return list
