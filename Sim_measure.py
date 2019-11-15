import scipy.spatial.distance
from math import *

"""
Several similarity measures for different types T
"""

# Dummy similarity
def trivial(x,y):
    return 1 if x==y else 0

# For T = Float
def euclidan(x,y):
    return euclidan(x,y)

# For T = Char
def sim_ASCII(a,b):
    return 1.0 - (abs(ord(a) - ord(b))) / 25.0

# For T = List<Float>
def cosine(X,Y) :
    if(sum(X) == 0 or sum(Y) == 0) :
        return 1
    else :
        dist = 0.0
        normX = 0
        normY = 0
        for i in range(len(X)) :
            dist += X[i] * Y[i]
            normX += X[i]**2
            normY += Y[i]**2
        return dist / (sqrt(normX) * sqrt(normY))
        #return 1 - spatial.distance.cosine(X, Y)