'''
Created on Dec 3, 2013

Acknowledgement: Some of this code is adapted from code found in repository: 
https://github.com/luispedro/BuildingMachineLearningSystemsWithPython

@author: micha
'''

import numpy as np
import scipy as sp
import timeit


a = np.array([0,1,2,3,4,5])

print "show array"
print a

print "dimensions of array"
print a.ndim

print "size"
print a.shape

b = a.reshape((3,2))

print "show b"
print b
c = a*2

print "INDEXING"
print c
print c[1]
print c[(2)]
print c[np.array([2,3,4])]
print c[([2,3,4])]
print type(c)
print c>4
print c[c>4]

d = np.array([2,3,4,5,4,3,2])
d[d>3] = 6
print d

# recode outliers
e = d.clip(0,3)
print e

c = np.array([1,2,np.NAN, 3, 4])
print c
print np.isnan(c)
print c[~np.isnan(c)]
print np.mean(c[~np.isnan(c)])

## runtime behavior python

normal_py_sec = timeit.timeit("sum(x * x for x in xrange(1000))",number=10000)

good_np_sec = timeit.timeit("m.dot(m)",setup="import numpy as np; m=np.arange(1000)",
                            number=10000)
print("Normal Py: %f sec" %  normal_py_sec)
print("Good NumPy: %f sec" % good_np_sec)

# calculate result
m=np.arange(10)
print m.dot(m)
print m * m



if __name__ == '__main__':
    pass