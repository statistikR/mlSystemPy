'''
Created on Dec 3, 2013

Acknowledgement: Some of this code is adapted from code found in repository: 
https://github.com/luispedro/BuildingMachineLearningSystemsWithPython

@author: micha
'''

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import os

from util import DATA_DIR, CHART_DIR
from plottingUtil import plot_models



print(DATA_DIR)

data = sp.genfromtxt(os.path.join(DATA_DIR,"web_traffic.tsv"), delimiter ="\t")
print(data[0:10])
print(data.shape)

x = data[:,0]
y = data[:,1]

# there are missing in the y variable
print sum(sp.isnan(y))

# exclude cases from both variables
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

# minimalist plot
#plt.scatter(x,y)
#plt.show()

#p22 regression fit

fp1 = sp.polyfit(x,y,1) # min output
fp1, residuals, rank, sv, rcond = sp.polyfit(x,y,1, full=True) # full output
print fp1

#function for predicting fitted values
f1 = sp.poly1d(fp1)
print(f1(1))
# errror calculation
def error(f,x,y):
    return sp.sum((f(x)-y)**2)
print(error(f1,x,y))

#p24 fit higher order model
fp2 = sp.polyfit(x,y,2)
f2 = sp.poly1d(fp2)

fp3 = sp.polyfit(x,y,3)
f3 = sp.poly1d(fp3)
## graph fitted line
# create x values from 0 to max x
fx = sp.linspace(0,x[-1], 1000)
fx = sp.linspace(0,1000, 1000) # increase x to plot further in future

#plt.scatter(x,y)
#plt.xticks([w*7*24 for w in range(10)],["week %i" %w for w in range(10)])
#plt.plot(fx,f1(fx), linewidth=4)
#plt.plot(fx,f2(fx), linewidth=4)
#plt.plot(fx,f3(fx), linewidth=4)
#plt.legend(["d=%i" % f1.order, "d=%i" % f2.order, "d=%i" % f3.order], loc= "upper left")

print("-------------------------------")
print("error d=1: %f" % error(f1,x,y))
print("error d=2: %f" % error(f2,x,y))
print("error d=3: %f" % error(f3,x,y))
plt.show()

###############################
## switch to inflection point models

# inflection seems to happen after 3.5 weeks

inflection = 3.5 * 7 * 24
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

fa = (sp.poly1d(sp.polyfit(xa,ya,1)))
fb = (sp.poly1d(sp.polyfit(xb,yb,1)))

fb2 = sp.poly1d(sp.polyfit(xb, yb, 2))

fa_error = error(fa,xa,ya)
fb_error = error(fb,xb,yb)

print("-------------------------------")
print("total error 2 lines: %f" % (fa_error + fb_error))

print("Trained only on data after inflection point")
fb1 = fb
fb2 = sp.poly1d(sp.polyfit(xb, yb, 2))
fb3 = sp.poly1d(sp.polyfit(xb, yb, 3))
fb10 = sp.poly1d(sp.polyfit(xb, yb, 10))
fb50 = sp.poly1d(sp.polyfit(xb, yb, 50))

print("Errors for only the time after inflection point")
for f in [fb1, fb2, fb3, fb10, fb50]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

plot_models(
    x, y, [fb1, fb2, fb3, fb10, fb50], os.path.join(CHART_DIR, "1400_01_07.png"),
    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

# separating training from testing data
frac = 0.3
split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
fbt1 = sp.poly1d(sp.polyfit(xb[train], yb[train], 1))
fbt2 = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))
fbt3 = sp.poly1d(sp.polyfit(xb[train], yb[train], 3))
fbt10 = sp.poly1d(sp.polyfit(xb[train], yb[train], 10))
fbt100 = sp.poly1d(sp.polyfit(xb[train], yb[train], 100))

print("Test errors for only the time after inflection point")
for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    print("Error d=%i: %f" % (f.order, error(f, xb[test], yb[test])))

plot_models(
    x, y, [fbt1, fbt2, fbt3, fbt10, fbt100], os.path.join(CHART_DIR,
                                                          "1400_01_08.png"),
    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

from scipy.optimize import fsolve
print(fbt2)
print(fbt2 - 100000)

reached_max = fsolve(fbt2 - 100000, 800) / (7 * 24)
print(reached_max)
reached_max = fsolve(fbt2 - 100000, 400) / (7 * 24)
print(reached_max)
reached_max = fsolve(fbt2 - 100000, 100000000) / (7 * 24)
print(reached_max)
print("100,000 hits/hour expected at week %f" % reached_max[0])

