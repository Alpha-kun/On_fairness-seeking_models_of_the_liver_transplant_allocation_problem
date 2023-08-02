import pandas as pd
import numpy as np
import gurobipy as grb
from geopy import distance
from gurobipy import GRB
import matplotlib.pylab as plt


####################################

# problem configuration
digit = 3  # zipcode digit (3 or 4)
cmin = 3
tmax = 600  # maximum allowed radius
TL = 3600  # time limit
fix = False  # use variable fixing

####################################

# input donor hospital info
dhdf = pd.read_csv('cluster' + str(digit) + '_info.csv')
s = dhdf['supply'].tolist()
n = len(s)
latitude = dhdf['latitude'].tolist()
longitude = dhdf['longitude'].tolist()
positions_dh = [(latitude[i], longitude[i]) for i in range(n)]

# input transplant center info
tcdf = pd.read_csv('location_demand_TC.csv')
d = tcdf['Number of Liver Registrations'].tolist()
m = len(d)
latitude = tcdf['Latitude'].tolist()
longitude = tcdf['Longitude'].tolist()
positions_tc = [(latitude[i], longitude[i]) for i in range(m)]

# compute pairwise distance between each supply and demand
dist = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        dist[i, j] = distance.geodesic(positions_dh[i], positions_tc[j], ellipsoid='GRS-80').nm

# print(sum(s)/sum(d))

####################################


Nbr = []
for i in range(n):
    N = sorted(list(range(m)), key=lambda j: dist[i, j])
    Nbr.append(list(filter(lambda j: dist[i, j] < tmax, N)))


# for i in range(n):
#     print(f'{i}:{Nbr[i]}')

# print([sum(c[j] for j in Nbr[i]) for i in I])

# b(i, r, j) = 1 if j is closer to i compared with r
def b(i, r, j):
    global Nbr
    for k in Nbr[i]:
        if j == k:
            return 1
        if r == k:
            return 0
    return 0


# D[i][j]: total demand from all TC in Nbr[0...j]
D = []
for i in range(n):
    pfs = {}
    sm = 0
    for j in Nbr[i]:
        sm += d[j]
        pfs[j] = sm
    D.append(pfs)

# Cr[i][j]: total number of TC from Nbr[0...j]
Cr = []
for i in range(n):
    pfs = {}
    sm = 0
    for j in Nbr[i]:
        sm += 1
        pfs[j] = sm
    Cr.append(pfs)

#######################################

# build stage 1 model
SP1 = grb.Model()

# add variables
varId = [(i, j) for i in range(n) for j in Nbr[i]]
x = SP1.addVars(varId, vtype=GRB.BINARY, name="x")
lmb = SP1.addVar(lb=0, vtype=GRB.CONTINUOUS, name="lmb")

# add objective
SP1.setObjective(lmb, GRB.MAXIMIZE)

# add s/d constraints
SP1.addConstrs(
    grb.quicksum(x[i, r] * b(i, r, j) * s[i] / D[i][r] for i in range(n) for r in Nbr[i]) >= lmb for j in range(m))
# add 1 radius constraints
SP1.addConstrs(grb.quicksum(x[i, r] for r in Nbr[i]) == 1 for i in range(n))
# add minimum TC constraints
SP1.addConstrs(grb.quicksum(x[i, r] * Cr[i][r] for r in Nbr[i]) >= cmin for i in range(n))

# variable fixing
if fix:
    for i in range(n):
        for j in range(cmin - 1):
            x[i, Nbr[i][j]].ub = 0

SP1.Params.TimeLimit = TL
SP1.optimize()

obj_sp1 = SP1.objVal
SP1.write("SP1_d" + str(digit) + "_tmax" + str(tmax) + ("_fixed" if fix else "_unfixed") + ".mst")

print('++++++++++++++++ STAGE-1 ENDS ++++++++++++++++')

#######################################


print('++++++++++++++++ STAGE-2 BEGIN ++++++++++++++++')
eps = 1e-5

# build model
SP2 = grb.Model()

# add variables
varId = [(i, j) for i in range(n) for j in Nbr[i]]
x = SP2.addVars(varId, vtype=GRB.BINARY, name="x")
beta = SP2.addVar(lb=0, vtype=GRB.CONTINUOUS, name="beta")

# add objective
SP2.setObjective(beta, GRB.MINIMIZE)

# add s/d constraints
SP2.addConstrs(grb.quicksum(x[i, r] * b(i, r, j) * s[i] / D[i][r] for i in range(n) for r in Nbr[i]) <= beta for j in range(m))
SP2.addConstrs(grb.quicksum(x[i, r] * b(i, r, j) * s[i] / D[i][r] for i in range(n) for r in Nbr[i]) >= obj_sp1 - eps for j in range(m))
# add 1 radius constraints
SP2.addConstrs(grb.quicksum(x[i, r] for r in Nbr[i]) == 1 for i in range(n))
# add minimum TC constraints
SP2.addConstrs(grb.quicksum(x[i, r] * Cr[i][r] for r in Nbr[i]) >= cmin for i in range(n))


# variable fixing
if fix:
    for i in range(n):
        for j in range(cmin-1):
            x[i, Nbr[i][j]].ub = 0


# load stage 1 incumbent solution
SP2.update()
SP2.read("SP1_d" + str(digit) + "_tmax" + str(tmax) + ("_fixed" if fix else "_unfixed") + ".mst")

SP2.Params.TimeLimit = TL
SP2.update()

SP2.optimize()
SP2.write("SP2_d" + str(digit) + "_tmax" + str(tmax) + ("_fixed" if fix else "_unfixed") + ".mst")
