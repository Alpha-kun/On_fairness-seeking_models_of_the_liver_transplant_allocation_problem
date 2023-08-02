import pandas as pd
import networkx as nx
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# input adjacency matrix
Adjdata = pd.read_csv('adjacency_boundary_1.csv')
A = Adjdata.to_numpy()
N = A.shape[0]
A = A - np.eye(N)
G = nx.from_numpy_array(A)
# nx.draw(G,node_size=10)
# plt.show()


# give special treatment to Hawaii and PR
special = np.zeros((N, N))

# strategy 1: connect HI and PR to their nearest mainland DSA
# special[15, 6] = special[6, 15] = 1
# special[46, 11] = special[11, 46] = 1

# strategy 2: connect HI to CA and OR and PR to FL
hinb = [3, 4, 5, 6, 43]  # Hawaii: 15, CA: 3-6, OR:43
for nb in hinb:
    special[15, nb] = special[nb, 15] = 1
prnb = [10, 11, 12, 13]  # PR: 46,  FL: 10-14
for nb in prnb:
    special[46, nb] = special[nb, 46] = 1

A_mod = A+special
G_mod = nx.from_numpy_array(A_mod)


# input distance matrix
distdata = pd.read_csv('opo_dist.csv')
dist = distdata.to_numpy()
dist = dist[0:, 1:]
# plt.imshow(dist.astype(float))
# plt.show()


# input DSA s/d info
sddata = pd.read_csv('DSA_sd.csv')
code = sddata['opo_ctr_cd'].tolist()
code2id = {}
for i in range(len(code)):
    code2id[code[i]] = i
s = sddata['donor_1317'].tolist()
d = sddata['incident_20_1317'].tolist()


# input DSA location info
opodata = pd.read_csv('OPO_location_list.csv')
lati = opodata['Latitude'].tolist()
longi = opodata['Longitude'].tolist()
loc = [np.array([longi[i], lati[i]]) for i in range(N)]

# draw us map
nx.draw(G_mod, pos=loc, node_size=20)
plt.show()


################################################################

# create indices for f,g variables
idx = []
for i in range(N):
    for k in range(N):
        if A_mod[i][k] == 1:
            for j in range(N):
                idx.append((i, k, j))


M1 = gp.Model()

# obj variable
lmb = M1.addVar(vtype=GRB.CONTINUOUS, lb=0, name="lmb")

# sharing variables
x = M1.addVars(N, N, vtype=GRB.BINARY, name="x")

# Aux variables
y = M1.addVars(N, N, vtype=GRB.CONTINUOUS, lb=0, name="y")
t = M1.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="t")

# flow variables
f = M1.addVars(idx, vtype=GRB.CONTINUOUS, lb=0, name="f")
g = M1.addVars(idx, vtype=GRB.CONTINUOUS, lb=0, name="g")

# setup lmd with constraint
M1.addConstrs(y[i, j] <= t[i] for i in range(N) for j in range(N))
M1.addConstrs(y[i, j] <= x[i, j] for i in range(N) for j in range(N))
M1.addConstrs(1 - x[i, j] + y[i, j] >= t[i] for i in range(N) for j in range(N))
M1.addConstrs(gp.quicksum(d[k] * y[i, k] for k in range(N)) == 1 for i in range(N))
M1.addConstrs(lmb <= gp.quicksum(s[i] * y[i, j] for i in range(N)) for j in range(N))
# self connectivity constraint
M1.addConstrs(x[i, i] == 1 for i in range(N))

# add f constrains
for i in range(N):
    for j in range(N):
        if i != j:
            M1.addConstr(gp.quicksum(f[i, k, j] for k in G_mod[i]) - gp.quicksum(f[k, i, j] for k in G_mod[i]) == x[i, j], name='f(a)' + str(i) + ',' + str(j))
for i in range(N):
    for j in range(N):
        M1.addConstr(gp.quicksum(f[k, i, j] for k in G_mod[i]) <= (N-1) * x[i, j], name='f(b)' + str(i) + ',' + str(j))
for j in range(N):
    M1.addConstr(gp.quicksum(f[j, k, j] for k in G_mod[j]) == 0, name='f(c)' + str(j))

# add g constrains
for i in range(N):
    for j in range(N):
        if i != j:
            M1.addConstr(gp.quicksum(g[j, k, i] for k in G_mod[j]) - gp.quicksum(g[k, j, i] for k in G_mod[j]) == x[i, j], name='g(a)' + str(i) + ',' + str(j))
for i in range(N):
    for j in range(N):
        M1.addConstr(gp.quicksum(g[k, j, i] for k in G_mod[j]) <= (N-1) * x[i, j], name='g(b)' + str(i) + ',' + str(j))
for i in range(N):
    M1.addConstr(gp.quicksum(g[i, k, i] for k in G_mod[i]) == 0, name='g(c)' + str(i))

# add max distance constraints
tau_max = 700
for i in range(N):
    for j in range(N):
        if special[i, j] == 0:
            M1.addConstr(dist[i, j] * x[i, j] <= tau_max)

# add objective
M1.setObjective(lmb, GRB.MAXIMIZE)

# set solver params
M1.Params.FeasibilityTol = 1e-7
M1.Params.TimeLimit = 1800
M1.update()

# load warm start solution (optional)
# M1.read("stage1_500_cold.mst")
# M1.update()

# solve and record solution
M1.optimize()
obj_s1 = M1.objVal
M1.write("stage1_" + str(tau_max) + ".mst")

# show distribution of s/d ratio
sdr = [sum(s[i] * y[i, j].X for i in range(N)) for j in range(N)]
plt.plot(sdr, len(sdr) * [0], "x")
plt.title('Stage 1')
plt.xlabel('s/d ratio')
plt.show()

# show stage-1 sharing network
DG = nx.DiGraph()
DG.add_nodes_from(range(N))
for i in range(N):
    for j in range(N):
        if i != j and x[i, j].X > 0.5:
            DG.add_edge(i, j)
nx.draw(DG, pos=loc, node_size=20)
plt.title('stage-1 network')
plt.show()

print('++++++++++++++++ STAGE-1 ENDS ++++++++++++++++')


#########################################


print('++++++++++++++++ STAGE-2 BEGIN ++++++++++++++++')

eps = 1e-5

M2 = gp.Model()

# obj variable
beta = M2.addVar(vtype=GRB.CONTINUOUS, lb=0, name="beta")

# sharing variables
x = M2.addVars(N, N, vtype=GRB.BINARY, name="x")

# Aux variables
y = M2.addVars(N, N, vtype=GRB.CONTINUOUS, lb=0, name="y")
t = M2.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="t")

# flow variables
f = M2.addVars(idx, vtype=GRB.CONTINUOUS, lb=0, name="f")
g = M2.addVars(idx, vtype=GRB.CONTINUOUS, lb=0, name="g")

# setup lmd with constraint
M2.addConstrs(y[i, j] <= t[i] for i in range(N) for j in range(N))
M2.addConstrs(y[i, j] <= x[i, j] for i in range(N) for j in range(N))
M2.addConstrs(1 - x[i, j] + y[i, j] >= t[i] for i in range(N) for j in range(N))
M2.addConstrs(gp.quicksum(d[k] * y[i, k] for k in range(N)) == 1 for i in range(N))
M2.addConstrs(obj_s1 - eps <= gp.quicksum(s[i] * y[i, j] for i in range(N)) for j in range(N))
# setup beta with constraint
M2.addConstrs(beta >= gp.quicksum(s[i] * y[i, j] for i in range(N)) for j in range(N))
# self connectivity constraint
M2.addConstrs(x[i, i] == 1 for i in range(N))

# add f constrains
for i in range(N):
    for j in range(N):
        if i != j:
            M2.addConstr(gp.quicksum(f[i, k, j] for k in G_mod[i]) - gp.quicksum(f[k, i, j] for k in G_mod[i]) == x[i, j], name='f(a)' + str(i) + ',' + str(j))
for i in range(N):
    for j in range(N):
        M2.addConstr(gp.quicksum(f[k, i, j] for k in G_mod[i]) <= (N-1) * x[i, j], name='f(b)' + str(i) + ',' + str(j))
for j in range(N):
    M2.addConstr(gp.quicksum(f[j, k, j] for k in G_mod[j]) == 0, name='f(c)' + str(j))

# add g constrains
for i in range(N):
    for j in range(N):
        if i != j:
            M2.addConstr(gp.quicksum(g[j, k, i] for k in G_mod[j]) - gp.quicksum(g[k, j, i] for k in G_mod[j]) == x[i, j], name='g(a)' + str(i) + ',' + str(j))
for i in range(N):
    for j in range(N):
        M2.addConstr(gp.quicksum(g[k, j, i] for k in G_mod[j]) <= (N-1) * x[i, j], name='g(b)' + str(i) + ',' + str(j))
for i in range(N):
    M2.addConstr(gp.quicksum(g[i, k, i] for k in G_mod[i]) == 0, name='g(c)' + str(i))

# add max distance constraints
for i in range(N):
    for j in range(N):
        if special[i, j] == 0:
            M2.addConstr(dist[i, j] * x[i, j] <= tau_max)

# add objective
M2.setObjective(beta, GRB.MINIMIZE)


# set solver params
M2.Params.FeasibilityTol = 1e-7
M2.Params.TimeLimit = 1800
M2.update()

# load incumbent solution from stage 1
M2.read("stage1_" + str(tau_max) + ".mst")
M2.update()

# solve and record solution
M2.optimize()
obj_s2 = M2.objVal
M2.write("stage2_" + str(tau_max) + ".mst")

# show distribution of s/d ratio
sdr = [sum(s[i] * y[i, j].X for i in range(N)) for j in range(N)]
plt.plot(sdr, len(sdr) * [0], "x")
plt.title('Stage 2')
plt.xlabel('s/d ratio')
plt.show()

# show stage-2 sharing network
DG = nx.DiGraph()
DG.add_nodes_from(range(N))
for i in range(N):
    for j in range(N):
        if i != j and x[i, j].X > 0.5:
            DG.add_edge(i, j)
nx.draw(DG, pos=loc, node_size=20)
plt.title('stage-2 network')
plt.show()
