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
# special[15, 6] = special[6, 15] = 1
# special[46, 11] = special[11, 46] = 1

hinb = [3, 4, 5, 6, 43]  # Hawaii: 15, CA: 3-6, OR:43
for nb in hinb:
    special[15, nb] = special[nb, 15] = 1
prnb = [10, 11, 12, 13]  # PR: 46,  FL: 10-14
for nb in prnb:
    special[46, nb] = special[nb, 46] = 1

G_mod = nx.from_numpy_array(A+special)


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


def find_fischetti_separator(DG, component, u):
    neighbors_component = [False for i in DG.nodes]
    for i in nx.node_boundary(DG, component, None):
        neighbors_component[i] = True

    visited = [False for i in DG.nodes]
    child = [u]
    visited[u] = True

    while child:
        parent = child
        child = []
        for i in parent:
            if not neighbors_component[i]:
                for j in DG.neighbors(i):
                    if not visited[j]:
                        child.append(j)
                        visited[j] = True

    C = [i for i in DG.nodes if neighbors_component[i] and visited[i]]
    return C


# provides lazy cuts
def cut_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        xval = model.cbGetSolution(model._x)

        # add receiving contiguity constraints
        for b in model._G.nodes:
            Vb = []
            for a in model._G.nodes:
                if xval[(a, b)] > 0.9:
                    Vb.append(a)
            Gb = model._G.subgraph(Vb)
            Cb = nx.descendants(Gb, b)
            Cb.add(b)
            inCb = [False] * len(model._G.nodes)
            for u in Cb:
                inCb[u] = True
            for a in Vb:
                if not inCb[a]:
                    C = find_fischetti_separator(model._G, list(Cb), a)
                    model.cbLazy(x[a, b] <= gp.quicksum(x[c, b] for c in C))

        # add sharing contiguity constraints
        for a in model._G.nodes:
            Va = []
            for b in model._G.nodes:
                if xval[(a, b)] > 0.9:
                    Va.append(b)
            Ga = model._G.subgraph(Va)
            Ca = nx.descendants(Ga, a)
            Ca.add(a)
            inCa = [False] * len(model._G.nodes)
            for u in Ca:
                inCa[u] = True
            for b in Va:
                if not inCa[b]:
                    C = find_fischetti_separator(model._G, list(Ca), b)
                    model.cbLazy(x[a, b] <= gp.quicksum(x[a, c] for c in C))


#################################################

M1 = gp.Model()

# obj variable
lmb = M1.addVar(vtype=GRB.CONTINUOUS, lb=0, name="lmb")

# sharing variables
x = M1.addVars(N, N, vtype=GRB.BINARY, name="x")

# Aux variables
y = M1.addVars(N, N, vtype=GRB.CONTINUOUS, lb=0, name="y")
t = M1.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="t")

# setup lmd with constraint
M1.addConstrs(y[i, j] <= t[i] for i in range(N) for j in range(N))
M1.addConstrs(y[i, j] <= x[i, j] for i in range(N) for j in range(N))
M1.addConstrs(1 - x[i, j] + y[i, j] >= t[i] for i in range(N) for j in range(N))
M1.addConstrs(gp.quicksum(d[k] * y[i, k] for k in range(N)) == 1 for i in range(N))
M1.addConstrs(lmb <= gp.quicksum(s[i] * y[i, j] for i in range(N)) for j in range(N))
# self connectivity constraint
M1.addConstrs(x[i, i] == 1 for i in range(N))


# add max distance constraints
tau_max = 500
for i in range(N):
    for j in range(N):
        if special[i, j] == 0:
            M1.addConstr(dist[i, j] * x[i, j] <= tau_max)


# add objective
M1.setObjective(lmb, GRB.MAXIMIZE)

# set solver and model params
M1.Params.lazyConstraints = 1
M1.Params.FeasibilityTol = 1e-7
M1.Params.TimeLimit = 1800
M1._x = x
M1._G = G_mod
M1.update()


# load warm start solution (optional)
# M1.read("stage1_500.mst")
# M1.update()


# solve and record solution
M1.optimize(cut_callback)
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

# add max distance constraints
for i in range(N):
    for j in range(N):
        if special[i, j] == 0:
            M2.addConstr(dist[i, j] * x[i, j] <= tau_max)

# add objective
M2.setObjective(beta, GRB.MINIMIZE)

# set solver and model params
M2.Params.lazyConstraints = 1
M2.Params.FeasibilityTol = 1e-7
M2.Params.TimeLimit = 300
M2._x = x
M2._G = G_mod
M2.update()

# load incumbent solution from stage 1
M2.read("stage1_" + str(tau_max) + ".mst")
M2.update()

# solve and record solution
M2.optimize(cut_callback)
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
