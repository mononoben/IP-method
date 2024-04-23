# calculate step length; alpha
import numpy as np
from numpy import linalg as LA

# Omega : [tau, gamma, eta]
# init = [x0, y0, z0, mu0]



# simple ver.
def simple(x, dx, z, dz, n, Omega):
    # glabal var. in primal-dual.py
    tau = Omega[0]
    # main part
    alpha_p = min([-x[i]/dx[i] for i in range(n) if dx[i] < 0])
    alpha_d = min([-z[i]/dz[i] for i in range(n) if dz[i] < 0])
    alpha = min(tau*alpha_p, tau*alpha_d, 1)
    return alpha

# introducing neighborhood
## feasible interior point method
def fesNlong(x, dx, z, dz, mu, Omega):
    # glabal var. in primal-dual.py
    tau, gamma = Omega[:2]
    # main part
    l, alpha = 0, 0
    while True:
        alpha = tau**l
        xnext = x + alpha*dx
        znext = z + alpha*dz
        if min(xnext * znext) >= (1 - gamma)*mu:
            break
        else:
            l = l + 1
    return alpha

## infeasible interior point method
def infesNlong(x, dx, y, dy, z, dz, mu, Omega, A, b, c, init):
    # global var. in primal-dual.py
    tau, gamma, eta = Omega
    x0, y0, z0, mu0 = init
    # main part
    def p_relax(xnext, A, b):
        return True if LA.norm(A@xnext - b) <= eta*mu/mu0 * LA.norm(A@x0-b) else False
    def d_relax(ynext, A, c):
        return True if LA.norm(A.T@y + z - c) <= eta*mu/mu0 * LA.norm(A.T@y0 + z0 - c) else False
    l, alpha = 0, 0
    while True:
        alpha = tau**l
        xnext = x + alpha*dx
        ynext = y + alpha*dy
        znext = z + alpha*dz
        if p_relax(xnext, A, b) and d_relax(ynext, A, c) and min(xnext * znext) >= (1 - gamma)*mu:
            break
        else: 
            l = l + 1
    return alpha