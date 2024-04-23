import numpy as np
from numpy import linalg as LA
# own package
import sys
sys.dont_write_bytecode = True
from mypackage import calstep as cs

# settings
np.set_printoptions(precision=3, suppress=True)


# input data
finPATH = "interior-Point methods/src/fin/lp01.csv"   # relative path (check os.getcwd())
coef = np.genfromtxt(finPATH, delimiter=",")

A = coef[1:, :-1]   # check if A is full-rank
b = coef[1:, -1].reshape((-1, 1))
c = coef[0, :-1].reshape((-1, 1))
m, n = A.shape

# inner product
def ip(a, b):
    return float(sum(a*b)) # np.dot(a.T, b) or a.T@b

# path-following 

# Omega.
beta = 0.1
tau, gamma, eta = 0.99, 0.95, 1.0   # for calculate step length; alpha
Omega = [tau, gamma, eta]
# initialize 
s = 100
x0 = s*np.ones((n, 1))
y0 = np.zeros((m, 1))
z0 = s*np.ones((n, 1))
mu0 = ip(x0, z0) / n

init = [x0, y0, z0, mu0]

x, y, z = x0, y0, z0
mu = mu0

# threshold, stopping criterion
iter = 0
epsilon = 1e-7  

# iteration
while iter < 100:
    # show process
    print(f"iteraion = {iter}, mu = {mu:.3g}, c^T x = {ip(c, x):.3g}, b^T y = {ip(b, y):.3g}")
    # stopping criterion
    if mu < epsilon : break
    # for convenience
    X = np.diag(x.flatten())
    Z = np.diag(z.flatten())
    e = np.ones((n, 1))
    # calculate residual, (rp, rd, rc)
    rp = b - A@x
    rd = c - A.T@y - z
    rc = beta*mu*np.ones((n,1)) - X@z
    # calculate serach direction, (dx, dy, dz)
    M = A@ LA.inv(Z) @ X @ A.T
    rhs = rp - A @ LA.inv(Z) @ (rc - X@rd)
    dy = np.linalg.solve(M, rhs)
    dz = rd - A.T@dy
    dx = LA.inv(Z) @ (rc - X@dz)
    # calculate step length, alpha
    alpha = cs.simple(x, dx, z, dz, n, Omega)  # simple ver.
    #alpha = cs.fesNlong(x, dx, z, dz, mu, Omega)  # introducing neighborhood
    #alpha = cs.infesNlong(x, dx, y, dy, z, dz, mu, Omega, A, b, c, init)
    # update
    x = x + alpha*dx
    y = y + alpha*dy
    z = z + alpha*dz
    mu = ip(x, z) / n
    iter = iter + 1

# output
print(f"optimal (obtained) \n x = {x.T}, \n y = {y.T}, \n z = {z.T}")
#print(f"duality gap = {ip(c, x) - ip(b, y):.3g}")