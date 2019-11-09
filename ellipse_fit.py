from scipy.optimize import minimize
import numpy as np

def mse_ellipse(theta, X1, X2, alpha):
    
    # theta.reshape((6,1))
    J = 0

    for x1, x2 in zip(X1, X2):
        x = np.array([[x1, x2]]).T
        u = np.array([[x1 ** 2, x1 * x2, x2 ** 2, x1, x2, 1]]).T
        A = np.dot(u, u.T)
        dux = np.array([[2 * x1, 0],
                        [x2,    x1],
                        [0, 2 * x2],
                        [1,      0],
                        [0,      1],
                        [0,      0]])
        l = np.dot(x, x.T)
        B = np.dot(dux, np.dot(l, dux.T))

        J += np.dot(theta, np.dot(A, theta)) / np.dot(theta, np.dot(B, theta))

    b = theta[1]
    a = theta[0]
    c = theta[2]

    penalty = (np.dot(theta.T, theta) ** 2) * (1/(b ** 2 - 4 * a * c)) ** 2

    return J + alpha * penalty

def ellipse_numerical(X1, X2, alpha):
    theta0 = np.array([2 for i in range(6)])

    return minimize(mse_ellipse, theta0, args=(X1, X2, alpha))

X_r = np.loadtxt('el.csv', delimiter=',')

theta = ellipse_numerical(X_r.T[0], X_r.T[1], 0.1)

ms = mse_ellipse(theta['x'], X_r.T[0], X_r.T[1], 0.1)

print(ms)
