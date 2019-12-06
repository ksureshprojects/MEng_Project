import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fit_ellipse(X, Y, plot=False, tol=10 ** -6, labels=None):
    """
        Solve Ax^2 + Bxy + Cy^2 + Dx +Ey = 1
    """
    # Formulate and solve the least squares problem ||Ax - b ||^2
    X = X.reshape((X.shape[0], 1))
    Y = Y.reshape((Y.shape[0], 1))
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b)[0].squeeze()
    r = min(x[0], x[2])/max(x[0], x[2])
    if not plot:
        return (np.linalg.norm(np.dot(A,x) - b, ord=2) ** 2)/(b.shape[0]), r
    
    # Plot the least squares ellipse
    x_coord = np.linspace(np.min(X) - 0.5, np.max(X) + 0.5, 300)
    y_coord = np.linspace(np.min(Y) - 0.5, np.max(Y) + 0.5, 300)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord) 
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
    plt.figure()
    plt.scatter(X,Y)
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.show()
    print("MSE general elliptical fit: {}".format((np.linalg.norm(np.dot(A,x) - b, ord=2) ** 2)/(b.shape[0])))
    
def order_ellipse(X, Y):
    X = X.reshape((X.shape[0], 1))
    Y = Y.reshape((Y.shape[0], 1))
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b)[0].squeeze()
#     r = min(x[0], x[2])/max(x[0], x[2])
    
    x_coord = np.linspace(np.min(X) - 0.5, np.max(X) + 0.5, 300)
    y_coord = np.linspace(np.min(Y) - 0.5, np.max(Y) + 0.5, 300)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord) 
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
    
    X_coord = X_coord[np.where(np.abs(Z_coord - 1) < 10 ** -3)]
    Y_coord = Y_coord[np.where(np.abs(Z_coord - 1) < 10 ** -3)]
    
    c_x = np.mean(X_coord)
    c_y = np.mean(Y_coord)
    
    df = pd.DataFrame(columns=['Time', 'Angle'])
    for i in range(X.shape[0]):
        x = X[i][0] - c_x
        y = Y[i][0] - c_y

        if x == 0:
            if y > 0:
                a = np.pi/2
            else:
                a = np.pi/2 + np.pi
        elif y == 0:
            if x > 0:
                a = 0
            else:
                a = np.pi
        elif x > 0 and y > 0:
            a = np.arctan(y/x)
        elif x > 0 and y < 0:
            a = np.arctan(y/x) + 2*np.pi
        else: 
            a = np.arctan(y/x) + np.pi

        
            
        line = pd.DataFrame(data=[[i, -a]], columns=['Time', 'Angle'])
        df = df.append(line, ignore_index=True)
        
    return df
        
    