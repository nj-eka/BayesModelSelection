import numpy as np
from scipy.stats import multivariate_normal

def EM_algorithm(X_train, X_test = None, y_train = None, count_of_clasters = 2, w = None, mu = None, cov = None, alpha = 0.9, Num = 50):
    if X_test is not None:
        X = np.vstack([X_train, X_test])
    if X_test is None:
        X = X_train
    count_of_data = X.shape[0]
    count_of_features = X.shape[1]

    if mu is None:
        mu = np.reshape(np.random.rand(count_of_clasters*count_of_features)- 0.5, [count_of_clasters, count_of_features])
    if w is None:
        w = np.ones(shape=[count_of_clasters])/count_of_clasters
    if cov is None:
        cov = []
        for i in range(count_of_clasters):
            cov.append(np.eye(X.shape[1]))
        cov = np.array(cov)
    
    g = np.zeros(shape=[count_of_clasters, count_of_data])
    
    for i in range(Num):
# E-step
        Sum = 0
        P = []
        for j in range(count_of_clasters):
            p = multivariate_normal.pdf(X, mean=mu[j], cov=cov[j])
            if X_test is not None:
                p[np.where(y_train==j)[0]] = alpha*p[np.where(y_train==j)[0]] + (1-alpha)
                p[np.where(y_train!=j)[0]] = alpha*p[np.where(y_train!=j)[0]]
            P.append(p)
            Sum += w[j]*p
        for j in range(count_of_clasters):
            p = P[j]
            g[j] = w[j]*p/Sum
            
        N = np.sum(g, axis=1)    
# M-step
        for j in range(count_of_clasters):
            mu[j] = 1.0/N[j] * np.sum(np.multiply(g[j], X.T), axis =1)
            cov[j] = 1.0/N[j] *(g[j]*(X - mu[j]).T@(X - mu[j]))
        
        w = N/count_of_data
        
    return np.array(w), np.array(mu), np.array(cov)

def SEM_algorithm(X_train, X_test = None, y_train = None, count_of_clasters = 2, w = None, mu = None, cov = None, alpha = 0.9, Num = 50):
    if X_test is not None:
        X = np.vstack([X_train, X_test])
    if X_test is None:
        X = X_train
    count_of_data = X.shape[0]
    count_of_features = X.shape[1]

    if mu is None:
        mu = np.reshape(np.random.rand(count_of_clasters*count_of_features)- 0.5, [count_of_clasters, count_of_features])
    if w is None:
        w = np.ones(shape=[count_of_clasters])/count_of_clasters
    if cov is None:
        cov = []
        for i in range(count_of_clasters):
            cov.append(np.eye(X.shape[1]))
        cov = np.array(cov)
    
    g = np.ones(shape=[count_of_clasters, count_of_data], dtype=np.float64)/count_of_clasters

    y = np.zeros(shape=[count_of_clasters, count_of_data], dtype=np.float64)

    nu = np.ones(shape=[count_of_clasters])
    
    for i in range(Num):
# S-step
        for t in range(count_of_data):
            y.T[t] = np.random.multinomial(1, g.T[t])
        for t in range(count_of_clasters):
            nu[t] = np.sum(y[t] == 1)
# E-step
        Sum = 0
        P = []
        for j in range(count_of_clasters):
            p = multivariate_normal.pdf(X, mean=mu[j], cov=cov[j])
            if X_test is not None:
                p[np.where(y_train==j)[0]] = alpha*p[np.where(y_train==j)[0]] + (1-alpha)
                p[np.where(y_train!=j)[0]] = alpha*p[np.where(y_train!=j)[0]]
            P.append(p)
            Sum += w[j]*p
        for j in range(count_of_clasters):
            p = P[j]
            g[j] = w[j]*p/Sum
                
# M-step
        w = np.array(nu/count_of_data)
        for j in range(count_of_clasters):
            mu[j] = 1.0/nu[j] * np.sum(X[np.where(y[j] == 1)[0]].T, axis =1)
            cov[j] = 1.0/nu[j] *((X[np.where(y[j] == 1)[0]] - mu[j]).T@(X[np.where(y[j] == 1)[0]] - mu[j]))
        
        # w = N/count_of_data
        
    return np.array(w), np.array(mu), np.array(cov)

def MSEM_algorithm(X_train, X_test = None, y_train = None, count_of_clasters = 2, w = None, mu = None, cov = None, alpha = 0.9, Num = 50):
    if X_test is not None:
        X = np.vstack([X_train, X_test])
    if X_test is None:
        X = X_train
    count_of_data = X.shape[0]
    count_of_features = X.shape[1]

    if mu is None:
        mu = np.reshape(np.random.rand(count_of_clasters*count_of_features)- 0.5, [count_of_clasters, count_of_features])
    if w is None:
        w = np.ones(shape=[count_of_clasters])/count_of_clasters
    if cov is None:
        cov = []
        for i in range(count_of_clasters):
            cov.append(np.eye(X.shape[1]))
        cov = np.array(cov)
    
    g = np.ones(shape=[count_of_clasters, count_of_data], dtype=np.float64)/count_of_clasters

    y = np.zeros(shape=[count_of_clasters, count_of_data], dtype=np.float64)

    nu = np.ones(shape=[count_of_clasters])
    
    for i in range(Num):
# S-step
        for t in range(count_of_data):
            y.T[t] = np.random.multinomial(1, g.T[t])
        for t in range(count_of_clasters):
            nu[t] = np.sum(y[t] == 1)
# E-step
        Sum = 0
        P = []
        for j in range(count_of_clasters):
            p = multivariate_normal.pdf(X, mean=mu[j], cov=cov[j])
            if X_test is not None:
                p[np.where(y_train==j)[0]] = alpha*p[np.where(y_train==j)[0]] + (1-alpha)
                p[np.where(y_train!=j)[0]] = alpha*p[np.where(y_train!=j)[0]]
            P.append(p)
            Sum += w[j]*p
        for j in range(count_of_clasters):
            p = P[j]
            g[j] = w[j]*p/Sum
                
# M-step
        w = np.array(nu/count_of_data)
        for j in range(count_of_clasters):
            mu[j] = 1.0/nu[j] * np.sum(X[np.where(y[j] == 1)[0]].T, axis =1)
            cov[j] = 1.0/nu[j] *((X[np.where(y[j] == 1)[0]] - mu[j]).T@(X[np.where(y[j] == 1)[0]] - mu[j]))
        
        # w = N/count_of_data
        
    return np.array(w), np.array(mu), np.array(cov)



