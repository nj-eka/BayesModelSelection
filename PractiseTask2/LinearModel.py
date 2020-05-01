import numpy as _np
import scipy.special as _sp
from tqdm import tqdm as _tqdm

class BayesMultiLinearModel:
    def __init__(self, n_models=20, mu = 0.1, epsilon = 10**(-30)):
        self.__n_models = n_models
        self.__mu = mu
        self.__epsilon = epsilon
        self.__A = None
        self.__b = None
        self.__beta = None
        
        
        self.__Z = None
        self.__gamma = None
        self.__B = None
        self.__m = None
        return
    
    def fit(self, X = None, p = None, epoch = 100):
        if X is None:
            return 1
        if p is None:
            return 1
        
        if self.__A is None:
            self.__A = _np.random.rand(self.__n_models, X.shape[1])
        if self.__b is None:
            self.__b = _np.random.rand(self.__n_models)*100000
        if self.__beta is None:
            self.__beta = 1
            
        return self.__fit(X, p, epoch)
    
    def __fit(self, X, p, epoch):
        self.__init_q(X, p)
        
        List_of_F = []
        
        for i in _tqdm(range(epoch)):
            self.__E_step(X, p)
            self.__delete_bad_model(X, p)
            self.__M_step(X, p)
            F.append(Func(X, p))
        return F
    
    def Func(self, X, p):
        res = 0
        
        for k in range(self.__n_models):
            temp_1 = _sp.digamma(self.__mu + self.__gamma[k]) - _sp.digamma(self.__n_models*self.__mu + X.shape[0])
            temp_2 = _np.sum(1.0/self.__A[k])
            temp_3 = _np.sum(_np.diag(self.__B[k]) + _np.diag(_np.reshape(self.__m[k], [-1,1])@_np.reshape(self.__m[k], [1,-1]))*self._A[k])
            
            temp_4 = 0
            temp_4_1 = temp_1 
            temp_4_2 = _np.log(self.__beta)
            temp_4_3 = _np.log(2*_np.pi)
            temp_4_6_1 = self.__B[k] + _np.reshape(self.__m[k], [-1,1])@_np.reshape(self.__m[k], [1,-1])
            for i in range(X.shape[0]):
                temp_4_4 = (p[i] - self.__b[k])**2
                temp_4_5 = (p[i] - self.__b[k])*_np.sum(self.__m[k]*X[i])
                temp_4_6_2 = _np.reshape(X[i], [-1,1])@_np.reshape(X[i], [1,-1])
                temp_4_6 = _np.sum(_np.diag(temp_4_6_1@temp_4_6_2))
                
                temp_4_0 = temp_4_1
                temp_4_0 += temp_4_2
                temp_4_0 -= temp_4_3
                temp_4_0 -= self.__beta*(temp_4_4 - 2*temp_4_5 + temp_4_6)
                temp_4_0 *= self.__Z[i, k]
                temp_4_0 = _np.reshape(temp_4_0, [-1])
                temp_4 += temp_4_0
            res += (self.__mu + 2*self.__gamma[k] - 1)*temp_1 +0.5*temp_2 - 0.5*temp_3 + temp_4
        return res
    
    def __delete_model(self, X, p, k):
        self.__A = _np.vstack([self.__A[:k,:], self.__A[k+1:,:]])
        self.__b = _np.hstack([self.__b[:k], self.__b[k+1:]])
        self.__n_models -= 1
        self.__Z = _np.hstack([self.__Z[:,:k], self.__Z[:,k+1:]])
        self.__Z = self.__Z/self.__Z.sum(1).reshape([-1,1])
        
        self.__gamma = _np.sum(self.__Z, axis=0)
        self.__B = _np.array([self.__B_matrix(X, p, k = i) for i in range(self.__n_models)])
        self.__m = _np.array([self.__m_vector(X, p, k = i) for i in range(self.__n_models)])
        return
    
    def __delete_bad_model(self, X, p):
        flag = 1
        while flag == 1:
            Estimation = self.__Z.sum(0)
            k = _np.argmin(Estimation)
            if Estimation[k] < self.__epsilon:
                self.__delete_model(X, p, k)
            else:
                flag = 0
            
        return
    
    def n_models(self):
        return self.__n_models
    
    def __M_step(self, X, p):
# пересчет A
        self.__A = _np.array([self.__A_matrix(k = i) for i in range(self.__n_models)])
# пересчет b
        self.__b = _np.reshape(_np.array([self.__b_scalar(X, p, k = i) for i in range(self.__n_models)]), [-1])
# пересчет beta
        self.__beta = self.__beta_scalar(X, p)
        return 0
    
    def __E_step(self, X, p):
# пересчет Z
        for i in range(X.shape[0]):
            for k in range(self.__n_models):
                temp_1 = _sp.digamma(self.__mu + self.__gamma[k]) - _sp.digamma(self.__n_models*self.__mu + X.shape[0])
                temp_2 = (p[i] - self.__b[k])**2
                temp_3 = _np.reshape(((p[i] - self.__b[k])*_np.reshape(X[i], [1, -1])@_np.reshape(self.__m[k], [-1, 1])), [-1])
                temp_4 = _np.reshape(_np.reshape(X[i], [1,-1])@(self.__B[k] + _np.reshape(self.__m[k], [-1,1])@_np.reshape(self.__m[k], [1,-1]))@_np.reshape(X[i], [-1,1]), [-1])
                self.__Z[i,k] = temp_1 - 0.5*self.__beta*(temp_2 - 2* temp_3 + temp_4) + 0.5*(_np.log(self.__beta) - _np.log(2*_np..pi))
            ex = _np.exp(self.__Z[i] - _np.max(self.__Z[i]))
            self.__Z[i] = ex/ex.sum()
        
# пересчет gamma
        self.__gamma = _np.sum(self.__Z, axis=0)
        
# пересчет w
        self.__B = _np.array([self.__B_matrix(X, p, k = i) for i in range(self.__n_models)])
        self.__m = _np.array([self.__m_vector(X, p, k = i) for i in range(self.__n_models)])
        return 0

    
    def __init_q(self, X, p):
        if self.__Z is None:
            m = X.shape[0]
            self.__Z = _np.random.rand(m, self.__n_models)
            self.__Z = self.__Z/self.__Z.sum(1).reshape([-1,1])
        if self.__gamma is None:
            self.__gamma = _np.sum(self.__Z, axis=0)
        if self.__B is None:
            self.__B = _np.array([self.__B_matrix(X, p, k = i) for i in range(self.__n_models)])
        if self.__m is None:
            self.__m = _np.array([self.__m_vector(X, p, k = i) for i in range(self.__n_models)])
        return 0
    
    def __B_matrix(self, X, p, k = 0):
        temp_1 = _np.diag(1/self.__A[k])
        temp_2 = 0
        for i in range(X.shape[0]):
            temp_2 += _np.reshape(X[i], [-1, 1])@_np.reshape(X[i], [1, -1])*self.__Z[i,k]
        return _np.linalg.inv(temp_1 + self.__beta*temp_2)
    
    def __m_vector(self, X, p, k = 0):
        temp_1 = 0
        for i in range(X.shape[0]):
            temp_1 += X[i]*(p[i] - self.__b[k])*self.__Z[i,k]
        return np.reshape(self.__beta*self.__B[k]@_np.reshape(temp_1, [-1, 1]), [-1])
    
    def __A_matrix(self, k = 0):
        return _np.diag(self.__B[k]) + _np.diag(_np.reshape(self.__m[k], [-1,1])@_np.reshape(self.__m[k], [1,-1]))
    
    def __b_scalar(self, X, p, k = 0):
        temp_1 = 0
        temp_2 = 0
        for i in range(X.shape[0]):
            temp_1 += p[i]*self.__Z[i,k]
            temp_2 += _np.reshape(_np.reshape(X[i], [1, -1])@_np.reshape(self.__m[k], [-1, 1]), [-1])*self.__Z[i,k]
        return (temp_1 - temp_2)/(self.__Z[:,k]).sum()
    
    def __beta_scalar(self, X, p):
        temp_1 = 0
        for i in range(X.shape[0]):
            for k in range(self.__n_models):
                temp_2 = 0
                temp_2 += (p[i] - self.__b[k])**2
                temp_2 += -2*_np.reshape(((p[i] - self.__b[k])*_np.reshape(X[i], [1, -1])@_np.reshape(self.__m[k], [-1, 1])), [-1])
                temp_2 += _np.reshape(_np.reshape(X[i], [1,-1])@(self.__B[k] + _np.reshape(self.__m[k], [-1,1])@_np.reshape(self.__m[k], [1,-1]))@_np.reshape(X[i], [-1,1]), [-1])
                temp_1 += temp_2*self.__Z[i,k]
        return _np.reshape(temp_1/X.shape[0], [-1])
    
    def parameters(self):
        return self.__A, self.__b , self.__beta
    
    def q_distribution(self):
        return self.__m, self.__B, self.__Z, self.__gamma
    
    def predict(self, X = None):
        if X is None:
            return None
        Z = _np.zeros(shape = [X.shape[0], self.__n_models])
        for i in range(X.shape[0]):
            for k in range(self.__n_models):
                temp_1 = _sp.digamma(self.__mu + self.__gamma[k]) - _sp.digamma(self.__n_models*self.__mu + X.shape[0])
                Z[i,k] = temp_1 - 0.5*self.__beta*self.__b[k]
            ex = _np.exp(Z[i] - _np.max(Z[i]))
            Z[i] = ex/ex.sum()

        k = _np.zeros(X.shape[0], dtype = _np.int64)
        for i in range(X.shape[0]):
            k[i] = _np.random.choice(self.__n_models, p = Z[i])
        
        y = _np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = _np.sum(X[i]*self.__m[k[i]]) + self.__b[k[i]]

        return y

    
    
class BPCA():
    def __init__(self):
        self.__n_components = None
        self.__mu = None
        self.__alpha = None
        self.__sigma = None
        self.__eye_d = None
        
        self.__M = None
        self.__A_WS = None
        self.__A = None
        self.__U = None
        self.__B_WS = None
        return
    
    def fit(self, X, epoch = 100):
        if self.__n_components is None:
            self.__n_components = X.shape[1]
        if self.__mu is None:
            self.__mu = _np.mean(X, axis = 0)
        if self.__sigma is None:
            self.__sigma = 1.0
        if self.__alpha is None:
            self.__alpha = _np.ones(shape = [X.shape[1]])
        if self.__eye_d is None:
            self.__eye_d = _np.eye(self.__n_components)
        return self.__fit(X, epoch = epoch)
    
    def __init_q(self, X):
        self.__X_mean = _np.mean(X, axis = 0)
        if self.__M is None:
            self.__M = _np.random.rand(X.shape[0], self.__n_components)
        if self.__A_WS is None:
            self.__A_WS = _np.eye(self.__n_components)
        if self.__A is None:
            self.__A = self.__sigma*self.__A_WS
        
        if self.__U is None:
            self.__U = _np.random.rand(X.shape[1], self.__n_components)
        if self.__B_WS is None:
            self.__B_WS = _np.zeros(shape = [X.shape[1], self.__n_components, self.__n_components])
            for k in range(X.shape[1]):
                self.__B_WS[k] = _np.eye(self.__n_components)
        self.__B = self.__sigma*self.__B_WS
        
# __C is expectation of W.TW
        self.__C = _np.diag(_np.sum(_np.diagonal(self.__B, axis1=1, axis2=2), axis = 1)) + _np.diag(_np.diagonal(self.__U.T@self.__U))

# __D is expectation of z_iz_i.T for all i 
        self.__D = _np.zeros(shape = [X.shape[0], self.__n_components, self.__n_components])
        for k in range(X.shape[0]):
            self.__D[k] = self.__A + _np.reshape(self.__M[k], [-1,1])@_np.reshape(self.__M[k], [1,-1])    
        return
    
    def __E_step(self, X):
        
# пересчет Z
        self.__A_WS = self.__find_A()
        self.__A = self.__sigma*self.__A_WS
        for k in range(X.shape[0]):
            self.__M[k] = self.__find_m(X[k])
        for k in range(X.shape[0]):
            self.__D[k] = self.__A + _np.reshape(self.__M[k], [-1,1])@_np.reshape(self.__M[k], [1,-1])
        
# пересчет W        
        temp_B = _np.sum(self.__D, axis = 0)
        for k in range(X.shape[1]):
            self.__B_WS[k] = _np.linalg.inv(temp_B + self.__sigma*_np.diag(self.__alpha))
        self.__B = self.__sigma*self.__B_WS
        
        temp_U_1 = 0
        for k in range(X.shape[0]):
            temp_U_1 += _np.reshape(X[k] - self.__mu, [-1,1])@_np.reshape(self.__M[k], [1,-1])
        temp_U_2 = _np.linalg.inv(_np.sum(self.__D, axis = 0) + self.__sigma*_np.diag(self.__alpha))
        self.__U = temp_U_1@temp_U_2
        
        self.__C = _np.diag(_np.sum(_np.diagonal(self.__B, axis1=1, axis2=2), axis = 1)) + _np.diag(_np.diagonal(self.__U.T@self.__U))

        return
    
    def __M_step(self, X):
# пересчитываем альфа
        for k in range(X.shape[1]):
            self.__alpha[k] = self.__n_components/(_np.sum(_np.diagonal(self.__B[k])) + _np.sum(self.__U[:,k]**2))

# пересчитываем mu
        self.__mu = self.__X_mean

# пересчитываем sigma
        temp_S_1 = _np.sum(_np.diagonal(_np.sum(self.__D, axis = 0)@self.__C))
        temp_S_2 = 0
        for k in range(X.shape[0]):
            temp_S_2 += _np.reshape(_np.reshape(self.__mu-X[k],[1,-1])@_np.reshape(self.__mu-X[k],[-1,1]), -1)
        temp_S_3 = 0
        for k in range(X.shape[0]):
            temp_S_3 += _np.reshape(self.__M[k], [1,-1])@self.__U.T@_np.reshape(self.__mu - X[k], [-1,1])
        temp_S_3 = 2*temp_S_3
        self.__sigma = float((temp_S_1+temp_S_2+temp_S_3)/(X.shape[0]*X.shape[1]))
        
        return
    
    def __fit(self, X, epoch = 100):
        
        self.__init_q(X)
        
        for i in _tqdm(range(epoch)):
            
            self.__E_step(X)
            self.__M_step(X)
        
        pass
    
    def __find_A(self):
        return _np.linalg.inv(self.__C + self.__sigma*self.__eye_d)
    
    def __find_m(self, x):
        return _np.reshape(self.__A_WS@self.__U.T@_np.reshape(x - self.__mu, [-1,1]), -1)
 

    def parameters(self):
        return self.__alpha, self.__mu , self.__sigma

    def __transform_proba(self, X):
        A_WS = self.__find_A()
        A = self.__sigma*A_WS
        Z = _np.zeros(shape = [X.shape[0], self.__n_components])
        for i in range(X.shape[0]):
            m = self.__find_m(X[i])
            z = _np.random.multivariate_normal(mean = m, cov = A)
            Z[i] = z
            
        return Z
    
    def __transform_predictable(self, X):
        Z = _np.zeros(shape = [X.shape[0], self.__n_components])
        for i in range(X.shape[0]):
            Z[i] = self.__find_m(X[i])
            
        return Z
        
    
    def transform(self, X, proba = False):
        if proba:
            return  self.__transform_proba(X)
        
        return self.__transform_predictable(X)
        