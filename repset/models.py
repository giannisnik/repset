import numpy as np
from lap import lapjv

class RepSet:
    def __init__(self, lr, n_hidden_sets, n_elements, d, n_classes):
        self.lr = lr
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements
        self.d = d
        self.n_classes = n_classes

        self.t = 0
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.m_t = [0]*n_hidden_sets
        self.v_t = [0]*n_hidden_sets
        self.m_t_c = 0
        self.v_t_c = 0

        self.Ws = np.random.randn(n_hidden_sets, n_elements, d)
        self.Wc = np.random.randn(n_hidden_sets+1, n_classes)


    def train(self, X, y):
        R = np.zeros((X.size, self.n_hidden_sets+1))
        R[:,-1] = 1

        Ds = list()

        for i in range(X.shape[0]):
            Ds.append(list())

            x = X[i]
    
            for j in range(self.n_hidden_sets):
                W = self.Ws[j]
                K = np.dot(W, x)
                K[K<0] = 0

                cost, x_lap, _ = lapjv(-K, extend_cost=True)

                D = np.zeros((self.n_elements, x.shape[1]))
                for k in range(self.n_elements):
                    if x_lap[k] != -1:
                        D[k, x_lap[k]] = 1

                Ds[i].append(D)

                cost_norm = cost/x.shape[1]
                R[i,j] = -cost_norm

        S = np.dot(R, self.Wc)
        y_pred = np.exp(S)/np.sum(np.exp(S), axis=1).reshape(-1, 1)

        E = y - y_pred

        ## Backprop
        upd_Ws = np.zeros((self.n_hidden_sets, self.n_elements, self.d))
        upd_Wc = np.zeros((self.n_hidden_sets+1, self.n_classes))

        for i in range(X.shape[0]):
            x = X[i]
    
            for j in range(self.n_hidden_sets):
                upd_Ws[j] = upd_Ws[j] + np.dot(Ds[i][j], x.T)*np.dot(E[i,:], self.Wc[j,:])

            upd_Wc += np.outer(R[i,:].T, E[i,:])
        
        
        self.t += 1
        for j in range(self.n_hidden_sets):
            g_t = upd_Ws[j]*1./x.shape[1]
            self.m_t[j] = self.beta_1*self.m_t[j] + (1-self.beta_1)*g_t
            self.v_t[j] = self.beta_2*self.v_t[j] + (1-self.beta_2)*(np.square(g_t))
            m_cap = self.m_t[j]/(1-(self.beta_1**self.t))
            v_cap = self.v_t[j]/(1-(self.beta_2**self.t))
            self.Ws[j] = self.Ws[j] + (self.lr*m_cap)/(np.sqrt(v_cap)+self.epsilon)
            
        g_t = upd_Wc*1./x.shape[1]
        self.m_t_c = self.beta_1*self.m_t_c + (1-self.beta_1)*g_t
        self.v_t_c = self.beta_2*self.v_t_c + (1-self.beta_2)*np.square(g_t)
        m_cap= self.m_t_c/(1-(self.beta_1**self.t))
        v_cap = self.v_t_c/(1-(self.beta_2**self.t))
        self.Wc = self.Wc + (self.lr*m_cap)/(np.sqrt(v_cap)+self.epsilon)

        return y_pred


    def test(self, X):
        R = np.zeros((X.size, self.n_hidden_sets+1))
        R[:,-1] = 1

        for i in range(X.shape[0]):
            x = X[i]
    
            for j in range(self.n_hidden_sets):
                W = self.Ws[j]
                K = np.dot(W, x)
                K[K<0] = 0

                cost, x_lap, _ = lapjv(-K, extend_cost=True)
                cost_norm = cost/x.shape[1]
                R[i,j] = -cost_norm

        R = np.dot(R, self.Wc)
        y_pred = np.exp(R)/np.sum(np.exp(R), axis=1).reshape(-1, 1)
        
        return y_pred