import numpy as np
from sklearn import datasets

def _normalize(v):
    if np.sum(v) == 0:
        w=np.copy(v)
        w[0]=1
        return w
    else:
        return v/(np.sum(v))

def _sample_parameters(num_clusters, num_dim, **sampler_x_args):

    parameters_gen={}
    parameters_gen['cluster_prop'] = _normalize(np.random.poisson(10,num_clusters))
    for cluster in np.arange(num_clusters):
        parameters_gen['mean'+str(cluster)] = np.random.multivariate_normal(mean=np.zeros(num_dim), cov = sampler_x_args.get('spread_means',10)*np.identity(num_dim)) 
        parameters_gen['cov'+str(cluster)] = datasets.make_spd_matrix(n_dim = num_dim) + sampler_x_args.get('extra_cor', 0)*np.ones(shape=(num_dim,num_dim))
    parameters_gen.update(sampler_x_args)
    
    return parameters_gen

class GMM_X(object): 
    
    def __init__(self, num_samples, num_dim, num_clusters, **sampler_x_args):
        self.num_samples = num_samples   
        self.num_dim = num_dim
        self.num_clusters = num_clusters               

        self.parameters = _sample_parameters(num_clusters = self.num_clusters, num_dim = self.num_dim, **sampler_x_args)
                                                            
        self._sample()        
        
    def _sample(self):

        self.X = np.empty((self.num_samples, self.num_dim))
        self.assignments = np.random.choice(self.num_clusters,size=self.num_samples,p=self.parameters['cluster_prop'])
        for cluster in np.arange(self.num_clusters):
            self.X[np.where(self.assignments == cluster)[0],:] = np.random.multivariate_normal(mean = self.parameters['mean'+str(cluster)], cov = self.parameters['cov'+str(cluster)], size = np.sum(self.assignments == cluster) )


for i in range(25):
    seed=i+1
    np.random.seed(seed)
    
    n=10000
    p0=100
    sample_X=GMM_X(num_samples=n,num_dim=p0,num_clusters=10)
    X=sample_X.X
    
    sig=np.random.choice(p0,30,replace=False)
    temp=np.dot(X[:,sig],np.ones(len(sig)))
    coef1=np.random.randn(4)
    coef2=np.random.randn(4)
    noise1=0.1*np.random.randn(n)
    noise2=0.1*np.random.randn(n)
    y1=coef1[0]*pow(temp,3)+coef1[1]*pow(temp,2)+coef1[2]*temp+coef1[3]+noise1
    y2=coef2[0]*pow(temp,3)+coef2[1]*pow(temp,2)+coef2[2]*temp+coef2[3]+noise2
    y=np.column_stack((y1,y2))
    Y=np.zeros((n,2))
    Y[np.argmax(y,1)==0,0]=1
    Y[np.argmax(y,1)==1,1]=1
    
    np.savetxt('data/X1_'+str(seed)+'.txt',X)
    np.savetxt('data/Y1_'+str(seed)+'.txt',Y)
    np.savetxt('data/sig1_'+str(seed)+'.txt',sig)