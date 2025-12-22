import numpy as np
from covariance_types import CovarianceType

class GMM:
    def __init__(self,k:int,covariance_type:CovarianceType,conv_limit:float, max_iter:int,seed:int=42):
        self.k = k
        self.covariance_type = covariance_type
        self.conv_limit = conv_limit
        self.max_iter = max_iter
        
        #  initialized in fit()
        self.means = None      # shape (n_components, D)
        self.covariances = None
        self.priors = None
        self.reg_term = None
        self.log_likelihood = []
        self.r = None
        self.rng = np.random.RandomState(42)
        np.random.seed(seed)
        
    def fit(self,X:np.ndarray,reg_term:float= 1e-6):
        # initialize parameters: means, priors and covariances
        self._init_parameters(X=X,reg_term=reg_term)
        prev_likelihood = None
        # iterate until convergence or until maximum number of iterations is reached
        for _ in range(self.max_iter):
            # E-step
            self.r = self._E_step(X)
            
            # M-step 
            self._M_step(X,self.r)
            
            log_likelihood = self._compute_log_likelihood(X=X)
            self.log_likelihood.append(log_likelihood)
            
            # check for convergence
            if prev_likelihood is not None and abs(prev_likelihood - log_likelihood) < self.conv_limit:
                break
            
            # update likelihood
            prev_likelihood = log_likelihood
            
    
    def _E_step(self,X:np.ndarray):
        N = X.shape[0]
        # unnormalized responsibility vector
        r = np.zeros((N,self.k))
        threshold = 1e-10
        
        for j in range(self.k):
            r[:, j] = self.priors[j] * self._multivar_normal_pdf(X, self.means[j], self.covariances[j])
        # normalized responsibility vector
        r /= (r.sum(axis=1,keepdims=True) + threshold)
        return r
    
    def _M_step(self,X:np.ndarray,r:np.ndarray):
        N,d = X.shape
        N_k = r.sum(axis=0)
        threshold = 1e-10
        N_k = np.maximum(N_k,threshold)
        
        self.priors = N_k / N
        self.means = r.T @ X / N_k[:,None]
        
        if self.covariance_type == CovarianceType.FULL: 
            self._full_covariance(X=X,r=r,N_k=N_k,d=d)
        elif self.covariance_type == CovarianceType.TIED:
            self._tied_covariance(X=X,r=r,N_k=N_k,d=d)
        elif self.covariance_type == CovarianceType.DIAG: 
            self._diag_covariance(X=X,r=r,N_k=N_k,d=d)    
        elif self.covariance_type == CovarianceType.SPHERICAL:
            self._spherical_covariance(X=X,r=r,N_k=N_k,d=d)
    
    def _multivar_normal_pdf(self,x,mean,cov):
        d = mean.shape[0]
        diff = x - mean # x - μ
        
        L = np.linalg.cholesky(cov)  # Σ = L L^T
        y = np.linalg.solve(L, diff.T)  # (x - μ)^T
        dist = np.sum(y ** 2, axis=0)  # (x - μ)^T Σ^-1 (x - μ)
        log_det = 2 * np.sum(np.log(np.diag(L)))  # log|Σ| = 2 * sum(log(diag(L)))
        log_pdf = -0.5 * (d * np.log(2 * np.pi) + log_det + dist)  # log N(x|μ,Σ)
        return np.exp(log_pdf)
    
    def _compute_log_likelihood(self,X:np.ndarray):
        N = X.shape[0]
        likelihood = 0
        threshold = 1e-10 #threshold to prevent log(0)
        for j in range(self.k):
            likelihood += self.priors[j] * self._multivar_normal_pdf(
                X, self.means[j], self.covariances[j]
            )
        return np.sum(np.log(likelihood + threshold))
    def _init_parameters(self, X:np.ndarray,reg_term:float):
        # set regularization term to prevent 0 covariance
        self.reg_term = reg_term
        N_samples, d = X.shape
        # initialize priors to equivalent values
        self.priors = np.full(self.k, 1 / self.k)
        # get random points from the dataset 
        indices = self.rng.choice(N_samples, self.k, replace=False)
        # declare the means to random values
        self.means = X[indices]
        
        # cov parameters
        sample_cov = np.cov(X, rowvar=False) + reg_term * np.eye(d)
        feature_var = np.var(X, axis=0) + reg_term
        avg_var = np.mean(feature_var)
        
        # covariances
        if self.covariance_type == CovarianceType.FULL: # each gaussian has an independent covariance
            self.covariances = np.array([sample_cov.copy() for _ in range(self.k)])
        
        elif self.covariance_type == CovarianceType.TIED: # all gaussians have the same covariance
            self.covariances = np.array([sample_cov.copy() for _ in range(self.k)])
            
        elif self.covariance_type == CovarianceType.DIAG: # all gaussians have values for the main diagonal only
            diag_cov = np.diag(feature_var)
            self.covariances = np.array([diag_cov.copy() for _ in range(self.k)])
            
        elif self.covariance_type == CovarianceType.SPHERICAL: # all covariances are scaled identitiy matrices
            spherical_cov = avg_var * np.eye(d)
            self.covariances = np.array([spherical_cov.copy() for _ in range(self.k)])

    
    def _full_covariance(self,X:np.ndarray,r:np.ndarray,N_k:np.ndarray,d:int):
        self.covariances = np.zeros((self.k, d, d))
        for j in range(self.k):
            diff = X - self.means[j]
            self.covariances[j] = (r[:, j][:, None] * diff).T @ diff / N_k[j]
            self.covariances[j] += self.reg_term * np.eye(d)
    
    def _tied_covariance(self,X:np.ndarray,r:np.ndarray,N_k:np.ndarray,d:int):
        temp_cov = np.zeros((d, d))
        for j in range(self.k):
            diff = X - self.means[j]
            temp_cov += (r[:, j][:, None] * diff).T @ diff

        temp_cov /= X.shape[0]
        temp_cov += self.reg_term * np.eye(d)

        self.covariances = np.array([temp_cov.copy() for _ in range(self.k)])
    
    def _diag_covariance(self,X:np.ndarray,r:np.ndarray,N_k:np.ndarray,d:int):
        self.covariances = np.zeros((self.k, d, d))
        for j in range(self.k):
            diff = X - self.means[j]
            var = (r[:, j][:, None] * (diff ** 2)).sum(axis=0) / N_k[j]
            self.covariances[j] = np.diag(var + self.reg_term)
        
    def _spherical_covariance(self,X:np.ndarray,r:np.ndarray,N_k:np.ndarray,d:int):
        self.covariances = np.zeros((self.k, d, d))
        for j in range(self.k):
            diff = X - self.means[j]
            var = (r[:, j] * (diff ** 2).sum(axis=1)).sum() / (N_k[j] * d)
            self.covariances[j] = (var + self.reg_term) * np.eye(d)
            
    def _count_params(self):
        d = self.means.shape[1]
        n_priors = self.k -1
        n_means = self.k * d
        
        # covariance parameters depend on covariance type
        if self.covariance_type == CovarianceType.FULL:
            # each has d(d+1)/2 free parameters 
            n_cov = self.k * d * (d + 1) // 2
        
        elif self.covariance_type == CovarianceType.TIED:
            # one covariance matrix has d(d+1)/2 parameters
            n_cov = d * (d + 1) // 2
        
        elif self.covariance_type == CovarianceType.DIAG:
            # k diagonal matrices each has d parameters (only diagonal elements)
            n_cov = self.k * d
        
        elif self.covariance_type == CovarianceType.SPHERICAL:
            # k scalar variances (one per component)
            n_cov = self.k
        
        return n_priors + n_means + n_cov
    
    def aic(self,X:np.ndarray):
        # Akaike Information Criterion
        # lower AIC means a better model
        log_likelihood = self._compute_log_likelihood(X)
        n_params = self._count_params()
        
        return 2 * n_params - 2 * log_likelihood
    
    def bic(self,X:np.ndarray):
        # Bayesian Information Criterion
        # lower BIC means a better model
        n_samples = X.shape[0]
        log_likelihood = self._compute_log_likelihood(X)
        n_params = self._count_params()
        
        return n_params * np.log(n_samples) - 2 * log_likelihood
