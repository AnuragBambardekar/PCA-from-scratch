import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components # n_components: Number of components to find. 0 < n_components <= no of features
        self.mean = None
        self.components = None
    
    def fit(self,X):
        #mean centering
        self.mean = np.mean(X,axis=0)
        X = X - self.mean

        #covariance, function needs samples as columns
        cov = np.cov(X.T)

        #eigenvectors , eigenvalues
        eigenvectors , eigenvalues = np.linalg.eig(cov)

        #eigenvectors v = [:, i] column vector, transpose this for easier calculation
        eigenvectors = eigenvectors.T

        #sort the eigenvectors according to eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        #choose first k eigenvectors
        self.components = eigenvectors[:self.n_components]



    def transform(self,X):
        #project data, Transform the X features based on the found Eigen Values and vectors
        X = X-self.mean

        return np.dot(X,self.components.T)

#Testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets

    data = datasets.load_iris()
    X = data.data
    y = data.target
    
    #Project data onto 2 primary principal components
    pca = PCA(3)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:,0] #PC1
    x2 = X_projected[:,1] #PC2
    x3 = X_projected[:,2] #PC3

    #fig,ax = plt.subplot(nrows=1, ncols=3, figsize=(15,5))

    plt.scatter(
        x1,x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis",3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()