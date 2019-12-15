import numpy as np

class PCA:
	def __init__(self, n_components):
		'''
		n_components: int, number of dimensions for new vector
		'''
		self.n_components = n_components

	def fit(self, X):
		'''
		cov_mat_: np.array, covariance matrix 
		eigen_vals_: np.array, eigen values of cov_mat ()
		eigen_vecs_: np.array , eigen vectors of cov_mat
		exp_var_: list, explained variance
		eigen_pairs_: list of tuples containing eigen_vals their eigen_vecs
		w_ = np.array: projection matrix 
		'''
		self.cov_mat_ = np.cov(X.T)
		self.eigen_vals_, self.eigen_vecs_ = np.linalg.eig(self.cov_mat_)

		self.tot_ = sum(self.eigen_vals_)
		self.exp_var_ = [(i / self.tot_) for i in sorted(self.eigen_vals_, reverse=True)]
		
		self.eigen_pairs_ = [(np.abs(self.eigen_vals_[i]), self.eigen_vecs_[:,i])
			for i in range(len(self.eigen_vals_))]
		self.eigen_pairs_.sort(key=lambda k: k[0], reverse=True)

		self.w_ = [self.eigen_pairs_[i][1][:, np.newaxis] for i in range(self.n_components)]
		self.w_ = np.hstack(self.w_)

	def transform(self, X):
		'''
		X_pca_: np.array, PCA array of n_components features
		'''
		self.X_pca_ = X.dot(self.w_)
		return self.X_pca_
		
	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)
		
		
if __name__ == '__main__':
	import pandas as pd, matplotlib.pyplot as plt
	from sklearn.preprocessing import StandardScaler
	from sklearn.model_selection import train_test_split
	from sklearn.neighbors import KNeighborsClassifier

	df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
		header=None)
	
	X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0, stratify=y)

	sc = StandardScaler()
	X_train_std = sc.fit_transform(X_train)
	X_test_std = sc.fit_transform(X_test)
	
	pca = PCA(n_components=5)
	X_pca = pca.fit_transform(X_train_std)
	print(X_pca)