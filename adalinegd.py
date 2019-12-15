import numpy as np
import matplotlib.pyplot as plt

class AdalineGD(object):
	"""Klasyfikator — ADAptacyjny LIniowy NEuron.
	GD = gradient descent - gradient prosty

	Parametry
	------------
	eta : zmiennoprzecinkowy
	Współczynnik uczenia (w zakresie pomiędzy 0.0 i 1.0).
	n_iter : liczba całkowita
	Liczba przebiegów po zestawie uczącym.
	random_state : liczba całkowita
    Zarodek dla generatorwa liczb losowych w celu inicjalizacji losowych wag początkowych.

	Atrybuty
	-----------
	w_ : jednowymiarowa tablica
	Wagi po dopasowaniu.
	errors_ : lista
	Liczba niewłaściwych klasyfikacji w każdej epoce.
	"""

	def __init__(self, eta=0.01, n_iter=50, random_state=1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self, X, y):
		""" Trenowanie za pomocą danych uczących.
		Parametry
		----------
		X : {tablicopodobny}, wymiary = [n_próbek, n_cech]
		Wektory uczenia,
		gdzie n_próbek oznacza liczbę próbek, a
		n_cech — liczbę cech.
		y : tablicopodobny, wymiary = [n_próbek]
		Wartości docelowe.
		Zwraca
		-------
		self : obiekt
		"""

		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			net_input = self.net_input(X)
			# Please note that the "activation" method has no effect
			# in the code since it is simply an identity function. We
			# could write `output = self.net_input(X)` directly instead.
			# The purpose of the activation is more conceptual, i.e.,  
			# in the case of logistic regression (as we will see later), 
			# we could change it to
			# a sigmoid function to implement a logistic regression classifier.
			output = self.activation(X)
			errors = (y - output)
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] = self.eta * errors.sum()
			cost = (errors**2).sum() / 2.0
			self.cost_.append(cost)
			self.plotting(X, y, i)
		plt.show()
		return self

	def net_input(self, X):
		"""Oblicza całkowite pobudzenie"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		"""Oblicza liniową funkcję aktywacji"""
		return self.net_input(X)

	def predict(self, X):
		"""Zwraca etykietę klas po wykonaniu skoku jednostkowego"""
		return np.where(self.activation(X) >= 0.0, 1, -1)

	def plotting(self, X, y, i):
		p = x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		p = np.arange(x1_min, x1_max, 0.01)
		w = ((self.w_[0] / self.w_[2]) / (-(self.w_[0] / self.w_[1]))) * p + (-self.w_[0] / self.w_[2])
		plt.plot(p, w, label=i)
		plt.legend(loc='upper left')

if __name__ == '__main__':
	from matplotlib.colors import ListedColormap
	from plot_decision_regions import plot_decision_regions 
	import pandas as pd

	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
	y = df.iloc[0:100, 4].values
	y = np.where(y == 'Iris-setosa', -1, 1)
	X = df.iloc[0:100, [0, 2]].values

	X_std = np.copy(X)
	X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
	X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

	ada = AdalineGD(n_iter=15, eta=0.01)
	ada.fit(X_std, y)

	plot_decision_regions(X_std, y, classifier=ada)
	plt.xlabel('Długość płatka [standaryzowana]')
	plt.ylabel('Szerokość płatka [standaryzowana]')
	plt.legend(loc='upper left')
	plt.tight_layout()
	# plt.savefig('images/02_14_1.png', dpi=300)
	plt.show()

	plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Sum-squared-error')

	plt.tight_layout()
	# plt.savefig('images/02_14_2.png', dpi=300)
	plt.show()