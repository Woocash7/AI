from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    Implementacja jądra RBF w analizie PCA.
    Parametry
    ------------
    X : {typ ndarray biblioteki NumPy}, wymiary = [n_próbek, n_cech]
    gamma : liczby zmiennoprzecinkowe
        Parametr strojenia jądra RBF
    n_components : liczby całkowite
        Liczba zwracanych głównych składowych
    
    Zwraca
    ------------
    X_pc : {typ ndarray biblioteki NumPy}, wymiary = [n_próbek, k_cech]
        Rzutowany zestaw danych
    """

    # oblicza kwadraty odległości euklidesowych par
    # w zestawie danych o rozmiarze MxN
    sq_dists = pdist(X, 'sqeuclidean')

    # przekształca wyliczone odległości na macierz kwadratową
    mat_sq_dists = squareform(sq_dists)

    # oblicza symetryczną macierz jądra
    K = exp(-gamma * mat_sq_dists)

    #centruje macierz jądra 
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # wydobywa pary własne z centrowanej macierzy jądra
    # funkcja  numpy.eigh zwraca je w rosnącej kolejności
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    ## wybiera  k największych wektorów własnych (rzutowanych próbek)
    alphas = np.column_stack((eigvecs[:, i]
                              for i in range(n_components)))

    # wybiera odpowiednie wartości własne
    lambdas = [eigvals[i] for i in range(n_components)]

    return alphas, lambdas