from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


MODELS = [
    (RandomForestClassifier(), "random_forest_classifier"),
    (KNeighborsClassifier(3), "nearest_neighbors"),
    (SVC(kernel="linear", C=0.025), "linear_svm"),
    (SVC(gamma=2, C=1), "rbf_svm"),
    (GaussianProcessClassifier(1.0 * RBF(1.0)), "gaussian_process"),
    (DecisionTreeClassifier(max_depth=10), "decision_tree"),
    (RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1), "random_forest"),
    (MLPClassifier(hidden_layer_sizes=(100, 10), alpha=0.0001, max_iter=100), "neural_net"),
    (AdaBoostClassifier(), "ada_boost"),
    (GaussianNB(), "naive_bayes"),
    (QuadraticDiscriminantAnalysis(), "qda"),
]
DEFAULT_MODEL = 0
