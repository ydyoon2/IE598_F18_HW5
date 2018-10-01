import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
df_wine.head()
print(df_wine.head())

#train_test_split for standard: 80% traning, 20% test, random_state=42
X, y = df_wine.iloc[:, 12:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#plot decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

#LR
lr1 = LogisticRegression()
lr1.fit(X_train_std, y_train)
plot_decision_regions(X_train_std, y_train, classifier=lr1)
plt.title('logistic classifier model: train')
plt.xlabel('LR 1')
plt.ylabel('LR 2')
plt.legend(loc='lower left')
plt.show()
acc_lr1 = round(lr1.score(X_train_std, y_train), 4)
print('accuracy: ',acc_lr1)

lr2 = LogisticRegression()
lr2.fit(X_test_std, y_test)
plot_decision_regions(X_test_std, y_test, classifier=lr2)
plt.title('logistic classifier model: test')
plt.xlabel('LR 1')
plt.ylabel('LR 2')
plt.legend(loc='lower left')
plt.show()
acc_lr2 = round(lr2.score(X_test_std, y_test), 4)
print('accuracy: ',acc_lr2)

#SVM
svm1 = SVC()
svm1.fit(X_train_std, y_train)
plot_decision_regions(X_train_std, y_train, classifier=svm1)
plt.title('Support Vector Machines: train')
plt.xlabel('SVM 1')
plt.ylabel('SVM 2')
plt.legend(loc='lower left')
plt.show()
acc_svm1 = round(svm1.score(X_train_std, y_train), 4)
print('accuracy: ',acc_svm1)

svm2 = SVC()
svm2.fit(X_test_std, y_test)
plot_decision_regions(X_test_std, y_test, classifier=svm2)
plt.title('Support Vector Machines: test')
plt.xlabel('SVM 1')
plt.ylabel('SVM 2')
plt.legend(loc='lower left')
plt.show()
acc_svm2 = round(svm2.score(X_test_std, y_test), 4)
print('accuracy: ',acc_svm2)

#train_test_split for PCA,SVM...: 80% traning, 20% test, random_state=42
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#eigenvalues
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)
sys.stdout.write(" \n")

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in
           sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

X_train_std[0].dot(w)
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
print('\narray', X_train_std[0].dot(w))

#PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr = LogisticRegression()
lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.title('logistic classifier model: PCA transformed train set')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
acc_pca1 = round(lr.score(X_train_pca, y_train), 4)
print('accuracy: ',acc_pca1)

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.title('logistic classifier model: PCA transformed test set')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()
acc_pca2 = round(lr.score(X_test_pca, y_test), 4)
print('accuracy: ',acc_pca2)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
svm = SVC()
svm.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=svm)
plt.title('SVM: PCA transformed train set')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
acc_pca3 = round(svm.score(X_train_pca, y_train), 4)
print('accuracy: ',acc_pca3)

plot_decision_regions(X_test_pca, y_test, classifier=svm)
plt.title('SVM: PCA transformed test set')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()
acc_pca4 = round(svm.score(X_test_pca, y_test), 4)
print('accuracy: ',acc_pca4)

#LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr = LogisticRegression()
lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.title('logistic classifier model: LDA transformed train set')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
acc_lda1 = round(lr.score(X_train_lda, y_train), 4)
print('accuracy: ',acc_lda1)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.title('logistic classifier model: LDA transformed test set')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
acc_lda2 = round(lr.score(X_test_lda, y_test), 4)
print('accuracy: ',acc_lda2)

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
svm = SVC()
svm.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=svm)
plt.title('SVM: LDA transformed train set')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
acc_lda3 = round(svm.score(X_train_lda, y_train), 4)
print('accuracy: ',acc_lda3)

plot_decision_regions(X_test_lda, y_test, classifier=svm)
plt.title('SVM: LDA transformed test set')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
acc_lda4 = round(svm.score(X_test_lda, y_test), 4)
print('accuracy: ',acc_lda4)

#kPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_train_kpca = kpca.fit_transform(X_train_std)
X_test_kpca = kpca.transform(X_test_std)
lr = LogisticRegression()
lr.fit(X_train_kpca, y_train)

plot_decision_regions(X_train_kpca, y_train, classifier=lr)
plt.title('logistic classifier model: kPCA transformed train set (gamma=0.1)')
plt.xlabel('kPCA 1')
plt.ylabel('kPCA 2')
plt.legend(loc='lower left')
plt.show()
acc_kpca1 = round(lr.score(X_train_kpca, y_train), 4)
print('accuracy: ',acc_kpca1)

plot_decision_regions(X_test_kpca, y_test, classifier=lr)
plt.title('logistic classifier model: kPCA transformed test set (gamma=0.1)')
plt.xlabel('kPCA 1')
plt.ylabel('kPCA 2')
plt.legend(loc='lower left')
plt.show()
acc_kpca2 = round(lr.score(X_test_kpca, y_test), 4)
print('accuracy: ',acc_kpca2)

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_train_kpca = kpca.fit_transform(X_train_std)
X_test_kpca = kpca.transform(X_test_std)
svm = SVC()
svm.fit(X_train_kpca, y_train)

plot_decision_regions(X_train_kpca, y_train, classifier=svm)
plt.title('SVM: kPCA transformed train set (gamma=0.1)')
plt.xlabel('kPCA 1')
plt.ylabel('kPCA 2')
plt.legend(loc='lower left')
plt.show()
acc_kpca3 = round(svm.score(X_train_kpca, y_train), 4)
print('accuracy: ',acc_kpca3)

plot_decision_regions(X_test_kpca, y_test, classifier=svm)
plt.title('SVM: kPCA transformed test set (gamma=0.1)')
plt.xlabel('kPCA 1')
plt.ylabel('kPCA 2')
plt.legend(loc='lower left')
plt.show()
acc_kpca4 = round(svm.score(X_test_kpca, y_test), 4)
print('accuracy: ',acc_kpca4)

print("My name is {James Yoon}")
print("My NetID is: {ydyoon2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")