# mnist classification
import torch
import torchvision
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
'''download mnist dataset '''
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                               ])),
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                               ])),
    shuffle=False)
X_train = train_loader.dataset.train_data.view(60000, 28*28).numpy()
X_train_label = train_loader.dataset.train_labels.numpy()
X_test = test_loader.dataset.test_data.view(10000, 28*28).numpy()
X_test_label = test_loader.dataset.test_labels.numpy()
print('Train:', X_train.shape, 'Label:', X_train_label.shape)
# Train: (60000, 784) Label: (60000,)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.fit_transform(X_test)
'''PCA Dimensionality Reduction'''
# 原始数维数为784，降维后的维数为N1,实际取N1={50,100,200,300,400}
M1 = np.array([50, 100, 200, 300, 400])
M2 = np.array([1, 3])
accuracy = np.zeros([5, 2])
time1 = np.zeros([5, 2])
accuracy_no = np.zeros([5, 2])
time_no = np.zeros([5, 2])
for i in range(len(M1)):
    for j in range(len(M2)):
        N1 = M1[i]
        N2 = M2[j]
        pca = PCA(n_components=N1)
        pca.fit(X_train)
        train_data_pca = pca.transform(X_train)
        test_data_pca = pca.transform(X_test)
        knn = KNeighborsClassifier(n_neighbors=N2, p=2, metric='minkowski')
        mms = MinMaxScaler()
        N2 = N1 / 2
        start_time = time.time()
        lda = LDA(n_components=N2)
        lda.fit(train_data_pca, X_train_label)
        train_data_lda = lda.transform(train_data_pca)
        test_data_lda = lda.transform(test_data_pca)

        train_data_lda_std = mms.fit_transform(train_data_lda)
        knn.fit(train_data_lda_std, X_train_label)
        test_pred = knn.predict(mms.fit_transform(test_data_lda))
        time1[i][j] = time.time() - start_time
        accuracy[i][j] = np.mean(np.equal(test_pred, X_test_label))

plt.figure()
plt.plot(M1, accuracy[:, 0], M1, accuracy[:, 1], M1, accuracy_no[:, 0], M1, accuracy_no[:, 1])
plt.legend(['k=1', 'k=3', 'k=1,without LDA', 'k=3,without LDA'])
plt.show()

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

pca = PCA(n_components=N1)
pca.fit(X_train)
train_data_pca = pca.transform(X_train)
test_data_pca = pca.transform(X_test)

lda = LDA(n_components=N1)
lda.fit(train_data_pca, X_train_label)
train_data_lda = lda.transform(train_data_pca)
test_data_lda = lda.transform(test_data_pca)

knn = KNeighborsClassifier(n_neighbors=N2, p=2, metric='minkowski')
mms = MinMaxScaler()
train_data_lda_std = mms.fit_transform(train_data_lda)
knn.fit(train_data_lda_std, X_train_label)
test_pred = knn.predict(mms.fit_transform(test_data_lda))



