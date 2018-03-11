from deepautoencoder import StackedAutoEncoder
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data, target = mnist.train.images, mnist.train.labels
test_x, test_s = mnist.test.images, mnist.test.labels
# train / test  split
#idx = np.random.rand(data.shape[0]) < 0.8
train_X, train_Y = data, target
#test_X, test_Y = data[~idx], target[~idx]

model = StackedAutoEncoder(dims=[784, 784 , 784], activations=['relu', 'linear','softmax'], epoch=[
                           3000, 3000, 3000], loss='rmse', lr=0.007, batch_size=100, print_step=200)
model.fit(train_X)
test_X_trans = model.transform(test_x)
residual_error = abs(test_X_trans - test_x)

#PCA 5 components
pca=PCA(n_components=5)
pca_error = pca.fit_transform(residual_error.T)

#mean error
mean_error = np.mean(residual_error, axis=0)

def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)


plt.imshow(mean_error.reshape((28, 28)), cmap=plt.cm.gray)
plt.show()

plt.imshow(pca_error[:,0].reshape((28, 28)), cmap=plt.cm.gray)
plt.show()

plt.imshow(pca_error[:,1].reshape((28, 28)), cmap=plt.cm.gray)
plt.show()

plt.imshow(pca_error[:,2].reshape((28, 28)), cmap=plt.cm.gray)
plt.show()

plt.imshow(pca_error[:,3].reshape((28, 28)), cmap=plt.cm.gray)
plt.show()

plt.imshow(pca_error[:,4].reshape((28, 28)), cmap=plt.cm.gray)
plt.show()


plt.imshow(mean_error.reshape((28, 28)), cmap=grayify_cmap('jet'))
plt.show()

plt.imshow(pca_error[:,0].reshape((28, 28)), cmap=grayify_cmap('jet'))
plt.show()

plt.imshow(pca_error[:,1].reshape((28, 28)), cmap=grayify_cmap('jet'))
plt.show()

plt.imshow(pca_error[:,2].reshape((28, 28)), cmap=grayify_cmap('jet'))
plt.show()

plt.imshow(pca_error[:,3].reshape((28, 28)), cmap=grayify_cmap('jet'))
plt.show()

plt.imshow(pca_error[:,4].reshape((28, 28)), cmap=grayify_cmap('jet'))
plt.show()