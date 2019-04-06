import numpy as np
import scipy.sparse as sp


def encode_onehot(labels):
    classes = set(labels)
    # print(classes)
    class_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # print(class_dict.get)
    labels_onehot = np.array(list(map(class_dict.get, labels)), dtype=np.int32)
    return labels_onehot


a = np.identity(5)[1, :]
print(a)
idx_features_labels = np.genfromtxt('E:\\CodingDocument\\Pycharm\\TorchTutorials\\Advanced\\GCNN\\dataset\\cora.content'
                                    '', dtype=np.dtype(str))
print('idx_features_labels: \n', idx_features_labels)
print(idx_features_labels[:, 1:-1])
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
print('features: \n', features)
labels = encode_onehot(idx_features_labels[:, -1])
print('labels: \n', labels)

# a = np.random.randn(3, 4)
# print(a)
# b = sp.csr_matrix(a, dtype=np.float32)
# print(b)
