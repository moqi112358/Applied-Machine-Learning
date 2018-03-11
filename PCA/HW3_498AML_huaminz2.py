# -*- coding: cp936 -*-
from sklearn.decomposition import PCA
import pickle
import numpy as np


# read data
wd = "C:\\Users\\98302\\Desktop\\cifar-10-batches-py\\data_batch_"
batch_file = ["1","2", "3", "4", "5"]
data = []
label = []
for file in batch_file:
    input_file = wd + file
    fo = open(input_file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    data.append(dict['data'])
    label.append(dict['labels'])

#############4.10(a)##################################################
class_data=[]
class_pca=[]
class_mean_image=[]
error = []
for i in range(0,10):
    x=[]
    y=[]
    x=np.array(x,dtype = np.uint8)
    new_dict = {'data':x,'labels':y}
    
    #classfier
    for j in range(0,5):
        for k in range(0,10000):
            if label[j][k] == i:
                new_dict['labels'].append(label[j][k])
                new_dict['data'] = np.concatenate((new_dict['data'],data[j][k]))
    new_dict['data'] = new_dict['data'].reshape(new_dict['data'].shape[0]/3072,3072)
    class_data.append(new_dict)
    
    #do pca
    pca = PCA(copy=True,n_components=20) #pca para
    new_data = pca.fit_transform(class_data[i]['data']) #transform 10000*3072 to 10000*20
    #mean image
    class_mean_image.append(pca.mean_) 
    class_pca.append(new_data)
    
    #Constructing a low-dimensional representation
    represent_data = pca.inverse_transform(new_data)

    #calculate error
    total_dis = 0
    for image in range(5000):
        dist = np.sqrt(np.sum(np.square(represent_data[image] - class_data[i]['data'][image])))
        total_dis = total_dis + dist

    ave_err = total_dis/5000
    error.append(ave_err)

#############4.10(b)##################################################

#create distance matrix
D = []
for i in range(10):
    for j in range(10):
        dist = np.sum(np.square(class_mean_image[i] - class_mean_image[j]))
        D.append(dist)
D = np.array(D).reshape(10,10)
#Form A,W and get the eigenvectors andeigenvalues of W
I = np.identity(10)
A = I - 0.1 * np.array([1]*100).reshape(10,10)
W = -0.5 * np.dot(np.dot(A,D),np.transpose(A))
eigval, eigvec = np.linalg.eig(W)

#the top left r × r block , r =2
sort1 = eigval.argsort()[-1]
sort2 = eigval.argsort()[-2]
v1 = eigval[sort1]
v2 = eigval[sort2]
vec1 = eigvec[:,sort1]
vec2 = eigvec[:,sort2]
diag = np.array([np.sqrt(v1),0,0,np.sqrt(v2)]).reshape(2,2)
eigvec_r = np.concatenate((vec1,vec2)).reshape(2,10)
#compute V
V_T = np.dot(diag,eigvec_r)
V = np.transpose(V_T)
#############4.10(c)##################################################

#create E matrix E(i->j) = E_ij
E = []
for i in range(10):
    for j in range(10):
        pca_i = PCA(copy=True,n_components=20) #pca para
        data_i = pca_i.fit_transform(class_data[i]['data']) #transform 10000*3072 to 10000*20
        pca_j = PCA(copy=True,n_components=20) #pca para
        data_j = pca_j.fit_transform(class_data[j]['data']) #transform 10000*3072 to 10000*20
        pca_i.components_ = pca_j.components_
        represent_data_i = pca_i.inverse_transform(data_i)
        total_dis = 0
        for image in range(5000):
            dist = np.sqrt(np.sum(np.square(represent_data_i[image] - class_data[i]['data'][image])))
            total_dis = total_dis + dist
        ave_err = total_dis/5000
        E.append(ave_err)
E = np.array(E).reshape(10,10)
D2 = []
for i in range(10):
    for j in range(10):
        similarity = 0.5 * (E[i][j] + E[j][i])
        if i == j:
            similarity = 0
        D2.append(similarity)
D2 = np.array(D2).reshape(10,10)
#Form A,W and get the eigenvectors andeigenvalues of W
I2 = np.identity(10)
A2 = I2 - 0.1 * np.array([1]*100).reshape(10,10)
W2 = -0.5 * np.dot(np.dot(A2,D2),np.transpose(A2))
eigval2, eigvec2 = np.linalg.eig(W2)
#the top left r × r block , r =2
sort_c1 = eigval2.argsort()[-1]
sort_c2 = eigval2.argsort()[-2]
v_c1 = eigval2[sort_c1]
v_c2 = eigval2[sort_c2]
vec_c1 = eigvec2[:,sort_c1]
vec_c2 = eigvec2[:,sort_c2]
diag2 = np.array([np.sqrt(v_c1),0,0,np.sqrt(v_c2)]).reshape(2,2)
eigvec_r2 = np.concatenate((vec_c1,vec_c2)).reshape(2,10)
#compute V
V_T2 = np.dot(diag2,eigvec_r2)
V2 = np.transpose(V_T2)

#############extra##################################################
D3 = []
for i in range(10):
    for j in range(10):
        similarity = 0.5 * (E[i][j] + E[j][i])
        D2.append(similarity)
D3 = np.array(D2).reshape(10,10)
#Form A,W and get the eigenvectors andeigenvalues of W
I3 = np.identity(10)
A3 = I2 - 0.1 * np.array([1]*100).reshape(10,10)
W3 = -0.5 * np.dot(np.dot(A3,D3),np.transpose(A3))
eigval3, eigvec3 = np.linalg.eig(W3)
#the top left r × r block , r =2
sort_d1 = eigval3.argsort()[-1]
sort_d2 = eigval3.argsort()[-2]
v_d1 = eigval3[sort_d1]
v_d2 = eigval3[sort_d2]
vec_d1 = eigvec3[:,sort_d1]
vec_d2 = eigvec3[:,sort_d2]
diag3 = np.array([np.sqrt(v_d1),0,0,np.sqrt(v_d2)]).reshape(2,2)
eigvec_r2 = np.concatenate((vec_d1,vec_d2)).reshape(2,10)
#compute V
V_T3 = np.dot(diag3,eigvec_r2)
V3 = np.transpose(V_T3)