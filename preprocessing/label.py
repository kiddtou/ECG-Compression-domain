 #导入相应的库（对数据库进行切分需要用到的库是sklearn.model_selection 中的 train_test_split）
import numpy as np
from sklearn.model_selection import train_test_split
my_matrix = np.loadtxt(open("N.csv"),delimiter=",",skiprows=0)
 #对于矩阵而言，将矩阵倒数第一列之前的数值给了X（输入数据），将矩阵大最后一列的数值给了y（标签）
X, y = my_matrix[:,:-1],my_matrix[:,-1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#add random noise
x_train_nosiy = x_train + 0.3 * np.random.normal(loc=0., scale=1., size=x_train.shape)
x_test_nosiy = x_test + 0.3 * np.random.normal(loc=0, scale=1, size=x_test.shape)
x_train_nosiy = np.clip(x_train_nosiy, 0., 1.)
x_test_nosiy = np.clip(x_test_nosiy, 0, 1.)
print(x_train_nosiy.shape, x_test_nosiy.shape)


# 压缩特征维度至2维
encoding_dim = 26
#build autoencoder model
input_img = Input(shape=(259,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)


encoded = Dense(500, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input=input_img, output=decoded)