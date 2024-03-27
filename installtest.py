from keras.utils import to_categorical
from keras import models, layers, regularizers
from keras.optimizers import RMSprop
from keras.datasets import mnist
import matplotlib.pyplot as plt
#加载数据集并处理
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()#归一化处理，注意必须进行归一化操作，否则准确率非常低，图片和标签
 
#将图片由二维铺开成一维
train_images = train_images.reshape((60000, 28*28)).astype('float')  #将28*28的二维数组转变为784的一维数组，浮点数类型
test_images = test_images.reshape((10000, 28*28)).astype('float')
train_labels = to_categorical(train_labels)  #to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
test_labels = to_categorical(test_labels)
#print(train_labels[0])
#搭建神经网络（全连接）
network = models.Sequential()   #选用的是Sequential 序贯模型sigmoid
network.add(layers.Dense(units=15, activation='sigmoid', input_shape=(28*28, ),))#添加一个(隐藏层)全连接层，神经元为15，激活函数是relu线性整流
#函数,输入形状为28*28
network.add(layers.Dense(units=10, activation='softmax'))#添加一个(输出层)全连接层，神经元为10，激活函数为softmax(Softmax 具有更好的解释性，
#这块通过softmax激活函数，最后的数组中，十个数哪个最大，计算机就认为是哪个
 
#神经网络的编译和训练
# 编译步骤，损失函数是模型优化的目标，优化器使用RMSporp,学习率为0.001，损失函数是categorical_crossentropy，评价函数为accuracy准确率
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])#RMSprop(lr=0.001)
# 训练网络，用fit函数（fit()方法用于执行训练过程）, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据，一次训练所选取的样本数。
network.fit(train_images, train_labels, epochs=22, batch_size=128, verbose=1)  #verbose：日志显示 0 为不在标准输出流输出日志信息 1 为输出进度条记录
                                                                                                                #2 为每个epoch输出一行记录
#测试集上测试模型性能
#y_pre = network.predict(test_images[:5])  #预测前五张图片的，model.predict 实际预测，其输出是目标值，根据输入数据预测。
#print(y_pre, test_labels[:5])
test_loss, test_accuracy = network.evaluate(test_images, test_labels)  #model.evaluate函数预测给定输入的输出
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
print(network.summary())  #查看神经网络model结构