import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MSE

import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
from initializer import InitFromFile
from GenerateData import generate_data


def load_data():

    data = np.loadtxt("data/new_data.txt")
    X = data[:, :-1]  # except last column
    y = data[:, -1]  # last column only
    return X, y


def test(X, y, initializer, ndim=3):

    title = f" test {type(initializer).__name__} "
    print("-"*20 + title + "-"*20)

    # # 数据归一化处理
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # create RBF network as keras sequential model
    model = Sequential()
    rbflayer = RBFLayer(80,
                        initializer=initializer,
                        betas=5.0,
                        input_shape=(ndim,))
    outputlayer = Dense(1, use_bias=False)

    model.add(rbflayer)
    model.add(outputlayer)

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop())

    # fit and predict
    model.fit(X, y,
              batch_size=30,
              epochs=2000,
              verbose=1)

    y_pred = model.predict(X)

    # 创建一个新的三维图像
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # show graph
    # ax.scatter(X[:, 0], X[:, 1], y_pred, label='Predicted', c='b')
    # ax.scatter(X[:, 0], X[:, 1], y, label='Actual', c='r')
    plt.scatter(y, y_pred)  # prediction
    plt.plot(y, y)       # response from data
    # plt.plot([-1, 1], [0, 0], color='black')  # zero line
    # plt.xlim([-1, 1])
    # ax.set_xlim([0, 1])
    # ax.set_ylim([-1, 1])

    # plot centers
    centers = rbflayer.get_weights()[0]
    widths = rbflayer.get_weights()[1]
    # ax.scatter(centers[:, 0], centers[:, 1], np.zeros(len(centers)),
    #            s=20*widths, label='RBF Centers')

    plt.show()

    # calculate and print MSE
    y_pred = y_pred.squeeze()
    print(f"MSE: {MSE(y, y_pred):.4f}")

    # saving to and loading from file
    filename = "rbf_{}.h5".format(type(initializer).__name__)
    print(f"Save model to file {filename} ... ", end="")
    model.save(filename)
    print("OK")

    print(f"Load model from file {filename} ... ", end="")
    newmodel = load_model(filename,
                          custom_objects={'RBFLayer': RBFLayer})
    print("OK")

    # check if the loaded model works same as the original
    y_pred2 = newmodel.predict(X).squeeze()
    print("Same responses: ", all(y_pred == y_pred2))
    # I know that I compared floats, but results should be identical

    # save, widths & weights separately
    np.save("centers", centers)
    np.save("widths", widths)
    np.save("weights", outputlayer.get_weights()[0])


def test_init_from_file(X, y):

    print("-"*20 + " test init from file " + "-"*20)

    # load the last model from file
    filename = "rbf_InitCentersRandom.h5"
    print(f"Load model from file {filename} ... ", end="")
    model = load_model(filename,
                       custom_objects={'RBFLayer': RBFLayer})
    print("OK")

    res = model.predict(X).squeeze()  # y was (50, ), res (50, 1); why?
    print(f"MSE: {MSE(y, res):.4f}")

    # load the weights of the same model separately
    rbflayer = RBFLayer(50,
                        initializer=InitFromFile("centers.npy"),
                        betas=InitFromFile("widths.npy"),
                        input_shape=(3,))
    print("rbf layer created")
    outputlayer = Dense(1,
                        kernel_initializer=InitFromFile("weights.npy"),
                        use_bias=False)
    print("output layer created")

    model2 = Sequential()
    model2.add(rbflayer)
    model2.add(outputlayer)

    res2 = model2.predict(X).squeeze()
    print(f"MSE: {MSE(y, res2):.4f}")
    print("Same responses: ", all(res == res2))

    plt.scatter(y, res2)  # prediction
    plt.plot(y, y)       # response from data
    plt.show()


if __name__ == "__main__":

    # X, y = load_data()
    X_train, X_val, y_train, y_val = generate_data()
    print("训练集特征数据形状:", X_train.shape, X_train[:10, :])
    print("训练集标签数据形状:", y_train.shape, y_train[:10])
    print("验证集特征数据形状:", X_val.shape, X_val[:10, :])
    print("验证集标签数据形状:", y_val.shape, y_val[:10])

    # test simple RBF Network with random  setup of centers
    # test(X_train, y_train, InitCentersRandom(X_train), ndim=3)

    # test simple RBF Network with centers set up by k-means
    test(X_train, y_train, InitCentersKMeans(X_train), ndim=3)

    # test simple RBF Networks with centers loaded from previous
    # computation
    # test(X_train, y_train, InitFromFile("centers.npy"))

    # test InitFromFile initializer
    # test_init_from_file(X_train, y_train)
