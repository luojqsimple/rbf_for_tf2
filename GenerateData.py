import numpy as np
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from mayavi import mlab


def generate_data():

    # 读取txt文档
    filename1 = r"C:\Users\luojq\OneDrive - zju.edu.cn\CloudDoc2023\230810_离心机运行状态实时演算系统\COMSOL_Centrifuge"
    filename2 = r'\SimulationData\LongArm\LongArm_Fn1_W1.txt'
    with open(filename1+filename2, "r") as file:
        lines = file.readlines()

    # 提取非%符号开头的所有行数据
    data_lines = []
    for line in lines:
        if not line.startswith("%"):
            data_lines.append(line.split())

    # 将数据转换为NumPy数组
    data_array = np.array(data_lines, dtype=float)

    # 提取特征数据和标签数据
    X = data_array[:, :3]  # 前三列为特征数据
    y = data_array[:, 3]  # 最后一列为标签数据

    # 划分训练集和验证集
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
    #                                                   random_state=1)

    return train_test_split(X, y, test_size=0.2, random_state=1)


# # 打印训练集和验证集的形状
# X_train, X_val, y_train, y_val = generate_data()
# print("训练集特征数据形状:", X_train.shape, X_train[:10, :])
# print("训练集标签数据形状:", y_train.shape, y_train[:10])
# print("验证集特征数据形状:", X_val.shape, X_val[:10, :])
# print("验证集标签数据形状:", y_val.shape, y_val[:10])

# train_data = np.column_stack((X_train, y_train))
# test_data = np.column_stack((X_val, y_val))

# # 创建一个新的三维图像
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制预测值和真实值
# ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2],
#            color='b', label='1', marker='o')
# ax.scatter(X_val[:, 0], X_val[:, 1], X_val[:, 2],
#            color='b', label='2')

# # 设置图例和标签
# ax.legend()
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # 显示图像
# plt.show()

# # 创建Mayavi场景
# mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))

# # 绘制点云
# radii = np.random.rand(X_train.shape[0]) * 0.5  # 生成随机半径，范围在0到0.5之间
# mlab.points3d(X_train[:, 0], X_train[:, 1], X_train[:, 2], radii,
#               mode='point')

# # 设置坐标轴标签
# mlab.xlabel('X')
# mlab.ylabel('Y')
# mlab.zlabel('Z')

# # 显示图形
# mlab.show()
