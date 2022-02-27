使用BP和CNN实现手写数字识别

由于数据集太大，这里没有上传，可在http://yann.lecun.com/exdb/mnist/上进行下载，下载好后，对数据集进行格式处理，详见“BP和CNN实验报告”

在src文件夹中，执行 python BP_network.py 训练BP神经网络
执行 python cnn.py 训练BP神经网络
执行 python GUI.py 进行手写数字识别（当前默认为CNN训练模型进行识别，如需换成BP，请将GUI中的# BP begin 到 # BP end取消注释，并注释掉# cnn begin 到 # cnn end）
