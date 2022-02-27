import os
import tensorflow as tf

from tensorflow.keras import datasets,layers,models

class CNN(object):
    def __init__(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        model.summary()
        self.model = model

    def create_dateset(self):
        # 下载数据集
        data_path = os.path.abspath(os.path.dirname(__file__)) + '/../tc/mnist.npz'
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)

        train_data = train_images.reshape((60000, 28, 28, 1))
        test_data = test_images.reshape((10000, 28, 28, 1))
        # 像素值映射到 0 - 1 之间
        train_data, test_data = train_data / 255.0, test_data / 255.0

        self.train_data, self.train_labels = train_data, train_labels
        self.test_data, self.test_labels = test_data, test_labels

    def train(self):
        print("start train...")
        self.create_dateset()
        check_path = './cnn_ckpt/cp-{epoch:04d}.ckpt'
        # period 每隔5epoch保存一次
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)

        self.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        self.model.fit(self.train_data, self.train_labels, epochs=5, callbacks=[save_model_cb])

    def test(self):
        print("start test...")
        test_loss, test_acc = self.model.evaluate(self.test_data, self.test_labels)
        print("accuracy = ", test_acc)




if __name__ == "__main__":
    cnn = CNN()
    cnn.train()
    cnn.test()
