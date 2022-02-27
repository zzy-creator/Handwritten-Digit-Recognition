from operator import ne
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

class BP:
    def __init__(self,input_layer,hiden_layer,output_layer,learning_rate):
        self.ilayer = input_layer #输入层
        self.hlayer = hiden_layer #隐藏层
        self.olayer = output_layer #输出层
        self.step = learning_rate #学习步长
        self.input_hidden_weight = np.random.normal(0.0,pow(self.hlayer,-0.5),(self.hlayer,self.ilayer)) #连接输入层和隐藏层的权重矩阵
        self.hidden_output_weight = np.random.normal(0.0,pow(self.olayer,-0.5),(self.olayer,self.hlayer)) #连接隐藏层和输出层的权重矩阵

        self.activation_function = lambda x : scipy.special.expit(x) #激活函数

    def forward(self,input_vector):
        #将输入向量形成矩阵作为输入层的输出
        inputs = np.array(input_vector,ndmin=2).T

        #将输入层的结果通过权重矩阵输入到隐藏层并用激活函数计算隐藏层的输出
        hidden_inputs = np.dot(self.input_hidden_weight,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #将隐藏层的输出通过权重矩阵作为输出层的输出并用激活函数计算输出层的输出
        output_inputs = np.dot(self.hidden_output_weight,hidden_outputs)
        output_outputs = self.activation_function(output_inputs)
        return inputs,hidden_outputs,output_outputs

    def backward(self,label_vector,output_outputs):
        labels = np.array(label_vector,ndmin=2).T
        #计算输出层损失
        output_errors = labels - output_outputs
        #反向传播到隐藏层,计算隐含层损失
        hidden_errors = np.dot(self.hidden_output_weight.T,output_errors)
        return output_errors,hidden_errors

    def update(self,inputs,hidden_outputs,output_outputs,hidden_errors,output_errors):
        #更新权重矩阵
        self.hidden_output_weight += self.step * np.dot((output_errors * output_outputs * (1.0-output_outputs)),np.transpose(hidden_outputs))
        self.input_hidden_weight += self.step * np.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)),np.transpose(inputs))

    def train(self,input_vector,label_vector):
        #训练过程：前向传播、反向传播、更新权重
        inputs,hidden_outputs,output_outputs = self.forward(input_vector)
        output_errors,hidden_errors = self.backward(label_vector,output_outputs)
        self.update(inputs,hidden_outputs,output_outputs,hidden_errors,output_errors)

    def predict(self,input_vector):
        input = np.array(input_vector,ndmin=2).T
        hidden_inputs = np.dot(self.input_hidden_weight,input) #得到隐藏层的输入
        hidden_outputs = self.activation_function(hidden_inputs) #得到隐藏层的输出
        output_inputs = np.dot(self.hidden_output_weight,hidden_outputs) #得到输出层的输入
        output_outputs = self.activation_function(output_inputs) #得到输出层的输出

        return np.argmax(output_outputs)

    def save_model(self):
        np.save("./BP_model/i_h_weight.npy",self.input_hidden_weight)
        np.save("./BP_model/h_o_weight.npy",self.hidden_output_weight)

    def load_model(self):
        self.input_hidden_weight = np.load("./BP_model/i_h_weight.npy")
        self.hidden_output_weight = np.load("./BP_model/h_o_weight.npy")

def start_train(epochs,network):
    #读入训练集
    train_file = open("../tc/mnist_train.csv","r")
    train_vectors = train_file.readlines()
    train_file.close()

    #进行训练
    print("start train...")
    for i in range(epochs):
        print("now is epoch %d" % (i))
        for sample in train_vectors:
            label_eigen = sample.split(',')
            inputs = (np.asfarray(label_eigen[1:])/255.0 * 0.99) + 0.01
            labels = np.zeros(output_node_num) + 0.01
            labels[int(label_eigen[0])] = 0.99
            network.train(inputs,labels)

    network.save_model()

def start_test(network):
    network.load_model()
    #读入测试集
    test_file = open("../tc/mnist_test.csv","r")
    test_vectors = test_file.readlines()
    test_file.close()

    print("start test...")
    #测试准确率
    scoreboard = []
    for sample in test_vectors:
        label_eigen = sample.split(',')
        expected_value = int(label_eigen[0])
        image_array = np.asfarray(label_eigen[1:]).reshape((28,28))
        plt.imshow(image_array,cmap='Greys',interpolation='None')
        inputs = (np.asfarray(label_eigen[1:])/255.0 * 0.99) + 0.01
        label = network.predict(inputs)
    
        if (label == expected_value):
            scoreboard.append(1)
        else:
            scoreboard.append(0)

    scoreboard_array = np.asarray(scoreboard)
    print("accuracy = ",scoreboard_array.sum() / scoreboard_array.size)


if __name__ == '__main__':
    input_node_num = 784 #24 * 24 =784
    hidden_node_num = 35 #隐藏层为35个神经元
    output_node_num = 10 #输出0-9所对应的概率分布
    learning_rate = 0.2 #学习的步长

    #创建神经网络
    network = BP(input_node_num,hidden_node_num,output_node_num,learning_rate)

    epochs = 5
    start_train(epochs,network)
    start_test(network)
    
