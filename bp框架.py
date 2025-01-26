import math
import random
import numpy
from sklearn import preprocessing
import time
import xlwt
import matplotlib.pyplot as plt
from matplotlib import rcParams

random.seed(0)

def read_data(dir_str):
    '''
    读取txt文件中的数据
    数据内容：科学计数法保存的多行多列数据
    输入：txt文件的路径
    输出：小数格式的数组，行列与txt文件中相同
    '''
    data_temp = []
    with open(dir_str) as fdata:
        while True:
            line=fdata.readline()
            if not line:
                break
            data_temp.append([float(i) for i in line.split()])
    return numpy.array(data_temp)


def randome_init_train_test(data, n_tr):
    ''' 随机划分训练集和测试集 '''
    # sklearn提供一个将数据集切分成训练集和测试集的函数train_test_split
    train_index = numpy.random.choice(data.shape[0], size=n_tr, replace=False, p=None)
    train_data = data[train_index]
    test_index = numpy.delete(numpy.arange(data.shape[0]),train_index)  # 删除train_index对应索引的行数
    test_data = data[test_index]
    return train_data, test_data


def min_max_normalization(np_array):
    ''' 离差标准化，(Xi-min(X))/(max(X)-min(X)) '''
    min_max_scaler = preprocessing.MinMaxScaler()
    ret = min_max_scaler.fit_transform(np_array)
    return ret


def label_to_value(label):
    ''' 标签转换为对应输出值 (由于输出层结构，需要修改输出数据结构)'''
    switch = {
        0.0: [1,0,0],
        1.0: [0,1,0],
        2.0: [0,0,1]
    }
    return switch[label]


def value_to_label(value):
    ''' 神经网络输出值转换为对应标签 '''
    return value.index(max(value))


def rand(min, max):
    ''' 随机取[a, b]范围内的值 '''
    return (max - min) * random.random() + min


def make_matrix(m, n, fill=0.0):    # 生成多维矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BPNeuralNetwork:
    def __init__(self):  # 设置在BP神经网络中用到的参数
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_values = []  # [1.0] * self.input_n
        self.hidden_values = []  # [1.0] * self.hidden_n
        self.output_values = []  # [1.0] * self.output_n
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []                     # dw1
        self.output_correction = []                    # dw2
        self.input_bias = []
        self.output_bias = []

    def setup(self, ni, nh, no):  # 参数设置
        self.input_n = ni
        self.hidden_n = nh
        self.output_n = no
        # init
        self.input_values = [1.0] * self.input_n    # 输入层神经元输出（输入特征）
        self.hidden_values = [1.0] * self.hidden_n  # 中间层神经元输出
        self.output_values = [1.0] * self.output_n  # 隐藏层神经元输出（预测结果）
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # 初始随机赋值，在范围[-1, +1]内
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-1, 1)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-1, 1)
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)
        self.input_bias = [0.0] * self.input_n
        self.output_bias = [0.0] * self.output_n

    def predict(self, inputs):  # 前向传播（在train中套在反向传播的train前面）
        # 输入层计算
        for i in range(self.input_n - 1):
            self.input_values[i] = inputs[i]
        # 隐藏层计算
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_values[i] * self.input_weights[i][j]
            self.hidden_values[j] = sigmoid(total + self.input_bias[i])
        # 输出层计算
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_values[j] * self.output_weights[j][k]
            self.output_values[k] = sigmoid(total + self.output_bias[j])
        return self.output_values[:]

    def back_propagate(self, case, label, learn, correct):
        # 前向预测
        self.predict(case)
        # 计算输出层的误差 w2
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_values[o]
            output_deltas[o] = sigmoid_derivative(self.output_values[o]) * error
        # 计算隐藏层的误差 w1
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_values[h]) * error
        # 更新隐藏-输出层权重 b2
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_values[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
                self.output_bias[o] += learn * change
        # 更新输入-隐藏层权重 b1
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_values[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
                self.input_bias[h] += learn * change

        # 计算样本的均方误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_values[o]) ** 2
        return error

    def train(self, datas, labels, epochs=5000, learn=0.05, correct=0.1, stop_error=0.001):
        for j in range(epochs):
            error = 0.0
            for i in range(len(datas)):
                label = labels[i]
                data = datas[i]
                error += self.back_propagate(data, label, learn, correct)
            if error <= stop_error:
                return j+1
        return epochs

def save_excel(datas, output_file):
    # 将数据保存到新的excel表格里
    # 因为xls文件支持最大数据行数为65536，所以大文件输出成几个小文件，每个小文件有MAX_EXCEL_ROWS行数据
    MAX_EXCEL_ROWS = 60000
    for no in range(0, datas.__len__()//MAX_EXCEL_ROWS + 1):
        sheet_name = 'sheet' + str(no+1)
        output_file_name = output_file.split('.')[0] + str(no+1) + '.' + output_file.split('.')[-1]
        print('输出文件：', output_file_name)
        excel = xlwt.Workbook()
        sh = excel.add_sheet(sheet_name)
        for i, data in enumerate(datas[no*MAX_EXCEL_ROWS:(no+1)*MAX_EXCEL_ROWS]):
            for j, d in enumerate(data):
                sh.write(i, j, d)
        try:
            excel.save(output_file_name)
        except:
            xxx = input('输出异常!!请检查输出路径是否异常或文件是否已存在(需删除已存在文件)。然后输入任意键即可...')
            no = no - 1

if __name__ == '__main__':
    n_tr=115
    input_nodes=4
    hidden_nodes=3
    output_nodes=3
    epochs=1000
    learn_rate=0.5
    momentum_rate=0.09
    correct_rate=0.1

    data=read_data(r"E:\BaiduNetdiskDownload\bp\数据\蝴蝶花数据.txt")
    normal_data=min_max_normalization(data[:,2:])
    data=numpy.concatenate((data[:,1:2],normal_data),axis=1)
    tr,te=randome_init_train_test(data,n_tr)#随即划分训练集和测试集
    tr_in=tr[:,1:]
    tr_out=[label_to_value(v[0]) for v in tr[:,:1]]#用于输出层使用3个节点，需要修改输出数据结构
    n_true=0#统计正确的预测数量

    nn=BPNeuralNetwork()
    nn.setup(input_nodes,hidden_nodes,output_nodes) #设置BP神经网络框架
    st=time.perf_counter()
    epoch=nn.train(tr_in,tr_out,epochs,learn_rate,correct_rate)
    print(f'eppch: {epoch}\n,Train_Time,{time.perf_counter()-st}')
    pre=[]
    for t in te:
        t_in=t[1:]
        label=value_to_label(nn.predict(t_in))
        if label==t[0]:
            n_true+=1
        pre.append([label])

    #输出统计结果
    accuracy=n_true/(data.shape[0]-n_tr)
    print('accuracy:',accuracy)
    print(nn.input_bias,nn.output_bias)

    #将数据保存到新的excel表格里
    te_pre=numpy.concatenate((te,numpy.array(pre)),axis=1)
    save_excel(te_pre,'test_result.xls')
    #绘制
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 假设数据
x_axis_data = [i for i in range(35)]
y_axis_data1 = te[:, :1].flatten()  # 转换为一维数组
y_axis_data2 = pre

# 绘制线条
plt.plot(x_axis_data, y_axis_data1, '-', color='blue', linewidth=3, label='真实结果', alpha=0.7)
plt.plot(x_axis_data, y_axis_data2, '--', color='red', linewidth=2, label='预测结果', alpha=0.7)

# 设置坐标轴标签
plt.xlabel('测试分类', fontproperties='SimHei')
plt.ylabel('预测分类', fontproperties='SimHei')

# 添加图例
plt.legend(loc='upper right')

# 显示图表
plt.show()

# 显示图像
plt.show()