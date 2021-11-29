import numpy as np
import matplotlib.pyplot as plt
import copy
import time


class FA:
    def __init__(self, D, N, Beta0, gama, alpha, T, bound):
        self.D = D  # 问题维数
        self.N = N  # 群体大小
        self.Beta0 = Beta0  # 最大吸引度
        self.gama = gama  # 光吸收系数
        self.alpha = alpha  # 步长因子
        self.T = T
        self.X = (bound[1] - bound[0]) * np.random.random([N, D]) + bound[0]  # 初始化种群
        self.X_origin = copy.deepcopy(self.X)  # 拷贝一份做原始种群
        self.FitnessValue = np.zeros(N)  # 适应度置零
        for n in range(N):  # 适应度初始化
            self.FitnessValue[n] = self.FitnessFunction(self.X[n, :])

    def alphat(self, t):
        self.alpha = (1 - t / self.T) * self.alpha  #自适应步长
    # def alphat(self):
    #     self.alpha = 0.99 * self.alpha
    # def alphat3(self, t):
    #     self.alpha = np.exp(-t/self.T) * self.alpha  #自适应步长

    def DistanceBetweenIJ(self, i, j):
        return np.linalg.norm(self.X[i, :] - self.X[j, :])

    def BetaIJ(self, i, j):  # AttractionBetweenIJ
        return self.Beta0 * \
               np.math.exp(-self.gama * (self.DistanceBetweenIJ(i, j) ** 2))

    def update(self, i, j):
        self.X[i, :] = self.X[i, :] + \
                       self.BetaIJ(i, j) * (self.X[j, :] - self.X[i, :]) + \
                       self.alpha * (np.random.rand(self.D) - 0.5)

    def FitnessFunction(self, x_):
        return np.linalg.norm(x_) ** 2

    def FindNewBest(self, i):
        FFi = self.FitnessFunction(self.X[i, :])
        x_ = self.X[i, :] + self.alpha * (np.random.rand(self.D) - 0.5)
        ffi = self.FitnessFunction(x_)
        if ffi < FFi:
            self.X[i, :] = x_
            self.FitnessValue[i] = ffi

    def iterate(self):
        t = 0
        while t < self.T:
            # self.alphat(t)
            for i in range(self.N):
                tag = 0
                FFi = self.FitnessValue[i]
                for j in range(self.N):
                    FFj = self.FitnessValue[j]
                    if FFj < FFi:
                        tag = 1
                        self.update(i, j)
                        self.FitnessValue[i] = self.FitnessFunction(self.X[i, :])
                        FFi = self.FitnessValue[i]
                if tag == 0:
                    self.FindNewBest(i)
            t += 1

    # def iterate(self):
    #     t = 0
    #     while t < self.T:
    #         self.alphat(t)
    #         for i in range(self.N):
    #             tag = 0
    #             FFi = self.FitnessValue[i]
    #             list = [(i-2)%self.N, (i-1)%self.N, (i+1)%self.N, (i+2)%self.N]
    #             for j in list:
    #                 FFj = self.FitnessValue[j]
    #                 if FFj < FFi:
    #                     tag = 1
    #                     self.update(i, j)
    #                     self.FitnessValue[i] = self.FitnessFunction(self.X[i,:])
    #                     FFi = self.FitnessValue[i]
    #             if tag == 0:
    #                 self.FindNewBest(i)
    #         t += 1

    # def iterate(self):
    #     t = 0
    #     while t < self.T:
    #         for i in range(self.N):
    #             j = np.random.randint(0,self.N)
    #             FFi = self.FitnessValue[i]
    #             FFj = self.FitnessValue[j]
    #             if FFj < FFi:
    #                 self.update(i, j)
    #                 self.FitnessValue[i] = self.FitnessFunction(self.X[i,:])
    #             else:
    #                 self.FindNewBest(i)
    #         t += 1

    def find_min(self):
        v = np.min(self.FitnessValue)
        n = np.argmin(self.FitnessValue)
        return v, self.X[n, :]


def plot(X_origin, X):
    fig_origin = plt.figure(0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X_origin[:, 0], X_origin[:, 1], c='r')
    plt.scatter(X[:, 0], X[:, 1], c='g')
    plt.show()

# def plot3(X, Y1, Y2, Y3, legend1, legend2, legend3):
#     Y1 = np.log(Y1)
#     Y2 = np.log(Y2)
#     Y3 = np.log(Y3)
#     X = np.array(range(int(X / 100) + 1))
#     plt.plot(X,Y1,marker='o',ls='-',mec='r',mfc='w',label=legend1)
#     plt.plot(X,Y2,marker='*',ls='-',ms=10,label=legend2)
#     plt.plot(X,Y3,marker='^',ls='-',mec='b',label=legend3)
#     plt.legend()
#     plt.xlabel('FEs/1000')
#     plt.ylabel('Best Fitness Value(log)')
#     plt.show()

if __name__ == '__main__':
    t = np.zeros(10)
    value = np.zeros(10)
    for i in range(10):
        fa = FA(2, 20, 1, 0.000001, 0.97, 100, [-100, 100])
        time_start = time.time()
        fa.iterate()
        time_end = time.time()
        t[i] = time_end - time_start
        value[i], n = fa.find_min()
        # plot(fa.X_origin, fa.X)
    print("平均值：", np.average(value))
    print("最优值：", np.min(value))
    print("最差值：", np.max(value))

    print("平均时间：", np.average(t))