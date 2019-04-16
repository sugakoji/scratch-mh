import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class MetroPolis(object):
    """
    メトロポリスヘイスティングの実装

    """

    def __init__(self, random_var=2, thin=10):

        # 正規乱数の分散
        self.random_var = random_var

        # まびく間隔
        self.thin = thin

        # 初期値
        self.param = 10

    def __call__(self, func, chain=3, iter_len=20000, burn_in=100):

        self.result = []
        for _c in range(0, chain + 1):
            self.result.append(self._sampling(func, iter_len, burn_in))

        return np.array(self.result)

    def _sampling(self, func, iter_len, burn_in):
        """サンプリングする関数

        :param func: サンプリングする関数
        :param iter_len: サンプリング回数
        :param burn_in: バーンイン期間
        :return: バーンイン期間を除いたサンプル
        """
        samples = []
        for i in range(iter_len):
            prob_dense = func(self.param)
            new_param = self.param + np.random.normal(loc=0,
                                                      scale=self.random_var)
            new_prob_dense = func(new_param)
            accept_prob = new_prob_dense / prob_dense
            if accept_prob > np.random.uniform(low=0.0, high=1.0):
                self.param = new_param
            if i % self.thin == 1:
                samples.append(self.param)

        return samples[burn_in:]

    def get_r_hat(self):
        """
        収束診断の為の値　r_hat

        完全なものではないのであとで修正

        :return:
        """

        result = np.array(self.result)

        # 全体平均
        overall_ave = np.mean(result.flatten())
        # 各列の平均
        np.mean(result, axis=1)
        # Nの数
        N = np.array([i.shape[0] for i in result])
        # Mの数
        M = result.shape[0]
        # Bの計算
        B = np.sum(N * (overall_ave - np.mean(result, axis=1)) ** 2) * 1 / (
                M - 1)
        # サンプル内分散
        s = np.sum((result - np.mean(result.reshape(1, -1), axis=1)) ** 2,
                   axis=1) * 1 / (N - 1)
        # Wの計算
        W = 1 / M * np.sum(s)
        ### todo Nの長さが違う場合を闔閭できていない
        n = N[0]
        var_hat = (n - 1) / n * W + 1 / n * B

        return (var_hat / W) ** 0.5
