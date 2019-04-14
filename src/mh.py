class MetroPolis():

    def __init__(self, random_var=2, thin=10):

        # 正規乱数の分散
        self.random_var = random_var

        # まびく間隔
        self.thin = thin

        # 初期値
        self.param = 10

        # サンプル列を入れておく
        self.samples = []

    def __call__(self, func, iter_len=10000, burn_in=100):
        # まず初期値の確率密度を計算

        for i in range(iter_len):
            prob_dense = func(self.param)
            new_param = self.param + np.random.normal(loc=0,
                                                      scale=self.random_var)
            new_prob_dense = func(new_param)

            accept_prob = new_prob_dense / prob_dense
            if accept_prob > np.random.uniform(low=0.0, high=1.0):
                self.param = new_param
                if i % self.thin == 1:
                    self.samples.append(self.param)

        return self.samples[burn_in:]