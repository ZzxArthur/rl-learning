class TrainConfig():
    def __init__(self):
        self.env_id = "CartPole-v1"
        self.train_seed = 42
        self.num_train = 1500
        self.gamma = 0.99
        self.lr = 0.01
