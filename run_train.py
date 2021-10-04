from train import Train_protonet


class TrainArgs(object):
    def __init__(self):
        # record saving path
        self.result_path = './runs/exp'
        self.checkpoints_dir = 'checkpoint'
        self.tensorboard_dir = 'tensorboard_log'

        # data loader args
        self.train_csv_path = './omniglot/train.csv'
        self.val_csv_path = './omniglot/val.csv'
        self.img_channels = 1
        self.img_size = 28

        self.way = 10
        self.shot = 10
        self.query = 5
        self.episodes = 10

        self.val_way = 2
        self.val_shot = 10
        self.val_query = 5
        self.val_episodes = 10

        # train args
        self.epochs = 600
        self.patience = 50
        self.learning_rate = 0.0001
        self.lr_decay_step = 1
        self.weight_decay_step = 0
        self.seed = 42

        # model args
        self.hidden_channels = 64


if __name__ == '__main__':
    args = TrainArgs()
    exp_p = Train_protonet(args)
    exp_p.train()
