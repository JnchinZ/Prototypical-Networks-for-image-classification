from dataset import ImageDataset, EpisodicBatchSampler
from protonet import PrototypicalNetwork
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import os
import json
import time
import numpy as np
from tqdm import tqdm
from utils import set_seed, metrics, AverageMeter, increment_path


class Train_protonet(object):
    def __init__(self, args):
        self.args = args
        self.args.seed = set_seed(self.args.seed)
        self.args.result_path = increment_path(self.args.result_path)
        print(f'This training log will be saved in {str(self.args.result_path)}')

        self.args.checkpoints_dir = self.args.result_path / self.args.checkpoints_dir
        self.args.tensorboard_dir = self.args.result_path / self.args.tensorboard_dir
        if not os.path.exists(self.args.checkpoints_dir):
            os.makedirs(self.args.checkpoints_dir)
        if not os.path.exists(self.args.tensorboard_dir):
            os.makedirs(self.args.tensorboard_dir)

        # save training args
        train_args = {k: str(v) for k, v in self.args.__dict__.copy().items()}
        with open(self.args.result_path / 'train_args.json', 'w', encoding='utf-8') as f:
            json.dump(train_args, f, ensure_ascii=False, indent=4)

        self.model = self._build_model()

    def _build_model(self):
        model = PrototypicalNetwork(self.args.img_channels, self.args.hidden_channels)
        model.cuda()
        return model

    def _get_data(self, mode='train'):
        if mode == 'train':
            data_path = self.args.train_csv_path
            way = self.args.way
            shot = self.args.shot
            query = self.args.query
            episodes = self.args.episodes
        else:
            data_path = self.args.val_csv_path
            way = self.args.val_way
            shot = self.args.val_shot
            query = self.args.val_query
            episodes = self.args.val_episodes

        data_set = ImageDataset(
            data_path=data_path,
            shot=shot,
            query=query,
            img_channels=self.args.img_channels,
            img_size=self.args.img_size,
        )
        sampler = EpisodicBatchSampler(n_classes=len(data_set), n_way=way, n_episodes=episodes)
        data_loader = DataLoader(data_set, shuffle=False, batch_sampler=sampler, num_workers=0)
        return data_loader

    def _select_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay_step)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_decay_step, gamma=0.1)
        return optimizer, scheduler

    def _select_criterion(self):
        criterion = nn.NLLLoss()
        return criterion

    def eval_model(self, epoch):
        self.model.eval()
        val_losses, accuracy, f_score = [AverageMeter() for i in range(3)]
        criterion = self._select_criterion()
        with torch.no_grad():
            val_loader = self._get_data(mode='val')
            for step, (x_support, x_query) in enumerate(tqdm(val_loader, desc='Evaluating model')):
                # data (n, x, c, w, h)
                x_support, x_query = x_support.cuda(), x_query.cuda()
                n = x_support.shape[0]
                q = x_query.shape[1]
                y_query = torch.arange(0, n, requires_grad=False).view(n, 1).expand(n, q).reshape(n * q, ).cuda()
                # infer
                output = self.model(x_support, x_query)
                # loss
                loss = criterion(output, y_query).item()
                val_losses.update(loss)
                # metrics of acc and f_score
                pred_ids = torch.argmax(output, dim=-1).cpu().numpy()
                y_query = y_query.cpu().numpy()
                acc, f_s = metrics(pred_ids, y_query)
                accuracy.update(acc)
                f_score.update(f_s)
        print(f'Epoch: {epoch} evaluation results: Val_loss: {val_losses.avg}, mAP: {accuracy.avg}, F_score: {f_score.avg}')
        return val_losses.avg, accuracy.avg, f_score.avg

    def train_one_epoch(self, optimizer, scheduler, epoch, total_epochs):
        self.model.train()
        train_losses = AverageMeter()
        criterion = self._select_criterion()
        train_loader = self._get_data(mode='train')
        epoch_iterator = tqdm(train_loader,
                              desc="Training [epoch X/X | episode X/X] (loss=X.X | lr=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, (x_support, x_query) in enumerate(epoch_iterator):
            # data (n, x, c, w, h)
            x_support, x_query = x_support.cuda(), x_query.cuda()
            n = x_support.shape[0]
            q = x_query.shape[1]
            y_query = torch.arange(0, n, requires_grad=False).view(n, 1).expand(n, q).reshape(n * q, ).cuda()

            # training one episode
            optimizer.zero_grad()
            output = self.model(x_support, x_query)
            loss = criterion(output, y_query)
            loss.backward()
            optimizer.step()

            # log
            train_losses.update(loss.item())
            epoch_iterator.set_description(
                "Training [epoch %d/%d | episode %d/%d] | (loss=%2.5f | lr=%f)" %
                (epoch, total_epochs, step + 1, len(epoch_iterator), loss.item(), scheduler.get_last_lr()[0])
            )
        scheduler.step()
        return train_losses.avg

    def train(self):
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f'[{start_time}] Start Training...')
        writer = SummaryWriter(self.args.tensorboard_dir)
        best_acc = 0
        early_stop_list = []
        optimizer, scheduler = self._select_optimizer()
        for epoch in range(self.args.epochs):
            train_loss = self.train_one_epoch(optimizer, scheduler, epoch, self.args.epochs)
            val_loss, accuracy, f_score = self.eval_model(epoch)

            # logging
            writer.add_scalars('Loss', {'TrainLoss': train_loss, 'ValLoss': val_loss}, epoch)
            writer.add_scalars('Metrics', {'mAP': accuracy, 'F_score': f_score}, epoch)

            # save checkpoint
            torch.save(self.model.state_dict(), self.args.checkpoints_dir / 'proto_last.pth')
            if best_acc < accuracy:
                torch.save(self.model.state_dict(), self.args.checkpoints_dir / 'proto_best.pth')
                best_acc = accuracy

            # early stop
            early_stop_list.append(val_loss)
            if len(early_stop_list)-np.argmin(early_stop_list) > self.args.patience:
                break
        writer.close()
        end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f'[{end_time}] End of all training. The highest accuracy is {best_acc}')
        return
