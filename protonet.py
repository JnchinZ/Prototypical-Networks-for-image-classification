import torch.nn as nn
import torch.nn.functional as F

from utils import cosine_similarity, euclidean_dist_similarity


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvEncoder(nn.Module):
    def __init__(self, input_img_channels, hidden_channels):
        super(ConvEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.input_img_channels = input_img_channels

        self.encoder = nn.Sequential(
            self.conv_block(self.input_img_channels, self.hidden_channels),
            self.conv_block(self.hidden_channels, self.hidden_channels),
            self.conv_block(self.hidden_channels, self.hidden_channels),
            self.conv_block(self.hidden_channels, self.hidden_channels),
            Flatten()
        )

    def forward(self, x):
        raw_shape = x.shape
        x = x.view(-1, *raw_shape[-3:])
        x = self.encoder(x)
        x = x.view(*raw_shape[:-3], -1)
        return x

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


class PrototypicalNetwork(nn.Module):
    def __init__(self, input_img_channels, hidden_channels):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = ConvEncoder(input_img_channels, hidden_channels)

    def forward(self, x_support, x_query):
        """
        infer an n-way k-shot task
        :param x_support: (n, k, c, w, h)
        :param x_query: (n, q, c, w, h) or (q, c, w, h)
        :return: (q, n)
        """
        x_proto = self.encoder(x_support)  # (n, k, embed_dim)
        x_proto = x_proto.mean(1)  # (n, embed_dim)
        x_q = self.encoder(x_query)  # (n, q, embed_dim)
        x_q = x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)

        sim_result = self.similarity(x_q, x_proto)  # (n*q, n)

        log_p_y = F.log_softmax(sim_result, dim=1)

        return log_p_y  # (n*q, n)

    @staticmethod
    def similarity(a, b, sim_type='cosine'):
        methods = {'euclidean': euclidean_dist_similarity, 'cosine': cosine_similarity}
        assert sim_type in methods.keys(), f'type must be in {methods.keys()}'
        return methods[sim_type](a, b)  # 值越大相似度越高
