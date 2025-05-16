import os
import numpy as np
import torch


class ISDataset:

    def __init__(self, data_root="F:/Imagined Speech/Extract_feature/graph_dataset"):
        self.data_root = data_root
        self.edgefea_fc, self.edgefea_sp, self.nodefea, self.graph_indicator, self.graph_label = self._load_data()

    def _load_data(self):
        print("Loading dataset from:", self.data_root)
        edgefea_fc = np.load(os.path.join(self.data_root, "edgefeafc_is.npy"))
        edgefea_sp = np.load(os.path.join(self.data_root, "edgefeasp_is.npy"))
        nodefea = np.load(os.path.join(self.data_root, "nodefea_is.npy"))
        graph_indicator = np.load(os.path.join(self.data_root, "graph_indicator.npy"))
        graph_label = np.load(os.path.join(self.data_root, "label_is.npy"))
        return edgefea_fc, edgefea_sp, nodefea, graph_indicator, graph_label


class EEGGraphBatch:
    def __init__(
        self,
        nodefea: np.ndarray,
        edgefea_fc: np.ndarray,
        edgefea_sp: np.ndarray,
        label: np.ndarray,
        batch_size: int = 128,
        batch_index: list = [],
        sample_num: int = 2236,
        channel_num: int = 128
    ):
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.channel_num = channel_num

        nodefea = nodefea[batch_index, :, :]
        edgefea_fc = edgefea_fc[batch_index, :, :]
        edgefea_sp = edgefea_sp[batch_index, :, :]
        label = label[batch_index]

        graph_indicator = np.repeat(np.arange(batch_size), channel_num)

        nodefea_ipt, edgefeafc_ipt, edgefeasp_ipt = self._transform_data(nodefea, edgefea_fc, edgefea_sp)

        self.nodefea_ipt = torch.tensor(nodefea_ipt, dtype=torch.float32)
        self.edgefeafc_ipt = torch.tensor(edgefeafc_ipt, dtype=torch.float32)
        self.edgefeasp_ipt = torch.tensor(edgefeasp_ipt, dtype=torch.float32)
        self.graph_indicator = torch.tensor(graph_indicator, dtype=torch.long)
        self.label = torch.tensor(label, dtype=torch.long)

    def _transform_data(self, nodefea, edgefea_fc, edgefea_sp):
        sample_num, channel_num, feature_dim = nodefea.shape
        total_nodes = sample_num * channel_num

        nodefea_ipt = nodefea.reshape((total_nodes, feature_dim))

        edgefeafc_ipt = np.zeros((total_nodes, total_nodes))
        edgefeasp_ipt = np.zeros((total_nodes, total_nodes))
        for i in range(sample_num):
            start = i * channel_num
            end = (i + 1) * channel_num
            edgefeafc_ipt[start:end, start:end] = edgefea_fc[i]
            edgefeasp_ipt[start:end, start:end] = edgefea_sp[i]

        return nodefea_ipt, edgefeafc_ipt, edgefeasp_ipt
