# Import Statements
import warnings
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from simclr import simclr
from proj_similarity import calculate_normalized_similarity
from IS_single import IS_dataset, EEG_batch
from model import StackingEnsemble
from strime_score import *

warnings.filterwarnings("ignore")

# ========================
# Data Preparation
# ========================
dataset = IS_dataset()
edgefea_fc = dataset.edgefea_fc
edgefea_sp = dataset.edgefea_sp
edgefea_sp[np.isinf(edgefea_sp)] = np.nan

# Normalize edge feature matrices (sp)
for i in range(edgefea_sp.shape[0]):
    sub_matrix = edgefea_sp[i]
    sub_matrix -= np.nanmin(sub_matrix)
    sub_matrix /= np.nanmax(sub_matrix) - np.nanmin(sub_matrix)
    edgefea_sp[i] = sub_matrix
edgefea_sp = np.where(np.isnan(edgefea_sp), 2, edgefea_sp)

nodefea = dataset.nodefea
graph_label = dataset.graph_label
dataset_num_features = nodefea.shape[2]
sample_num = len(graph_label)
channel_num = 128

# ========================
# Training Configuration
# ========================
kernel_size = 2
stride = 2
epochs = 1
batch_size = 64
hidden_dim = 32
lrr = 0.01
num_gc_layers = 2
important_channels = [45, 46, 47, 48, 54, 55, 56, 57, 58, 59, 60, 61, 103, 104, 105, 106,
                      115, 116, 117, 118, 119, 120, 121, 122, 87, 88, 89, 90, 91, 94, 95,
                      96, 97, 98, 99, 100, 101, 107, 108]
device = torch.device('cuda')
print(f'Device: {device}\nLearning rate: {lrr}\nHidden dim: {hidden_dim}\nGC layers: {num_gc_layers}\n')

# ========================
# Training Loop
# ========================
for seed in range(100):
    acc_best = 0
    mean_best = 0
    accuracies = []

    print(f"\n=== Seed: {seed} ===")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = simclr(hidden_dim, num_gc_layers, channel_num=channel_num,
                   sample_num=batch_size, important_channels=important_channels,
                   rand_seed=seed).to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=lrr, weight_decay=0.001)
    model.train()

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}')
        for _ in range(15):
            # Random sampling of a batch
            index_batch = np.random.randint(0, len(graph_label), size=batch_size)
            batch = EEG_batch(nodefea, edgefea_fc, edgefea_sp, graph_label,
                              batch_size=len(index_batch), batch_index=index_batch,
                              sample_num=sample_num, channel_num=channel_num)

            optimizer.zero_grad()

            # Move data to device
            nodefea_ipt = batch.nodefea_ipt.to(device)
            edgefeafc_ipt = batch.edgefeafc_ipt.to(device)
            edgefeasp_ipt = batch.edgefeasp_ipt.to(device)
            graph_indicator = batch.graph_indicator.to(device)
            label = batch.label.to(device)

            # Generate views with various augmentations
            x1, nf1, idx1, attr1, b1, _, eig1, m1 = model(nodefea_ipt, edgefeafc_ipt, edgefeasp_ipt, graph_indicator, len(index_batch), aug=20)
            x2, nf2, idx2, attr2, b2, _, eig2, m2 = model(nodefea_ipt, edgefeafc_ipt, edgefeasp_ipt, graph_indicator, len(index_batch), aug=30)
            x_aug1, _, _, _, _, _, eig_aug1, m_aug1 = model(nodefea_ipt, edgefeafc_ipt, edgefeasp_ipt, graph_indicator, len(index_batch), aug=2)
            x_aug2, _, _, _, _, _, eig_aug2, m_aug2 = model(nodefea_ipt, edgefeafc_ipt, edgefeasp_ipt, graph_indicator, len(index_batch), aug=3)

            # Compute similarity-based weights
            weight_node = torch.tensor(calculate_normalized_similarity(torch.cat((eig2, eig_aug2), dim=1), 4, seed))
            weight_edge = torch.tensor(calculate_normalized_similarity(torch.cat((m2, m_aug2), dim=1), 4, seed))
            weight_emb = torch.tensor(calculate_normalized_similarity(torch.cat((x2, x_aug2), dim=1), 4, seed))
            weight = (weight_node + weight_edge + weight_emb) / 3
            weight.fill_diagonal_(0)

            # Compute loss
            loss_node = model.compute_loss(nf2, batch_size=batch_size)
            loss_edge = model.kop_loss()
            loss = model.loss_cal(x2, x_aug2, weight.to(device), loss_node, loss_edge)
            loss.backward()
            optimizer.step()

            # Evaluation
            emb1, y = model.encoder.get_embeddings(nf1, idx1, attr1, b1, label)
            emb2, _ = model.encoder.get_embeddings(nf2, idx2, attr2, b2, label)
            emb3 = torch.cat((eig2.cpu(), m2.cpu(), x2.cpu()), dim=1).detach().numpy()

            X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(
                emb1, emb2, emb3, y, test_size=0.3, random_state=seed)

            stacking_model = StackingEnsemble()
            stacking_model.train_svm_models(X1_train, X2_train, X3_train, y_train)
            stacking_model.train_stacking_model(X1_test, X2_test, X3_test, y_test)

            acc, pre, rec, f1, kpa, mat = stacking_model.evaluate_stacking_model(X1_test, X2_test, X3_test, y_test)
            accuracies.append(acc)

            if acc > acc_best:
                acc_best = acc

            mean_acc = sum(accuracies) / len(accuracies)
            print(f"mean_accuracy: {round(mean_acc, 4)}")
            if acc > acc_best:
                print(f"Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kpa:.4f}")
                conf_norm = mat.astype('float') / mat.sum(axis=1, keepdims=True)
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_norm, annot=True, fmt='.2f', cmap='Blues',
                            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Normalized Confusion Matrix')
                plt.show()

            if mean_acc > mean_best:
                mean_best = mean_acc


