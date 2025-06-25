import math
import random
import time
from copy import deepcopy
from typing import List, Union

import torch
import torch.optim as optim
import torch.nn.functional as F
from flatbuffers.builder import np

import dhg.data
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer,cooking_200
from dhg.models import HGNN, HGNNP, HNHN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

torch.cuda.empty_cache()

def create_e_list(incidence_matrix: np.ndarray) -> List[Union[List[int], int]]:
    e_list = []
    num_hyperedges = incidence_matrix.shape[1]

    for hyperedge_index in range(num_hyperedges):
        # 找到属于当前超边的所有节点的索引
        nodes_in_hyperedge = np.where(incidence_matrix[:, hyperedge_index] == 1)[0]

        # 如果只有一个节点，直接添加节点索引；如果有多个节点，添加节点索引列表
        if len(nodes_in_hyperedge) == 1:
            e_list.append(nodes_in_hyperedge[0])
        else:
            e_list.append(nodes_in_hyperedge.tolist())

    return e_list

def train(net, X, G, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, G)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, G, lbls, idx, test=False):
    net.eval()
    outs = net(X, G)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res


if __name__ == "__main__":
    set_seed(2022)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    data = Cora()
    # data = Pubmed()
    # data = Citeseer()
    # data = dhg.data.Cooking200
    X, lbl = data["features"], data["labels"]
    G = Graph(data["num_vertices"], data["edge_list"])
    HG = Hypergraph.from_graph_kHop(G, k=1)
    # HG = Hypergraph.from_feature_kNN(X,k=1)
    print(f"Hypergraph: {HG}")


    # att_rate = 2707 * 10858 * 0.1
    # #攻击比例
    # position_row  = random.randint(0, 2707) * att_rate
    # position_column = random.randint(0, 10858) * att_rate
    # H_dense = HG.H.to_dense().to(device)
    # #转换为稠密矩阵
    # H_new = H_dense
    # # hyperedge_features = HG.compute_hyperedge_features(X)
    # # print(hyperedge_features)

    H_dense = HG.H.to_dense().to(device)

    # 攻击比例
    att_rate = int(2708 * 2590 * 0.0011)
    print(HG.H)

    # 获取矩阵的行和列
    num_rows, num_cols = H_dense.shape

    x_range = (0, num_rows - 1)  # x坐标范围
    y_range = (0, num_cols - 1)  # y坐标范围

    coordinates = HG.generate_point(x_range, y_range, att_rate)
    print(coordinates.__len__())

    # H_new = HG.update_edge_node_matrix(coordinates,H_dense)
    H_new = HG.invert_matrix(H_dense,coordinates)
    # 将修改后的矩阵转移到设备上
    H_cpu = H_new.cpu()
    H = H_cpu.numpy()
    print("输出破坏后H")
    print(H)
    e_v_list = create_e_list(H)
    # print("新超边节点结构")
    # print(e_v_list)
    e_v = [e if isinstance(e, list) else [e] for e in e_v_list]
    hg = Hypergraph(num_v=2708, e_list=e_v, e_weight=[1.0] * 2590,
                    v_weight=[1.0] * 2708,
                    merge_op="mean", device=torch.device("cuda"))
    print("更新后超图关联矩阵")
    print(hg.H)

    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    net = HGNN(data["dim_features"], 16, data["num_classes"])
    # net = HNHN(data["dim_features"], 16, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    hg = hg.to(X.device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, hg, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, hg, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, hg, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
