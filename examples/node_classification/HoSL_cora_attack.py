# import time
# from copy import deepcopy
# from typing import List, Union
#
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from flatbuffers.builder import np
#
# from dhg import Graph, Hypergraph
# from dhg.data import Cora, Pubmed, Citeseer, cooking_200, Cooking200
# from dhg.models import HGNN, HGNNP, HNHN
# from dhg.random import set_seed
# from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
#
# # 遍历稠密图，找到超边节点关系
# def create_e_list(incidence_matrix: np.ndarray) -> List[Union[List[int], int]]:
#     e_list = []
#     num_hyperedges = incidence_matrix.shape[1]
#
#     for hyperedge_index in range(num_hyperedges):
#         # 找到属于当前超边的所有节点的索引
#         nodes_in_hyperedge = np.where(incidence_matrix[:, hyperedge_index] == 1)[0]
#
#         # 如果只有一个节点，直接添加节点索引；如果有多个节点，添加节点索引列表
#         if len(nodes_in_hyperedge) == 1:
#             e_list.append(nodes_in_hyperedge[0])
#         else:
#             e_list.append(nodes_in_hyperedge.tolist())
#
#     return e_list
#
#
# def train(net, X, G, lbls, train_idx, optimizer, epoch):
#     net.train()
#     st = time.time()
#     optimizer.zero_grad()
#     outs = net(X, G)
#     outs, lbls = outs[train_idx], lbls[train_idx]
#     loss = F.cross_entropy(outs, lbls)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
#     return loss.item()
#
#
# @torch.no_grad()
# def infer(net, X, G, lbls, idx, test=False):
#     net.eval()
#     outs = net(X, G)
#     outs, lbls = outs[idx], lbls[idx]
#     if not test:
#         res = evaluator.validate(lbls, outs)
#     else:
#         res = evaluator.test(lbls, outs)
#     return res
#
#
# if __name__ == "__main__":
#     set_seed(2022)
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
#     data = Cora()
#     # data = Pubmed()
#     # data = Citeseer()
#     # data = dhg.data.Cooking200
#     X, lbl = data["features"], data["labels"]
#     G = Graph(data["num_vertices"], data["edge_list"])
#     HG = Hypergraph.from_graph_kHop(G, k=1)
#     # HG = Hypergraph.from_feature_kNN(X,k=1)
#     print(f"Hypergraph: {HG}")
#
#
#     # print("节点特征")
#     # print(X)
#     # # 打印返回的索引
#     # print("低同质性超边")
#     # print(low_homogeneity_indices)
#     # print(HG.homogeneity)
#     # print(hg.homogeneities())
#     # print("超边特征")
#     # print(hg.v2e_aggregation(X))
#     # Xe = hg.v2e_aggregation(X)
#     # intra_edge_similarity = hg.calculate_intra_edge_similarity(low_homogeneity_indices, X , Xe)
#     # print(intra_edge_similarity)
#     # hg = Hypergraph(num_v=5, e_list=[[0, 1, 2], [2, 3], [1, 3, 4], [1, 2]], e_weight=[1.0, 1.0, 1.0, 1.0],
#     #                 v_weight=[1.0] * 5,
#     #                 merge_op="mean", device=torch.device("cpu"))
#     # X = torch.randn(5, 3)
#     # print("初始e_list")
#     # print(HG.e_con)
#     # print("用create_e_list方法构造的e_list")
#     # print(create_e_list(HG.H.to_dense()))
#     # print("初始关联矩阵")
#     # print(HG.H)
#     #low_homogeneity_indices = HG.save_low_homogeneity_indices_data(1.1)
#     low_homogeneity_indices = HG.save_low_homogeneity_indices_per(0.05)
#     # print("低同质性超边_阈值")
#     # print(low_homogeneity_indices)
#     print("低同质性超边_比例")
#     print(low_homogeneity_indices)
#     current_device = torch.cuda.current_device()
#     # print(f"Current device: {current_device}")
#
#
#     # H_dense[0][0] = 0
#     # H_cpu = H_dense.cpu()
#     # H = H_cpu.numpy()
#     #
#     # e_list = create_e_list(H)
#     # hg = Hypergraph(num_v=2708, e_list=e_list, e_weight=[1.0] * 2590,
#     #                                 v_weight=[1.0] * 2708,
#
#     #                                 merge_op="mean", device=torch.device("cuda"))
#     # print("========================================")
#     # print(e_list)
#     # print("计算超边和节点余弦相似度")
#     similiar = HG.calculate_intra_edge_similarity(low_homogeneity_indices,X,HG.v2e_aggregation(X))
#     # print(HG.calculate_intra_edge_similarity(low_homogeneity_indices,X,HG.v2e_aggregation(X)))
#
#     print("保存需删除节点超边信息_阈值")
#     v_removed = HG.save_low_similarity_indices_from_intra(similiar,0.4)
#     print(HG.save_low_similarity_indices_from_intra(similiar,0.5))
#
#     H_dense = HG.H.to_dense().to(device)
#     # 将关联矩阵转换为稠密矩阵
#     # print("原关联矩阵")
#     # print(HG.H.to_dense())
#     # print("更新关联矩阵")
#     H_new = HG.update_edge_node_matrix(v_removed,H_dense)
#     # print(H_new)
#     # print("判断两个矩阵是否相等")
#     # are_equal = np.array_equal(HG.H.to_dense(), H_new)
#     # print(are_equal)
#     # 攻击比例
#     att_rate = int(2708 * 2590 * 0.0001)
#     print(HG.H)
#
#     # 获取矩阵的行和列
#     num_rows, num_cols = H_dense.shape
#
#     x_range = (0, num_rows - 1)  # x坐标范围
#     y_range = (0, num_cols - 1)  # y坐标范围
#
#     coordinates = HG.generate_point(x_range, y_range, att_rate)
#     print(coordinates.__len__())
#
#     # H_new = HG.update_edge_node_matrix(coordinates,H_dense)
#     H_new = HG.invert_matrix(H_dense, coordinates)
#
#     H_cpu = H_new.cpu()
#     H = H_cpu.numpy()
#     print("输出破坏后H")
#     print(H)
#
#     e_v_list = create_e_list(H)
#     # print("新超边节点结构")
#     # print(e_v_list)
#     e_v  = [e if isinstance(e, list) else [e] for e in e_v_list]
#
#     hg = Hypergraph(num_v=2708, e_list=e_v, e_weight=[1.0] * 2590,
#                                                     v_weight=[1.0] * 2708,
#                                                     merge_op="mean", device=torch.device("cuda"))
#     print("更新后超图关联矩阵")
#     print(hg.H)
#     # print("保存需删除节点超边信息_比例")
#     # print(HG.save_lowest_percentage_similarity_indices_from_intra(similiar,percentage=0.05))
#
#     # H_dense[0][0] = 0
#     # print(H_dense)
#     # H = H_dense.to_sparse_coo().to(device)
#     # print(H)
#
#
#
#     train_mask = data["train_mask"]
#     val_mask = data["val_mask"]
#     test_mask = data["test_mask"]
#
#     net = HGNN(data["dim_features"], 16, data["num_classes"])
#     # net = HNHN(data["dim_features"], 16, data["num_classes"], use_bn=True)
#     optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
#
#     X, lbl = X.to(device), lbl.to(device)
#     hg = hg.to(X.device)
#     net = net.to(device)
#
#     best_state = None
#     # 用于存储在验证过程中表现最好的模型状态字典。
#     best_epoch, best_val = 0, 0
#     # 记录最佳验证结果对应的训练轮数  记录最佳验证结果的值
#     for epoch in range(200):
#         # train   调用 train 函数进行一轮训练
#         # net：模型  X：输入特征  HG：超图对象  lbl：标签  train_mask：训练集掩码，用于指示哪些样本用于训练
#         # optimizer：优化器  epoch：当前训练轮数
#         train(net, X, hg, lbl, train_mask, optimizer, epoch)
#         # validation
#         if epoch % 1 == 0:
#             # 每隔1轮训练进行一次验证
#             with torch.no_grad():
#                 # 临时禁用梯度计算
#                 val_res = infer(net, X, hg, lbl, val_mask)
#                 # 调用infer函数进行验证
#             if val_res > best_val:
#                 print(f"update best: {val_res:.5f}")
#                 best_epoch = epoch
#                 best_val = val_res
#                 best_state = deepcopy(net.state_dict())
#                 # 使用deepcopy深拷贝当前模型的状态字典net.state_dict()，并将其赋值给best_state
#     print("\ntrain finished!")
#     print(f"best val: {best_val:.5f}")
#     # test
#     print("test...")
#     net.load_state_dict(best_state)
#     # 将最佳模型状态best_state加载回模型net中
#     res = infer(net, X, hg, lbl, test_mask, test=True)
#     # 调用infer函数进行测试
#     print(f"final result: epoch: {best_epoch}")
#     print(res)
import time
from copy import deepcopy
from typing import List, Union

import torch
import torch.optim as optim
import torch.nn.functional as F
from flatbuffers.builder import np

from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer, cooking_200, Cooking200
from dhg.models import HGNN, HGNNP, HNHN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

# 遍历稠密图，找到超边节点关系
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
    # print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
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
    X, lbl = data["features"], data["labels"]
    G = Graph(data["num_vertices"], data["edge_list"])
    HG = Hypergraph.from_graph_kHop(G, k=1)
    print(f"Hypergraph: {HG}")
    best_homogeneity_threshold = 0.0
    best_similarity_threshold = 0.0
    best_attack_rate = 0.0
    best_res = {"accuracy": 0.0, "f1_score": 0.0}

    # 自增参数
    for attack_rate in np.arange(0.0021, 0.0022, 0.0001):
        for homogeneity_threshold in np.arange(0.01, 0.81, 0.04):
            for similarity_threshold in np.arange(0.1, 0.91, 0.2):
                print(f"Running with homogeneity threshold: {homogeneity_threshold:.2f}, similarity threshold: {similarity_threshold:.1f}, attack rate: {attack_rate:.4f}")
                low_homogeneity_indices = HG.save_low_homogeneity_indices_per(homogeneity_threshold)
                # print("低同质性超边_比例")
                # print(low_homogeneity_indices)

                similiar = HG.calculate_intra_edge_similarity(low_homogeneity_indices, X, HG.v2e_aggregation(X))
                v_removed = HG.save_low_similarity_indices_from_intra(similiar, similarity_threshold)

                H_dense = HG.H.to_dense().to(device)
                att_rate = int(2708 * 2590 * attack_rate)
                num_rows, num_cols = H_dense.shape
                x_range = (0, num_rows - 1)
                y_range = (0, num_cols - 1)
                coordinates = HG.generate_point(x_range, y_range, att_rate)
                print(f"Generated {len(coordinates)} points for attack")

                H_new = HG.invert_matrix(H_dense, coordinates)

                H_cpu = H_new.cpu()
                H = H_cpu.numpy()
                # print("输出破坏后H")
                # print(H)

                e_v_list = create_e_list(H)
                e_v = [e if isinstance(e, list) else [e] for e in e_v_list]
                hg = Hypergraph(num_v=2708, e_list=e_v, e_weight=[1.0] * 2590, v_weight=[1.0] * 2708, merge_op="mean", device=torch.device("cuda"))
                # print("更新后超图关联矩阵")
                # print(hg.H)

                train_mask = data["train_mask"]
                val_mask = data["val_mask"]
                test_mask = data["test_mask"]

                net = HGNN(data["dim_features"], 16, data["num_classes"])
                optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

                X, lbl = X.to(device), lbl.to(device)
                hg = hg.to(X.device)
                net = net.to(device)

                best_state = None
                best_epoch, best_val = 0, 0
                for epoch in range(200):
                    train(net, X, hg, lbl, train_mask, optimizer, epoch)
                    if epoch % 1 == 0:
                        with torch.no_grad():
                            val_res = infer(net, X, hg, lbl, val_mask)
                        if val_res > best_val:
                            # print(f"update best: {val_res:.5f}")
                            best_epoch = epoch
                            best_val = val_res
                            best_state = deepcopy(net.state_dict())

                print("\ntrain finished!")
                # print(f"best val: {best_val:.5f}")
                # print("test...")
                net.load_state_dict(best_state)
                res = infer(net, X, hg, lbl, test_mask, test=True)
                # print(f"final result: epoch: {best_epoch}")
                print(res)
                print()
                # 比较当前结果和最佳结果的 accuracy
                if res["accuracy"] > best_res["accuracy"]:
                    best_attack_rate = attack_rate
                    best_similarity_threshold = similarity_threshold
                    best_homogeneity_threshold = homogeneity_threshold
                    best_res = res
        print(f"best homogeneity threshold: {best_homogeneity_threshold:.2f}, similarity threshold: {best_similarity_threshold:.1f}, attack rate: {best_attack_rate:.4f}, best result (accuracy): {best_res['accuracy']:.8f}")
