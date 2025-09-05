import torch
'''最大准确数/总数=最大准确率'''
features = torch.load("features.pth")
# qf = query features，查询集的特征矩阵，形状 [num_query, feature_dim]
# ql = query labels，查询集的标签 [num_query]
# gf = gallery features，图库特征 [num_gallery, feature_dim]
# gl = gallery labels [num_gallery]
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]
# qf.mm(gf.t()) = 矩阵乘法，计算每个 query 对 gallery 的相似度
# 假设 query 特征 [N_q, D]，gallery 特征 [N_g, D]，得到 scores [N_q, N_g]
# 分数越大 = 相似度越高（通常特征已经 L2 归一化，所以相当于余弦相似度）
scores = qf.mm(gf.t())
# topk(k, dim=1) 会在 dim=1 维度上取前 k 个最大值。
# 返回一个二元组 (values, indices)：
# [0] 是前 k 个最大值（value）。
# [1] 是对应的索引（index）。
# [:,0]
# 在 dim=1 方向上取第一个元素，也就是每行的最大值索引
res = scores.topk(5, dim=1)[1][:,0]
# .item() 会把这个 0维张量转成 Python 标量
# 否则如果你不 .item()，结果还是 tensor(123)
top1correct = gl[res].eq(ql).sum().item()

print("Acc top1:{:.3f}".format(top1correct/ql.size(0)))


