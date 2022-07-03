import matplotlib.pyplot as plt
import numpy as np

import geomstats.backend as gs
import pandas as pd
import geomstats._backend.numpy as gs_np
# import geomstats._backend.numpy.assignment as gs_asign
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.kmeans import RiemannianKMeans
from geomstats.learning.online_kmeans import OnlineKMeans
from geomstats.geometry.hypersphere import Hypersphere
import pickle
from geomstats.geometry.riemannian_metric import RiemannianMetric
import torch
from geomstats.geometry.spd_matrices import SPDMatrices
from sklearn.preprocessing import StandardScaler

with open('/home/smorandv/DynamicalSystems/DynamicalSystems/running_scripts/single_exp/x_y.pkl', 'rb') as f:
    e = pickle.load(f)
    data_tr = e[0:4]
    # data_ts = e[4:-1]
    max_age = e[-1]
for dt in data_tr:
    dt.drop(['AVNN'], axis=1, inplace=True)
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
n = len(dt.columns)   # Integer representing the shape of the matrices: n x n (meaning features)0
res = pd.Series([])
for k, tag in enumerate(np.unique(dt.index)):
    if k == 10:
        break
    temp = dt.loc[tag]
    curr_age = temp.loc[temp['Age'] == '9']
    HRV_temp = curr_age.iloc[np.random.randint(0, curr_age.shape[0], 3)]
    res = pd.concat([res, HRV_temp], axis=0)
res = res.loc[:, 'SD1':'SD2']
res = res.values
scaler = StandardScaler()
res = scaler.fit_transform(res)
s = gs_np.from_numpy(res)
new_data = gs_np.zeros((res.shape[0], res.shape[1], res.shape[1]))
for i in range(res.shape[0]):
    new_data[i, :, :] = s[i:i+1, :].T@s[i:i+1, :]
manifold = SPDMatrices(n)
metric = manifold.metric
# metric = RiemannianMetric(n, signature='dist')  # not sure that n is the dimension of the manifold

K = 10  # num of mice
kmeans = RiemannianKMeans(metric, K, init='kmeans++')
# kmeans = OnlineKMeans(metric, K)

kmeans.fit(new_data)
labels = kmeans.predict(new_data)
centroids = kmeans.centroids

# sphere = Hypersphere(dim=2)
# cluster = sphere.random_von_mises_fisher(kappa=20, n_samples=140)
#
# SO3 = SpecialOrthogonal(3)
# rotation1 = SO3.random_uniform()
# rotation2 = SO3.random_uniform()
#
# cluster_1 = cluster @ rotation1
# cluster_2 = cluster @ rotation2
#
# fig = plt.figure(figsize=(15, 15))
# ax = visualization.plot(
#     cluster_1, space="S2", color="red", alpha=0.7, label="Data points 1 "
# )
# ax = visualization.plot(
#     cluster_2, space="S2", ax=ax, color="blue", alpha=0.7, label="Data points 2"
# )
# ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
# ax.legend()
# plt.show()
#
# manifold = Hypersphere(dim=2)
# metric = manifold.metric
#
# data = gs.concatenate((cluster_1, cluster_2), axis=0)
#
# kmeans = RiemannianKMeans(metric, 2, tol=1e-3, init_step_size=1.0)
# kmeans.fit(data)
# labels = kmeans.predict(data)
# centroids = kmeans.centroids
#
# fig = plt.figure(figsize=(15, 15))
# colors = ["red", "blue"]
#
# ax = visualization.plot(data, space="S2", marker=".", color="grey")
#
# for i in range(2):
#     ax = visualization.plot(
#         points=data[labels == i], ax=ax, space="S2", marker=".", color=colors[i]
#     )
#
# for i, c in enumerate(centroids):
#     ax = visualization.plot(c, ax=ax, space="S2", marker="*", s=2000, color=colors[i])
#
# ax.set_title("Kmeans on Hypersphere Manifold")
# ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
# plt.show()
#
#
