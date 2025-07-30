# Script for experiments
import openml
from dirty_cat import SuperVectorizer
import ott
from ott.geometry import pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation, IncrementalPCA, FastICA
import sys
import warnings
import re
import wandb
import numpy as np
from sklearn.impute import SimpleImputer
import jax.numpy as jnp

wandb.init(project="automated-clustering", entity="prabhant", group="distance_jnp")
wandb.config.rank = 6
wandb.config.datasuite = 'openmlcc-18'

data_cleaner = SuperVectorizer()
cc18 = openml.study.get_suite(99)

#arg = sys.argv[1]
#dataset1 = cc18.data[int(arg)]
d1 = openml.datasets.get_dataset(40923)
X_s, y_s, *_ = d1.get_data()
source_dataset = data_cleaner.fit_transform(X_s)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
source_dataset = imp.fit_transform(source_dataset)

distance = "gwlr"
preprocessing = "ica"
if type(source_dataset)!=np.ndarray:
    source_dataset = FastICA().fit_transform(source_dataset.toarray())
else:
    source_dataset = FastICA().fit_transform(source_dataset)
#geom_xx = pointcloud.PointCloud(source_dataset)
geom_xx = pointcloud.PointCloud(jnp.asarray(source_dataset))

costs=[]


for i in range(len(cc18.data)):
    if distance == 'gwlr':
        if preprocessing == 'ica':
            dataset2 = cc18.data[i]
            d2 = openml.datasets.get_dataset(dataset2)
            print(d2.name)
            print(d2.id)
            X_t, y_t, *_ = d2.get_data()
            target_dataset = data_cleaner.fit_transform(X_t)
            target_dataset = imp.fit_transform(target_dataset)
            if type(target_dataset)!=np.ndarray:
                target_dataset = FastICA().fit_transform(target_dataset.toarray())
            else:
                target_dataset = FastICA().fit_transform(target_dataset)
            #geom_yy = pointcloud.PointCloud(target_dataset)
            geom_yy = pointcloud.PointCloud(jnp.asarray(target_dataset))
            prob = ott.problems.quadratic.quadratic_problem.QuadraticProblem(geom_xx, geom_yy)
            solver = gromov_wasserstein.GromovWasserstein(rank=6)
            ot_gwlr = solver(prob)
            cost = ot_gwlr.costs[ot_gwlr.costs > 0][-1]
            costs.append(cost)
distance = min(costs)
dataset = cc18.data[costs.index(distance)]
wandb.log({"similar_dataset_id": dataset})
wandb.log({"distance": distance})
