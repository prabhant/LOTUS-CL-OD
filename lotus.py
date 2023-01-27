import gama
from gama import GamaClassifier
import pandas as pd
import numpy as np
import os
import sys
import re
import time
import json
import pickle
import warnings
import sklearn
import ott
from ott.geometry import pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation, IncrementalPCA, FastICA

#define a class
class LotusMetaData:
    #define a constructor
    # self.eval_function = None
    
    def __init__(self, dataset_list, eval_metric, time_budget, 
                 out=None, dataloader=None):
        self.dataset_list = dataset_list
        self.eval_metric = eval_metric
        self.time_budget = time_budget
        self.datasets, self.models, self.scores = [], [], []
        self.dataloader = dataloader
        self.out = out
    #define a method
    def get_dataset_list(self):
        return self.dataset_list
    
    def get_eval_metric(self):
        return self.eval_metric

    def create_lotus_metadata(self):

        for dataset in self.dataset_list:
            
            if dataloader != None:
              dataset = dataloader(dataset)
            model = GamaClassifier(max_total_time = self.time_budget)
            model.fit(dataset['X_train'], dataset['y_train'])
            model = model.model['0'].fit(dataset['X_train'])
            score = model.decision_function(dataset['X_test'])
            self.datasets.append(dataset)
            self.models.append(model)
            self.scores.append(score)
        if out == 'csv':
            df = pd.DataFrame({'datasets': self.datasets, 'models': self.models, 'scores': self.scores})
            df.to_csv('lotus_metadata.csv', index=False)
            print('lotus_metadata.csv is created')
        elif out == 'json':
            with open('lotus_metadata.json', 'w') as f:
                json.dump({'datasets': self.datasets, 'models': self.models, 'scores': self.scores}, f)
            print('lotus_metadata.json is created')
        elif out == 'pickle':
            with open('lotus_metadata.pickle', 'wb') as f:
                pickle.dump({'datasets': self.datasets, 'models': self.models, 'scores': self.scores}, f)
            print('lotus_metadata.pickle is created')
        else:
            print('No output file is created')
            return self.datasets, self.models, self.scores
        

class LotusModel:
    def __init__(self, new_dataset, meta_data_obj, distance, preprocessing):
        self.new_dataset = new_dataset
        self.meta_data_obj = meta_data_obj
        self.distance = distance
        self.preprocessing = preprocessing
    
    def find_model(self):
        # find the best model
        best_model = None
        best_score = None
        for i in range(len(datasets)):
            if self.distance == 'gwlr':
                if self.preprocessing == 'ica':
                    anchor_dataset = new_dataset
                    anchor_dataset = FastICA().fit_transform(anchor_dataset)
                    geom_xx = ointcloud.PointCloud(anchor_dataset)
                    costs = []
                    for dataset in meta_data_obj.datasets:
                        dataset = FastICA().fit_transform(dataset)
                        geom_yy = pointcloud.PointCloud(dataset)
                        prob = ott.problems.quadratic.QuadraticProblem(geom_xx, geom_yy)
                        solver = gromov_wasserstein.GromovWasserstein(rank=6)
                        ot_gwlr = solver(prob)
                        cost = ot_gwlr.costs[ot_gwlr.costs > 0][-1]
                        costs.append(cost)

        distance = min(costs)
        best_model = models.index(costs.index(distance))
        score = scores.index(costs.index(distance))
        dataset = datasets.index(costs.index(distance))
        return best_model, distance, score, dataset