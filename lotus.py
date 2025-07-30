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
        if self.out == 'csv':
            df = pd.DataFrame({'datasets': self.datasets, 'models': self.models, 'scores': self.scores})
            df.to_csv('lotus_metadata.csv', index=False)
            print('lotus_metadata.csv is created')
        elif self.out == 'json':
            with open('lotus_metadata.json', 'w') as f:
                json.dump({'datasets': self.datasets, 'models': self.models, 'scores': self.scores}, f)
            print('lotus_metadata.json is created')
        elif self.out == 'pickle':
            with open('lotus_metadata.pickle', 'wb') as f:
                pickle.dump({'datasets': self.datasets, 'models': self.models, 'scores': self.scores}, f)
            print('lotus_metadata.pickle is created')
        else:
            print('No output file is created')
            return self.datasets, self.models, self.scores
        

class LotusModel:
    def __init__(self, new_dataset, meta_data_obj, distance='gwlr', preprocessing='ica'):
        """
        Initialize LotusModel for finding the best model based on distance metrics.
        
        Args:
            new_dataset: The new dataset to find a model for
            meta_data_obj: LotusMetaData object containing pre-computed models
            distance: Distance metric to use ('gwlr' currently supported)
            preprocessing: Preprocessing method ('ica' currently supported)
        """
        self.new_dataset = new_dataset
        self.meta_data_obj = meta_data_obj
        self.distance = distance
        self.preprocessing = preprocessing
        
        # Initialize results
        self.best_model = None
        self.best_score = None
        self.best_dataset = None
        self.best_distance = None

    def _preprocess_data(self, data):
        """Apply preprocessing to the data."""
        if self.preprocessing == 'ica':
            return FastICA().fit_transform(data)
        else:
            raise ValueError(f"Unsupported preprocessing method: {self.preprocessing}")

    def _calculate_gwlr_distance(self, anchor_data, target_data):
        """Calculate Gromov-Wasserstein distance between two datasets."""
        geom_xx = pointcloud.PointCloud(anchor_data)
        geom_yy = pointcloud.PointCloud(target_data)
        prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy)
        solver = gromov_wasserstein.GromovWasserstein(rank=6)
        ot_gwlr = solver(prob)
        return ot_gwlr.costs[ot_gwlr.costs > 0][-1]

    def _calculate_distances(self):
        """Calculate distances between new dataset and all metadata datasets."""
        if self.distance != 'gwlr':
            raise ValueError(f"Unsupported distance metric: {self.distance}")
        
        # Preprocess anchor dataset
        anchor_dataset = self._preprocess_data(self.new_dataset)
        
        costs = []
        for dataset in self.meta_data_obj.datasets:
            # Preprocess target dataset
            target_data = self._preprocess_data(dataset['X_train'])
            
            # Calculate distance
            cost = self._calculate_gwlr_distance(anchor_dataset, target_data)
            costs.append(cost)
        
        return costs

    def find_model(self):
        """
        Find the best model based on distance metrics.
        
        Returns:
            tuple: (best_model, distance, score, dataset)
        """
        costs = self._calculate_distances()
        
        # Find minimum distance and corresponding model
        min_distance = min(costs)
        best_index = costs.index(min_distance)
        
        # Store results
        self.best_model = self.meta_data_obj.models[best_index]
        self.best_score = self.meta_data_obj.scores[best_index]
        self.best_dataset = self.meta_data_obj.datasets[best_index]
        self.best_distance = min_distance
        
        return self.best_model, self.best_distance, self.best_score, self.best_dataset

    def get_results(self):
        """Get the results of the model selection."""
        if self.best_model is None:
            raise RuntimeError("Call find_model() first")
        
        return {
            'model': self.best_model,
            'distance': self.best_distance,
            'score': self.best_score,
            'dataset': self.best_dataset
        }