#Unfinished runs only
# Experimetn maetadata sOpenml

import os
import pandas as pd
import wandb
import sys
import warnings
import re
from sklearn.pipeline import Pipeline
import openml
from gama import GamaCluster
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from dirty_cat import SuperVectorizer

wandb.init(project="automated-clustering", entity="prabhant", group="All_metadata_creation_gama_clean_full")
wandb.config.time = 3600
wandb.config.eval_time = 1200
wandb.config.datasuite = 'openml-clustering-94'

def read_list_from_file(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content
#clustering_list = read_list_from_file('openml_clustering.txt')
#selecting only 40 datasets
#clustering_list = [1137, 1162, 52, 381, 1123, 1134, 24, 1132, 40, 378, 1500, 1165, 1131, 1120, 1147, 1161, 382, 1158, 1511, 1127, 1038, 1144, 1145, 1042, 204, 1166, 1143, 1136, 49, 1138, 1485, 1460, 1139, 1135, 1141, 1499, 51, 1130, 1128, 1489]
#unfinished one
#clustering_list = [1489, 1485, 1038, 1120, 24, 1137, 1128, 1138, 1166, 1158, 1165, 1134, 1130, 1139, 1145, 1161, 40, 52, 51, 49, 1042, 382, 378, 381, 1499, 1460, 204, 1511, 1500, 1123, 1162, 1141, 1143, 1127, 1131, 1132, 1135, 1136, 1144, 1147, 1153, 1156, 1122, 1124, 1129, 1154, 1142, 1149, 1163, 1155, 1159, 1164, 1133, 1140, 1148, 1150, 1151, 1152, 1157, 1160, 1125, 1126, 1146, 488, 231, 565, 538, 540, 1432, 41465, 40976, 42997, 42110, 42111, 42112, 42113, 42464, 40589, 41545, 43542, 43618, 43577, 43562, 43704, 43973, 43653, 43657, 43658, 43712, 43451, 43515, 43388, 43804, 45087]
#clustering_list = [24, 40, 49, 51, 52, 204, 231, 378, 381, 382, 488, 538, 540, 565, 1038, 1042, 1120, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1432, 1460, 1485, 1489, 1499, 1500, 1511, 40589, 40976, 41465, 41545, 42110, 42111, 42112, 42113, 42464, 42997, 43388, 43451, 43515, 43542, 43562, 43577, 43618, 43653, 43657, 43658, 43704, 43712, 43804, 43973, 45087]
#unfinished-clean
clustering_list = [24, 51, 52, 204, 231, 378, 381, 382, 488, 540, 1038, 1042, 1122, 1123, 1124, 1125, 1126, 1128, 1129, 1130, 1131, 1132, 1134, 1136, 1138, 1139, 1140, 1141, 1142, 1145, 1146, 1148, 1149, 1150, 1154, 1155, 1158, 1160, 1161, 1162, 1163, 1165, 1166]
arg = sys.argv[1]



dataset_id = int(clustering_list[int(arg)])
print(dataset_id)
dataset_obj = openml.datasets.get_dataset(dataset_id, download_qualities=False)
dataset, y, *_ = dataset_obj.get_data(dataset_format="dataframe", target=dataset_obj.default_target_attribute)
#data_cleaner = SuperVectorizer()
#dataset = data_cleaner.fit_transform(dataset)
#dataset.columns = dataset.columns.astype(str)
print(dataset_obj)
wandb.log({"dataset_id": dataset_id})

automl = GamaCluster(max_total_time = int(wandb.config.time), store = "all", n_jobs = 1, scoring = "adjusted_mutual_info_score", verbosity=0, max_eval_time=int(wandb.config.eval_time))
automl.fit(dataset, y)
label_predictions = automl.predict(dataset)
wandb.log({"pipeline": str(automl.model)})
print(automl.model)
print("CH", calinski_harabasz_score(dataset,y))
wandb.log({"CH": calinski_harabasz_score(dataset,y)})
print("AMI:", adjusted_mutual_info_score(y, label_predictions))
wandb.log({"AMI": adjusted_mutual_info_score(y, label_predictions)})
print("ARI:", adjusted_rand_score(y, label_predictions))
wandb.log({"ARI": adjusted_rand_score(y, label_predictions)})
