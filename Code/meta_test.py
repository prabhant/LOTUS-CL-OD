"""
File: meta_test.py
File to test the meta model on the meta test set.
"""
import numpy as np

from sklearn.cluster import (KMeans,MiniBatchKMeans,
                             SpectralClustering,
                             MeanShift,
                             AgglomerativeClustering,
                             DBSCAN,
                             OPTICS,
                             Birch,
                             FeatureAgglomeration)

from sklearn.preprocessing import (MaxAbsScaler,
                                   MinMaxScaler,
                                   RobustScaler,
                                   Normalizer,
                                   PolynomialFeatures)

from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from dirty_cat import SuperVectorizer
import openml
import wandb
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
SuperVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

wandb.init(project="automated-clustering", entity="prabhant", group="meta-testing-experiment-clean")
wandb.config.datasuite = 'clustering-openml-59'

df = pd.read_csv('metadata-clean.csv')
#df = df[['AMI', 'ARI', 'dataset_id', 'pipeline']]
distances = pd.read_csv('dist-gama-sucess-clean.csv')
#distances = distances[['dataset_id', 'distance', 'similar_dataset_id']]
arg = sys.argv[1]

#clustering_list = [1148, 44072, 1124, 1038, 42110, 488, 44087, 1156, 1127, 231, 49, 44075, 44125, 42113, 42464, 1150, 45087, 1500, 44127, 1120, 1489, 1499, 44070, 382, 1123, 1159, 1145, 1511, 43973, 43971, 1140, 1141, 1131, 42112, 1125, 44118, 1129, 1154, 1147, 1158, 44116, 540, 204, 378, 1155, 51, 42111, 1152, 40, 1149, 1122, 1136, 381, 1165, 1151, 24, 1485, 1460, 1042, 52, 44073, 1432, 1137, 1126]
clustering_list =list(distances['dataset_id'])
meta_test_dataset_id = int(clustering_list[int(arg)])
#clustering_list.remove(meta_test_dataset_id)
wandb.log({'dataset_id': meta_test_dataset_id})
meta_test_dataset = openml.datasets.get_dataset(meta_test_dataset_id)
meta_test_dataset, y, *_ = meta_test_dataset.get_data(dataset_format="dataframe", target=meta_test_dataset.default_target_attribute)
similar_dataset = distances[distances['dataset_id'] == meta_test_dataset_id]
similar_dataset_id = similar_dataset['similar_dataset_id'].values[0]
#trying to encode it according to gama
from gama.utilities.preprocessing import basic_encoding
enc_pipeline = basic_encoding(meta_test_dataset, True)
meta_test_dataset = enc_pipeline[1].transform(meta_test_dataset)
##
#data_cleaner = SuperVectorizer()
#meta_test_dataset = data_cleaner.fit_transform(meta_test_dataset)
optimal_pipeline = df[df['dataset_id'] == similar_dataset_id]['pipeline'].values[0]
wandb.log({'optimal_pipeline': optimal_pipeline})
print(optimal_pipeline)
optimal_pipeline = eval(optimal_pipeline)
optimal_pipeline.fit(meta_test_dataset)
ami, ari = list(), list()
for i in range(5):
    label_predictions = optimal_pipeline.fit_predict(meta_test_dataset)
    print(f"AMI repeat {i}:", adjusted_mutual_info_score(y, label_predictions))
#    wandb.log({f"AMI_r_{i}": adjusted_mutual_info_score(y, label_predictions)})
    print(f"ARI repeat {i}:", adjusted_rand_score(y, label_predictions))
#    wandb.log({f"ARI_r_{i}": adjusted_rand_score(y, label_predictions)})
    ami.append(adjusted_mutual_info_score(y, label_predictions))
    ari.append(adjusted_rand_score(y, label_predictions))
#np.mean(ami), np.mean(ari)
wandb.log({'Mean_AMI': np.mean(ami), 'Mean_ARI': np.mean(ari)})
wandb.log({'AMI_max': np.amax(ami), 'ARI_max': np.amax(ari)})
# Now baselines
baseline_models = [KMeans,DBSCAN,
                             AgglomerativeClustering,
                             DBSCAN,
                             OPTICS,
                             Birch,MiniBatchKMeans]
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
meta_test_dataset=imp.fit_transform(meta_test_dataset)
for model_ in baseline_models:
    ami_, ari_ = list(), list()
    m = model_()
    print(model_)
    for i in range(5):
        try:
            preds = m.fit_predict(meta_test_dataset)
        except TypeError:
            preds = m.fit_predict(imputed_dataset.toarray())
 #       wandb.log({f"AMI_{model_.__name__}_r_{i}": adjusted_mutual_info_score(y, preds)})
 #       wandb.log({f"ARI_{model_.__name__}_r_{i}": adjusted_rand_score(y, preds)})
        ami_.append(adjusted_mutual_info_score(y, preds))
        ari_.append(adjusted_rand_score(y, preds))
    wandb.log({'Mean_AMI_'+model_.__name__: np.mean(ami_), 'Mean_ARI_'+model_.__name__: np.mean(ari_)})
    wandb.log({'AMI_max_'+model_.__name__: np.amax(ami_), 'ARI_max_'+model_.__name__: np.amax(ari_)})
