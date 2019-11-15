from Sim_measure import *
from CED import *
import sklearn.metrics
import pandas as pd
from Clustering import *
from Context_function import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

def compute_sequences(df):
    """
    :param df: Dataframe -- Dafaframe of explorations sequences and queries
    :return: List<List<Float>> * List<String> -- List of all explorations * List of names/Id of explorations
    """
    sequences = []  # All sequences
    seq_i = []
    sequences_name = []
    buff_name = df['ExplorationSID'][0]
    for index, row in df.iterrows():
        if (row['ExplorationSID'] != buff_name):
            if (len(seq_i) > 1):  # Filter sequences of size 1
                sequences_name += [buff_name]
                sequences += [seq_i]
            seq_i = []
            buff_name = row['ExplorationSID']
        seq_i += [[row[1], row[2], row[3], row[4], row[5], row[6], row[7]]]
    if (len(seq_i) > 1):
        sequences_name += [buff_name]
        sequences += [seq_i]
    return [sequences, sequences_name]



"""""""""""""""""
!!! MAIN HERE !!!
"""""""""""""""""

### /!\ Paramètre d'initialisation de CED  /!\ ###

f_k = gaussianOlap
sim = cosine
alpha = 0

FILE_PATH_OPERATIONS = "/DATA-operations/artificial-operations.csv"
FILE_PATH_CED_DIST_MATRIX = "/CED-matrices/ced-artificial.csv"
###

df = pd.read_csv(FILE_PATH_OPERATIONS, sep=";")
df = df.drop(["QuerySID"], axis=1)

### Compute explorations sequences from a file  ###
seq_and_names = compute_sequences(df)

sequences = seq_and_names[0]
sequences_name = seq_and_names[1]

###
# -> TWO SOLUTIONS
#
# 1) COMPUTE CED FROM SCRATCH
#
# 2) IMPORT CED DISTANCE MATRIX
###

####
# COMPUTE FROM SCRATCH CED
###

CED_matrix = np.zeros((len(sequences), len(sequences))) # Matrice des distance CED

### Computation of CED
"""
for i in range(len(sequences)) :
    for j in range(i+1, len(sequences)):
        CED_matrix[i, j] = CED(sequences[i], sequences[j], sim, f_k, alpha)
        CED_matrix[j, i] = CED_matrix[i, j]
        print(i, j)

print(CED_matrix)
"""
###

####
# IMPORT CED DISTANCE MATRIX
###
CED_matrix  = pd.read_csv(FILE_PATH_CED_DIST_MATRIX, sep=";").to_numpy()


## TO export CED matrix
# CED_csv = pd.DataFrame(data=CED_matrix, columns=sequences_name)


###
# Compute the hierarchical clustering
###

dists = squareform(CED_matrix)
linkage_matrix = linkage(dists, "ward")

cut = 3.5 # TO AUTOMATIZED !!!

dendrogram(linkage_matrix, color_threshold=cut, labels=sequences_name)
plt.show()

### Quality indicators
groupes_subset_cah = fcluster(linkage_matrix,t=cut, criterion='distance')
clusters = clusters_identity(groupes_subset_cah, sequences_name)
medoid_point = medoid(clusters, CED_matrix, sequences_name)
silhouette_score = sklearn.metrics.silhouette_score(CED_matrix, groupes_subset_cah, metric="precomputed")
print("Silhouette mean Score : ", silhouette_score)
print("Silhouette Score for each clusters : ",silhouette_cluster(clusters, CED_matrix, groupes_subset_cah))