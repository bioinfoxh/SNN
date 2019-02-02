#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 01:07:31 2018

@author: Hui
"""

import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt

from evaluation.evaluation_metrics import *
    
from evaluation.plot_confussion_matrices import plot_confussion_matrix_as_heatmap_for_cancer_data, plot_confussion_matrix_as_heatmap

#from gene_clustering.hierarchical_clustering import plot_dendogram

'''from epigenetic_data.embryo_development_data.embryo_development_data import \
    EmbryoDevelopmentData, EmbryoDevelopmentDataWithClusters, EmbryoDevelopmentDataWithSingleCluster '''
    
from cancer_data.cancer_data import CancerPatientsData, CancerPatientsDataWithModalities, CancerPatientsDataMethylationLevels, CancerPatientsDataExpressionLevels

from neural_network_models.multilayer_perceptron import MultilayerPerceptron

#from neural_network_models.recurrent_neural_network import RecurrentNeuralNetwork

from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork

#from evaluation.nested_cross_validation import nested_cross_validation_on_MLP, nested_cross_validation_on_SNN, nested_cross_validation_on_RNN
from evaluation.nested_cross_validation import nested_cross_validation_on_MLP, nested_cross_validation_on_SNN




########## get cancer data set

noise_mean=0 
noise_stddev=0
print ("Noise Characteristics")
print (noise_mean)
print (noise_stddev)

epigenetic_data = CancerPatientsData(num_folds=5, num_folds_hyperparameters_tuning=3)

epigenetic_data_with_clusters = CancerPatientsDataWithModalities(num_folds=5, num_folds_hyperparameters_tuning=3)

epigenetic_data_for_methylation = CancerPatientsDataMethylationLevels(num_folds=5, num_folds_hyperparameters_tuning=3)
 
epigenetic_data_for_expression = CancerPatientsDataExpressionLevels(num_folds=5, num_folds_hyperparameters_tuning=3)


##########

def evaluate_feed_forward_neural_network(epigenetic_data):

    print ("-------------------------------------------------------------------------------")
    print ("-------------------Evaluating Feed Forward Neural Network----------------------")
    print ("-------------------------------------------------------------------------------")

    input_data_size = epigenetic_data.input_size
    print ("input data size")
    print (input_data_size)
    output_size = epigenetic_data.output_size

    feed_forward_neural_network = MultilayerPerceptron(input_data_size, [256, 128, 64, 32], output_size)
    """feed_forward_neural_network = MultilayerPerceptron(128, [256, 128, 64, 32], output_size)"""

    #feed_forward_neural_network = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)
    
    confussion_matrix, ROC_points, performance_metrics, best_parameters = nested_cross_validation_on_MLP(feed_forward_neural_network, epigenetic_data)

    label_to_one_hot_encoding = epigenetic_data.label_to_one_hot_encoding
    #plot_confussion_matrix_as_heatmap(confussion_matrix, label_to_one_hot_encoding, 'Confusion Matrix for MLP')

    #plot_ROC_curves(ROC_points)

    return confussion_matrix, ROC_points, performance_metrics, best_parameters



    
mlp_confussion_matrix, mlp_ROC_points, mlp_performance_metrics, mlp_best_parameters = evaluate_feed_forward_neural_network(epigenetic_data)



#########################


def evaluate_superlayered_neural_network(epigenetic_data_with_clusters):

    print ("-------------------------------------------------------------------------------")
    print ("-------------------Evaluating Superlayered Neural Network----------------------")
    print ("-------------------------------------------------------------------------------")

    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size

    superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[128, 64, 32, 16], [128, 64, 32, 16]],
        [64, 16], output_size)

    """superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[512, 256, 128, 64], [512, 256, 128, 64]],
        [256, 64], output_size)"""

    """superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[512, 256, 128, 64], [512, 256, 128, 64]],
        [256, 64], output_size)"""

    confussion_matrix, ROC_points, performance_metrics = nested_cross_validation_on_SNN(
        superlayered_neural_network, epigenetic_data_with_clusters)

    label_to_one_hot_encoding = epigenetic_data_with_clusters.label_to_one_hot_encoding
    
    class_id_to_symbol_id = compute_class_id_to_class_symbol(label_to_one_hot_encoding)
    
    class_symbol_to_evaluation_matrix = compute_evaluation_metrics_for_each_class(confussion_matrix, class_id_to_symbol_id)

    #plot_confussion_matrix_as_heatmap(confussion_matrix, label_to_one_hot_encoding, 'Confusion Matrix for SNN')
    #plot_ROC_curves(ROC_points)

    return confussion_matrix, ROC_points, performance_metrics



snn_confussion_matrix, snn_ROC_points, snn_performance_metrics = evaluate_superlayered_neural_network(
        epigenetic_data_with_clusters)





