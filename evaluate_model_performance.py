#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 18:02:49 2018

@author: Hui
"""

import math
import numpy as np
import tensorflow as tf
import json

#from evaluation.evaluation_metrics import *

from evaluation.evaluation_metrics import paired_t_test_binary_classification, compute_average_performance_metrics_for_binary_classification, \
    compute_class_id_to_class_symbol, compute_evaluation_metrics_for_each_class, \
    compute_performance_metrics_for_multiclass_classification, plot_mean_ROC_curves, paired_t_test_multiclass_classification, \
    plot_ROC_curves, plot_mean_ROC_curves_for_two_models
    
from evaluation.plot_confussion_matrices import plot_confussion_matrix_as_heatmap_for_cancer_data, plot_confussion_matrix_as_heatmap

#from gene_clustering.hierarchical_clustering import plot_dendogram
from neural_network_models.multilayer_perceptron import MultilayerPerceptron 
from neural_network_models.superlayered_neural_network_CL1 import SuperlayeredNeuralNetwork as SuperlayeredNeuralNetwork_CL1
from neural_network_models.superlayered_neural_network_CL2 import SuperlayeredNeuralNetwork as SuperlayeredNeuralNetwork_CL2
from neural_network_models.superlayered_neural_network_CL3 import SuperlayeredNeuralNetwork as SuperlayeredNeuralNetwork_CL3
from neural_network_models.superlayered_neural_network_NCL import SuperlayeredNeuralNetwork as SuperlayeredNeuralNetwork_NCL

#from neural_network_models.superlayered_neural_network_LRdecay import SuperlayeredNeuralNetwork as SuperlayeredNeuralNetworkLRdecay

#from evaluation.nested_cross_validation import nested_cross_validation_on_MLP, nested_cross_validation_on_SNN

from cancer_data.cancer_data import CancerPatientsData, CancerPatientsDataWithModalities, \
    CancerPatientsDataMethylationLevels, CancerPatientsDataExpressionLevels



########## get cancer data

epigenetic_data = CancerPatientsData(num_folds=5, num_folds_hyperparameters_tuning=3)

epigenetic_data_with_clusters = CancerPatientsDataWithModalities(num_folds=5, num_folds_hyperparameters_tuning=3)

epigenetic_data_for_methylation = CancerPatientsDataMethylationLevels(num_folds=5, num_folds_hyperparameters_tuning=3)

epigenetic_data_for_expression = CancerPatientsDataExpressionLevels(num_folds=5, num_folds_hyperparameters_tuning=3)



############ functions for nested cross-validation evaluation  
    

def nested_cross_validation_on_MLP(network, epigenetic_data):

    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()
    output_size = epigenetic_data.output_size
    keys = k_fold_datasets.keys()

    confussion_matrix = np.zeros(shape=(output_size, output_size))
    validation_accuracy_list = list()
    ROC_points = dict()

    class_id_to_class_symbol = compute_class_id_to_class_symbol(epigenetic_data.label_to_one_hot_encoding)
    performance_metrics = dict()

    """ Outer cross-validation """

    #best_parameters = dict()

    for key in keys:
        print ("key number" + str(key))

#        learning_rate, weight_decay, keep_probability = choose_hyperparameters(
#           network, k_fold_datasets_hyperparameters_tuning[key])

        #best_parameters[key] = [learning_rate, weight_decay, keep_probability]
        
        learning_rate = 0.01
        weight_decay = 0.05
        keep_probability = 0.9

        print ("Learning rate" + str(learning_rate))
        print ("Weight decay" + str(weight_decay))
        print ("Keep probability" + str(keep_probability))

        training_dataset = k_fold_datasets[key]["training_dataset"]

        print (len(training_dataset["training_data"]))
        print (len(training_dataset["training_data"][0]))

        validation_dataset = k_fold_datasets[key]["validation_dataset"]
        print (len(validation_dataset["validation_data"]))

        validation_accuracy, ffnn_confussion_matrix, MLP_ROC_points = network.train_and_evaluate(
            training_dataset, validation_dataset,
            learning_rate, weight_decay, keep_probability)

        performance_metrics[key] = compute_evaluation_metrics_for_each_class(
            ffnn_confussion_matrix, class_id_to_class_symbol)

        print (performance_metrics[key])

        """micro_average[key] = compute_micro_average(performance_metrics[key])
        macro_average[key] = compute_macro_average(performance_metrics[key])

        micro_average[key]['accuracy'] = validation_accuracy
        macro_average[key]['accuracy'] = validation_accuracy"""

        print (ffnn_confussion_matrix)
        validation_accuracy_list.append(validation_accuracy)
        confussion_matrix = np.add(confussion_matrix, ffnn_confussion_matrix)
        ROC_points[key] = MLP_ROC_points

    average_validation_accuracy = np.mean(validation_accuracy_list)
    average_performance_metrics = compute_average_performance_metrics_for_binary_classification(performance_metrics)

    print(average_performance_metrics)
    print(average_validation_accuracy)

    """performance_metrics['micro'] = micro_average
    performance_metrics['macro'] = macro_average

    print "Micro"
    average_micro = compute_performance_metrics_for_multiclass_classification(micro_average)
    print "Macro"
    average_macro = compute_performance_metrics_for_multiclass_classification(macro_average)"""

    return confussion_matrix, ROC_points, performance_metrics, average_validation_accuracy, average_performance_metrics 






def nested_cross_validation_on_SNN(network, epigenetic_data_with_clusters):

    k_fold_datasets_with_clusters, k_fold_datasets_hyperparameters_tuning = \
        epigenetic_data_with_clusters.get_k_fold_datasets()

    output_size = epigenetic_data_with_clusters.output_size

    keys = k_fold_datasets_with_clusters.keys()

    confussion_matrix = np.zeros(shape=(output_size, output_size))
    validation_accuracy_list = list()
    ROC_points = dict()

    class_id_to_class_symbol = compute_class_id_to_class_symbol(epigenetic_data_with_clusters.label_to_one_hot_encoding)
    performance_metrics = dict()

    """ Outer cross-validation """

    for key in keys:

        """ Inner cross-validation """
        #learning_rate, weight_decay, keep_probability = choose_hyperparameters(
            #network, k_fold_datasets_hyperparameters_tuning[key])

        learning_rate = 0.01
        weight_decay = 0.05
        keep_probability = 0.9

        print(learning_rate)
        print(weight_decay)
        print(keep_probability)

        training_dataset = k_fold_datasets_with_clusters[key]["training_dataset"]
        validation_dataset = k_fold_datasets_with_clusters[key]["validation_dataset"]

        validation_accuracy, snn_confussion_matrix, snn_ROC_points = network.train_and_evaluate(
            training_dataset, validation_dataset, learning_rate, weight_decay, keep_probability)

        print(snn_confussion_matrix)

        performance_metrics[key] = compute_evaluation_metrics_for_each_class(
            snn_confussion_matrix, class_id_to_class_symbol)

        print(performance_metrics[key])

        validation_accuracy_list.append(validation_accuracy)
        confussion_matrix = np.add(confussion_matrix, snn_confussion_matrix)
        ROC_points[key] = snn_ROC_points

    average_validation_accuracy = np.mean(validation_accuracy_list)
    average_performance_metrics = compute_average_performance_metrics_for_binary_classification(performance_metrics)

    print(average_performance_metrics)
    print(average_validation_accuracy)

    return confussion_matrix, ROC_points, performance_metrics, average_validation_accuracy, average_performance_metrics





#######################################
#######################################
    

print ("-------------------------------------------------------------------------------")
print ("-----------------------------Evaluating MLP------------------------------------")
print ("-------------------------------------------------------------------------------")

input_data_size = epigenetic_data.input_size
output_size = epigenetic_data.output_size

print("input_data_size:")
print(input_data_size)
print("output_size:")
print(output_size)

MLP = MultilayerPerceptron(input_data_size, [256, 128, 64, 32], output_size)

confussion_matrix_MLP, \
ROC_points_MLP, \
performance_metrics_MLP, \
avg_validation_acc_MLP, \
avg_performance_MLP = nested_cross_validation_on_MLP(MLP, epigenetic_data)

jsObj = json.dumps(performance_metrics_MLP)  
fileObj = open('body_performance_metrics_MLP.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close() 

jsObj = json.dumps(avg_performance_MLP)  
fileObj = open('body_avg_performance_MLP.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close() 






print ("-------------------------------------------------------------------------------")
print ("-----------------------------Evaluating MLP_exp--------------------------------")
print ("-------------------------------------------------------------------------------")

input_data_size = epigenetic_data_for_expression.input_size
output_size = epigenetic_data_for_expression.output_size

print("input_data_size:")
print(input_data_size)
print("output_size:")
print(output_size)

MLP_exp = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)

confussion_matrix_MLP_exp, \
ROC_points_MLP_exp, \
performance_metrics_MLP_exp, \
avg_validation_acc_MLP_exp, \
avg_performance_MLP_exp = nested_cross_validation_on_MLP(MLP_exp, epigenetic_data_for_expression)

jsObj = json.dumps(performance_metrics_MLP_exp)  
fileObj = open('body_performance_metrics_MLP_exp.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close() 

jsObj = json.dumps(avg_performance_MLP_exp)  
fileObj = open('body_avg_performance_MLP_exp.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close() 




print ("-------------------------------------------------------------------------------")
print ("-----------------------------Evaluating MLP_meth-------------------------------")
print ("-------------------------------------------------------------------------------")

input_data_size = epigenetic_data_for_methylation.input_size
output_size = epigenetic_data_for_methylation.output_size

print("input_data_size:")
print(input_data_size)
print("output_size:")
print(output_size)

MLP_meth = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)

confussion_matrix_MLP_meth, \
ROC_points_MLP_meth, \
performance_metrics_MLP_meth, \
avg_validation_acc_MLP_meth, \
avg_performance_MLP_meth = nested_cross_validation_on_MLP(MLP_meth, epigenetic_data_for_methylation)

jsObj = json.dumps(performance_metrics_MLP_meth)  
fileObj = open('body_performance_metrics_MLP_meth.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close() 

jsObj = json.dumps(avg_performance_MLP_meth)  
fileObj = open('body_avg_performance_MLP_meth.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close()








print ("-------------------------------------------------------------------------------")
print ("-----------------------------Evaluating SNN_CL1--------------------------------")
print ("-------------------------------------------------------------------------------")

clusters_size = epigenetic_data_with_clusters.clusters_size
output_size = epigenetic_data_with_clusters.output_size

print("clusters_size:")
print(clusters_size)
print("output_size:")
print(output_size)

SNN_CL1 = SuperlayeredNeuralNetwork_CL1(
    [clusters_size[0], clusters_size[1]],
    [[128, 64, 32, 16], [128, 64, 32, 16]],
    [64, 16], output_size)

confussion_matrix_SNN_CL1, \
ROC_points_SNN_CL1, \
performance_metrics_SNN_CL1, \
avg_validation_acc_SNN_CL1, \
avg_performance_SNN_CL1 = nested_cross_validation_on_SNN(SNN_CL1, epigenetic_data_with_clusters)

jsObj = json.dumps(performance_metrics_SNN_CL1)  
fileObj = open('body_performance_metrics_SNN_CL1.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close() 

jsObj = json.dumps(avg_performance_SNN_CL1)  
fileObj = open('body_avg_performance_SNN_CL1.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close() 


#def evaluate_superlayered_neural_network(epigenetic_data_with_clusters):

print ("-------------------------------------------------------------------------------")
print ("-----------------------------Evaluating SNN_CL2--------------------------------")
print ("-------------------------------------------------------------------------------")

clusters_size = epigenetic_data_with_clusters.clusters_size
output_size = epigenetic_data_with_clusters.output_size

print("clusters_size:")
print(clusters_size)
print("output_size:")
print(output_size)

SNN_CL2 = SuperlayeredNeuralNetwork_CL2(
    [clusters_size[0], clusters_size[1]],
    [[128, 64, 32, 16], [128, 64, 32, 16]],
    [64, 16], output_size)

confussion_matrix_SNN_CL2, \
ROC_points_SNN_CL2, \
performance_metrics_SNN_CL2, \
avg_validation_acc_SNN_CL2, \
avg_performance_SNN_CL2 = nested_cross_validation_on_SNN(SNN_CL2, epigenetic_data_with_clusters)

jsObj = json.dumps(performance_metrics_SNN_CL2)  
fileObj = open('body_performance_metrics_SNN_CL2.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close() 

jsObj = json.dumps(avg_performance_SNN_CL2)  
fileObj = open('body_avg_performance_SNN_CL2.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close()





print ("-------------------------------------------------------------------------------")
print ("-----------------------------Evaluating SNN_CL3--------------------------------")
print ("-------------------------------------------------------------------------------")

clusters_size = epigenetic_data_with_clusters.clusters_size
output_size = epigenetic_data_with_clusters.output_size

print("clusters_size:")
print(clusters_size)
print("output_size:")
print(output_size)

SNN_CL3 = SuperlayeredNeuralNetwork_CL3(
    [clusters_size[0], clusters_size[1]],
    [[128, 64, 32, 16], [128, 64, 32, 16]],
    [64, 16], output_size)

confussion_matrix_SNN_CL3, \
ROC_points_SNN_CL3, \
performance_metrics_SNN_CL3, \
avg_validation_acc_SNN_CL3, \
avg_performance_SNN_CL3 = nested_cross_validation_on_SNN(SNN_CL3, epigenetic_data_with_clusters)

jsObj = json.dumps(performance_metrics_SNN_CL3)  
fileObj = open('body_performance_metrics_SNN_CL3.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close() 

jsObj = json.dumps(avg_performance_SNN_CL3)  
fileObj = open('body_avg_performance_SNN_CL3.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close()






print ("-------------------------------------------------------------------------------")
print ("-----------------------------Evaluating SNN_NCL--------------------------------")
print ("-------------------------------------------------------------------------------")

clusters_size = epigenetic_data_with_clusters.clusters_size
output_size = epigenetic_data_with_clusters.output_size

print("clusters_size:")
print(clusters_size)
print("output_size:")
print(output_size)

SNN_NCL = SuperlayeredNeuralNetwork_NCL(
    [clusters_size[0], clusters_size[1]],
    [[128, 64, 32, 16], [128, 64, 32, 16]],
    [64, 16], output_size)

confussion_matrix_SNN_NCL, \
ROC_points_SNN_NCL, \
performance_metrics_SNN_NCL, \
avg_validation_acc_SNN_NCL, \
avg_performance_SNN_NCL = nested_cross_validation_on_SNN(SNN_NCL, epigenetic_data_with_clusters)

jsObj = json.dumps(performance_metrics_SNN_NCL)  
fileObj = open('body_performance_metrics_SNN_NCL.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close() 

jsObj = json.dumps(avg_performance_SNN_NCL)  
fileObj = open('body_avg_performance_SNN_NCL.json', 'w+')  
fileObj.write(jsObj)  
fileObj.close()


















