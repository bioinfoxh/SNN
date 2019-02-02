
from evaluation_metrics import *

from evaluation_metrics import paired_t_test_binary_classification, compute_average_performance_metrics_for_binary_classification, \
    compute_class_id_to_class_symbol, compute_evaluation_metrics_for_each_class, \
    compute_performance_metrics_for_multiclass_classification, plot_mean_ROC_curves, paired_t_test_multiclass_classification, \
    plot_ROC_curves, plot_mean_ROC_curves_for_two_models
from plot_confussion_matrices import plot_confussion_matrix_as_heatmap_for_cancer_data, plot_confussion_matrix_as_heatmap
from gene_clustering.hierarchical_clustering import plot_dendogram

from epigenetic_data.embryo_development_data.embryo_development_data import \
    EmbryoDevelopmentData, EmbryoDevelopmentDataWithClusters, EmbryoDevelopmentDataWithSingleCluster
from epigenetic_data.cancer_data.cancer_data import CancerPatientsData, CancerPatientsDataWithModalities, \
    CancerPatientsDataDNAMethylationLevels

from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork

from evaluation.nested_cross_validation import nested_cross_validation_on_MLP, nested_cross_validation_on_SNN, nested_cross_validation_on_RNN





def nested_cross_validation_on_SNN(network, epigenetic_data_with_clusters):

    k_fold_datasets_with_clusters, k_fold_datasets_hyperparameters_tuning = \
        epigenetic_data_with_clusters.get_k_fold_datasets()

    output_size = epigenetic_data_with_clusters.output_size

    keys = k_fold_datasets_with_clusters.keys()

    confussion_matrix = np.zeros(shape=(output_size, output_size))
    validation_accuracy_list = list()
    ROC_points = dict()

    macro_average = dict()
    micro_average = dict()

    class_id_to_class_symbol = compute_class_id_to_class_symbol(epigenetic_data_with_clusters.label_to_one_hot_encoding)
    performance_metrics = dict()

    """ Outer cross-validation """

    for key in keys:

        """ Inner cross-validation """
        #learning_rate, weight_decay, keep_probability = choose_hyperparameters(
            #network, k_fold_datasets_hyperparameters_tuning[key])

        learning_rate = 0.05
        weight_decay = 0.01
        keep_probability = 0.5

        print learning_rate
        print weight_decay
        print keep_probability

        training_dataset = k_fold_datasets_with_clusters[key]["training_dataset"]
        validation_dataset = k_fold_datasets_with_clusters[key]["validation_dataset"]

        validation_accuracy, snn_confussion_matrix, snn_ROC_points = network.train_and_evaluate(
            training_dataset, validation_dataset, learning_rate, weight_decay, keep_probability)

        print snn_confussion_matrix

        performance_metrics[key] = compute_evaluation_metrics_for_each_class(
            snn_confussion_matrix, class_id_to_class_symbol)

        print performance_metrics[key]

        micro_average[key] = compute_micro_average(performance_metrics[key])
        macro_average[key] = compute_macro_average(performance_metrics[key])

        micro_average[key]['accuracy'] = validation_accuracy
        macro_average[key]['accuracy'] = validation_accuracy

        validation_accuracy_list.append(validation_accuracy)
        confussion_matrix = np.add(confussion_matrix, snn_confussion_matrix)
        ROC_points[key] = snn_ROC_points

    average_validation_accuracy = np.mean(validation_accuracy_list)
    #average_performance_metrics = compute_average_performance_metrics_for_binary_classification(performance_metrics)


    print "Micro"
    performance_metrics['micro'] = micro_average
    print "Macro"
    performance_metrics['macro'] = macro_average

    print "Micro"
    average_micro = compute_performance_metrics_for_multiclass_classification(micro_average)
    print "Macro"
    average_macro = compute_performance_metrics_for_multiclass_classification(macro_average)

    return confussion_matrix, ROC_points, performance_metrics




def evaluate_superlayered_neural_network(epigenetic_data_with_clusters):

    print "-------------------------------------------------------------------------------"
    print "-------------------Evaluating Superlayered Neural Network----------------------"
    print "-------------------------------------------------------------------------------"

    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size

    """superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[256, 128, 64, 32], [256, 128, 64, 32]],
        [128, 32], output_size)"""

    superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[512, 256, 128, 64], [512, 256, 128, 64]],
        [256, 64], output_size)

    """superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[512, 256, 128, 64], [512, 256, 128, 64]],
        [256, 64], output_size)"""

    confussion_matrix, ROC_points, performance_metrics = nested_cross_validation_on_SNN(
        superlayered_neural_network, epigenetic_data_with_clusters)

    label_to_one_hot_encoding = epigenetic_data_with_clusters.label_to_one_hot_encoding
    class_id_to_symbol_id = compute_class_id_to_class_symbol(label_to_one_hot_encoding)
    class_symbol_to_evaluation_matrix = compute_evaluation_metrics_for_each_class(
        confussion_matrix, class_id_to_symbol_id)

    #plot_confussion_matrix_as_heatmap(confussion_matrix, label_to_one_hot_encoding, 'Confusion Matrix for SNN')
    #plot_ROC_curves(ROC_points)




    return confussion_matrix, ROC_points, performance_metrics



def get_embryo_development_data(clustering_algorithm):

    noise_mean = 0
    noise_stddev = 1

    print "Noise Characteristics"

    """epigenetic_data = EmbryoDevelopmentData(
        num_folds=6, num_folds_hyperparameters_tuning=3, max_num_genes=256, gene_entropy_threshold=6.2) """
    epigenetic_data = EmbryoDevelopmentData(
        num_folds=6, num_folds_hyperparameters_tuning=3, max_num_genes=128, gene_entropy_threshold=6.3)
    #epigenetic_data.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_with_clusters = EmbryoDevelopmentDataWithClusters(
        num_clusters=4, clustering_algorithm=clustering_algorithm,
        num_folds=6, num_folds_hyperparameters_tuning=3,
        max_num_genes=1000, gene_entropy_threshold=6.0)
    epigenetic_data_with_clusters.add_Gaussian_noise(noise_mean, noise_stddev)
    print epigenetic_data_with_clusters.clusters_size

    epigenetic_data_for_single_cluster = EmbryoDevelopmentDataWithSingleCluster(
        num_clusters=5, clustering_algorithm=clustering_algorithm,
        num_folds=6, num_folds_hyperparameters_tuning=3,
        max_num_genes=1000, gene_entropy_threshold=6.0)
    print"Gene entropy threshold for epigenetic data with single cluster"
    print epigenetic_data_with_clusters.gene_entropy_threshold
    epigenetic_data_for_single_cluster.add_Gaussian_noise(noise_mean, noise_stddev)
    print epigenetic_data_with_clusters.clusters_size

    return epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster




epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = get_embryo_development_data('k')

evaluate_superlayered_neural_network(epigenetic_data_with_clusters)