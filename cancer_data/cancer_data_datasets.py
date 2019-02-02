import numpy as np


#def compute_probability_distribution(input_values):
#    """
#    Normalizes the gene expressions profile to obtain a probability distribution which will be used as the input
#    to the neural network architectures.
#
#    :param (list) input_values :  The un-normalized gene expression profile for a training example
#    :return (list): normalized_input_values: The normalized gene expression profile for a training
#             example
#    """
#
#    input_values_sum = 0.0
#
#    for input_value in input_values:
#        input_values_sum += float(input_value)
#    normalized_input_values = range(len(input_values))
#
#    if input_values_sum != 0:
#        for index in range(len(input_values)):
#            normalized_input_values[index] = float(input_values[index])/input_values_sum
#
#    return normalized_input_values


#def __extract_training_validation_test_patient_ids(labels_to_patient_ids):
#    training_patient_ids = []
#    validation_patient_ids = []
#    test_patient_ids = []
#
#    labels = labels_to_patient_ids.keys()
#    for label in labels:
#        patient_ids = labels_to_patient_ids[label]
#
#        num_training_patients = len(patient_ids) * 70/100
#        num_validation_patients = len(patient_ids) * 15/100
#
#        training_patient_ids += patient_ids[:num_training_patients]
#        validation_patient_ids += patient_ids[num_training_patients + 1 : num_training_patients + num_validation_patients]
#        test_patient_ids += patient_ids[num_training_patients + num_validation_patients:]
#
#    return training_patient_ids, validation_patient_ids, test_patient_ids


def create_dataset(
        patient_ids, input_data_size, output_size,
        patient_id_to_input_values, label_to_one_hot_encoding, patient_id_to_label):

    data = np.ndarray(shape=(len(patient_ids), input_data_size),
                                 dtype=np.float32)
    labels = np.ndarray(shape=(len(patient_ids), output_size),
                               dtype=np.float32)

    np.random.shuffle(patient_ids)
    index = 0
    for patient_id in patient_ids:
        data[index, :] = patient_id_to_input_values[patient_id]
        labels[index, :] = label_to_one_hot_encoding[patient_id_to_label[patient_id]]
        index += 1


    return data, labels



def create_dataset_with_clusters(
        patient_ids, clusters_size, output_size,
        patient_id_to_input_values_clusters, label_to_one_hot_encoding, patient_id_to_label):

    data = dict()

    for cluster_id in range(len(clusters_size)):
        data[cluster_id] = np.ndarray(shape=(len(patient_ids), clusters_size[cluster_id]),
                                                 dtype=np.float32)

    labels = np.ndarray(shape=(len(patient_ids), output_size),
                                   dtype=np.float32)

    np.random.shuffle(patient_ids)
    index = 0
    for patient_id in patient_ids:
        for cluster_id in range(len(clusters_size)):
            data[cluster_id][index, :] = \
                patient_id_to_input_values_clusters[patient_id][cluster_id]
        labels[index, :] = label_to_one_hot_encoding[patient_id_to_label[patient_id]]
        index += 1

    return data, labels


def create_k_fold_patient_ids(k, label_to_patient_ids):
    """
    Separates the patient_ids into k folds. (k-1) folds will be used to training and one fold for validation.
    """
    k_fold_patient_ids = dict()
    for index in range(k):
        k_fold_patient_ids[index] = []

    labels = label_to_patient_ids.keys()
    for label in labels:
        patient_ids = label_to_patient_ids[label]
        group_size = len(patient_ids)/k
        for index in range(k-1):
            k_fold_patient_ids[index] += patient_ids[index*group_size:(index+1)*group_size]
        k_fold_patient_ids[k-1] += patient_ids[(k-1)*group_size:]

    keys = k_fold_patient_ids.keys()
    for key in keys:
        patient_ids = k_fold_patient_ids[key]
        np.random.shuffle(patient_ids)
        k_fold_patient_ids[key] = patient_ids

    return k_fold_patient_ids


def create_k_fold_datasets(
        k, k_fold_patient_ids, input_data_size, output_size,
        patient_id_to_input_values, label_to_one_hot_encoding, patient_id_to_label):

    k_fold_datasets = dict()
    for index in range(k):
        k_fold_datasets[index] = dict()

    for index_i in range(k):
        validation_patient_ids = k_fold_patient_ids[index_i]
        training_patient_ids = []
        for index_j in range(k):
            if index_j != index_i:
                training_patient_ids += k_fold_patient_ids[index_j]

        training_dataset = dict()
        training_data, training_labels = create_dataset(
            training_patient_ids, input_data_size, output_size,
            patient_id_to_input_values, label_to_one_hot_encoding, patient_id_to_label)
        training_dataset["training_data"] = training_data
        training_dataset["training_labels"] = training_labels

        validation_dataset = dict()
        validation_data, validation_labels = create_dataset(
            validation_patient_ids, input_data_size, output_size,
            patient_id_to_input_values, label_to_one_hot_encoding, patient_id_to_label)
        validation_dataset["validation_data"] = validation_data
        validation_dataset["validation_labels"] = validation_labels

        k_fold_datasets[index_i]["training_dataset"] = training_dataset
        k_fold_datasets[index_i]["validation_dataset"] = validation_dataset

    return k_fold_datasets
#0

def create_k_fold_datasets_with_clusters(
        k, k_fold_patient_ids, clusters_size, output_size,
        patient_id_to_input_values_clusters,
        label_to_one_hot_encoding, patient_id_to_label):
    """
    Creates the datasets_old corresponding to each fold.
    """

    k_fold_datasets = dict()
    for index in range(k):
        k_fold_datasets[index] = dict()

    for index_i in range(k):
        validation_patient_ids = k_fold_patient_ids[index_i]
        training_patient_ids = []
        for index_j in range(k):
            if index_j != index_i:
                training_patient_ids += k_fold_patient_ids[index_j]

        training_dataset = dict()
        training_data, training_labels = create_dataset_with_clusters(
            training_patient_ids, clusters_size, output_size,
            patient_id_to_input_values_clusters, label_to_one_hot_encoding, patient_id_to_label)

        training_dataset["training_data"] = training_data
        training_dataset["training_labels"] = training_labels


        validation_dataset = dict()
        validation_data, validation_labels = create_dataset_with_clusters(
            validation_patient_ids, clusters_size, output_size,
            patient_id_to_input_values_clusters, label_to_one_hot_encoding, patient_id_to_label)

        validation_dataset["validation_data"] = validation_data
        validation_dataset["validation_labels"] = validation_labels

        k_fold_datasets[index_i]["training_dataset"] = training_dataset
        k_fold_datasets[index_i]["validation_dataset"] = validation_dataset

    return k_fold_datasets

