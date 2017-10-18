import numpy as np


def unpickle(file):
    import cPickle
    with open('cifar100_dataset/' + file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
#
# for key, value in unpickle('train').iteritems() :
#     print key
# mydict = unpickle('train')
# print max(mydict['fine_labels']), min(mydict['fine_labels'])

def unpickle_data(file, sample=None):
    import cPickle
    with open('cifar100_dataset/' + file, 'rb') as fo:
        dict = cPickle.load(fo)
    if sample is not None:
        sample_data = [dict['data'][i] for i in range(len(dict['data'])) if i in sample]
        return sample_data
    return dict['data']


def unpickle_labels(file, sample=None):
    import cPickle
    with open('cifar100_dataset/' + file, 'rb') as fo:
        dict = cPickle.load(fo)
    labels_value = dict['fine_labels']
    label_set = [[0 for i in range(100)] for j in range(len(labels_value))]
    for i in range(len(labels_value)):
        label_set[i][labels_value[i]] = 1
    if sample is not None:
        sample_data = [label_set[i] for i in range(len(label_set)) if i in sample]
        return sample_data
    return np.array(label_set)


def generate_cifar100_images_batch(batch_size):
    image_set = unpickle('train')['data']
    max_index = len(image_set)
    actual_index = 0
    while True:
        if actual_index + batch_size < max_index:
            yield np.array(image_set[actual_index:actual_index + batch_size])
            actual_index += batch_size
        else:
            last_part = image_set[actual_index:]
            first_part = image_set[:batch_size - (max_index - actual_index)]
            tmp = []
            tmp.extend(last_part)
            tmp.extend(first_part)
            yield np.array(tmp)
            actual_index = batch_size - (max_index - actual_index)


def generate_cifar100_labels_batch(batch_size):
    labels_value =  unpickle('train')['fine_labels']
    image_set = [[0 for i in range(100)] for j in range(len(labels_value))]
    for i in range(len(labels_value)):
        image_set[i][labels_value[i]] = 1
    actual_index = 0
    max_index = len(image_set)
    while True:
        if actual_index + batch_size < max_index:
            yield np.array(image_set[actual_index:actual_index + batch_size])
            actual_index += batch_size
        else:
            last_part = image_set[actual_index:]
            first_part = image_set[:batch_size - (max_index - actual_index)]
            tmp = []
            tmp.extend(last_part)
            tmp.extend(first_part)
            yield np.array(tmp)
            actual_index = batch_size - (max_index - actual_index)


# print(unpickle_labels('test_batch'))


def my_gienerator(batch_size):
    image_set = [i for i in range(15)]
    actual_index = 0
    max_index = len(image_set)
    while True:
        if actual_index + batch_size < max_index:
            yield np.array(image_set[actual_index:actual_index + batch_size])
            actual_index += batch_size
        else:
            last_part = image_set[actual_index:]
            first_part = image_set[:batch_size - (max_index - actual_index)]
            yield np.array(last_part + first_part)
            actual_index = batch_size - (max_index - actual_index)
