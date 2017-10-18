import numpy as np


def unpickle(file):
    import cPickle
    with open('cifar10_dataset/' + file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def unpickle_data(file, sample=None, multiply_factor=1):
    import cPickle
    with open('cifar10_dataset/' + file, 'rb') as fo:
        dict = cPickle.load(fo)
    sample_data = dict['data']
    if sample is not None:
        sample_data = [dict['data'][i] for i in range(len(dict['data'])) if i in sample]
        sample_data = [sample_data[int(i / multiply_factor)] for i in range(len(sample_data) * multiply_factor)]
    # print ('shape of data', np.array(sample_data).shape)
    return sample_data


def unpickle_labels(file, sample=None, multiply_factor=1):
    import cPickle
    # print ('multiply_factor =%s' % multiply_factor)
    with open('cifar10_dataset/' + file, 'rb') as fo:
        dict = cPickle.load(fo)
    labels_value = dict['labels']
    label_set = [[0 for i in range(10)] for j in range(len(labels_value))]
    for i in range(len(labels_value)):
        label_set[i][labels_value[i]] = 1
    sample_data = np.array(label_set)
    if sample is not None:
        sample_data = [label_set[i] for i in range(len(label_set)) if i in sample]
        sample_data = [sample_data[int(i / multiply_factor)] for i in range(len(sample_data) * multiply_factor)]
        # print ("sample labels:", sample_data)
    # print ('shape of labels', np.array(sample_data).shape)
    return sample_data


def generate_cifar10_images_batch(batch_size):
    image_set = []
    for i in range(5):
        diki = unpickle('data_batch_%s' % str(i + 1))
        image_set.extend(diki['data'])
    # print (type(image_set))
    # print (len(image_set))
    # print np.array(image_set)
    # print np.size(np.array(image_set))
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


def noise_to_images(images_batch, corelation, distribution, noise_param):
    images_batch = np.array(images_batch)
    images_batch_shape = images_batch.shape
    ones_mat = np.ones(shape=images_batch_shape)
    if corelation == 'absolut' and distribution == 'uniform':
        print ("Absolut-Uniform Noise")
        images_batch = np.add(images_batch,
                              np.random.uniform(low=-noise_param, high=noise_param, size=images_batch_shape))
        print ("Finish Add Noise")
    elif corelation == 'linear' and distribution == 'uniform':
        print ("Linear-Uniform Noise")
        images_batch = np.multiply(images_batch,
                                   np.add(ones_mat, np.random.uniform(low=-noise_param, high=noise_param,
                                                                      size=images_batch_shape)))
        print ("Finish Add Noise")
    elif corelation == 'absolut' and distribution == 'normal':
        print ("Absolut-Normal Noise")
        images_batch = np.add(images_batch, np.random.normal(loc=0.0, scale=noise_param, size=images_batch_shape))
        print ("Finish Add Noise")
    elif corelation == 'linear' and distribution == 'normal':
        print ("Linear-Normal Noise")
        images_batch = np.multiply(images_batch,
                                   np.add(ones_mat,
                                          np.random.normal(loc=0.0, scale=noise_param, size=images_batch_shape)))
        print ("Finish Add Noise")
    else:
        print ("No Noise")
    del ones_mat
    del images_batch_shape
    return images_batch
def generate_cifar10_images_batch_modified(batch_size, data_files=5, corelation=0, distribution=0, noise_param=0):


    image_set = []
    for i in range(5):
        diki = unpickle('data_batch_%s' % str(i + 1))
        image_set.extend(diki['data'])
    # print (type(image_set))
    # print (len(image_set))
    # print np.array(image_set)
    # print np.size(np.array(image_set))
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


def generate_cifar10_labels_batch(batch_size):
    labels_value = []
    for i in range(5):
        diki = unpickle('data_batch_%s' % str(i + 1))
        labels_value.extend(diki['labels'])
    image_set = [[0 for i in range(10)] for j in range(len(labels_value))]
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
            yield np.array(last_part + first_part)
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
