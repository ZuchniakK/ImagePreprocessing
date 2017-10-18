# from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function
import datetime
import sys
import tensorflow as tf
import numpy as np
import cifar10_reader

ORIGINAL_IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 10000


def _variable_with_weight_decay(name, shape, stddev, wd):
    initial = tf.truncated_normal(shape, stddev=stddev)
    var = tf.Variable(initial, name=name)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def train_cifar10(conv1size=32,
                  conv2size=32,
                  local3size=192,
                  local4size=96,
                  training_epoch=2.1,
                  batch_size=128,
                  test_frequency_per_epoch=0.2,
                  testing_part=0.95,
                  cropped_size=24,
                  interpolation_method=0):
    file_group_name = '_cifar10_cropp_to_%s_resized_m%s_c%s_c%s_f%s_f%s_e%s_b%s_tf%s_tp%s' % (
        str(cropped_size),
        str(interpolation_method),
        str(conv1size),
        str(conv2size),
        str(local3size),
        str(local4size),
        str(training_epoch),
        str(batch_size),
        str(test_frequency_per_epoch),
        str(testing_part),
    )

    IMAGE_SIZE = cropped_size
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    training_loops = int(num_batches_per_epoch * training_epoch)
    test_frequency = int(num_batches_per_epoch * test_frequency_per_epoch)
    print (test_frequency,'testfeqwqwq')
    print('Plan: Training Epoch %s (loops %s * batch size %s)'
          % (str(training_epoch), str(training_loops), str(batch_size)))
    epoch_to_save_model_state = [1, 2, 15, 40, 80, 130, 180]
    loop_to_save_model_state = [int(num_batches_per_epoch * i) for i in epoch_to_save_model_state]
    x = tf.placeholder(tf.float32, shape=[None, ORIGINAL_IMAGE_SIZE * ORIGINAL_IMAGE_SIZE * 3])
    labels = tf.placeholder(tf.float32, shape=[None, 10])
    is_train = tf.placeholder(tf.bool)
    x_r = tf.reshape(x, [tf.shape(x)[0], 3, ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE])
    images = tf.transpose(x_r, [0, 2, 3, 1])

    def train_preprocesing(train_batch):
        # only cropping
        return tf.map_fn(lambda img: tf.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3]), train_batch)

    def test_preprocessing(test_batch):
        return tf.map_fn(lambda img: tf.image.central_crop(img,
                                                           float(IMAGE_SIZE / ORIGINAL_IMAGE_SIZE)),
                         test_batch)

    preproc_images = tf.cond(is_train, lambda: train_preprocesing(images),
                             lambda: test_preprocessing(images))

    resized_images = tf.image.resize_images(preproc_images,
                                            [ORIGINAL_IMAGE_SIZE,
                                             ORIGINAL_IMAGE_SIZE],
                                            method=interpolation_method)
    print (resized_images)

    W_conv1 = _variable_with_weight_decay('W_conv1',
                                          shape=[5, 5, 3, conv1size],
                                          stddev=5e-2,
                                          wd=0.0)
    b_conv1 = tf.Variable(tf.constant(0.0, shape=[conv1size]), name='b_conv1')
    conv1 = tf.nn.relu(tf.nn.conv2d(resized_images, W_conv1, [1, 1, 1, 1], padding='SAME') + b_conv1
                       , name='conv1')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    W_conv2 = _variable_with_weight_decay('W_conv2',
                                          shape=[5, 5, conv1size, conv2size],
                                          stddev=5e-2,
                                          wd=0.0)
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[conv2size]), name='b_conv2')
    conv2 = tf.nn.relu(tf.nn.conv2d(norm1, W_conv2, [1, 1, 1, 1], padding='SAME') + b_conv2
                       , name='conv2')
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    reshape = tf.reshape(pool2, [tf.shape(x)[0], -1])
    dim = int(ORIGINAL_IMAGE_SIZE / 4) ** 2 * conv2size
    print(dim)
    W_local3 = _variable_with_weight_decay('W_local3', shape=[dim, local3size],
                                           stddev=0.04, wd=0.004)
    b_local3 = tf.Variable(tf.constant(0.1, shape=[local3size]), name='b_local3')
    local3 = tf.nn.relu(tf.matmul(reshape, W_local3) + b_local3, name='local3')
    W_local4 = _variable_with_weight_decay('W_local4', shape=[local3size, local4size],
                                           stddev=0.04, wd=0.004)
    b_local4 = tf.Variable(tf.constant(0.1, shape=[local4size]), name='b_local4')
    local4 = tf.nn.relu(tf.matmul(local3, W_local4) + b_local4, name='local4')
    W_softmax = _variable_with_weight_decay('W_softmax', [local4size, NUM_CLASSES],
                                            stddev=1 / float(local4size), wd=0.0)
    b_softmax = tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES]), name='b_softmax')
    softmax_linear = tf.add(tf.matmul(local4, W_softmax), b_softmax, name='softmax_linear')
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=softmax_linear))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    argmax_from_softmax = tf.argmax(softmax_linear, 1)
    argmax_from_labels = tf.argmax(labels, 1)
    correct_prediction = tf.equal(argmax_from_softmax, argmax_from_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    labels_gen = cifar10_reader.generate_cifar10_labels_batch(batch_size)
    images_gen = cifar10_reader.generate_cifar10_images_batch(batch_size)

    time = []
    accuracy_list = []
    softmax_pred_all_test = []
    if testing_part == 1:
        test_sample_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TEST / batch_size) - 1
        data_to_test = cifar10_reader.unpickle_data('test_batch')
        labels_to_test = cifar10_reader.unpickle_labels('test_batch')
        real_sample_size = NUM_EXAMPLES_PER_EPOCH_FOR_TEST
    else:
        sample_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TEST * testing_part)
        test_sample_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TEST / batch_size * testing_part)
        real_sample_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TEST * testing_part)
    saver = tf.train.Saver()
    for i in range(training_loops):
        # print('iteration ', i, datetime.datetime.now())
        x_sample = images_gen.next()
        labels_sample = labels_gen.next()
        train_step.run(feed_dict={
            x: x_sample, labels: labels_sample, is_train: True})
        # if i % test_frequency == 0 and i > 1:
        # ima = sess.run(resized_images, feed_dict={x: x_sample, labels: labels_sample, is_train: True})
        # labs = sess.run(labels, feed_dict={x: x_sample, labels: labels_sample})
        # print(ima[1].shape)
        # for l in range(128):
        #     tmp = [[[ima[l][m][k][0] / 256, ima[l][m][k][1] / 256, ima[l][m][k][2] / 256] for k in range(ORIGINAL_IMAGE_SIZE)]
        #            for m in range(ORIGINAL_IMAGE_SIZE)]
        #     print(ima[1])
        #     interpol = ['none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
        #                 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
        #     for interp in interpol:
        #         plt.imshow(tmp, interpolation=interp)
        #         plt.title(interp)
        #
        #         plt.show()
        if i % test_frequency == 0:
            total_accu = 0
            if testing_part == 1:
                pass
            else:
                testing_sample = np.random.choice(NUM_EXAMPLES_PER_EPOCH_FOR_TEST, sample_size, replace=False)
                data_to_test = cifar10_reader.unpickle_data('test_batch', sample=testing_sample)
                labels_to_test = cifar10_reader.unpickle_labels('test_batch', sample=testing_sample)

            softmax_pred_from_epoch_test = []
            for test_batch in range(int(real_sample_size / batch_size)):
                accu, softmax_fully_test = sess.run([accuracy, softmax_linear],
                                                    feed_dict={
                                                        x: data_to_test[
                                                           batch_size * test_batch:batch_size * (test_batch + 1)],
                                                        labels: labels_to_test[
                                                                batch_size * test_batch:batch_size * (test_batch + 1)],
                                                        is_train: False}
                                                    )
                total_accu += accu
                del accu
                softmax_pred_from_epoch_test.extend(list(softmax_fully_test))
            softmax_pred_all_test.append(softmax_pred_from_epoch_test)
            total_accu /= test_sample_size
            time.append(i)
            accuracy_list.append(total_accu)
            print("step %d, training accuracy on test%g" % (i, total_accu))
            print(datetime.datetime.now())
            del testing_sample
            del data_to_test
            del labels_to_test

        if i in loop_to_save_model_state:
            actual_epoch = str(np.around(i * batch_size / NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, decimals=2))
            save_path = saver.save(sess, "models/model_at_%s_epoch" % actual_epoch + file_group_name + ".ckpt")

    print('len of softmax pred all test', len(softmax_pred_all_test))
    print('len of softmax pred all test', len(softmax_pred_all_test[0]))
    print('len of softmax pred all test', len(softmax_pred_all_test[0][0]))
    np.save('params/softmax_test_values_vs_time' + file_group_name, np.array(softmax_pred_all_test))
    training_in_time = np.array([time, accuracy_list])
    np.save('params/accu_vs_time' + file_group_name, training_in_time)
    save_path = saver.save(sess, "models/model_final" + file_group_name + ".ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()


if __name__ == "__main__":
    train_cifar10()
    train_cifar10(conv1size=int(sys.argv[1]),
                  conv2size=int(sys.argv[2]),
                  local3size=int(sys.argv[3]),
                  local4size=int(sys.argv[4]),
                  training_epoch=int(sys.argv[5]),
                  batch_size=int(sys.argv[6]),
                  test_frequency=int(sys.argv[7]),
                  testing_part=float(sys.argv[8]),

                  cropped_size=int(sys.argv[9]),
                  interpolation_method=int(sys.argv[13]),
                  )
