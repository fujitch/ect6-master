# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import random

batch_size = 100
training_epochs = 15000
display_epochs = 100
dataset_size = 2000

for coil in [7]:
    for liftmin in [1, 3]:
        frequency = 25
        fname = "dataset_plus_fre" + str(frequency)
        dataset = pickle.load(open("data/" + fname + ".pickle", "rb"))
        dataset = dataset[coil]

        # 導電率すべて使用
        datasetMatrix = np.zeros((dataset_size, len(dataset[1][0])))
        for i in range(dataset_size):
            data1 = dataset[1][i%486]
            data3 = dataset[3][i%486]
            data5 = dataset[5][i%486]
            for k in range(99):
                if k < 6:
                    if k == 3:
                        datasetMatrix[i, k] = 0
                    else:
                        datasetMatrix[i, k] = data1[k]
                else:
                    if liftmin == 1:
                        datasetMatrix[i, k] = data3[k] + (random.random() * (data1[k] - data3[k]))
                    elif liftmin == 3:
                        datasetMatrix[i, k] = data5[k] + (random.random() * (data3[k] - data5[k]))
        dataset = datasetMatrix

        widthRMS = 0.5
        depthRMS = 10
        coilRMS = 7
        liftRMS = 5
        frequencyRMS = 400
        conductivityRMS = 100
        # 規格化
        dataset[:, 0] /= widthRMS
        dataset[:, 1] /= depthRMS
        dataset[:, 2] /= coilRMS
        dataset[:, 3] /= liftRMS
        dataset[:, 4] /= frequencyRMS
        dataset[:, 5] /= conductivityRMS
        dataset[:, 6:] -= np.min(dataset[:, 6:])
        dataset[:, 6:] /= np.max(dataset[:, 6:])
        # 型変換
        dataset = np.array(dataset, dtype=np.float32)
        # datasetを分ける
        datasetTrain = dataset[:dataset_size-1000, :]
        datasetTest = dataset[dataset_size-1000:, :]
        pickle.dump(datasetTrain, open("data/" + fname + "_" + str(coil) + "_0_Train_NoiseRandomLift_" + str(liftmin) + "_ConductivityAll.pickle", "wb"))
        pickle.dump(datasetTest, open("data/" + fname + "_" + str(coil) + "_0_Test_NoiseRandomLift_" + str(liftmin) + "_ConductivityAll.pickle", "wb"))

        tf.reset_default_graph()

        x = tf.placeholder("float", shape=[None, 93])
        y_ = tf.placeholder("float", shape=[None, 1])

        # 荷重作成
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        # バイアス作成
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # 畳み込み処理を定義
        def conv2d_pad(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        # プーリング処理を定義
        def max_pool_2_2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        # 畳み込み層1
        W_conv1 = weight_variable([3, 3, 1, 256])
        b_conv1 = bias_variable([256])
        x_image = tf.reshape(x, [-1, 31, 3, 1])
        h_conv1 = tf.nn.relu(conv2d_pad(x_image, W_conv1) + b_conv1)
        # プーリング層1
        h_pool1 = max_pool_2_2(h_conv1)

        # 畳み込み層2
        W_conv2 = weight_variable([3, 2, 256, 256])
        b_conv2 = bias_variable([256])
        h_conv2 = tf.nn.relu(conv2d_pad(h_pool1, W_conv2) + b_conv2)
        # プーリング層2
        h_pool2 = max_pool_2_2(h_conv2)

        # 全結合層1
        W_fc1 = weight_variable([256*8, 1024])
        b_fc1 = bias_variable([1024])
        h_flat = tf.reshape(h_pool2, [-1, 256*8])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        # ドロップアウト
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 全結合層2
        W_fc2 = weight_variable([1024, 1])
        b_fc2 = bias_variable([1])
        y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # 学習誤差を求める
        loss = tf.reduce_mean(tf.square(y_ - y_out))

        # 最適化処理
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        sess = tf.Session()

        saver = tf.train.Saver()

        sess.run(tf.initialize_all_variables())

        for i in range(training_epochs):
            for k in range(0, len(datasetTrain), batch_size):
                batch = np.zeros((batch_size, 93))
                batch = np.array(batch, dtype=np.float32)
                batch[:, :] = datasetTrain[k:k+batch_size, 6:]
                output = datasetTrain[k:k+batch_size, 1:2]
                train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})
            
            if i%display_epochs == 0:
                batch = np.zeros((datasetTrain.shape[0], 93))
                batch = np.array(batch, dtype=np.float32)
                batch[:, :] = datasetTrain[:, 6:]
                output = datasetTrain[:, 1:2]
                train_loss = loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
                batch = np.zeros((datasetTest.shape[0], 93))
                batch = np.array(batch, dtype=np.float32)
                batch[:, :] = datasetTest[:, 6:]
                output = datasetTest[:, 1:2]
                test_loss = loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
                print(str(i) + "epochs_finished!")
                print("train_loss===" + str(train_loss))
                print("test_loss===" + str(test_loss))
        saver.save(sess, "./model3/20190110CNNmodeldepthNoiseRandomLift_" + str(liftmin) + "_" + fname + "_" + str(coil) + "_byall")
        sess.close()
# pickle.dump(lossDict, open("lossDict20181114CNNmodel_dataset_plus_con0_noise_merge_lift.pickle", "wb"))