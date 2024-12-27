import math

import numpy as np
import keras
from keras.models import load_model, Model
import os
from tqdm import tqdm
from scipy import stats
import skimage.io as io
from scipy.spatial.distance import mahalanobis

def save_npy(dataset, save_path):
    path = os.path.join('/dataset', dataset, 'picture/train')
    for i in range(10):
        class_path = os.path.join(path, str(i))
        pic_list = os.listdir(class_path)
        if dataset =='mnist':
            temp_arr = np.empty((len(pic_list), 28, 28, 1))
            temp1_arr = np.empty(len(pic_list), dtype='uint8')
            for j in range(len(pic_list)):
                img = io.imread(os.path.join(class_path, pic_list[j]))
                img = img[:, :, 0]
                img = np.expand_dims(img, axis=0)
                img = img.transpose(1, 2, 0)
                temp_arr[j] = img
                temp1_arr[j] = i
        elif dataset =='cifar':
            temp_arr = np.empty((len(pic_list), 32, 32, 3))
            temp1_arr = np.empty(len(pic_list), dtype='uint8')
            for j in range(len(pic_list)):
                img = io.imread(os.path.join(class_path, pic_list[j]))
                temp_arr[j] = img
                temp1_arr[j] = i

        if i == 0:
            img_arr = temp_arr
            label_arr = temp1_arr
        else:
            img_arr = np.concatenate((img_arr, temp_arr), axis=0)
            label_arr = np.concatenate((label_arr, temp1_arr), axis=0)

    img_arr = img_arr.astype('float32') / 255.0
    print(img_arr.shape)
    print(label_arr.shape)
    np.save(os.path.join(save_path, 'x_train.npy'), img_arr)
    np.save(os.path.join(save_path, 'y_train.npy'), label_arr)

def load_npy(path, flag):
    for i in range(10):
        if flag == 0:
            pic_path = os.path.join(path, str(i) + '_pictures.npy')
        else:
            pic_path = os.path.join(path, str(i) + '_pictures.npy')
            # pic_path = os.path.join(path, str(i) + '_edited_pictures.npy')
        temp_arr = np.load(pic_path)
        temp1_arr = np.empty(len(temp_arr), dtype='uint8')
        for j in range(len(temp1_arr)):
            temp1_arr[j] = i
        if i == 0:
            pic_arr = temp_arr
            label_arr = temp1_arr
        else:
            pic_arr = np.concatenate((pic_arr, temp_arr), axis=0)
            label_arr = np.concatenate((label_arr, temp1_arr), axis=0)
    return pic_arr, label_arr

def gen_sadl_layers(model_name):
    model_path = '/model' + os.sep + model_name + os.sep + model_name + '.h5'
    model = load_model(model_path)
    if model_name == 'lenet1':
        input = model.layers[0].output
        layers = [model.layers[3].output]
        layers = list(zip(1 * ['conv'], layers))
    elif model_name == 'lenet5':
        input = model.layers[0].output
        layers = [model.layers[7].output]
        layers = list(zip(1 * ['dense'], layers))
    elif model_name == 'cifarmodel':
        input = model.layers[0].output
        layers = [model.layers[10].output]
        layers = list(zip(1 * ['dense'], layers))
    elif model_name == 'vgg16':
        input = model.layers[0].output
        layers = [model.layers[34].output]
        layers = list(zip(1 * ['dense'], layers))
    return input, layers

def gen_model(layers, input):
    model = []
    index = []
    for name, layer in layers:
        m = Model(inputs=input, outputs=layer)
        model.append(m)
        index.append(name)
    models = list(zip(index, model))
    return models

def gen_neuron_activate(models, x, std, period='train'):
    neuron_activate = []
    mask = []
    for index, model in models:
        if index == 'conv':
            temp = model.predict(x).reshape(len(x), -1, model.output.shape[-1])
            temp = np.mean(temp, axis=1)
        if index == 'dense':
            temp = model.predict(x).reshape(len(x), model.output.shape[-1])
        neuron_activate.append(temp)
        mask.append(np.array(np.std(temp, axis=0)) > std)
    neuron_activate = np.concatenate(neuron_activate, axis=1)
    mask = np.concatenate(mask, axis=0)
    # print(neuron_activate.shape)
    if period == 'train':
        return neuron_activate, mask
    else:
        return neuron_activate

def com_LSA(model_name, x_train, x_test, std):
    input, layers = gen_sadl_layers(model_name)
    models = gen_model(layers, input)
    train_neuron_activate, mask = gen_neuron_activate(models, x_train, std, 'train')
    test_neuron_activate = gen_neuron_activate(models, x_test, std, 'test')
    test_score = []
    train = []
    for train_neuron_activate in train_neuron_activate[:, mask]:
        train.append(train_neuron_activate)
    train = np.transpose(train)
    kde = stats.gaussian_kde(train, bw_method='scott')
    print("compute kde")
    for test_neuron_activate in tqdm(test_neuron_activate[:, mask]):
        if model_name != 'vgg16':
            temp = np.empty((len(train), 1), dtype='float32')
            # lenet1,lenet5,cifarmodel:
            for i in range(len(train)):
                temp[i][0] = test_neuron_activate[i]
            score = -kde.logpdf(temp)
            test_score.append(score[0])

        else:
            # vgg16 use log lead to nan, thus we use kde:
            score = 0.0
            for i in range(len(train)):
                score = score + test_neuron_activate[i]
            test_score.append(score)

    test_score = np.array(test_score)
    return test_score

def com_DSA(model_name, x_train, y_train, x_test, y_test):
    input, layers = gen_sadl_layers(model_name)
    models = gen_model(layers, input)
    train_neuron_activate, mask = gen_neuron_activate(models, x_train, 0.05, 'train')
    test_neuron_activate = gen_neuron_activate(models, x_test, 0.05, 'test')
    test_score = []
    for i in tqdm(range(len(test_neuron_activate))):
        dis_a = 1000
        x1 = train_neuron_activate[0]
        for j in range(len(train_neuron_activate)):
            if y_train[j] == y_test[i] and ((train_neuron_activate[j]-test_neuron_activate[i])**2).sum() < dis_a:
                dis_a = ((train_neuron_activate[j]-test_neuron_activate[i])**2).sum()
                x1 = train_neuron_activate[j]
        dis_b = 1000
        for k in range(len(train_neuron_activate)):
            if y_train[k] != y_test[i] and ((train_neuron_activate[k]-x1)**2).sum() < dis_b:
                dis_b = ((train_neuron_activate[k]-x1)**2).sum()
        if dis_b == 1000 or dis_b == 0:
            # print('out!!!')
            dis_b = 1
        dis = (dis_a / dis_b) ** 0.5
        test_score.append(dis)
    test_score = np.array(test_score)
    return test_score

def com_MDSA(model_name, x_train, x_test):
    input, layers = gen_sadl_layers(model_name)
    models = gen_model(layers, input)
    train_neuron_activate, mask = gen_neuron_activate(models, x_train, 0.05, 'train')
    test_neuron_activate = gen_neuron_activate(models, x_test, 0.05, 'test')
    test_score = []

    mean_value = np.mean(train_neuron_activate, axis=0)
    cov_value = np.cov(train_neuron_activate.transpose((1,0)))

    for i in tqdm(range(len(test_neuron_activate))):
        temp = test_neuron_activate[i]
        dis = mahalanobis(temp, mean_value, cov_value)
        test_score.append(dis)
        print(dis)
    test_score = np.array(test_score)
    return test_score



def com_DDR(model_name, x_test, y_test):
    model_path = '/model' + os.sep + model_name + os.sep + model_name + '.h5'
    model = load_model(model_path)
    y_test = keras.utils.to_categorical(y_test, 10)
    print(y_test.shape)
    if model_name == 'lenet1' or model_name == 'lenet5':
        loss, acc = model.evaluate(x_test, y_test, batch_size=128)
        # DDR_value = acc
        DDR_value = 1 - acc
    elif model_name == 'cifarmodel' or model_name == 'vgg16':
        loss, acc = model.evaluate(x_test, y_test, batch_size=256)
        DDR_value = 1 - acc

    value = []
    value.append(DDR_value)
    value = np.array(value)
    return value


def com_coverage(model, path, state_i, rate):
    print(state_i, ':')
    if state_i == 'ori':
        LSA_save_path = os.path.join(path, state_i + '_LSA_value.npy')
        DSA_save_path = os.path.join(path, state_i + '_DSA_value.npy')
        MDSA_save_path = os.path.join(path, state_i + '_MDSA_value.npy')
        DDR_save_path = os.path.join(path, state_i + '_DDR_value.npy')
        LSA_value = np.load(LSA_save_path)
        DSA_value = np.load(DSA_save_path)
        MDSA_value = np.load(MDSA_save_path)
        DDR_value = np.load(DDR_save_path)
        # print(np.max(LSA_value))
        # print(np.max(DSA_value))
        # print(np.max(MDSA_value))
        # print(np.max(DDR_value))

        # LSA
        temp = np.zeros(2000)
        # cifarmodel
        # temp_arr = LSA_value / 0.2
        # vgg16
        temp_arr = LSA_value
        for i in range(len(temp_arr)):
            t = math.ceil(temp_arr[i])
            if t < 0:
                t = -t
            elif t > 2000:
                # print(temp_arr[i], "惊喜度爆炸")
                continue
            else:
                temp[t] = 1
        num1 = sum(temp == 1)
        rate1 = num1 / 2000.0
        print('LSA:', str(rate1))

        #DSA
        temp = np.zeros(2000)
        # cifarmodel
        # temp_arr = DSA_value *1000
        # vgg16
        temp_arr = DSA_value * 800
        for i in range(len(temp_arr)):
            t = math.ceil(temp_arr[i])
            if t < 0:
                t = -t
            elif t >= 2000:
                # print(temp_arr[i], "惊喜度爆炸")
                continue
            else:
                temp[t] = 1
        num1 = sum(temp == 1)
        rate1 = num1 / 2000.0
        print('DSA:', str(rate1))

        # MDSA
        temp = np.zeros(2000)
        # cifarmodel
        # temp_arr = MDSA_value * 6
        # vgg16
        temp_arr = MDSA_value * 25

        for i in range(len(temp_arr)):
            t = math.ceil(temp_arr[i])
            if t < 0:
                t = -t
            elif t >= 2000:
                # print(temp_arr[i], "惊喜度爆炸")
                continue
            else:
                temp[t] = 1
        num1 = sum(temp == 1)
        rate1 = num1 / 2000.0
        print('MDSA:', str(rate1))

    else:
        for r in range(len(rate)):
            print(str(rate[r]), ":")
            LSA_save_path = os.path.join(path, state_i + '_' + str(rate[r]) + '_LSA_value.npy')
            DSA_save_path = os.path.join(path, state_i + '_' + str(rate[r]) + '_DSA_value.npy')
            MDSA_save_path = os.path.join(path, state_i + '_' + str(rate[r]) + '_MDSA_value.npy')
            DDR_save_path = os.path.join(path, state_i + '_' + str(rate[r]) + '_DDR_value.npy')
            LSA_value = np.load(LSA_save_path)
            DSA_value = np.load(DSA_save_path)
            MDSA_value = np.load(MDSA_save_path)
            DDR_value = np.load(DDR_save_path)
            # print(np.max(LSA_value))
            # print(np.max(DSA_value))
            # print(np.max(MDSA_value))
            # print(np.max(DDR_value))

            # LSA
            temp = np.zeros(2000)
            # cifarmodel
            # temp_arr = LSA_value / 0.2
            # vgg16
            temp_arr = LSA_value
            for i in range(len(temp_arr)):
                t = math.ceil(temp_arr[i])
                # print(t)
                if t < 0:
                    t = -t
                elif t >= 2000:
                    # print(temp_arr[i], "惊喜度爆炸")
                    continue
                else:
                    temp[t] = 1
            num1 = sum(temp == 1)
            rate1 = num1 / 2000.0
            print('LSA:', str(rate1))
            #
            # # DSA
            temp = np.zeros(2000)
            temp_arr = DSA_value * 1000
            for i in range(len(temp_arr)):
                t = math.ceil(temp_arr[i])
                if t < 0:
                    t = -t
                elif t >= 2000:
                    # print(temp_arr[i], "惊喜度爆炸")
                    continue
                else:
                    temp[t] = 1
            num1 = sum(temp == 1)
            rate1 = num1 / 2000.0
            print('DSA:', str(rate1))
            #
            # # MDSA
            temp = np.zeros(2000)
            # # cifarmodel
            # temp_arr = MDSA_value * 6
            # vgg16
            temp_arr = MDSA_value * 25

            for i in range(len(temp_arr)):
                t = math.ceil(temp_arr[i])
                if t < 0:
                    t = -t
                elif t >= 2000:
                    # print(temp_arr[i], "惊喜度爆炸")
                    continue
                else:
                    temp[t] = 1
            num1 = sum(temp == 1)
            rate1 = num1 / 2000.0
            print('MDSA:', str(rate1))

def com_combined_coverage(model, path_1, path_2, state_i, rate):
    print(state_i, ':')
    if state_i == 'ori':
        LSA_save_path_1 = os.path.join(path_1, state_i + '_LSA_value.npy')
        DSA_save_path_1 = os.path.join(path_1, state_i + '_DSA_value.npy')
        DDR_save_path_1 = os.path.join(path_1, state_i + '_DDR_value.npy')
        LSA_save_path_2 = os.path.join(path_2, state_i + '_LSA_value.npy')
        DSA_save_path_2 = os.path.join(path_2, state_i + '_DSA_value.npy')
        DDR_save_path_2 = os.path.join(path_2, state_i + '_DDR_value.npy')
        LSA_value = np.concatenate((np.load(LSA_save_path_1), np.load(LSA_save_path_2)), axis=0)
        DSA_value = np.concatenate((np.load(DSA_save_path_1), np.load(DSA_save_path_2)), axis=0)
        DDR_value = (np.load(DDR_save_path_1)[0] + np.load(DDR_save_path_2)[0]) / 2.0
        print(np.max(LSA_value))
        print(np.max(DSA_value))
        print(DDR_value)

        #LSA
        temp = np.zeros(2000)
        temp_arr = LSA_value / 0.2
        for i in range(len(temp_arr)):
            t = math.ceil(temp_arr[i])
            if t < 0:
                t = -t
            if t >= 2000:
                # print(temp_arr[i], "惊喜度爆炸")
                continue
            else:
                temp[t] = 1
        num1 = sum(temp == 1)
        rate1 = num1 / 2000.0
        print('LSA:', str(rate1))

        #DSA
        temp = np.zeros(2000)
        temp_arr = DSA_value * 1000
        for i in range(len(temp_arr)):
            t = math.ceil(temp_arr[i])
            if t < 0:
                t = -t
            if t >= 2000:
                # print(temp_arr[i], "惊喜度爆炸")
                continue
            else:
                temp[t] = 1
        num1 = sum(temp == 1)
        rate1 = num1 / 2000.0
        print('DSA:', str(rate1))
    else:
        for r in range(len(rate)):
            print(str(rate[r]), ":")
            LSA_save_path_1 = os.path.join(path_1, state_i + '_' + str(rate[r]) + '_LSA_value.npy')
            DSA_save_path_1 = os.path.join(path_1, state_i + '_' + str(rate[r]) + '_DSA_value.npy')
            DDR_save_path_1 = os.path.join(path_1, state_i + '_' + str(rate[r]) + '_DDR_value.npy')
            LSA_save_path_2 = os.path.join(path_2, state_i + '_' + str(rate[r]) + '_LSA_value.npy')
            DSA_save_path_2 = os.path.join(path_2, state_i + '_' + str(rate[r]) + '_DSA_value.npy')
            DDR_save_path_2 = os.path.join(path_2, state_i + '_' + str(rate[r]) + '_DDR_value.npy')

            LSA_value = np.concatenate((np.load(LSA_save_path_1), np.load(LSA_save_path_2)), axis=0)
            DSA_value = np.concatenate((np.load(DSA_save_path_1), np.load(DSA_save_path_2)), axis=0)
            DDR_value = (np.load(DDR_save_path_1)[0] + np.load(DDR_save_path_2)[0]) / 2.0

            print(np.max(LSA_value))
            print(np.max(DSA_value))
            print(DDR_value)

            # LSA
            temp = np.zeros(2000)
            temp_arr = LSA_value / 0.2
            for i in range(len(temp_arr)):
                t = math.ceil(temp_arr[i])
                if t < 0:
                    t = -t
                if t >= 2000:
                    # print(temp_arr[i], "惊喜度爆炸")
                    continue
                else:
                    temp[t] = 1
            num1 = sum(temp == 1)
            rate1 = num1 / 2000.0
            print('LSA:', str(rate1))

            # DSA
            temp = np.zeros(2000)
            temp_arr = DSA_value * 1000
            for i in range(len(temp_arr)):
                t = math.ceil(temp_arr[i])
                if t < 0:
                    t = -t
                if t >= 2000:
                    # print(temp_arr[i], "惊喜度爆炸")
                    continue
                else:
                    temp[t] = 1
            num1 = sum(temp == 1)
            rate1 = num1 / 2000.0
            print('DSA:', str(rate1))

if __name__ == '__main__':
    # model_name = 'cifarmodel'
    model_name = 'vgg16'
    dataset = 'cifar'

    # path = '/data' + os.sep + model_name + os.sep + 'step_9_cal_SADL_value'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # save_npy(dataset, path)

    # train_path = '/data' + os.sep + model_name + os.sep + 'step_9_cal_SADL_value'
    # x_train = np.load(os.path.join(train_path, 'x_train.npy'))
    # y_train = np.load(os.path.join(train_path, 'y_train.npy'))
    # path = 'data' + os.sep + model_name + os.sep + 'step_7_test'
    # state = ['ori', 'fgsm', 'bim', 'jsma', 'cw']
    # rate = [0.3, 0.6, 0.9]
    # for i in range(len(state)):
    #     # if i!=0:
    #     #     continue
    #     print(state[i])
    #     for j in range(len(rate)):
    #         if i == 0:
    #             LSA_save_path = os.path.join(train_path, state[i] + '_LSA_value.npy')
    #             DSA_save_path = os.path.join(train_path, state[i] + '_DSA_value.npy')
    #             MDSA_save_path = os.path.join(train_path, state[i] + '_MDSA_value.npy')
    #             DDR_save_path = os.path.join(train_path, state[i] + '_DDR_value.npy')
    #             path1 = path + os.sep + state[i]
    #             path2 = path1
    #             x_test, y_test = load_npy(path2, 0)
    #         else:
    #             print(rate[j])
    #             LSA_save_path = os.path.join(train_path, state[i] + '_' + str(rate[j]) + '_LSA_value.npy')
    #             DSA_save_path = os.path.join(train_path, state[i] + '_' + str(rate[j]) + '_DSA_value.npy')
    #             MDSA_save_path = os.path.join(train_path, state[i] + '_' + str(rate[j]) + '_MDSA_value.npy')
    #             DDR_save_path = os.path.join(train_path, state[i] + '_' + str(rate[j]) + '_DDR_value.npy')
    #             path2 = os.path.join(path, state[i], str(rate[j]))
    #             x_test, y_test = load_npy(path2, 1)
    #         value = com_LSA(model_name, x_train, x_test, 0.05)
    #         np.save(LSA_save_path, value)
    #         value = com_DSA(model_name, x_train, y_train, x_test, y_test)
    #         np.save(DSA_save_path, value)
    #         value = com_MDSA(model_name, x_train[:5000], x_test)
    #         np.save(MDSA_save_path, value)
    #         value = com_DDR(model_name, x_test, y_test)
    #         np.save(DDR_save_path, value)
    #         if i == 0:
    #             break

    load_path = '/data' + os.sep + model_name + os.sep + 'step_9_cal_SADL_value'
    state = ['ori', 'fgsm', 'bim', 'jsma', 'cw']
    rate = [0.3, 0.6, 0.9]
    for i in range(len(state)):
        com_coverage(model_name, load_path, state[i], rate)
    #
    # load_path = '/data' + os.sep + model_name + os.sep + 'step_11_cal_SADL_value'
    # state = ['ori', 'fgsm', 'bim', 'jsma', 'cw']
    # rate = [0.3, 0.6, 0.9]
    # for i in range(len(state)):
    #     com_coverage(model_name, load_path, state[i], rate)

    # path_1 = '/data' + os.sep + model_name + os.sep + 'step_9_cal_SADL_value'
    # path_2 = '/data' + os.sep + model_name + os.sep + 'step_11_cal_SADL_value'
    # state = ['ori', 'fgsm', 'bim', 'jsma', 'cw']
    # rate = [0.3, 0.6, 0.9]
    # for i in range(len(state)):
    #     com_combined_coverage(model_name, path_1, path_2, state[i], rate)
