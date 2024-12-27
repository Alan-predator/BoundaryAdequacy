import math
import os
import numpy as np
from keras.models import load_model

from scipy.spatial.distance import mahalanobis
def cal_Md(output, i, k, coordinate_path):
    train_coordinates = np.load(os.path.join(coordinate_path, str(i) + '_' + str(k) + '_boundary_coordinate.npy'))
    # # change nan to 0
    train_coordinates[np.isnan(train_coordinates)] = 0

    #cal mean and cov of coordinate ima
    # 2*800, following the row to compute mean value
    mean_value = np.mean(train_coordinates, axis=1)
    cov_value = np.cov(train_coordinates)

    Md_per_img_with_coordinate = 0
    #get the col of 2-dim array
    coo_length = len(train_coordinates[0])
    for coo in range(coo_length):
        md_img_with_img = mahalanobis((output[i], output[k]), mean_value, np.linalg.inv(cov_value))
        Md_per_img_with_coordinate += md_img_with_img
    # print(Md_per_img_with_coordinate)
    return Md_per_img_with_coordinate/10000.0



def cal_ori_MdDB(dataset, model, load_path, save_path, path):
    model_path = os.path.join('/mnt/sda/project_flc/DeepBoundary/model', model, model + '.h5')
    model = load_model(model_path)
    for i in range(10):
        print(str(i), ":")
        npy_path = os.path.join(load_path, str(i) + '_pictures.npy')
        pic = np.load(npy_path)
        test_outputs = model.predict(pic)
        scores = []
        #visit predict result of all pic(npy)
        for j in range(len(test_outputs)):
            # compute each pic ed value with other boundary
            score = 0
            for k in range(10):
                if k == i:
                    continue
                temp_score = cal_Md(test_outputs[j], i, k, path)
                score += temp_score
            score = score / 9.0
            # print(score)
            scores.append(score)
        scores = np.array(scores)
        print(scores.shape)
        score_path = os.path.join(save_path, str(i) + '_MdDB.npy')
        np.save(score_path, scores)


def cal_adv_MdDB(dataset, model, load_path, save_path, path, rate):
    model_path = os.path.join('/mnt/sda/project_flc/DeepBoundary/model', model, model + '.h5')
    model = load_model(model_path)
    for r in range(len(rate)):
        print(rate[r])
        state_path = os.path.join(save_path, str(rate[r]))
        if not os.path.exists(state_path):
            os.mkdir(state_path)
        for i in range(10):
            print(str(i), ':')
            # npy_path = os.path.join(load_path, str(rate[r]), 'npy', str(i) + '_pictures.npy')
            npy_path = os.path.join(load_path, str(rate[r]), str(i) + '_pictures.npy')
            pic = np.load(npy_path)
            test_outputs = model.predict(pic)
            scores = []
            for j in range(len(test_outputs)):
                score = 0
                for k in range(10):
                    if k == i:
                        continue
                    temp_score = cal_Md(test_outputs[j], i, k, path)
                    score += temp_score
                score = score / 9.0
                scores.append(score)
            scores = np.array(scores)
            print(scores.shape)
            score_path = os.path.join(state_path, str(i) + '_MdDB.npy')
            np.save(score_path, scores)

def com_coverage(dataset, load_path, save_path, state, rate):
    for s in range(len(state)):
        print(state[s])
        if s == 0:
            path1 = os.path.join(load_path, state[s])
            path2 = os.path.join(save_path, state[s] + '_coverage.npy')
            for i in range(10):
                temp_score = np.load(os.path.join(path1, str(i) + '_MdDB.npy'))
                if i == 0:
                    scores = temp_score
                else:
                    scores = np.concatenate((scores, temp_score), axis=0)
            # cifarmodel
            # scores = scores * 0.9
            # vgg16
            scores = scores * 1.8

            temp = np.zeros(2000)

            for i in range(len(scores)):
                t = math.ceil(scores[i])
                if t < 0:
                    t = -t
                if t >= 2000:
                    print(scores[i], "惊喜度爆炸")
                    continue
                else:
                    temp[t] = 1
            num1 = sum(temp == 1)
            coverage = num1 / 2000.0
            cover_arr = []
            cover_arr.append(coverage)
            cover_arr = np.array(cover_arr)
            np.save(path2, cover_arr)
            print(cover_arr)
        else:
            for r in range(len(rate)):
                print(rate[r])
                path1 = os.path.join(load_path, state[s], str(rate[r]))
                path2 = os.path.join(save_path, state[s] + '_' + str(rate[r]) + '_coverage.npy')
                for i in range(10):
                    temp_score = np.load(os.path.join(path1, str(i) + '_MdDB.npy'))
                    if i == 0:
                        scores = temp_score
                    else:
                        scores = np.concatenate((scores, temp_score), axis=0)
                for i in range(10):
                    temp_score = np.load(os.path.join(load_path, state[0], str(i) + '_MdDB.npy'))
                    scores = np.concatenate((scores, temp_score), axis=0)
                # print(scores)

                # cifarmodel
                # scores = scores * 0.9
                # vgg16
                scores = scores * 1.8

                temp = np.zeros(2000)
                for i in range(len(scores)):
                    t = math.ceil(scores[i])
                    if t < 0:
                        t = -t
                    if t >= 2000:
                        print(scores[i], "惊喜度爆炸")
                        continue
                    else:
                        temp[t] = 1
                # print(temp)
                num1 = sum(temp == 1)
                coverage = num1 / 2000.0
                cover_arr = []
                cover_arr.append(coverage)
                cover_arr = np.array(cover_arr)
                np.save(path2, cover_arr)
                print(cover_arr)

if __name__ == '__main__':
    dataset = 'cifar'
    # model = 'cifarmodel'
    model = 'vgg16'

    #compute EdDB
    # load_path = os.path.join('/mnt/sda/project_flc/DeepBoundary/data', model, 'step_7_test')
    # path = os.path.join('/mnt/sda/project_flc/DeepBoundary/data', model, 'step_3_save_boundary_pic')
    # save_path = os.path.join('/mnt/sda/project_flc/DeepBoundary/data', model, 'step_13_MdDB')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # state = ['ori', 'fgsm', 'bim', 'jsma', 'cw']
    # rate = [0.3, 0.6, 0.9]
    # for i in range(len(state)):
    #     print(state[i], ':')
    #     data_path = load_path + os.sep + state[i]
    #     save_state_path = os.path.join(save_path, state[i])
    #     if not os.path.exists(save_state_path):
    #         os.mkdir(save_state_path)
    #     if i == 0:
    #         continue
    #         # cal_ori_MdDB(dataset, model, data_path, save_state_path, path)
    #     else:
    #         cal_adv_MdDB(dataset, model, data_path, save_state_path, path, rate)
    #         # pass
    #
    load_path = os.path.join('/mnt/sda/project_flc/DeepBoundary/data', model, 'step_13_MdDB')
    save_path = os.path.join('/mnt/sda/project_flc/DeepBoundary/data', model, 'step_13_MdDB_coverage')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    state = ['ori', 'fgsm', 'bim', 'jsma', 'cw']
    rate = [0.3, 0.6, 0.9]
    com_coverage(dataset, load_path, save_path, state, rate)