import keras
from keras.optimizers import SGD
k=10

from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.datasets import cifar10

def load_cifar10():
    (train_datas, train_labels), (test_datas, test_labels) = cifar10.load_data()
    # 将数据转换为浮点型并归一化到 [0,1] 范围
    train_datas = train_datas.astype('float32') / 255
    test_datas = test_datas.astype('float32') / 255
    # 将标签转换为 one-hot 编码
    train_labels = keras.utils.to_categorical(train_labels, k)
    test_labels = keras.utils.to_categorical(test_labels, k)
    return (train_datas, train_labels), (test_datas, test_labels)

def cifarmodel():
    input_tensor = Input(shape=[32, 32, 3])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(280, activation='relu', name='fc1')(x)
    x = Dense(160, activation='relu', name='fc2')(x)
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    model = keras.Model(input_tensor, x, name='cifarmodel')
    return model

def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # (train_datas, train_labels), (test_datas, test_labels) = load_cifar10()
    # model = cifarmodel()
    # model = compile_model(model)
    # model.fit(train_datas, train_labels, epochs=40, batch_size=256, callbacks=None, verbose=1)
    # model.save("cifarmodel.h5")
    # loss, acc = model.evaluate(test_datas, test_labels, batch_size=128)
    # print('Normal model accurancy: {:5.2f}%'.format(100 * acc))
    #
    (train_datas, train_labels), (test_datas, test_labels) = load_cifar10()
    model = keras.models.load_model('/mnt/sda/project_flc/DeepBoundary/model/vgg16/vgg16.h5')
    model = compile_model(model)
    loss, acc = model.evaluate(test_datas, test_labels, batch_size=128)
    print('Normal model accurancy: {:5.2f}%'.format(100 * acc))

    model.summary()