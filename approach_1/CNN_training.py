'''
this shit will be called to train the cnn model

steps that it will be performing:
1. define hyperparams, like batch_size, learning rate, optimizer, etc. We can also pass them while calling .py file
2. split into train-val and create data generator objects
3. create model object, compile it and train by using keras' fit
4. save the model somewhere in repo (we will need to add it .gitignore because its gonna be huge probably)

below is an inspiration
'''


from config_sc import *
from utils_sc import *
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from models.resnet import res_net_model
from strat_data_generator import DataGenerator
from losses import *
import datetime as dt


def main():
    imb_ratio = imb_ratios[1]
    strat_dims = ['age_mapped']
    train_split = 0.8
    batch_size = 32
    data_path = project_path

    metadf = load_metadf(data_path)
    data_train, data_val, labels, strat_classes_num, cls_num_list = draw_data(
        metadf, imb_ratio, strat_dims, train_split)

    print(strat_classes_num)
    print(cls_num_list)

    params_generator = {'dim': (450, 600, 3),
                        'batch_size': batch_size,
                        'n_classes': 7,
                        'shuffle': True}

    training_generator = DataGenerator(
        data_train, labels, strat_classes_num, imgs_path, **params_generator)
    validation_generator = DataGenerator(
        data_val, labels, strat_classes_num, imgs_path, **params_generator)

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True, histogram_freq=1),
    ]

    model = res_net_model(strat_classes_num, num_res_net_blocks=2)
    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=LDAMLoss(cls_num_list),
                  metrics=['accuracy'])

    model.fit(training_generator, epochs=10,
              validation_data=validation_generator, callbacks=callbacks)
    
    model.save(data_path)


if __name__ == '__main__':
    main()