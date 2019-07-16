"""The code to test training process for pnet"""

import tensorflow as tf
from mtcnn import train_net, PNet


def train_Pnet(training_data, base_lr, loss_weight,
               train_mode, num_epochs,
               load_model=False, load_filename=None,
               save_model=False, save_filename=None,
               num_iter_to_save=10000,
               device='/cpu:0', gpu_memory_fraction=1):

    pnet = tf.Graph()
    with pnet.as_default():
        with tf.device(device):
            train_net(Net=PNet,
                      training_data=training_data,
                      base_lr=base_lr,
                      loss_weight=loss_weight,
                      train_mode=train_mode,
                      num_epochs=num_epochs,
                      load_model=load_model,
                      load_filename=load_filename,
                      save_model=save_model,
                      save_filename=save_filename,
                      num_iter_to_save=num_iter_to_save,
                      gpu_memory_fraction=gpu_memory_fraction)


if __name__ == '__main__':

    load_filename = './pretrained/initial_weight_pnet.npy'
    save_filename = './save_model/new_saver/pnet/pnet'
    training_data = ['./prepare_data/pnet_data_for_cls.tfrecords',
                     './prepare_data/pnet_data_for_bbx.tfrecords']
    device = '/gpu:0'
    train_Pnet(training_data=training_data,
               base_lr=0.0001,
               loss_weight=[1.0, 0.5, 0.5],
               train_mode=2,
               num_epochs=[200, 200, 200],
               load_model=False,
               load_filename=load_filename,
               save_model=True,
               save_filename=save_filename,
               num_iter_to_save=1000,
               device=device,
               gpu_memory_fraction=0.3)
