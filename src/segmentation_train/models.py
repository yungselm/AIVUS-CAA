from typing import Optional, Tuple

from deep_utils import split_extension
from keras_unet_collection import models as unet_models
from tensorflow import keras
from tensorflow.keras import layers

from configs import *


def SimpleConv(shape=SHAPE, n_classses=1):
    input_img = keras.Input(shape=shape)

    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
    x5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
    x5 = x3 + x5
    x6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x5)
    decoded = layers.Conv2D(n_classses, (3, 3), activation=None, padding='same')(x6)

    model = keras.Model(input_img, decoded)
    return model


################################################################################
# 2
################################################################################
def att_unet_2d(shape=SHAPE, output_activation=LACT):
    model = unet_models.att_unet_2d(shape, filter_num=[16, 32, 64], n_labels=1,
                                    stack_num_down=2, stack_num_up=2, activation='ReLU',
                                    atten_activation='ReLU', attention='add',
                                    output_activation=output_activation,
                                    batch_norm=True, pool=False, unpool=False,
                                    # backbone='VGG16', weights='imagenet',
                                    # freeze_backbone=True, freeze_batch_norm=True,
                                    name='attunet')
    return model


################################################################################
# 3
################################################################################
def u2net_2d(shape=SHAPE, output_activation=LACT):
    model = unet_models.u2net_2d(shape, n_labels=1,
                                 filter_num_down=[16, 32], filter_num_up=[16, 32],
                                 filter_mid_num_down=[16, 32], filter_mid_num_up=[16, 32],
                                 filter_4f_num=[16, 32], filter_4f_mid_num=[16, 32],
                                 activation='ReLU', output_activation=output_activation,
                                 batch_norm=True, pool=False, unpool=False, deep_supervision=True, name='u2net')
    return model


################################################################################
# 3' or 11 Deep U2Net
################################################################################
def u2net_2dD(shape=SHAPE, output_activation=LACT):
    model = unet_models.u2net_2d(shape, n_labels=1,
                                 filter_num_down=[16, 32], filter_num_up=[16, 32],
                                 filter_mid_num_down=[16, 32], filter_mid_num_up=[16, 32],
                                 filter_4f_num=[16, 32], filter_4f_mid_num=[16, 32],
                                 activation='ReLU', output_activation=output_activation,
                                 batch_norm=True, pool=False, unpool=False, deep_supervision=True, name='u2net',
                                 freeze_batch_norm=True)
    return model


################################################################################
# 4
################################################################################
def unet_3plus_2d(shape=SHAPE, output_activation=LACT):
    model = unet_models.unet_3plus_2d(shape, n_labels=1, filter_num_down=[16, 32, 64],
                                      filter_num_skip='auto', filter_num_aggregate='auto',
                                      stack_num_down=2, stack_num_up=2, activation='ReLU',
                                      output_activation=output_activation,
                                      batch_norm=True, pool='max', unpool=False, deep_supervision=True,
                                      name='unet3plus')
    return model


################################################################################
# 5
################################################################################
def transunet_2d(shape=SHAPE, output_activation: str = LACT):
    model = unet_models.transunet_2d(shape, filter_num=[16, 32], n_labels=1, stack_num_down=1, stack_num_up=1,
                                     # proj_dim=16,
                                     # embed_dim=384,
                                     num_heads=1, num_transformer=1,
                                     num_mlp=50,
                                     # num_heads=1, num_transformer=1,
                                     activation='ReLU', mlp_activation='ReLU',
                                     output_activation=output_activation,
                                     batch_norm=True, pool=True, unpool='bilinear', name='transunet')
    return model


################################################################################
# 6
################################################################################
def vnet_2d(shape=SHAPE, output_activation=LACT):
    model = unet_models.vnet_2d(shape, filter_num=[16, 32, 64], n_labels=1,
                                res_num_ini=1, res_num_max=3,
                                activation='PReLU', output_activation=output_activation,
                                batch_norm=True, pool=False, unpool=False, name='vnet')
    return model


################################################################################    
# 7
################################################################################    
def unet_plus_2d(shape=SHAPE, output_activation=LACT):
    model = unet_models.unet_plus_2d(shape, [16, 32, 64], n_labels=1,
                                     stack_num_down=2, stack_num_up=2,
                                     activation='LeakyReLU', output_activation=output_activation,
                                     batch_norm=False, pool='max', unpool=False, deep_supervision=True, name='xnet')
    return model


def r2_unet_2d(shape=SHAPE, output_activation=LACT):
    model = unet_models.r2_unet_2d(shape, [16, 32, 64], n_labels=1,
                                   stack_num_down=2, stack_num_up=1, recur_num=2,
                                   activation='ReLU', output_activation=output_activation,
                                   batch_norm=True, pool='max', unpool='nearest', name='r2unet')
    return model


################################################################################
# 9
################################################################################
# def resunet_a_2d():
#     shape=SHAPE
#     model=unet_models.resunet_a_2d(shape, [16, 32, 64], 
#                             dilation_num=[1, 3, 15, 31], 
#                             n_labels=1, aspp_num_down=256, aspp_num_up=128, 
#                             activation='ReLU', output_activation=LACT, 
#                             batch_norm=True, pool=False, unpool='nearest', name='resunet')
#     return model


def resunet_a_2d(shape, output_activation=LACT):
    model = unet_models.resunet_a_2d(shape, [8, 16, 32],
                                     dilation_num=[1, 3, 15, 31],
                                     n_labels=1, aspp_num_down=256, aspp_num_up=128,
                                     activation='ReLU', output_activation=output_activation,
                                     batch_norm=True, pool=False, unpool='nearest', name='resunet')
    return model


################################################################################
# 10
################################################################################
def swin_unet_2d(shape, output_activation):
    model = unet_models.swin_unet_2d(shape, filter_num_begin=32, n_labels=1, depth=2, stack_num_down=2, stack_num_up=2,
                                     patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=176,
                                     output_activation=output_activation, shift_window=True, name='swin_unet')
    return model


MODELS = dict(
    SimpleConv=SimpleConv,
    att_unet_2d=att_unet_2d,
    u2net_2d=u2net_2d,
    u2net_2dD=u2net_2dD,
    unet_3plus_2d=unet_3plus_2d,
    transunet_2d=transunet_2d,
    vnet_2d=vnet_2d,
    unet_plus_2d=unet_plus_2d,
    r2_unet_2d=r2_unet_2d,
    resunet_a_2d=resunet_a_2d,
    swin_unet_2d=swin_unet_2d,
)


def get_model(model_name: str, postfix: Optional[str] = None):
    print(f"[INFO] Loading model: {model_name}")
    model_weight_name = split_extension(f'{model_name}.h5', suffix=postfix) if postfix else f"{model_name}.h5"
    model = MODELS[model_name]()
    model.summary()
    return model, model_weight_name
