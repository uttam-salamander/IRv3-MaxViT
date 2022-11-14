import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from tensorflow import keras

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras import backend
from tensorflow.keras.models import Model

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 16
num_epochs = 100
num_classes = 5
image_size = 512  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]


def conv2d(x,numfilt,filtsz,strides=1,pad='same',act=True,name=None):
  x = Conv2D(numfilt,filtsz,strides,padding=pad,data_format='channels_last',use_bias=False,name=name+'conv2d')(x)
  x = BatchNormalization(axis=3,scale=False,name=name+'conv2d'+'bn')(x)
  if act:
    x = Activation('relu',name=name+'conv2d'+'act')(x)
  return x

from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    mhsa_with_multi_head_relative_position_embedding,
    MultiHeadRelativePositionalEmbedding,
    se_module,
    output_block,
    window_attention,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.mlp_family.res_mlp import ChannelAffine

PRETRAINED_DICT = {
    "maxvit_tiny": {"imagenet": {224: "e5cfd6a6bd4dea939860b6d8a29a911a"}},
    "maxvit_small": {"imagenet": {224: "6bbaff1c6316486c3ac29b607d9ebb13"}},
    "maxvit_base": {"imagenet": {224: "00c833043b87ef2861ecf79820d827e0"}},
    "maxvit_large": {"imagenet": {224: "93d079fa8171986cc272f6fb4e9b0255"}},
}


def res_MBConv(inputs, output_channel, conv_short_cut=True, strides=1, expansion=4, se_ratio=0, use_torch_mode=False, drop_rate=0, activation="gelu", name=""):
    if use_torch_mode:
        use_torch_padding, epsilon, momentum = True, 1e-5, 0.9
    else:
        use_torch_padding, epsilon, momentum = False, 0.001, 0.99

    if strides > 1:
        shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs)
        shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, use_bias=True, name=name + "shortcut_") if conv_short_cut else shortcut
    else:
        shortcut = inputs

    # MBConv
    preact = batchnorm_with_activation(inputs, activation=None, zero_gamma=False, epsilon=epsilon, momentum=momentum, name=name + "preact_")
    nn = conv2d_no_bias(preact, output_channel * expansion, 1, strides=1, padding="same", name=name + "expand_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=epsilon, momentum=momentum, name=name + "expand_")
    nn = depthwise_conv2d_no_bias(nn, 3, strides=strides, padding="SAME", use_torch_padding=use_torch_padding, name=name + "MB_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, epsilon=epsilon, momentum=momentum, name=name + "MB_dw_")
    if se_ratio:
        nn = se_module(nn, se_ratio=se_ratio / expansion, activation="swish", name=name + "se/")
    nn = conv2d_no_bias(nn, output_channel, 1, strides=1, use_bias=True, padding="same", name=name + "MB_pw_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    # print(f"{shortcut.shape = }, {nn.shape = }, {strides = }")
    return keras.layers.Add(name=name + "output")([shortcut, nn])


def res_attn_ffn(inputs, output_channel, head_dimension=32, window_size=7, expansion=4, is_grid=False, drop_rate=0, layer_scale=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]
    attn = layer_norm(inputs, name=name + "attn_preact_")
    num_heads = attn.shape[-1] // head_dimension
    attention_block = lambda inputs, num_heads, name: mhsa_with_multi_head_relative_position_embedding(
        inputs, num_heads=num_heads, qkv_bias=True, out_bias=True, name=name
    )
    attn = window_attention(attn, window_size=window_size, num_heads=num_heads, is_grid=is_grid, attention_block=attention_block, name=name + "window_mhsa/")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    # print(f"{name = }, {inputs.shape = }, {shortcut.shape = }, {attn.shape = }")
    attn = keras.layers.Add(name=name + "attn_output")([inputs, attn])

    ffn = layer_norm(attn, name=name + "ffn_preact_")
    ffn = keras.layers.Dense(input_channel * expansion, name=name + "ffn/1_dense")(ffn)
    ffn = activation_by_name(ffn, activation=activation, name=name)
    ffn = keras.layers.Dense(input_channel, name=name + "ffn/2_dense")(ffn)
    ffn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "2_gamma")(ffn) if layer_scale >= 0 else ffn
    ffn = drop_block(ffn, drop_rate=drop_rate, name=name + "ffn_")
    return keras.layers.Add(name=name + "ffn_output")([attn, ffn])

def MaxViT_Custom(x,
    num_blocks,
    out_channels,
    stem_width=64,
    strides=[1,1,1,1],
    expansion=4,
    se_ratio=0.25,
    head_dimension=32,
    window_ratio=32,
    output_filter=-1,  # -1 for out_channels[-1], 0 to disable
    use_torch_mode=False,
    layer_scale=-1,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu/app",  # means tf.nn.gelu(approximate=True)
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="maxvit",
    kwargs=None,):
  

    window_size = [int(tf.math.ceil(input_shape[0] / window_ratio)), int(tf.math.ceil(input_shape[1] / window_ratio))]

    attn_ffn_common_kwargs = {
        "head_dimension": head_dimension,
        "window_size": window_size,
        "expansion": expansion,
        "layer_scale": layer_scale,
        "activation": activation,
    }

    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
        stack_se_ratio = se_ratio[stack_id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        stack_strides = strides[stack_id] if isinstance(strides, (list, tuple)) else strides
        for block_id in range(num_block):
            name = "stack_{}_block_{}/".format(stack_id + 1, block_id + 1)
            stride = stack_strides if block_id == 0 else 1
            conv_short_cut = True if block_id == 0 and x.shape[-1] != out_channel else False
            block_se_ratio = stack_se_ratio[block_id] if isinstance(stack_se_ratio, (list, tuple)) else stack_se_ratio
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            x = res_MBConv(
                x, out_channel, conv_short_cut, stride, expansion, block_se_ratio, use_torch_mode, block_drop_rate, activation, name=name + "mbconv/"
            )
            x = res_attn_ffn(x, out_channel, is_grid=False, drop_rate=block_drop_rate, name=name + "block_", **attn_ffn_common_kwargs)
            x = res_attn_ffn(x, out_channel, is_grid=True, drop_rate=block_drop_rate, name=name + "grid_", **attn_ffn_common_kwargs)


    return x
  
 
def incresA(x,scale,name=None):   #block35
    pad = 'same'
    branch0 = conv2d(x,32,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,32,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,32,3,1,pad,True,name=name+'b1_2')
    branch2 = conv2d(x,32,1,1,pad,True,name=name+'b2_1')
    branch2 = conv2d(branch2,48,3,1,pad,True,name=name+'b2_2')
    branch2 = conv2d(branch2,64,3,1,pad,True,name=name+'b2_3')
    branches = [branch0,branch1,branch2]
    mixed = Concatenate(axis=3, name=name + '_concat')(branches)
    filt_exp_1x1 = conv2d(mixed,backend.int_shape(x)[channel_axis],1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresB(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,128,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,160,[1,7],1,pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,192,[7,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d(mixed,backend.int_shape(x)[channel_axis],1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresC(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,192,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,224,[1,3],1,pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,256,[3,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d(mixed,backend.int_shape(x)[channel_axis],1,1,pad,False,name=name+'fin1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_saling')([x, filt_exp_1x1])
    return final_lay

################            Stem block          ##########################
img_input = Input(shape=(299,299,3))

x = conv2d(img_input,32,3,2,'valid',True,name='conv1')
x = conv2d(x,32,3,1,'valid',True,name='conv2')
x = conv2d(x,64,3,1,'valid',True,name='conv3')
x = MaxPooling2D(3, padding = "same", strides=2)(x)

x_tA = conv2d(x,128,1,1,'same',True,name='conv_trans')
x_tA = MaxPooling2D(3, padding = "valid", strides=2)(x_tA)
 
x = conv2d(x,80,1,1,'valid',True,name='conv1_2')
x = conv2d(x,192,3,1,'valid',True,name='conv2_2')
x = MaxPooling2D(3, strides=2)(x)


x_11 = conv2d(x,96,1,1,'valid',True,name='stem_br_11')

x_21 = conv2d(x,48,1,1,'same',True,name='stem_br_211')
x_21 = conv2d(x_21,64,5,1,'same',True,name='stem_br_212')

x_31 = conv2d(x,64,1,1,'valid',True,name='stem_br_31')
x_31 = conv2d(x_31,96,3,1,'same',True,name='stem_br_32')
x_31 = conv2d(x_31,96,3,1,'same',True,name='stem_br_33')

x_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
x_pool = conv2d(x_pool,64,1,1,'valid',True,name='stem_br_pool')
branches = [x_11,x_21,x_31,x_pool]

channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
x = layers.Concatenate(axis=channel_axis, name='stem_concat')(branches)

############################################################################

#Inception-ResNet-A modules
x = incresA(x,0.15,name='incresA_1')
x = incresA(x,0.15,name='incresA_2')
x = incresA(x,0.15,name='incresA_3')
x = incresA(x,0.15,name='incresA_4')

x_tA = MaxViT_Custom(x_tA,input_shape= x.shape[1:], num_blocks = [0,4,0,0], out_channels = [64 ,128, 256, 512],  stem_width = 64, num_classes=0, drop_connect_rate=0)

#35 × 35 to 17 × 17 reduction module.
x_red_11 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_1')(x)

x_red_12 = conv2d(x,384,3,2,'valid',True,name='x_red1_c1')

x_red_13 = conv2d(x,256,1,1,'same',True,name='x_red1_c2_1')
x_red_13 = conv2d(x_red_13,256,3,1,'same',True,name='x_red1_c2_2')
x_red_13 = conv2d(x_red_13,384,3,2,'valid',True,name='x_red1_c2_3')

x_tA = MaxPooling2D(3,strides=2,padding='valid',name='x_tA_red_maxpool_1')(x_tA)

x = Concatenate(axis=3, name='red_concat_1')([x_red_11,x_red_12,x_red_13,x_tA])

#Inception-ResNet-B modules

x_tB = conv2d(x,256,1,1,'same',True,name='x_tB_conv')
x_tB = MaxViT_Custom(x_tB,input_shape= x.shape[1:], num_blocks = [0,0,5,0], out_channels = [64 ,128, 256, 512],  stem_width = 64, num_classes=0, drop_connect_rate=0)
 

x = incresB(x,0.1,name='incresB_1')
x = incresB(x,0.1,name='incresB_2')
x = incresB(x,0.1,name='incresB_3')
x = incresB(x,0.1,name='incresB_4')
x = incresB(x,0.1,name='incresB_5')
x = incresB(x,0.1,name='incresB_6')
x = incresB(x,0.1,name='incresB_7')

#17 × 17 to 8 × 8 reduction module.
x_red_21 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_2')(x)

x_red_22 = conv2d(x,256,1,1,'same',True,name='x_red2_c11')
x_red_22 = conv2d(x_red_22,384,3,2,'valid',True,name='x_red2_c12')

x_red_23 = conv2d(x,256,1,1,'same',True,name='x_red2_c21')
x_red_23 = conv2d(x_red_23,256,3,2,'valid',True,name='x_red2_c22')

x_red_24 = conv2d(x,256,1,1,'same',True,name='x_red2_c31')
x_red_24 = conv2d(x_red_24,256,3,1,'same',True,name='x_red2_c32')
x_red_24 = conv2d(x_red_24,256,3,2,'valid',True,name='x_red2_c33')

x_tB = MaxPooling2D(3,strides=2,padding='valid',name='x_tB_red_maxpool_2')(x_tB)

x = Concatenate(axis=3, name='red_concat_2')([x_red_21,x_red_22,x_red_23,x_red_24,x_tB])

#Inception-ResNet-C modules

x_tC = conv2d(x,512,1,1,'same',True,name='conv4')
x_tC = MaxViT_Custom(x_tC,input_shape= x.shape[1:], num_blocks = [0,0,0,2], out_channels = [64 ,128, 256, 512],  stem_width = 64, num_classes=0, drop_connect_rate=0)

x = incresC(x,0.2,name='incresC_1')
x = incresC(x,0.2,name='incresC_2')
x = incresC(x,0.2,name='incresC_3')

x = Concatenate(axis=3, name = "Final_concat")([x,x_tC])

#TOP
x = GlobalAveragePooling2D(data_format='channels_last')(x)
x = Dropout(0.6)(x)
x = Dense(num_classes, activation='softmax')(x)


model = Model(img_input,x,name="Iv3-MaxViT")

import datetime, os
filepath = "/storage/Model/"
def train_model(model): 

  model.compile(optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate, decay=weight_decay),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[ tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                         tf.keras.metrics.TopKCategoricalAccuracy(2, name="top-2-accuracy")])
  
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath= filepath, 
                                                          save_weights_only=True, verbose =1)
  
  logdir = os.path.join("/storage/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

  model.fit(train_batches,
            batch_size=batch_size,
            epochs=100,
            validation_data=val_batches, 
            callbacks=[tensorboard_callback,checkpoint_callback])

!kill 176
%tensorboard --logdir logs --bind_all

model.load_weights("./storage/model_final")
train_model(model)
