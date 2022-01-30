import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras import activations
from tensorflow.keras import regularizers

from tensorflow.keras.applications import VGG16


class ConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, input_shape, use_bias, kernel_initializer, kernel_l2, bias_l2):
        super(ConvModule, self).__init__()
        param_dict = {
            'filters': filters, 
            'kernel_size': kernel_size, 
            'padding': padding, 
            'input_shape': input_shape, 
            'use_bias': use_bias, 
            'kernel_initializer': kernel_initializer
        }
        assert kernel_l2 != 0 or kernel_l2 == bias_l2, "It is not possible to set bias_l2 alone"
        if bias_l2 != 0:
            param_dict['bias_regularizer'] = regularizers.l2(bias_l2)
        if kernel_l2 != 0:
            param_dict['kernel_regularizer'] = regularizers.l2(kernel_l2)
        
        self.conv = layers.Conv2D(**param_dict)
        
    def call(self, x):
        return self.conv(x)

class VanillaModel(tf.keras.Model):
    def __init__(self, use_bn, dropout, drop_rate,vanilla_conv_count, init, kernel_l2, bias_l2,
            in_channels=3, base_out_channels=16):

        """
        use_bn: Whether to use Batch Normalization
        """
        super(VanillaModel, self).__init__()

        

        if init=='xavier':
            kernel_initializer='glorot_normal'
        elif init=='gaussian':
            kernel_initializer='random_normal'
        else:
            kernel_initializer=None

        out_channels = base_out_channels
        use_bias = False if use_bn else True #bias is not usually used with Batch Norm
        repeat_layers = 2

        #First block: 3,80,80 -> 64,80,80
        self.first_block = FirstBlock(in_channels, out_channels, use_bn, dropout, drop_rate, use_bias, kernel_initializer, kernel_l2, bias_l2)

        #Intermediate Blocks:   C,H,W -> C*2, H/2, W/2. 
        #Exp: 32,80,80 -> 64, 40,40
        in_channels = 64
        out_channels = 16
        # self.intermediate_blocks = []
        # for i in range(repeat_layers):
        #     #in_channels = #out_channels
        #     #out_channels = #out_channels *2
        #     self.intermediate_blocks.append(IntermediateBlock(in_channels, out_channels, 
        #                                 use_bn, dropout, drop_rate, use_bias, vanilla_conv_count, kernel_initializer, kernel_l2, bias_l2))
        #     in_channels = out_channels
        #     out_channels = out_channels*2
        #Last Block: Conv to Fully Connected Layers
        self.last_block = LastBlock(in_channels*2, out_channels, use_bn, dropout, drop_rate, use_bias, kernel_initializer, kernel_l2, bias_l2)

        self.downsample = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

        #self.dropout = layers.Dropout(drop_rate) if dropout else layers.Lambda(lambda x: x) # Identity layer

    def call(self, x):
        x = self.first_block(x)
        #x_np = x.numpy()
        # for block in self.intermediate_blocks:
        #     #x_np = x.numpy()
        #     x = block(x)
        #     #x_np = x.numpy()
        #     x = self.downsample(x)
        #     #x_np = x.numpy()
        #     #x = self.dropout(x)
        #     #x_np = x.numpy()
        x = self.downsample(x)
        x = self.last_block(x)
        #x_np = x.numpy()
        return x

class FirstBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels , use_bn =False, use_dropout=False, drop_rate=0, use_bias=True, kernel_initializer='glorot_normal', kernel_l2=0, bias_l2=0):
        super(FirstBlock, self).__init__()
        # self.conv = ConvModule(filters=out_channels, kernel_size=(3,3), padding="same", input_shape=(in_channels,), use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_l2=kernel_l2, bias_l2=bias_l2)
        # self.bn = layers.BatchNormalization(axis=-1) if use_bn else layers.Lambda(lambda x: x) #Identity Layer
        #self.dropout = layers.Dropout(drop_rate) if use_dropout else layers.Lambda(lambda x: x) #Identity Layer

        vgg = VGG16(include_top=False, weights='imagenet')
        self.vgg= tf.keras.Sequential(vgg.layers[:7])
        self.vgg.summary()

        for layer in self.vgg.layers[:7]:
            layer.trainable = False


    def call(self, x):
        # x = self.conv(x)
        # x = self.bn(x)
        # #x = self.dropout(x)
        # x = activations.relu(x)
        x = self.vgg(x)
        x = activations.relu(x)
        return x

class IntermediateBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, use_bn =False, use_dropout=False, drop_rate=0, use_bias=True,vanilla_conv_count=1, kernel_initializer='glorot_normal', kernel_l2=0, bias_l2=0):
        super(IntermediateBlock, self).__init__()
        self.conv1 = ConvModule(filters=out_channels, kernel_size=(3,3), padding="same", input_shape=(in_channels,), use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_l2=kernel_l2, bias_l2=bias_l2)
        self.bn1 = layers.BatchNormalization(axis=-1) if use_bn else layers.Lambda(lambda x: x) #Identity Layer        
        self.dropout = layers.Dropout(drop_rate) if use_dropout else layers.Lambda(lambda x: x) #Identity Layer
        self.blocks = []
        for i in range(vanilla_conv_count):

            self.blocks.append(ConvModule(filters=out_channels, kernel_size=(3,3), padding="same", 
                            input_shape=(out_channels,), use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_l2=kernel_l2, bias_l2=bias_l2))
            bn_layer = layers.BatchNormalization(axis=-1) if use_bn else layers.Lambda(lambda x: x) # Identity layer
            self.blocks.append (bn_layer)

            dropout_layer = layers.Dropout(drop_rate) if use_dropout else layers.Lambda(lambda x: x) # Identity Layer
            self.blocks.append(dropout_layer)

    def call(self,x):
        x = activations.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        for block_idx in range(0,len(self.blocks),3):
            x = self.blocks[block_idx](x)      #Conv Block
            x = self.blocks[block_idx+1](x)    #Norm Block
            x = self.blocks[block_idx+2](x)    #Dropout Block
            x = activations.relu(x)
        return x

class LastBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, use_bn=False, use_dropout=False, drop_rate=0, use_bias=True, kernel_initializer='glorot_normal', kernel_l2=0, bias_l2=0):
        super(LastBlock, self).__init__()
        down_list = [64,16,8]
        down_length = len(down_list)
        in_channels = 128
        current_in_channels = in_channels
        
        self.down_blocks = []
        self.pool_blocks = []
        self.bn_blocks = []
        self.drop_blocks = []
        for i in range(0, down_length,1):
            conv = ConvModule(filters=down_list[i], kernel_size=(3,3), padding="same", input_shape=(current_in_channels,), use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_l2=kernel_l2, bias_l2=bias_l2)
            #downsample = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
            bn = layers.BatchNormalization(axis=-1) if use_bn else layers.Lambda(lambda x: x) #Identity Layer
            dropout = layers.Dropout(drop_rate) if use_dropout else layers.Lambda(lambda x: x) #Identity Layer
            current_in_channels=down_list[i]
            self.down_blocks.append(conv)
            #self.pool_blocks.append(downsample)
            self.bn_blocks.append(bn)
            self.drop_blocks.append(dropout)

        in_shape = 10*10*down_list[-1]
        self.output_layer = ConvModule(filters=1, kernel_size=(1,1), padding="same", input_shape=(in_shape,), use_bias=True, kernel_initializer=kernel_initializer, kernel_l2=kernel_l2, bias_l2=bias_l2)
        # self.flatten = layers.Flatten()
        # self.fully1 = layers.Dense(units=3000, kernel_initializer=kernel_initializer)
        # self.bn1 = layers.BatchNormalization(axis=-1) if use_bn else layers.Lambda(lambda x: x) #Identity Layer
        # self.dropout1 = layers.Dropout(drop_rate) if use_dropout else layers.Lambda(lambda x: x) #Identity Layer

        # self.fully2 = layers.Dense(units=1000, kernel_initializer=kernel_initializer)
        # self.bn2 = layers.BatchNormalization(axis=-1) if use_bn else layers.Lambda(lambda x: x) #Identity Layer
        # self.dropout2 = layers.Dropout(drop_rate) if use_dropout else layers.Lambda(lambda x: x) #Identity Layer

        # self.fully3 = layers.Dense(units=1, kernel_initializer=kernel_initializer)
    def call(self,x):

        for conv, bn, drop in zip(self.down_blocks,self.bn_blocks, self.drop_blocks):
            x = activations.relu( bn(drop(conv(x))))
            #x = pool(x)
        x = tf.reshape(x, (x.shape[0], 1,1, x.shape[1] * x.shape[2]* x.shape[3]))
        x = activations.relu(self.output_layer(x))
        #x = self.output_layer(x)
        # x = activations.relu(self.bn(self.conv(x)))
        # x = self.dropout(x)
        #x_np = x.numpy()
        #x = self.flatten(x)
        #x_np = x.numpy()
        
        # x = activations.relu(self.bn1(self.fully1(x)))
        # x = self.dropout1(x)

        # x = activations.relu(self.bn2(self.fully2(x)))
        # x = self.dropout2(x)

        # x = activations.relu(self.fully3(x))
        #x_np = x.numpy()
        x = tf.reshape(x, (x.shape[0], ))
        return x