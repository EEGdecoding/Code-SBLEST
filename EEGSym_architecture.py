from tensorflow.keras.layers import Activation, Input, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Conv3D, Add, AveragePooling3D
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

def EEGSym(input_time=3000, fs=128, ncha=8, filters_per_branch=8,
           scales_time=(500, 250, 125), dropout_rate=0.25, activation='elu',
           n_classes=2, learning_rate=0.001, ch_lateral=3,
           spatial_resnet_repetitions=1, residual=True, symmetric=True):

    """Keras implementation of EEGSym.

    This model was initially designed for MI decodification of either
    left/right hand.
    Hyperparameters and architectural choices are explained in the
    original article.

    Parameters
    ----------
    input_time : int
        EEG epoch time in milliseconds.
    fs : int
        Sample rate of the EEG.
    ncha :
        Number of input channels.
    filters_per_branch : int
        Number of filters in each Inception branch. The number should be
        multiplies of 8.
    scales_time : list
        Temporal scale of the temporal convolutions on first Inception module.
        This parameter determines the kernel sizes of the filters.
    dropout_rate : float
        Dropout rate
    activation : str
        Activation
    n_classes : int
        Number of output classes
    learning_rate : float
        Learning rate
    ch_lateral : int
        Number of channels that are attributed to one hemisphere of the head.
    spatial_resnet_repetitions: int
        Number of repetitions of the operations of spatial analysis at each
        step of the spatiotemporal analysis. In the original publication this
        value was set to 1 and not tested its variations.
    residual : Bool
        If the residual operations are present in EEGSym architecture.
    symmetric : Bool
        If the architecture considers the parameter ch_lateral to create two
        symmetric inputs of the electrodes.

    Returns
    -------
    model : keras.models.Model
        Keras model already compiled and ready to work
    """

    # ======================================================================== #
    # ================== GENERAL INCEPTION/RESIDUAL MODULE =================== #
    def general_module(input, scales_samples, filters_per_branch, ncha,
                         activation, dropout_rate, average,
                         spatial_resnet_repetitions=1, residual=True,
                         init=False):
        """General inception/residual module.

            This function returns the input with the operations of a
            inception or residual module from the publication applied.

            Parameters
            ----------
            input : list
                List of input blocks to the module.
            scales_samples : list
                List of samples size of the temporal operations kernels.
            filters_per_branch : int
                Number of filters in each Inception branch. The number should be
                multiplies of 8.
            ncha :
                Number of input channels.
            activation : str
                Activation
            dropout_rate : float
                Dropout rate
            spatial_resnet_repetitions: int
                Number of repetitions of the operations of spatial analysis at
                each step of the spatiotemporal analysis. In the original
                publication this value was set to 1 and not tested its
                variations.
            residual : Bool
                If the residual operations are present in EEGSym architecture.
            init : Bool
                If the module is the first one applied to the input, to apply a
                channel merging operation if the architecture does not include
                residual operations.

            Returns
            -------
            block_out : list
                List of outputs modules
        """
        block_units = list()
        unit_conv_t = list()
        unit_batchconv_t = list()

        for i in range(len(scales_samples)):
            unit_conv_t.append(Conv3D(filters=filters_per_branch,
                                      kernel_size=(1, scales_samples[i], 1),
                                      kernel_initializer='he_normal',
                                      padding='same'))
            unit_batchconv_t.append(BatchNormalization())

        if ncha != 1:
            unit_dconv = list()
            unit_batchdconv = list()
            unit_conv_s = list()
            unit_batchconv_s = list()
            for i in range(spatial_resnet_repetitions):
                # 3D Implementation of DepthwiseConv
                unit_dconv.append(Conv3D(kernel_size=(1, 1, ncha),
                                         filters=filters_per_branch * len(
                                             scales_samples),
                                         groups=filters_per_branch * len(
                                             scales_samples),
                                         use_bias=False,
                                         padding='valid'))
                unit_batchdconv.append(BatchNormalization())

                unit_conv_s.append(Conv3D(kernel_size=(1, 1, ncha),
                                          filters=filters_per_branch,
                                          # groups=filters_per_branch,
                                          use_bias=False,
                                          strides=(1, 1, 1),
                                          kernel_initializer='he_normal',
                                          padding='valid'))
                unit_batchconv_s.append(BatchNormalization())

            unit_conv_1 = Conv3D(kernel_size=(1, 1, 1),
                                 filters=filters_per_branch,
                                 use_bias=False,
                                 kernel_initializer='he_normal',
                                 padding='valid')
            unit_batchconv_1 = BatchNormalization()

        for j in range(len(input)):
            block_side_units = list()
            for i in range(len(scales_samples)):
                unit = input[j]
                unit = unit_conv_t[i](unit)

                unit = unit_batchconv_t[i](unit)
                unit = Activation(activation)(unit)
                unit = Dropout(dropout_rate)(unit)

                block_side_units.append(unit)
            block_units.append(block_side_units)
        # Concatenation
        block_out = list()
        for j in range(len(input)):
            if len(block_units[j]) != 1:
                block_out.append(
                    keras.layers.concatenate(block_units[j], axis=-1))
            else:
                block_out.append(block_units[j][0])

            if residual:
                if len(block_units[j]) != 1:
                    block_out_temp = input[j]
                else:
                    block_out_temp = input[j]
                    block_out_temp = unit_conv_1(block_out_temp)

                    block_out_temp = unit_batchconv_1(block_out_temp)
                    block_out_temp = Activation(activation)(block_out_temp)
                    block_out_temp = Dropout(dropout_rate)(block_out_temp)

                block_out[j] = Add()([block_out[j], block_out_temp])

            if average != 1:
                block_out[j] = AveragePooling3D((1, average, 1))(block_out[j])

        if ncha != 1:
            for i in range(spatial_resnet_repetitions):
                block_out_temp = list()
                for j in range(len(input)):
                    if len(scales_samples) != 1:
                        if residual:
                            block_out_temp.append(block_out[j])

                            block_out_temp[j] = unit_dconv[i](block_out_temp[j])

                            block_out_temp[j] = unit_batchdconv[i](
                                block_out_temp[j])
                            block_out_temp[j] = Activation(activation)(
                                block_out_temp[j])
                            block_out_temp[j] = Dropout(dropout_rate)(
                                block_out_temp[j])

                            block_out[j] = Add()(
                                [block_out[j], block_out_temp[j]])

                        elif init:
                            block_out[j] = unit_dconv[i](block_out[j])
                            block_out[j] = unit_batchdconv[i](block_out[j])
                            block_out[j] = Activation(activation)(block_out[j])
                            block_out[j] = Dropout(dropout_rate)(block_out[j])
                    else:
                        if residual:
                            block_out_temp.append(block_out[j])

                            block_out_temp[j] = unit_conv_s[i](
                                block_out_temp[j])
                            block_out_temp[j] = unit_batchconv_s[i](
                                block_out_temp[j])
                            block_out_temp[j] = Activation(activation)(
                                block_out_temp[j])
                            block_out_temp[j] = Dropout(dropout_rate)(
                                block_out_temp[j])

                            block_out[j] = Add()(
                                [block_out[j], block_out_temp[j]])
        return block_out
    # ============================= CALCULATIONS ============================= #
    input_samples = int(input_time * fs / 1000)
    scales_samples = [int(s * fs / 1000) for s in scales_time]

    # ================================ INPUT ================================= #
    input_layer = Input((input_samples, ncha, 1))
    input = tf.expand_dims(input_layer, axis=1)
    if symmetric:
        superposition = False
        if ch_lateral < ncha // 2:
            superposition = True
        ncha = ncha - ch_lateral

        left_idx = list(range(ch_lateral))
        ch_left = tf.gather(input, indices=left_idx, axis=-2)
        right_idx = list(np.array(left_idx) + int(ncha))
        ch_right = tf.gather(input, indices=right_idx, axis=-2)

        if superposition:
            # ch_central = crop(3, self.ch_lateral, -self.ch_lateral)(input)
            central_idx = list(
                np.array(range(ncha - ch_lateral)) + ch_lateral)
            ch_central = tf.gather(input, indices=central_idx, axis=-2)

            left_init = keras.layers.concatenate((ch_left, ch_central),
                                                 axis=-2)
            right_init = keras.layers.concatenate((ch_right, ch_central),
                                                  axis=-2)
        else:
            left_init = ch_left
            right_init = ch_right

        input = keras.layers.concatenate((left_init, right_init), axis=1)
        division = 2
    else:
        division = 1
    # ======================== TEMPOSPATIAL ANALYSIS ========================= #
    # ============================ Inception (x2) ============================ #
    b1_out = general_module([input],
                              scales_samples=scales_samples,
                              filters_per_branch=filters_per_branch,
                              ncha=ncha,
                              activation=activation,
                              dropout_rate=dropout_rate, average=2,
                              spatial_resnet_repetitions=spatial_resnet_repetitions,
                              residual=residual, init=True)

    b2_out = general_module(b1_out, scales_samples=[int(x / 4) for x in
                                                      scales_samples],
                              filters_per_branch=filters_per_branch,
                              ncha=ncha,
                              activation=activation,
                              dropout_rate=dropout_rate, average=2,
                              spatial_resnet_repetitions=spatial_resnet_repetitions,
                              residual=residual)
    # ============================== Residual (x3) =========================== #
    b3_u1 = general_module(b2_out, scales_samples=[16],
                             filters_per_branch=int(
                                 filters_per_branch * len(
                                     scales_samples) / 2),
                             ncha=ncha,
                             activation=activation,
                             dropout_rate=dropout_rate, average=2,
                             spatial_resnet_repetitions=spatial_resnet_repetitions,
                             residual=residual)
    b3_u1 = general_module(b3_u1,
                             scales_samples=[8],
                             filters_per_branch=int(
                                 filters_per_branch * len(
                                     scales_samples) / 2),

                             ncha=ncha,
                             activation=activation,
                             dropout_rate=dropout_rate, average=2,
                             spatial_resnet_repetitions=spatial_resnet_repetitions,
                             residual=residual)
    b3_u2 = general_module(b3_u1, scales_samples=[4],
                             filters_per_branch=int(
                                 filters_per_branch * len(
                                     scales_samples) / 4),
                             ncha=ncha,
                             activation=activation,
                             dropout_rate=dropout_rate, average=2,
                             spatial_resnet_repetitions=spatial_resnet_repetitions,
                             residual=residual)
    # ========================== TEMPORAL REDUCTION ========================== #
    t_red = b3_u2[0]
    for _ in range(1):
        t_red_temp = t_red
        t_red_temp = Conv3D(kernel_size=(1, 4, 1),
                            filters=int(filters_per_branch * len(
                                scales_samples) / 4),
                            use_bias=False,
                            strides=(1, 1, 1),
                            kernel_initializer='he_normal',
                            padding='same')(t_red_temp)
        t_red_temp = BatchNormalization()(t_red_temp)
        t_red_temp = Activation(activation)(t_red_temp)
        t_red_temp = Dropout(dropout_rate)(t_red_temp)

        if residual:
            t_red = Add()([t_red, t_red_temp])
        else:
            t_red = t_red_temp

    t_red = AveragePooling3D((1, 2, 1))(t_red)

    # =========================== CHANNEL MERGING ============================ #
    ch_merg = t_red
    if residual:
        for _ in range(2):
            ch_merg_temp = ch_merg
            ch_merg_temp = Conv3D(kernel_size=(division, 1, ncha),
                                  filters=int(filters_per_branch * len(
                                      scales_samples) / 4),
                                  use_bias=False,
                                  strides=(1, 1, 1),
                                  kernel_initializer='he_normal',
                                  padding='valid')(ch_merg_temp)
            ch_merg_temp = BatchNormalization()(ch_merg_temp)
            ch_merg_temp = Activation(activation)(ch_merg_temp)
            ch_merg_temp = Dropout(dropout_rate)(ch_merg_temp)

            ch_merg = Add()([ch_merg, ch_merg_temp])

        ch_merg = Conv3D(kernel_size=(division, 1, ncha),
                         filters=int(
                             filters_per_branch * len(scales_samples) / 4),
                         groups=int(
                             filters_per_branch * len(scales_samples) / 8),
                         use_bias=False,
                         padding='valid')(ch_merg)
        ch_merg = BatchNormalization()(ch_merg)
        ch_merg = Activation(activation)(ch_merg)
        ch_merg = Dropout(dropout_rate)(ch_merg)
    else:
        if symmetric:
            ch_merg = Conv3D(kernel_size=(division, 1, 1),
                             filters=int(
                                 filters_per_branch * len(
                                     scales_samples) / 4),
                             groups=int(
                                 filters_per_branch * len(
                                     scales_samples) / 8),
                             use_bias=False,
                             padding='valid')(ch_merg)
            ch_merg = BatchNormalization()(ch_merg)
            ch_merg = Activation(activation)(ch_merg)
            ch_merg = Dropout(dropout_rate)(ch_merg)
    # ========================== TEMPORAL MERGING ============================ #
    t_merg = ch_merg
    for _ in range(1):
        if residual:
            t_merg_temp = t_merg
            t_merg_temp = Conv3D(kernel_size=(1, input_samples // 64, 1),
                                 filters=int(filters_per_branch * len(
                                     scales_samples) / 4),
                                 use_bias=False,
                                 strides=(1, 1, 1),
                                 kernel_initializer='he_normal',
                                 padding='valid')(t_merg_temp)
            t_merg_temp = BatchNormalization()(t_merg_temp)
            t_merg_temp = Activation(activation)(t_merg_temp)
            t_merg_temp = Dropout(dropout_rate)(t_merg_temp)

            t_merg = Add()([t_merg, t_merg_temp])
        else:
            t_merg_temp = t_merg
            t_merg_temp = Conv3D(kernel_size=(1, input_samples // 64, 1),
                                 filters=int(filters_per_branch * len(
                                     scales_samples) / 4),
                                 use_bias=False,
                                 strides=(1, 1, 1),
                                 kernel_initializer='he_normal',
                                 padding='same')(t_merg_temp)
            t_merg_temp = BatchNormalization()(t_merg_temp)
            t_merg_temp = Activation(activation)(t_merg_temp)
            t_merg_temp = Dropout(dropout_rate)(t_merg_temp)
            t_merg = t_merg_temp

    t_merg = Conv3D(kernel_size=(1, input_samples // 64, 1),
                    filters=int(
                        filters_per_branch * len(scales_samples) / 4) * 2,
                    groups=int(
                        filters_per_branch * len(scales_samples) / 4),
                    use_bias=False,
                    padding='valid')(t_merg)
    t_merg = BatchNormalization()(t_merg)
    t_merg = Activation(activation)(t_merg)
    t_merg = Dropout(dropout_rate)(t_merg)
    # =============================== OUTPUT ================================= #
    output = t_merg
    for _ in range(4):
        output_temp = output
        output_temp = Conv3D(kernel_size=(1, 1, 1),
                             filters=int(
                                 filters_per_branch * len(
                                     scales_samples) / 2),
                             use_bias=False,
                             strides=(1, 1, 1),
                             kernel_initializer='he_normal',
                             padding='valid')(output_temp)
        output_temp = BatchNormalization()(output_temp)
        output_temp = Activation(activation)(output_temp)
        output_temp = Dropout(dropout_rate)(output_temp)
        if residual:
            output = Add()([output, output_temp])
        else:
            output = output_temp
    output = Flatten()(output)
    output_layer = Dense(n_classes, activation='softmax')(output)
    # Create and compile model
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate,
                                      beta_1=0.9, beta_2=0.999,
                                      amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model