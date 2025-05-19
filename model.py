#import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Conv1D, Conv2DTranspose, \
    LeakyReLU, Activation, Flatten, Reshape, BatchNormalization, MultiHeadAttention, Add, LayerNormalization, Dropout

from keras.models import Model



def RICR(X_train, y_train, coeffs=(2, 10,), semi=False, label_ind=None, prop_dim=None):
    K.clear_session()

    if not semi:
        coeff_KL, coeff_prop = coeffs
    else:
        coeff_KL, coeff_prop, coeff_prop_semi = coeffs

    latent_dim = 256
    max_filters = 128
    filter_size = [5, 3, 3]
    strides = [2, 2, 1]

    input_dim = X_train.shape[1]
    channel_dim = X_train.shape[2]
    regression_dim = y_train.shape[1]

    encoder_inputs = Input(shape=(input_dim, channel_dim,))
    regression_inputs = Input(shape=(regression_dim,))



    def attention_layer(inputs, latent_dim, num_heads=4, dropout_rate=0.1):

        mha = MultiHeadAttention(num_heads=num_heads, key_dim=latent_dim // num_heads)
        attention_output = mha(query=inputs, value=inputs, key=inputs)
        attention_output = Add()([inputs, attention_output])
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        attention_output = Dropout(dropout_rate)(attention_output)

        return attention_output

    def residual_block(x, filters, kernel_size, strides):
        skip = x
        x = Conv1D(filters, kernel_size, strides=strides, padding='SAME')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Conv1D(filters, kernel_size, strides=1, padding='SAME')(x)  # Keep strides=1 for second conv
        x = BatchNormalization()(x)


        if strides > 1 or x.shape[-1] != skip.shape[-1]:  # If filter dimension differs
            skip = Conv1D(filters, kernel_size=1, strides=strides, padding='SAME')(skip)

        x = Add()([x, skip])
        return LeakyReLU(0.2)(x)

    x = residual_block(encoder_inputs, max_filters // 4, filter_size[0], strides[0])
    x = BatchNormalization()(x)

    x = attention_layer(x, latent_dim)

    x = residual_block(x, max_filters // 2, filter_size[1], strides[1])
    x = BatchNormalization()(x)

    x = residual_block(x, max_filters, filter_size[2], strides[2])
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(1024, activation='sigmoid')(x)
    z_mean = Dense(latent_dim, activation='linear')(x)
    z_log_var = Dense(latent_dim, activation='linear')(x)


    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(encoder_inputs, z, name='encoder')

    if not semi:
        x = Activation('relu')(z_mean)
        x = Dense(128, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        y_hat = Dense(regression_dim, activation='sigmoid')(x)
        regression = Model(encoder_inputs, y_hat, name='target-learning branch')
    else:
        x = Activation('relu')(z_mean)
        x = Dense(128, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        y_hat = Dense(prop_dim, activation='sigmoid')(x)

        x = Activation('relu')(z_mean)
        x = Dense(128, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        y_semi_hat = Dense(semi_prop_dim, activation='sigmoid')(x)
        regression = Model(encoder_inputs, [y_hat, y_semi_hat], name='target-learning branch')

    latent_inputs = Input(shape=(latent_dim,))
    map_size = K.int_shape(encoder.layers[-6].output)[1]
    x = Dense(max_filters * map_size, activation='relu')(latent_inputs)
    x = Reshape((map_size, 1, max_filters))(x)

    x = attention_layer(x, latent_dim)

    x = BatchNormalization()(x)
    x = Conv2DTranspose(max_filters // 2, (filter_size[2], 1), strides=(strides[2], 1), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(max_filters // 4, (filter_size[1], 1), strides=(strides[1], 1), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(channel_dim, (filter_size[0], 1), strides=(strides[0], 1), padding='SAME')(x)
    x = Activation('sigmoid')(x)

    decoder_outputs = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')

    reconstructed_outputs = decoder(z)
    VAE = Model(inputs=[encoder_inputs, regression_inputs], outputs=reconstructed_outputs)

    VAE.summary()

    def vae_loss(x, decoded_x):
        loss_recon = K.sum(K.square(encoder_inputs - reconstructed_outputs))
        loss_KL = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        loss_prop = K.sum(K.square(regression_inputs[:, :prop_dim] - y_hat))

        if semi:
            loss_prop_semi = K.sum(K.square(y_semi_hat - y_semi[:, prop_dim:prop_dim + semi_prop_dim]))
            vae_loss = K.mean(
                loss_recon + coeff_KL * loss_KL + coeff_prop * loss_prop + coeff_prop_semi * loss_prop_semi)
        else:
            vae_loss = K.mean(loss_recon + coeff_KL * loss_KL + coeff_prop * loss_prop)
        return vae_loss

    return VAE, encoder, decoder, regression, vae_loss
