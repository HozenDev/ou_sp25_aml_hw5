from mesonet_support import SinhArcsinh
import tf_keras as keras
from tf_keras.models import Model
from tf_keras.layers import Input, Dense, BatchNormalization, Concatenate
from mesonet_support import SinhArcsinh

def create_inner_model(input_dim, hidden_layers=[128, 64]):
    inputs = Input(shape=(input_dim,))
    x = BatchNormalization()(inputs)
    for units in hidden_layers:
        x = Dense(units, activation='elu')(x)

    mu = Dense(1)(x)
    std = Dense(1, activation='softplus')(x)
    skew = Dense(1)(x)
    tail = Dense(1, activation='softplus')(x)

    params = Concatenate(name='concat_params')([mu, std, skew, tail])
    return Model(inputs=inputs, outputs=params, name="inner_model")

def create_outer_model(inner_model):
    inputs = inner_model.input
    param_vector = inner_model(inputs)
    dist = SinhArcsinh.create_layer()(param_vector)
    return Model(inputs=inputs, outputs=dist, name='outer_model')
