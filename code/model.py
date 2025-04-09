from mesonet_support import SinhArcsinh
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization

def create_inner_model(input_dim, hidden_layers=[128, 64]):
    inputs = Input(shape=(input_dim,))
    x = BatchNormalization()(inputs)
    for units in hidden_layers:
        x = Dense(units, activation='elu')(x)
        x = BatchNormalization()(x)

    mu = Dense(1)(x)
    sigma = Dense(1, activation='softplus')(x)
    skew = Dense(1)(x)
    tail = Dense(1, activation='softplus')(x)

    return Model(inputs=inputs, outputs=[mu, sigma, skew, tail], name="inner_model")

def create_outer_model(inner_model):
    inputs = inner_model.input
    dist_params = inner_model(inputs)
    dist = SinhArcsinh.create_layer()(dist_params)
    return Model(inputs=inputs, outputs=dist, name="outer_model")
