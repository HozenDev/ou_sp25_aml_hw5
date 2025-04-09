from mesonet_support import SinhArcsinh

from tf_keras.layers import Dense, BatchNormalization, Concatenate, Lambda, Dropout
from tf_keras import Input, Model
from tf_keras.optimizers import Adam
from tf_keras.regularizers import l1, l2

def create_inner_model(input_dim, hidden_layers=[128, 64]):
    inputs = Input(shape=(input_dim,))
    x = BatchNormalization()(inputs)
    for units in hidden_layers:
        x = Dense(units, activation='elu')(x)
        x = BatchNormalization()(x)

    epsilon = 1e-6
        
    mu = Dense(1)(x)
    
    std = Dense(1, activation='softplus')(x)
    std = Lambda(lambda t: t + epsilon)(std)
    
    skew = Dense(1)(x)
    
    tail = Dense(1, activation='softplus')(x)
    tail = Lambda(lambda t: t + epsilon)(tail)

    params = Concatenate(name='concat_params')([mu, std, skew, tail])
    return Model(inputs=inputs, outputs=params, name="inner_model")

def create_outer_model(inner_model, lrate):
    inputs = inner_model.input
    param_vector = inner_model(inputs)
    dist = SinhArcsinh.create_layer()(param_vector)

    outer_model = Model(inputs=inputs, outputs=dist, name='outer_model')
    
    outer_model.compile(
        optimizer=Adam(learning_rate=lrate, clipnorm=1.0),
        loss=SinhArcsinh.mdn_loss
    )
    
    return outer_model

def fully_connected_stack(n_inputs, 
                          n_hidden, 
                          n_output, 
                          activation='elu', 
                          activation_out=['linear'],
                          dropout=None, 
                          dropout_input=None, 
                          kernel_regularizer_L2=None, 
                          kernel_regularizer_L1=None):

    '''
    General network building code that creates a stack of Dense layers.
    The output of the stack is a list of output Keras Tensors, each with its own activation function.
    If the output activation function is 'positive', then the output is elu()+1.1
    
    :param n_inputs: Number of input units
    :param n_hidden: List of hidden layer sizes
    :param n_output: List of the number of output units for each output Tensor
    :param activation: Activation function for hidden layers
    :param activation_out: List of the activation functions for each output Tensor
    :param lrate: Learning rate
    :param dropout: Dropout probability for hidden layers
    :param dropout_input: Dropout probability for input layer
    :param kernel_regularizer_L2: L2 regularization param
    :param kernel_regularizer_L1: L1 regularization param
    :return: (input tensor, list of output tensors)
    '''
    
    if dropout is not None:
        print("DENSE: DROPOUT %f"%dropout)

    # L2 or L1 regularization?
    kernel_regularizer = None
    if kernel_regularizer_L2 is not None:
        print("DENSE: L2 Regularization %f"%kernel_regularizer)
        kernel_regularizer=l2(kernel_regularizer)
    elif kernel_regularizer_L1 is not None:
        # Only us L1 if specified *and* L2 is not active
        print("DENSE: L1 Regularization %f"%kernel_regularizer_L1)
        kernel_regularizer=l1(kernel_regularizer_L1)

    # Input layer
    input_tensor = tensor = Input(shape=(n_inputs,))
    
    # Dropout input features?
    if dropout_input is not None:
        tensor = Dropout(rate=dropout_input, name="dropout_input")(tensor)
            
    # Loop over hidden layers
    for i, n in enumerate(n_hidden):             
        tensor = Dense(n, use_bias=True, name="hidden_%02d"%(i), activation=activation,
                 kernel_regularizer=kernel_regularizer)(tensor)
        
        if dropout is not None:
            tensor = Dropout(rate=dropout, name="dropout_%02d"%(i))(tensor)
    
    # Output layers
    outputs = []
    for i, (n, act) in enumerate(zip(n_output, activation_out)):
        o = Dense(n, use_bias=True, name="output%d"%i, activation=act)(tensor)
        outputs.append(o)

    return input_tensor, outputs
