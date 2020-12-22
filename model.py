def simple_regression():
    
    xi=Input(shape=(31,))
    x=BatchNormalization()(xi)
    x=Dense(1)(x)
    model = Model(inputs=xi, outputs=x)
    return model

def nn_model(X_train, y_train, X_test, y_test, dependent_name, epochs=20):
    xi=Input(shape=(31,))

    x=Dense(128)(xi)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    
    x=Dense(64)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)

    x=Dense(32)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)

    x=Dense(1)(x)

    model=Model(inputs=xi, outputs=x)
    return model