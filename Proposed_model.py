from keras import Input
from keras.src.layers import GlobalAveragePooling1D, Dense

from keras_transformer.keras_transformer.transformer import TransformerBlock

def build_model(x_train,x_test,y_train,y_test):

    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2]*x_train.shape[3])

    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2]*x_test.shape[3])

    input_layer = Input(shape=(x_train.shape[2],64))

    transformer1 = TransformerBlock(
        name="transformer",
        num_heads=8,
        residual_dropout=0.1,
        attention_dropout=0.1
    )(input_layer)

    x = GlobalAveragePooling1D()(transformer1)

    # -------- Classification head --------
    x = Dense(256, activation="relu")(x)