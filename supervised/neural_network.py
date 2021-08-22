import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc


class NeuralNetwork:
    def __init__(self, model):
        self.model_history = None
        self.model = model

    @classmethod
    def build(cls, input_dim, hidden_layers, nodes_per_layer, hidden_activation, loss,
              optimizer, metrics, output_activation=None):
        return cls(cls._build_nn(input_dim, nodes_per_layer, hidden_activation,
                                 hidden_layers, output_activation, loss, optimizer, metrics))


    @staticmethod
    def _build_nn(input_dim, nodes_per_layer, hidden_activation, hidden_layers, output_activation,
                  loss, optimizer, metrics):
        model = keras.Sequential()
        model.add(keras.layers.Dense(nodes_per_layer, input_dim=input_dim, activation=hidden_activation))

        for i in range(hidden_layers - 1):
            model.add(keras.layers.Dense(nodes_per_layer, activation=hidden_activation))

        model.add(keras.layers.Dense(1, activation=output_activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        self.model_history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                            validation_data=(X_val, y_val), verbose=1)

    def predict(self, df):
        return self.model.predict(df).transpose()[0]

    def evaluate(self, inputs, labels):
        mse, l2_error, max_error = self.model.evaluate(inputs, labels, verbose=0)
        return mse, l2_error, max_error

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load_network(path, custom):
        return NeuralNetwork(keras.models.load_model(path, custom_objects=custom))

    @staticmethod
    def dummy_regressor_results(y_train, y_test):
        """
        dummy regressor result by assigning all prices the average price (baseline result)
        """
        y_pred = np.full_like(y_test, np.mean(y_train))
        mse = keras.losses.MeanSquaredError()
        return mse(y_test, y_pred).numpy()

    def plot_history(self, mae_low_ylim, mae_high_ylim, mse_low_ylim, mse_high_ylim, path1, path2):
        plt.style.use('default')
        plt.rcParams.update({'font.size': 12.5})
        rc('text', usetex=True)
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

        mae = self.model_history.history['mean_absolute_error']
        val_mae = self.model_history.history['val_mean_absolute_error']
        mse = self.model_history.history['loss']
        val_mse = self.model_history.history['val_loss']

        x = range(1, len(mae) + 1)

        fig1, ax1 = plt.subplots(figsize=(6, 4), tight_layout=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4), tight_layout=True)

        ax1.plot(x, mae, label='Training MAE', color='royalblue')
        ax1.plot(x, val_mae, label='Validation MAE', color='darkorange')
        ax1.set_title('Training and Validation MAE')
        ax1.set_ylim((mae_low_ylim, mae_high_ylim))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MAE')
        ax1.legend(loc='upper right')

        ax2.plot(x, mse, color='royalblue', label='Training MSE')
        ax2.plot(x, val_mse, color='darkorange', label='Validation MSE')
        ax2.set_title('Training and Validation MSE')
        ax2.set_ylim((mse_low_ylim, mse_high_ylim))
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.legend(loc='upper right')

        fig1.savefig(path1)
        fig2.savefig(path2)
