from pickle import dump, load

import QuantLib as ql
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from enums import OptionType
from generator import STRIKE_COL, UNDERLYING_COL, RF_RATE_COL, DAYS_TO_MATURITY_COL, DIV_COL, SIGMA_COL, \
    DataGenerator, UniformGenerator, RandIntGenerator
from pricer import BinomialAmericanPricer
from supervised.neural_network import NeuralNetwork

YEARS_TO_MATURITY_COL = 'years_to_maturity'


class SupervisedPricingProblem:
    def __init__(self, generator, pricer, network, scaler=MinMaxScaler()):
        self._generator = generator
        self._pricer = pricer
        self._network = network

        self._full_data = None
        self._scaler = scaler

    def generate_data(self, samples=1000):
        df = self._generator.generate(samples=samples)
        df['prices'] = self._pricer.price(df, use_tqdm=True)
        self._full_data = df

    def save_data(self, path):
        self._full_data.to_csv(path, index=False)

    def load_data(self, path):
        self._full_data = pd.read_csv(path)
        print(self._full_data)

    def save_scaler(self, path):
        dump(self._scaler, open(path, 'wb'))

    @staticmethod
    def load_scaler(path):
        return load(open(path, 'rb'))

    def save_history(self, path):
        df = pd.DataFrame()
        df['l2_error'] = self._network.model_history.history['l2_error']
        df['max_error'] = self._network.model_history.history['max_error']
        df['val_l2_error'] = self._network.model_history.history['val_l2_error']
        df['val_max_error'] = self._network.model_history.history['val_max_error']
        df['loss'] = self._network.model_history.history['loss']
        df['val_loss'] = self._network.model_history.history['val_loss']
        df.to_csv(path, index=False)

    @staticmethod
    def load_history(path):
        return pd.read_csv(path)

    def set_scaler(self, scaler):
        self._scaler = scaler

    def load_network(self, path, custom):
        self._network = NeuralNetwork.load_network(path, custom)

    def preprocess_and_split(self):
        inputs_to_norm = [UNDERLYING_COL, STRIKE_COL, RF_RATE_COL, DIV_COL, SIGMA_COL,
                          YEARS_TO_MATURITY_COL]

        new_df = self._full_data.copy()
        new_df[YEARS_TO_MATURITY_COL] = new_df[DAYS_TO_MATURITY_COL] / 365
        new_df[inputs_to_norm] = self._scaler.fit_transform(new_df[inputs_to_norm])

        X_train, X_test, y_train, y_test = train_test_split(new_df[inputs_to_norm], new_df['prices'],
                                                            test_size=0.2, random_state=0)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                          random_state=0)

        return X_train, y_train, X_test, y_test, X_val, y_val

    def train(self, epochs=800, batch_size=128):
        X_train, y_train, X_test, y_test, X_val, y_val = self.preprocess_and_split()
        self._network.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, epochs=epochs,
                          batch_size=batch_size)

        training_mse, training_l2, training_max = self._network.evaluate(inputs=X_train, labels=y_train)
        print('Training MSE: {}'.format(training_mse), 'L2: {}'.format(training_l2),
              'max: {}'.format(training_max))
        testing_mse, testing_l2, testing_max = self._network.evaluate(inputs=X_test, labels=y_test)
        print('Testing MSE: {}'.format(testing_mse), 'L2: {}'.format(testing_l2),
              'max: {}'.format(testing_max))

    def predict(self, df):
        if YEARS_TO_MATURITY_COL not in df:
            inputs_to_norm = [UNDERLYING_COL, STRIKE_COL, RF_RATE_COL, DIV_COL, SIGMA_COL,
                              YEARS_TO_MATURITY_COL]
            new_df = df.copy()
            new_df[YEARS_TO_MATURITY_COL] = new_df[DAYS_TO_MATURITY_COL] / 365
            new_df[inputs_to_norm] = self._scaler.transform(new_df[inputs_to_norm])
            pred = self._network.predict(new_df[inputs_to_norm])
        else:
            pred = self._network.predict(df)
        return pred


if __name__ == '__main__':
    method = 'binomial'
    dividend_type = 'continuous_dividend'
    expiry_type = 'american'
    option_type = OptionType.Put
    epochs = 800
    batch_size = 128
    hidden_layers = 2
    nodes_per_layer = 128

    unique_name = f'{method}_{dividend_type}_{expiry_type}_{option_type.name}_{epochs}epochs_' \
                  f'{hidden_layers}x{nodes_per_layer}'


    # gen = DataGenerator(
    #     [ConstantGenerator(15, STRIKE_COL), LinspaceGenerator((0.01, 60), UNDERLYING_COL),
    #      ConstantGenerator(0.04, RF_RATE_COL), ConstantGenerator(365, DAYS_TO_MATURITY_COL),
    #      ConstantGenerator(0.0, DIV_COL), ConstantGenerator(0.25, SIGMA_COL)], seed=42)

    gen = DataGenerator(
        [UniformGenerator((0.01, 100), STRIKE_COL), UniformGenerator((0.01, 100), UNDERLYING_COL),
         UniformGenerator((-0.02, 0.08), RF_RATE_COL), RandIntGenerator((1, 1095), DAYS_TO_MATURITY_COL),
         UniformGenerator((0.0, 0.08), DIV_COL), UniformGenerator((0.01, 0.5), SIGMA_COL)], seed=42)

    # pricer = MCAmericanPricer(use_tqdm=True, calculation_date=ql.Date(8, 5, 2015),
    #                           option_type=ql.Option.Put, steps=20, num_paths=10000, seed=42)
    pricer = BinomialAmericanPricer(use_tqdm=True, calculation_date=ql.Date(8, 5, 2015),
                                    option_type=ql.Option.Put, steps=1000)
    # pricer = DiscreteDividendsPricer(use_tqdm=True, pricing_date='20150508', option_type='Put',
    #                                  expiry_type='american', method='Binomial')

    nn = NeuralNetwork.build(input_dim=len(gen.generators), hidden_layers=hidden_layers,
                             nodes_per_layer=nodes_per_layer, hidden_activation='softplus', loss='mse',
                             optimizer='adam', metrics=['mae'], output_activation='softplus')

    problem = SupervisedPricingProblem(gen, pricer, nn)
    problem.generate_data(samples=10000)
    problem.save_data(f'supervised/data/test_{method}_{dividend_type}_{expiry_type}_{option_type.name}.csv')
    # problem.load_data(f'data/{method}_{dividend_type}_{expiry_type}_{option_type.name}.csv')

    problem.train(epochs=epochs, batch_size=batch_size)

    # save trained model and scaler
    nn.save(f'supervised/models/{unique_name}')
    problem.save_scaler(f'supervised/scaler/scaler_{unique_name}.pkl')

    nn.plot_history(0, 2, 0, 5, f'supervised/figures/{unique_name}.pdf')
