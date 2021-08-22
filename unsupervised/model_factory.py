import config

from unsupervised.neural_network import NeuralNetwork
from unsupervised.pde.black_scholes import BSSt, BSAll, BSSigmaSt, BSStrikeSt, BSStrikeSigmaSt
from unsupervised.pde.american_options import AmericanSt, AmericanOptionsSigmaSt, \
    AmericanOptionsStrikeSt, AmericanOptionsSigmaStrikeSt
from enums import ActivationFunction, OptionType
from pathlib import Path


class ModelFactory:
    def __init__(self, activation_func=ActivationFunction.Softplus, nodes_per_layer=20, layers=4,
                 option_type=OptionType.Put, rf_rate=0.04, div_yield=0.02, div_yield_1=0.0, div_yield_2=0.0,
                 sigma=0.25, sigma_1=0.25, sigma_2=0.20, strike_price=20, maturity=1.0, t=0.0,
                 kappa=3.0, theta=0.2, rho=-0.8, initial_vol=0.2):
        self.activation_func = activation_func
        self.nodes_per_layer = nodes_per_layer
        self.layers = layers
        self.option_type = option_type
        self.rf_rate = rf_rate
        self.div_yield = div_yield
        self.div_yield_1 = div_yield_1
        self.div_yield_2 = div_yield_2
        self.sigma = sigma
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.strike_price = strike_price
        self.maturity = maturity
        self.t = t
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.initial_vol = initial_vol

    def create(self, name):
        params = config.get_unsupervised_model_params(name)
        if name == 'BSSt':
            return self.create_blackscholes_st(params)
        elif name == 'BSSigmaSt':
            return self.create_blackscholes_sigmast(params)
        elif name == 'BSStrikeSt':
            return self.create_blackscholes_strikest(params)
        elif name == 'BSFixedrDivT':
            return self.create_blackscholes_fixed_rdivt(params)
        elif name == 'BSAll':
            return self.create_blackscholes_all(params)
        elif name == 'AmericanSt':
            return self.create_american_st(params)
        elif name == 'AmericanSigmaSt':
            return self.create_american_sigmast(params)
        elif name == 'AmericanStrikeSt':
            return self.create_american_strikest(params)
        elif name == 'AmericanSigmaStrikeSt':
            return self.create_american_sigmastrikest(params)
        else:
            raise ValueError(f'Invalid model name: {name}')

    def create_blackscholes_st(self, params):
        network = self.build_network(BSSt.input_count)
        return BSSt(**params, network=network)

    def create_blackscholes_sigmast(self, params):
        network = self.build_network(BSSigmaSt.input_count)
        return BSSigmaSt(**params, network=network)

    def create_blackscholes_strikest(self, params):
        network = self.build_network(BSStrikeSt.input_count)
        return BSStrikeSt(**params, network=network)

    def create_blackscholes_fixed_rdivt(self, params):
        network = self.build_network(BSStrikeSigmaSt.input_count)
        return BSStrikeSigmaSt(**params, network=network)

    def create_blackscholes_all(self, params):
        network = self.build_network(BSAll.input_count)
        return BSAll(**params, network=network)

    def create_american_st(self, params):
        network = self.build_network(AmericanSt.input_count)
        return AmericanSt(**params, network=network)

    def create_american_sigmast(self, params):
        network = self.build_network(AmericanOptionsSigmaSt.input_count)
        return AmericanOptionsSigmaSt(**params, network=network)

    def create_american_strikest(self, params):
        network = self.build_network(AmericanOptionsStrikeSt.input_count)
        return AmericanOptionsStrikeSt(**params, network=network)

    def create_american_sigmastrikest(self, params):
        network = self.build_network(AmericanOptionsSigmaStrikeSt.input_count)
        return AmericanOptionsSigmaStrikeSt(**params, network=network)

    def build_network(self, inputs):
        return NeuralNetwork(input_dimension=inputs, hidden_layers=[self.nodes_per_layer] * self.layers,
                             activation_function=self.activation_func)


class ModelFile:
    def __init__(self, model_name, train_mode, iterations, activation_func, layers, nodes_per_layer,
                 option_type):
        self.model_name = model_name
        self.train_mode = train_mode.name
        self.iterations = iterations
        self.activation_func = activation_func.name
        self.layers = layers
        self.nodes_per_layer = nodes_per_layer
        if option_type is None:
            self.option_type = None
        else:
            self.option_type = option_type.name

    def dir(self):
        return f'{self.model_name}/{self.train_mode}'

    def file_name(self, extension=None, prefix=None):
        s = f'{self.iterations}_{self.activation_func}_{self.layers}x{self.nodes_per_layer}_' \
            f'{self.option_type}'
        if extension is not None:
            s = f'{s}.{extension}'
        if prefix is not None:
            s = f'{prefix}_{s}'
        return s

    def path(self, extension=None, prefix=None):
        return f'{self.dir()}/{self.file_name(extension, prefix)}'

    @staticmethod
    def make_path(path):
        Path(path).mkdir(parents=True, exist_ok=True)
