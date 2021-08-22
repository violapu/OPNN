import ast
import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enums import OptionType
from generator import build_constant_generator
from matplotlib import rc
from pricer_factory import PricerFactory
from supervised.neural_network import NeuralNetwork
from supervised.problem import SupervisedPricingProblem
from errors import dependencies


def check_accuracy(y_train_true, y_train_pred, y_test_true, y_test_pred, path_1, path_2):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 13})
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

    stats = dict()
    stats['train_diff'] = y_train_true - y_train_pred
    stats['test_diff'] = y_test_true - y_test_pred

    plt.figure(figsize=(6, 4))
    plt.scatter(y_train_true.sample(n=1000, random_state=42),
                pd.Series(data=y_train_pred).sample(n=1000, random_state=42),
                s=8, alpha=0.8, color='royalblue', label='Training Data')
    plt.scatter(y_test_true.sample(n=1000, random_state=42),
                pd.Series(data=y_test_pred).sample(n=1000, random_state=42),
                s=8, alpha=0.6, color='darkorange', label='Test Data')
    plt.title('Predicted Prices vs Binomial Tree Prices')
    plt.xlabel('Binomial Tree Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_1)

    plt.figure(figsize=(6, 4))
    plt.hist(stats['train_diff'].sample(n=y_test_pred.shape[0], random_state=42), bins=60,
             label='Training Data', alpha=0.8, color='royalblue')
    plt.hist(stats['test_diff'], alpha=0.7, color='darkorange', bins=60, label='Test Data')
    plt.title('Differences between Predicted and Binomial Tree Prices')
    plt.xlim(-0.8, 0.8)
    plt.xlabel('Difference')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_2)

    return stats


def plot_prices_against_params(models):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 13})
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

    fig1, ax1 = plt.subplots(figsize=(6, 4), tight_layout=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4), tight_layout=True)

    params = config.get_supervised_compare_graphing_params()
    gen = build_constant_generator([ast.literal_eval(params['strike']),
                                    ast.literal_eval(params['underlying']),
                                    ast.literal_eval(params['rf_rate']),
                                    ast.literal_eval(params['days_to_maturity']),
                                    ast.literal_eval(params['div_yield']),
                                    ast.literal_eval(params['sigma'])])
    prices_df = gen.generate(samples=int(params['samples']))
    prices_df['intrinsic_val'] = np.maximum(prices_df['strike'] - prices_df['underlying'], 0)

    xlabel = params['xlabel']
    path1 = "_".join(["1"] + models + [xlabel]) + ".pdf"
    path2 = "_".join(["2"] + models + [xlabel]) + ".pdf"

    for model in models:
        model_params = config.get_supervised_params(model)
        option_type = OptionType.of(model_params['option_type'])
        samples = int(model_params['samples'])

        unique_name = f'{model}_{model_params["option_type"]}_{model_params["epochs"]}epochs_' \
                      f'{model_params["hidden_layers"]}x{model_params["nodes_per_layer"]}_{samples}samples'
        trained_nn = NeuralNetwork.load_network(f'supervised/models/{unique_name}', custom=dependencies)
        problem = SupervisedPricingProblem(None, None, trained_nn, SupervisedPricingProblem.load_scaler(
            f'supervised/scaler/scaler_{unique_name}.pkl'))

        pricer_factory = PricerFactory(option_type=option_type, steps=int(model_params['steps']),
                                       num_paths=int(model_params['num_paths']))
        pricer = pricer_factory.create(model)

        predicted_prices = problem.predict(prices_df)
        numerical_prices = pricer.price(prices_df, use_tqdm=True)

        ax1.plot(prices_df[xlabel.lower()], predicted_prices, label=f'NN predicted {model}',
                 color='royalblue')
        ax1.plot(prices_df[xlabel.lower()], numerical_prices, label=f'{model} prices',
                 linestyle='dashed', color='darkorange')
        if xlabel != 'sigma':
            ax1.plot(prices_df[xlabel.lower()], prices_df['intrinsic_val'], linestyle='dotted',
                     label='Payoff', color='dimgrey')

        ax2.plot(prices_df[xlabel.lower()], abs(predicted_prices - numerical_prices),
                 label=f'NN predicted {model} vs {model} prices', color='royalblue')

        if model == 'AnalyticalBS':
            pricer_factory_binomial = PricerFactory(option_type=option_type, steps=1000, num_paths=20)
            binomial_pricer = pricer_factory_binomial.create('BinomialEuropean')
            pricer_factory_mc = PricerFactory(option_type=option_type, steps=20, num_paths=10000)
            mc_pricer = pricer_factory_mc.create('MCEuropean')
            numerical_prices_binomial = binomial_pricer.price(prices_df, use_tqdm=True)
            numerical_prices_mc = mc_pricer.price(prices_df, use_tqdm=True)
            ax2.plot(prices_df[xlabel.lower()], abs(numerical_prices_binomial - numerical_prices),
                     label=f'BinomialEuropean vs {model} prices', color='darkorange', linestyle='dashed',
                     linewidth=2.5)
            ax2.plot(prices_df[xlabel.lower()], abs(numerical_prices_mc - numerical_prices),
                     label=f'MCEuropean vs {model} prices', color='gold', linestyle='dotted',
                     linewidth=2)

        ax2.set_ylim([0.0, 0.6])
        ax2.grid(True)

    ax1.set_title(f'Prices against {xlabel}')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Price')
    ax1.legend(prop={'size': 12})
    ax1.grid(True)

    ax2.set_title(f'Differences in Prices against {xlabel}')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Price')
    ax2.legend(prop={'size': 11}, loc='upper left')

    fig1.savefig(path1)
    fig2.savefig(path2)

