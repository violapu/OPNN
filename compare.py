import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config

from enums import ActivationFunction, TrainMode, OptionType
from generator import STRIKE_COL, UNDERLYING_COL, build_constant_generator
from pricer_factory import PricerFactory
from unsupervised.model_factory import ModelFile

SUPERVISED_COL = 'supervised_prices'
UNSUPERVISED_COL = 'unsupervised_prices'
NUMERICAL_COL = 'numerical_prices'


def compare_models(models, iterations):
    comparison_config = config.get_comparison_config()

    option_type = OptionType[comparison_config['option_type']]

    gen = build_constant_generator(ast.literal_eval(comparison_config['domain']))
    prices_df = gen.generate(samples=int(comparison_config['samples']))

    pricer_factory = PricerFactory(option_type=option_type, steps=int(comparison_config['steps']),
                                   num_paths=int(comparison_config['num_paths']))

    pricer = pricer_factory.create(comparison_config['pricer'])

    predicted_df = pd.DataFrame()
    predicted_df[SUPERVISED_COL] = load_supervised_data(models[1]).iloc[:, 0]
    predicted_df[UNSUPERVISED_COL] = load_unsupervised_data(models[0], iterations)
    predicted_df[NUMERICAL_COL] = pricer.price(prices_df, use_tqdm=True)

    comparison_plot(df=prices_df, predicted_df=predicted_df, param=comparison_config['param'],
                    numerical_method=comparison_config['numerical_method'], xlabel=comparison_config['xlabel'],
                    path1=f'comparison_figures/{comparison_config["xlabel"]}_super_{models[1]}_unsuper_{models[0]}.pdf',
                    path2=f'comparison_figures/{comparison_config["xlabel"]}_super_{models[1]}_unsuper_{models[0]}.pdf')


def load_unsupervised_data(model, iterations):
    training_params = config.get_unsupervised_training_params(model)
    model_params = config.get_unsupervised_model_params(model)

    train_mode = TrainMode[training_params['train_mode']]
    nodes_per_layer = int(training_params['nodes_per_layer'])
    layers = int(training_params['layers'])

    iterations = iterations or int(training_params['iterations'])

    option_type = None
    if 'option_type' in model_params:
        option_type = OptionType[model_params['option_type']]

    model_file = ModelFile(model, train_mode, iterations,
                           ActivationFunction.Softplus, layers, nodes_per_layer,
                           option_type)
    return pd.read_csv(f'comparison_data/{model_file.path("csv", "unsupervised")}')


def load_supervised_data(model):
    params = config.get_supervised_params(model)

    unique_name = f'{model}_{params["option_type"]}_{params["epochs"]}epochs_' \
                  f'{params["hidden_layers"]}x{params["nodes_per_layer"]}_{params["samples"]}samples'

    return pd.read_csv(f'comparison_data/supervised_{unique_name}.csv')


def comparison_plot(df, predicted_df, param, numerical_method, xlabel, path1, path2, intrinsic=False):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 14})
    # rc('text', usetex=True)
    # rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

    plt.figure()
    fig1, ax1 = plt.subplots(figsize=(6, 4), tight_layout=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4), tight_layout=True)

    ax1.plot(df[param], predicted_df[SUPERVISED_COL], label='Supervised Prices', color='royalblue')
    ax1.plot(df[param], predicted_df[NUMERICAL_COL], label=f'{numerical_method} prices', color='green', linestyle='dashdot')
    ax1.plot(df[param], predicted_df[UNSUPERVISED_COL], label='Unsupervised Prices', color='darkorange', linestyle='dashed')
    print("supervised", max(abs(predicted_df[NUMERICAL_COL]-predicted_df[SUPERVISED_COL])))
    print("unsupervised", max(abs(predicted_df[NUMERICAL_COL] - predicted_df[UNSUPERVISED_COL])))
    if intrinsic:
        ax1.plot(df[param], np.maximum(df[STRIKE_COL]-df[UNDERLYING_COL], 0), label='Payoff', color='dimgrey', linestyle='dotted')
    ax1.set_title(f'Prices against {xlabel}')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2.plot(df[param], abs(predicted_df[SUPERVISED_COL] - predicted_df[NUMERICAL_COL]),
             label=f'Supervised vs {numerical_method} Prices', color='royalblue')
    ax2.plot(df[param], abs(predicted_df[UNSUPERVISED_COL] - predicted_df[NUMERICAL_COL]),
             label=f'Unsupervised vs {numerical_method} Prices', color='darkorange', linestyle='dashed')
    ax2.set_title(f'Differences in Prices against {xlabel}')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Price')
    ax2.grid(True)
    ax2.set_ylim([0, 1.6])
    ax2.legend(loc='upper left', prop={'size': 13})

    fig1.savefig(path1)
    fig2.savefig(path2)
