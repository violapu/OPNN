import argparse
import ast
import compare
import config
import importlib
import time

from enums import TrainMode, ActivationFunction, OptionType
from generator import build_random_generator, STRIKE_COL, RF_RATE_COL, DIV_COL, UNDERLYING_COL, \
    DAYS_TO_MATURITY_COL, SIGMA_COL, build_constant_generator
from unsupervised.graphing.numerical_graphing import build_prediction_data, numerical_solution_plotting
from unsupervised.graphing.analytical_graphing import analytical_solution_plotting
from pricer_factory import PricerFactory
from supervised.graphing import check_accuracy, plot_prices_against_params
from errors import dependencies, max_error, l2_error

SUPERVISED_MODELS = ['BinomialAmerican', 'BinomialEuropean', 'MCAmerican', 'MCEuropean', 'AnalyticalBS']


def main():
    parser = argparse.ArgumentParser(
        description='Pricing options using supervised and unsupervised learning')
    parser.add_argument('-t', '--train', dest='train', action='store_true',
                        help='train a specified model')
    parser.add_argument('-g', '--graph', dest='graph', action='store_true',
                        help='graph a specified model')
    parser.add_argument('-i', '--iterations', dest='iterations', action='store',
                        help='number of iterations to train the model', type=int)
    parser.add_argument('-cs', '--compare_supervised', dest='compare_supervised', action='store_true',
                        help='graph a set of supervised models')
    parser.add_argument('-e', '--error', dest='error', action='store_true',
                        help='plot absolute error between actual and predicted prices')
    parser.add_argument('-p', '--predict', dest='predict', action='store_true',
                        help='predict prices using trained neural network')
    parser.add_argument('-c', '--compare', dest='compare', action='store_true',
                        help='plot the comparison of predicted prices by supervised and unsupervised '
                             'learning')
    parser.add_argument('models', metavar='MODEL', nargs='+', help='models to train or graph')

    args = parser.parse_args()

    if args.compare_supervised:
        all_in = [model in SUPERVISED_MODELS for model in args.models]
        if not all(all_in):
            raise ValueError("Cannot use -cs with unsupervised model")
        plot_prices_against_params(args.models)

    for model in args.models:
        run_for_model(args, model)

    if args.compare:
        if len(args.models) != 2 or args.models[0] in SUPERVISED_MODELS \
                or args.models[1] not in SUPERVISED_MODELS:
            raise ValueError("For -c must specify two models: the first unsupervised, "
                             "the second supervised")

        compare.compare_models(args.models, args.iterations)


def run_for_model(args, model):
    if model in SUPERVISED_MODELS:
        run_supervised(args, model)
    else:
        if args.train:
            train_model, ModelFile, ModelFactory, predict = init_unsupervised_training_imports()

            training_params = config.get_unsupervised_training_params(model)
            model_params = config.get_unsupervised_model_params(model)

            train_mode = TrainMode[training_params['train_mode']]
            iterations = args.iterations or int(training_params['iterations'])
            nodes_per_layer = int(training_params['nodes_per_layer'])
            layers = int(training_params['layers'])

            option_type = None
            if 'option_type' in model_params:
                option_type = OptionType[model_params['option_type']]

            train_model.train(model_name=model,
                              model_file=ModelFile(model, train_mode, iterations,
                                                   ActivationFunction.Softplus, layers, nodes_per_layer,
                                                   option_type),
                              factory=ModelFactory(layers=layers, nodes_per_layer=nodes_per_layer),
                              train_mode=train_mode,
                              iterations=iterations)

        if args.predict:
            train_model, ModelFile, ModelFactory, predict = init_unsupervised_training_imports()

            training_params = config.get_unsupervised_training_params(model)
            model_params = config.get_unsupervised_model_params(model)
            prediction_params = config.get_unsupervised_prediction_params(model)

            iterations = args.iterations or int(training_params['iterations'])

            predict.predict(model, training_params, model_params, prediction_params, iterations)

        if args.graph:
            train_model, ModelFile, ModelFactory, predict = init_unsupervised_training_imports()

            training_params = config.get_unsupervised_training_params(model)
            model_params = config.get_unsupervised_model_params(model)
            graphing_params = config.get_unsupervised_graphing_params(model)

            train_mode = TrainMode[training_params['train_mode']]
            iterations = args.iterations or int(training_params['iterations'])
            nodes_per_layer = int(training_params['nodes_per_layer'])
            layers = int(training_params['layers'])

            option_type = None
            if 'option_type' in model_params:
                option_type = OptionType[model_params['option_type']]

            all_params = config.merge_params(training_params,
                                             config.merge_params(model_params, graphing_params))

            all_params.update({'iterations': iterations})

            if graphing_params['mode'] == 'analytical':
                analytical_solution_plotting(model_name=model,
                                             model_file=ModelFile(model, train_mode, iterations,
                                                                  ActivationFunction.Softplus, layers,
                                                                  nodes_per_layer, option_type),
                                             factory=ModelFactory(layers=layers,
                                                                  nodes_per_layer=nodes_per_layer),
                                             domain_3D=ast.literal_eval(graphing_params['domain_3D']),
                                             x_label_3D=graphing_params['x_label_3D'],
                                             y_label_3D=graphing_params['y_label_3D'],
                                             title_3D=title_preprocess(graphing_params['title_3D'],
                                                                       all_params),
                                             domain_2D=ast.literal_eval(graphing_params['domain_2D']),
                                             x_label_2D=graphing_params['x_label_2D'],
                                             title_2D=title_preprocess(graphing_params['title_2D'],
                                                                       all_params),
                                             param=graphing_params['x_axis_param'],
                                             underlying=str_to_bool(graphing_params['boundary_plot']))
            elif graphing_params['mode'] == 'numerical':
                samples = int(graphing_params['samples'])
                num_paths = None
                if 'num_paths' in graphing_params:
                    num_paths = int(graphing_params['num_paths'])

                gen = build_constant_generator([ast.literal_eval(graphing_params['strike_range']),
                                                ast.literal_eval(graphing_params['underlying_range']),
                                                ast.literal_eval(graphing_params['rf_rate_range']),
                                                ast.literal_eval(
                                                    graphing_params['days_to_maturity_range']),
                                                ast.literal_eval(graphing_params['div_yield_range']),
                                                ast.literal_eval(graphing_params['sigma_range'])])
                prices_df = gen.generate(samples=samples)
                data, index = build_prediction_data(prices_df, ast.literal_eval(
                    graphing_params['prediction_ranges']), samples)

                price_factory = PricerFactory(option_type=option_type,
                                              steps=int(graphing_params['steps']), num_paths=num_paths)
                numerical_solution_plotting(model_name=model,
                                            model_file=ModelFile(model, train_mode, iterations,
                                                                 ActivationFunction.Softplus, layers,
                                                                 nodes_per_layer, option_type),
                                            factory=ModelFactory(nodes_per_layer=nodes_per_layer,
                                                                 option_type=option_type),
                                            prices_df=prices_df,
                                            pricer=price_factory.create(graphing_params['pricer_name']),
                                            data=data, index=index,
                                            domain_3D=ast.literal_eval(graphing_params['domain_3D']),
                                            x_label_3D=graphing_params['x_label_3D'],
                                            y_label_3D=graphing_params['y_label_3D'],
                                            title_3D=title_preprocess(graphing_params['title_3D'],
                                                                      all_params),
                                            x_label_2D=graphing_params['x_label_2D'],
                                            title_2D=title_preprocess(graphing_params['title_2D'],
                                                                      all_params),
                                            legend_label=graphing_params['legend_label'],
                                            elevation=float_or_none(graphing_params['elevation']),
                                            angle=float_or_none(graphing_params['angle']),
                                            underlying=str_to_bool(graphing_params['boundary_plot']))
            else:
                raise ValueError(f'Graphing mode {graphing_params["mode"]} not recognised.')


def run_supervised(args, model):
    NeuralNetwork, SupervisedPricingProblem, predict = init_supervised_training_imports()

    params = config.get_supervised_params(model)
    prediction_params = config.get_supervised_prediction_params(model)

    unique_name = f'{model}_{params["option_type"]}_{params["epochs"]}epochs_' \
                  f'{params["hidden_layers"]}x{params["nodes_per_layer"]}_{params["samples"]}samples'

    gen = build_random_generator({
        STRIKE_COL: ast.literal_eval(params['strike']),
        UNDERLYING_COL: ast.literal_eval(params['underlying']),
        RF_RATE_COL: ast.literal_eval(params['rf_rate']),
        DAYS_TO_MATURITY_COL: ast.literal_eval(params['days_to_maturity']),
        DIV_COL: ast.literal_eval(params['div_yield']),
        SIGMA_COL: ast.literal_eval(params['sigma'])
    })

    option_type = OptionType.of(params['option_type'])

    pricer_factory = PricerFactory(option_type=option_type, steps=int(params['steps']),
                                   num_paths=int(params['num_paths']))

    pricer = pricer_factory.create(model)

    hidden_layers = int(params['hidden_layers'])
    nodes_per_layer = int(params['nodes_per_layer'])
    epochs = int(params['epochs'])
    batch_size = int(params['batch_size'])

    nn = NeuralNetwork.build(input_dim=len(gen.generators), hidden_layers=hidden_layers,
                             nodes_per_layer=nodes_per_layer, hidden_activation='softplus',
                             loss='mse', optimizer='adam', metrics=[l2_error, max_error],
                             output_activation='softplus')

    problem = SupervisedPricingProblem(gen, pricer, nn)
    samples = int(params['samples'])

    if args.train:
        start_time = time.time()
        problem.generate_data(samples=samples)
        end_time = time.time()
        print('Generation time elapsed: ', end_time - start_time)
        problem.save_data(f'supervised/data/{model}_{option_type.name}_{samples}samples.csv')

        start_time = time.time()
        problem.train(epochs=epochs, batch_size=batch_size)
        end_time = time.time()
        print('Training time elapsed: ', end_time - start_time)

        nn.save(f'supervised/models/{unique_name}')
        problem.save_scaler(f'supervised/scaler/scaler_{unique_name}.pkl')
        problem.save_history(f'supervised/history/scaler_{unique_name}.pkl')

    if args.predict:
        problem.load_network(f'supervised/models/{unique_name}', custom=dependencies)
        problem.set_scaler(problem.load_scaler(f'supervised/scaler/scaler_{unique_name}.pkl'))
        predict.predict(problem, prediction_params, unique_name)

    if args.graph:
        problem.load_network(f'supervised/models/{unique_name}', custom=dependencies)
        problem.set_scaler(problem.load_scaler(f'supervised/scaler/scaler_{unique_name}.pkl'))
        problem.load_history(f'supervised/history/scaler_{unique_name}.pkl')
        nn.plot_history(0, 2, 0, 5, f'supervised/figures/1_{unique_name}.pdf',
                        f'supervised/figures/2_{unique_name}.pdf')

    if args.error:
        problem.load_network(f'supervised/models/{unique_name}', custom=dependencies)
        problem.set_scaler(problem.load_scaler(f'supervised/scaler/scaler_{unique_name}.pkl'))
        problem.load_data(f'supervised/data/{model}_{option_type.name}_{samples}samples.csv')
        X_train, y_train, X_test, y_test, X_val, y_val = problem.preprocess_and_split()
        check_accuracy(y_train, problem.predict(X_train),
                       y_test, problem.predict(X_test),
                       f'supervised/figures/pred_vs_actual_{model}_{option_type.name}_{samples}samples.pdf',
                       f'supervised/figures/hist_{model}_{option_type.name}_{samples}samples.pdf')


def init_unsupervised_training_imports():
    tf = importlib.import_module('tensorflow.compat.v1')
    tf.disable_v2_behavior()

    train_model = importlib.import_module('unsupervised.train_model')
    model_factory = importlib.import_module('unsupervised.model_factory')
    predict = importlib.import_module('unsupervised.predict')
    ModelFile = model_factory.ModelFile
    ModelFactory = model_factory.ModelFactory

    return train_model, ModelFile, ModelFactory, predict


def init_supervised_training_imports():
    neural_network = importlib.import_module('supervised.neural_network')
    problem = importlib.import_module('supervised.problem')
    NeuralNetwork = neural_network.NeuralNetwork
    SupervisedPricingProblem = problem.SupervisedPricingProblem
    predict = importlib.import_module('supervised.predict')

    return NeuralNetwork, SupervisedPricingProblem, predict


def str_to_bool(s: str):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert {s} to bool.")


def float_or_none(s: str):
    if s == 'None':
        return None
    else:
        return float(s)


def title_preprocess(title, params):
    return title.strip('"').replace('\\n', '\n').format(**params)


if __name__ == "__main__":
    main()
