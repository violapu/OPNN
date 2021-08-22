import ast
import time
import numpy as np
import pandas as pd

from enums import ActivationFunction, TrainMode, OptionType
from generator import build_constant_generator
from pathlib import Path
from unsupervised.model_factory import ModelFactory, ModelFile


def predict(model, training_params, model_params, prediction_params, iterations):
    train_mode = TrainMode[training_params['train_mode']]
    nodes_per_layer = int(training_params['nodes_per_layer'])
    layers = int(training_params['layers'])

    option_type = None
    if 'option_type' in model_params:
        option_type = OptionType[model_params['option_type']]

    factory = ModelFactory(layers=layers, nodes_per_layer=nodes_per_layer)

    model_file = ModelFile(model, train_mode, iterations,
                           ActivationFunction.Softplus, layers, nodes_per_layer,
                           option_type)

    model = factory.create(model)
    model.network.load_weights(f'unsupervised/weights/{model_file.path()}/weights')

    domain = ast.literal_eval(prediction_params['domain'])
    gen = build_constant_generator(domain, ["NONE"] * len(domain))
    gen_data = gen.generate_arrays(samples=int(prediction_params['samples']))
    data = np.array([np.array([d]).transpose() for d in gen_data])

    start_time = time.time()
    unsupervised_prices = model.network.predict(data)
    end_time = time.time()
    print('time elapsed: ', end_time - start_time)

    Path(f'comparison_data/{model_file.dir()}').mkdir(parents=True, exist_ok=True)

    pd.DataFrame(unsupervised_prices.transpose()[0]).to_csv(
        f'comparison_data/{model_file.path("csv", "unsupervised")}', index=False)
