import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from unsupervised.graphing.graphing import surface_plot, single_plot_with_generator, \
    avg_losses_against_iterations, \
    losses_against_iterations
from unsupervised.model_factory import ModelFile

tf.disable_v2_behavior()


def numerical_solution_plotting(model_name, model_file, factory, prices_df, pricer, data,
                                index, domain_3D, x_label_3D, y_label_3D, title_3D, x_label_2D, title_2D,
                                legend_label, elevation=None, angle=None, underlying=False):
    model = factory.create(model_name)
    model.network.load_weights(f'unsupervised/weights/{model_file.path()}/weights')

    ModelFile.make_path(f'unsupervised/figures/{model_file.dir()}')
    history = pd.read_csv(f'unsupervised/histories/{model_file.path(extension="csv")}')

    losses_against_iterations(history,
                              f'unsupervised/figures/{model_file.path(extension="pdf", prefix="losses")}')
    avg_losses_against_iterations(history,
                                  f'unsupervised/figures/{model_file.path(extension="pdf", prefix="avg_losses")}')
    surface_plot(model, model.network, domain_3D, x_label_3D, y_label_3D, title_3D,
                 f'unsupervised/figures/{model_file.path(extension="pdf", prefix="3D")}',
                 elevation=elevation, angle=angle, boundary_plot=underlying)
    single_plot_with_generator(data, index, model.network, pricer.price(prices_df, use_tqdm=True),
                               title_2D, x_label_2D, legend_label,
                               f'unsupervised/figures/{model_file.path(extension="pdf", prefix=f"1_{x_label_2D}")}',
                               f'unsupervised/figures/{model_file.path(extension="pdf", prefix=f"2_{x_label_2D}")}')


def build_prediction_data(prices_df, prediction_ranges, samples):
    data = []
    index = 0
    for i, item in enumerate(prediction_ranges):
        if isinstance(item, str):
            data.append(np.array([prices_df[item]]).transpose())
            index = i
        else:
            data.append(np.full((samples, 1), item))
    return np.array(data), index
