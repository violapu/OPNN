import pandas as pd
import tensorflow.compat.v1 as tf

from unsupervised.graphing.graphing import surface_plot, single_plot, avg_losses_against_iterations, \
    losses_against_iterations
from unsupervised.model_factory import ModelFile

tf.disable_v2_behavior()


def analytical_solution_plotting(model_name, model_file, factory, domain_3D, x_label_3D, y_label_3D,
                                 title_3D, domain_2D, x_label_2D, title_2D, param, underlying=False):
    model = factory.create(model_name)
    model.network.load_weights(f'unsupervised/weights/{model_file.path()}/weights')

    ModelFile.make_path(f'unsupervised/figures/{model_file.dir()}')

    history = pd.read_csv(f'unsupervised/histories/{model_file.path(extension="csv")}')
    # losses_against_iterations(history,
    #                           f'unsupervised/figures/{model_file.path(extension="pdf", prefix=f"losses")}')
    # avg_losses_against_iterations(history,
    #                               f'unsupervised/figures/{model_file.path(extension="pdf", prefix=f"avg_losses")}')
    surface_plot(model, model.network, domain_3D, x_label_3D, y_label_3D, title_3D,
                 f'unsupervised/figures/{model_file.path(extension="pdf", prefix=f"{param}_3D")}',
                 boundary_plot=underlying)
    # implied_vol_surface_plot(model, model.network, domain_3D, x_label_3D, y_label_3D, title_3D,
    #                          f'figures/{model_file.path(extension="pdf", prefix=f"{param}_3Dvol")}')
    single_plot(model, domain_2D, x_label_2D, title_2D,
                f'unsupervised/figures/{model_file.path(extension="pdf", prefix=f"1_{param}")}',
                f'unsupervised/figures/{model_file.path(extension="pdf", prefix=f"2_{param}")}')
