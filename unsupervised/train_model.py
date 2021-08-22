from unsupervised.model_factory import ModelFactory, ModelFile
from enums import ActivationFunction, TrainMode, OptionType
from unsupervised.history import History


def train(model_name, model_file, factory, train_mode, iterations):
    model = factory.create(model_name)

    model.train(train_mode=train_mode, iterations=iterations)
    ModelFile.make_path(f'unsupervised/weights/{model_file.path()}')
    model.network.save_weights(f'unsupervised/weights/{model_file.path()}/weights')

    l2_error = model.compute_l2_error()
    max_error = model.compute_max_error()

    ModelFile.make_path(f'unsupervised/histories/{model_file.dir()}')
    History.save_history(model.history, l2_error, max_error,
                         f'unsupervised/histories/{model_file.path(extension="csv")}')

    model.network.cleanup()


if __name__ == '__main__':
    model = 'BSAll'
    train_mode = TrainMode.DefaultAdaptive
    iters = 200
    activation = ActivationFunction.Softplus
    layers = 4
    nodes_per_layer = 128
    option_type = OptionType.Put

    train(model_name=model,
          model_file=ModelFile(model, train_mode, iters, activation, layers, nodes_per_layer, option_type),
          factory=ModelFactory(layers=layers, nodes_per_layer=nodes_per_layer, option_type=option_type),
          train_mode=train_mode,
          iterations=iters)
