from configparser import ConfigParser

DEFAULT_SUPERVISED_CONFIG_FILE_PATH = 'supervised.conf'
DEFAULT_UNSUPERVISED_CONFIG_FILE_PATH = 'unsupervised.conf'
DEFAULT_COMPARISON_CONFIG_FILE_PATH = 'compare.conf'
CONFIGURATIONS = {}


def get_config(name: str):
    if name in CONFIGURATIONS:
        return CONFIGURATIONS[name]
    else:
        config = ConfigParser()
        config.read(name)
        CONFIGURATIONS[name] = config
        return config


def get_comparison_config():
    return get_config(DEFAULT_COMPARISON_CONFIG_FILE_PATH)['GraphingParams']


def get_default_supervised_config():
    return get_config(DEFAULT_SUPERVISED_CONFIG_FILE_PATH)


def get_supervised_params(name: str):
    return merge_params(get_supervised_default_params(), get_default_supervised_config()[name])


def get_supervised_prediction_params(name: str):
    return get_default_supervised_config()[f'{name}.PredictionParams']


def get_supervised_compare_graphing_params():
    return get_default_supervised_config()['ComparisonGraphingParams']


def get_supervised_default_params():
    return get_default_supervised_config()['Default']


def get_default_unsupervised_config():
    return get_config(DEFAULT_UNSUPERVISED_CONFIG_FILE_PATH)


def get_unsupervised_graphing_params(name: str):
    return get_default_unsupervised_config()[f'{name}.GraphingParams']


def get_unsupervised_model_params(name: str):
    return get_default_unsupervised_config()[f'{name}.ModelParams']


def get_unsupervised_prediction_params(name: str):
    return get_default_unsupervised_config()[f'{name}.PredictionParams']


def get_unsupervised_training_params(name: str):
    return merge_params(get_unsupervised_default_training_params(),
                        get_default_unsupervised_config()[f'{name}.TrainingParams'])


def get_unsupervised_default_training_params():
    return get_default_unsupervised_config()['Default.TrainingParams']


def merge_params(param_a, param_b):
    new_dict = dict(param_a)
    new_dict.update(param_b)
    return new_dict
