import QuantLib as ql

from enum import Enum
from typing import Union


class ActivationFunction(Enum):
    Tanh = 0
    Sigmoid = 1
    Sin = 2
    Cos = 3
    Atan = 4
    Relu = 5
    Softplus = 6


class TrainMode(Enum):
    Default = 0
    Optimal = 1
    Magnitude = 2
    DefaultAdaptive = 3
    OptimalAdaptive = 4
    MagnitudeAdaptive = 5


class OptionType(Enum):
    Call = 0
    Put = 1

    @staticmethod
    def of(val: Union[str, 'OptionType']):
        if isinstance(val, OptionType):
            return val
        else:
            return OptionType[val]

    @staticmethod
    def ql_type(option_type):
        if option_type == OptionType.Call:
            return ql.Option.Call
        elif option_type == OptionType.Put:
            return ql.Option.Put
