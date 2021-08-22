import numpy as np
import pandas as pd

from enums import OptionType

STRIKE_COL = 'strike'
UNDERLYING_COL = 'underlying'
RF_RATE_COL = 'rf_rate'
DIV_COL = 'div'
SIGMA_COL = 'sigma'
DAYS_TO_MATURITY_COL = 'days_to_maturity'


class ColumnGenerator:
    def __init__(self, col_name):
        self.col_name = col_name

    def generate_array(self, samples):
        pass

    def generate(self, df, samples):
        df[self.col_name] = self.generate_array(samples)


class ConstantGenerator(ColumnGenerator):
    def __init__(self, constant, col_name):
        ColumnGenerator.__init__(self, col_name)
        self.constant = constant
        self.col_name = col_name

    def generate_array(self, samples):
        return [self.constant] * samples


class LinspaceGenerator(ColumnGenerator):
    def __init__(self, gen_range, col_name):
        ColumnGenerator.__init__(self, col_name)
        self.gen_range = gen_range
        self.col_name = col_name

    def generate_array(self, samples):
        return np.linspace(*self.gen_range, num=samples)


class UniformGenerator:
    def __init__(self, gen_range, col_name):
        self.gen_range = gen_range
        self.col_name = col_name

    def generate(self, df, samples):
        df[self.col_name] = np.random.uniform(*self.gen_range, size=samples)


class RandIntGenerator:
    def __init__(self, gen_range, col_name):
        self.gen_range = gen_range
        self.col_name = col_name

    def generate(self, df, samples):
        df[self.col_name] = np.random.randint(*self.gen_range, size=samples)


class DiscreteDividendGenerator:
    def __init__(self, underlying_multiplier, max_divs, col_name):
        self.col_name = col_name
        self.underlying_multiplier = underlying_multiplier
        self.max_divs = max_divs

    def generate(self, df, samples):
        days_to_maturity = df[DAYS_TO_MATURITY_COL]
        underlying = df[UNDERLYING_COL]
        divs = []
        for i in range(samples):
            div_num = np.random.randint(1, self.max_divs + 1)
            divs.append(np.array([np.sort(np.random.randint(0, days_to_maturity[i], div_num)),
                                  np.random.randint(0, underlying[i] * self.underlying_multiplier,
                                                    div_num)]))
        df[DIV_COL] = divs


class IntrinsicValues:
    def __init__(self, optionType):
        self.optionType = optionType

    def generate(self, df):
        if self.optionType == OptionType.Call:
            df['intrinsic_val'] = np.maximum(df[UNDERLYING_COL] - df[STRIKE_COL], 0)
        elif self.optionType == OptionType.Put:
            df['intrinsic_val'] = np.maximum(df[STRIKE_COL] - df[UNDERLYING_COL], 0)


class DataGenerator:
    def __init__(self, generators, seed=None):
        self.generators = generators
        self.seed = seed

    def generate(self, samples=100):
        df = pd.DataFrame()

        for i, generator in enumerate(self.generators):
            np.random.seed(self.seed + i)
            generator.generate(df, samples=samples)
        return df

    def generate_arrays(self, samples=100):
        arrays = []
        for i, generator in enumerate(self.generators):
            np.random.seed(self.seed + i)
            arrays.append(generator.generate_array(samples=samples))
        return arrays


def build_random_generator(ranges_dict, seed=42):
    column_generators = []
    for name in ranges_dict.keys():
        domain = ranges_dict[name]
        if isinstance(domain, tuple):
            if isinstance(domain[0], int) and isinstance(domain[1], int):
                column_generators.append(RandIntGenerator(domain, name))
            else:
                column_generators.append(UniformGenerator(domain, name))
        else:
            column_generators.append(ConstantGenerator(domain, name))
    return DataGenerator(column_generators, seed)


def build_constant_generator(values, columns=None, seed=42):
    if columns is None:
        columns = [STRIKE_COL, UNDERLYING_COL, RF_RATE_COL, DAYS_TO_MATURITY_COL, DIV_COL, SIGMA_COL]

    column_generators = []
    for val, col in zip(values, columns):
        if isinstance(val, tuple):
            column_generators.append(LinspaceGenerator(val, col))
        else:
            column_generators.append(ConstantGenerator(val, col))

    return DataGenerator(column_generators, seed)
