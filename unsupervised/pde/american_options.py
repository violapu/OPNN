import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import QuantLib as ql

from enums import OptionType
from abc import abstractmethod
from unsupervised.pde.PDE import PDENeuralNetwork
from generator import STRIKE_COL, UNDERLYING_COL, RF_RATE_COL, DAYS_TO_MATURITY_COL, DIV_COL, SIGMA_COL
from pricer import BinomialAmericanPricer


class AmericanOptionsBase(PDENeuralNetwork):
    def __init__(self, domain, pricer, option_type, network):
        self.pricer = pricer
        self.option_type = option_type

        PDENeuralNetwork.__init__(self, domain, network)

        # Regular losses
        loss_int, magnitude_int, loss_bound, magnitude_bound = \
            self.compute_loss_terms(self.network.x_int, self.network.y_int, self.network.y_bound,
                                    self.network.boundary_condition)

        self.default_loss = tf.add(loss_int, loss_bound)
        self.optimal_loss = (self.loss_weight * self.interior_domain_size) * loss_int + \
                            ((1 - self.loss_weight) * self.total_boundary_domain_size) * loss_bound
        self.magnitude_loss = loss_int / magnitude_int + loss_bound / magnitude_bound

        # Validation losses
        loss_int_validate, magnitude_int_validate, loss_bound_validate, magnitude_bound_validate = \
            self.compute_loss_terms(self.network.x_int_validate, self.network.y_int_validate,
                                    self.network.y_bound_validate,
                                    self.network.boundary_condition_validate)

        self.default_loss_validate = tf.add(loss_int_validate, loss_bound_validate)
        self.optimal_loss_validate = (self.loss_weight * self.interior_domain_size) * loss_int_validate + \
                                     ((1 - self.loss_weight) * self.total_boundary_domain_size) * \
                                     loss_bound_validate
        self.magnitude_loss_validate = loss_int_validate / magnitude_int_validate \
                                       + loss_bound_validate / magnitude_bound_validate

        # Create fetch lists
        self.fetch_list = [loss_int, loss_bound]
        self.fetch_list_validate = [loss_int, loss_bound, loss_int_validate, loss_bound_validate]

    def compute_loss_terms(self, x_int, y_int, y_bound, boundary_condition):
        dvds = PDENeuralNetwork.partial_derivative(y_int, self.underlying(x_int), 1)
        dvdt = PDENeuralNetwork.partial_derivative(y_int, self.real_time(x_int), 1)
        d2vds2 = PDENeuralNetwork.partial_derivative(y_int, self.underlying(x_int), 2)

        L = dvdt + 0.5 * (self.sigma(x_int) ** 2) * tf.square(self.underlying(x_int)) * d2vds2 + (
                self.rf_rate(x_int) - self.div_yield(x_int)) * self.underlying(
            x_int) * dvds - self.rf_rate(x_int) * y_int

        if self.option_type == OptionType.Call:
            loss_int = tf.reduce_mean(
                tf.square(
                    tf.maximum(tf.maximum(self.underlying(x_int) - self.strike_price(x_int), 0) - y_int,
                               L)))
            magnitude_int = tf.reduce_mean(
                tf.square(
                    tf.maximum(
                        tf.abs(tf.maximum(self.underlying(x_int) - self.strike_price(x_int), 0) - y_int),
                        L)))
        elif self.option_type == OptionType.Put:
            loss_int = tf.reduce_mean(
                tf.square(
                    tf.maximum(tf.maximum(self.strike_price(x_int) - self.underlying(x_int), 0) - y_int,
                               L)))
            magnitude_int = tf.reduce_mean(
                tf.square(
                    tf.maximum(
                        tf.abs(tf.maximum(self.strike_price(x_int) - self.underlying(x_int), 0) - y_int),
                        L)))
        else:
            loss_int = None
            magnitude_int = None

        loss_bound = tf.reduce_mean(tf.square(y_bound - boundary_condition))
        magnitude_bound = tf.reduce_mean(tf.square(boundary_condition))

        return loss_int, magnitude_int, loss_bound, magnitude_bound

    def boundary_condition(self, x):
        if self.option_type == OptionType.Call:
            return np.maximum(self.underlying(x) - self.strike_price(x), 0)
        elif self.option_type == OptionType.Put:
            return np.maximum(self.strike_price(x) - self.underlying(x), 0)

    def analytical_solution(self, x):
        df = pd.DataFrame()
        df[UNDERLYING_COL] = self.underlying(x)
        df[STRIKE_COL] = self.strike_price(x)
        df[RF_RATE_COL] = self.rf_rate(x)
        df[DAYS_TO_MATURITY_COL] = self.maturity(x) * 365
        df[DIV_COL] = self.div_yield(x)
        df[SIGMA_COL] = self.sigma(x)

        prices = self.pricer.price(df, use_tqdm=True)
        return prices

    @abstractmethod
    def strike_price(self, x):
        pass

    @abstractmethod
    def rf_rate(self, x):
        pass

    @abstractmethod
    def div_yield(self, x):
        pass

    @abstractmethod
    def sigma(self, x):
        pass

    @abstractmethod
    def maturity(self, x):
        pass

    @abstractmethod
    def underlying(self, x):
        pass

    @abstractmethod
    def real_time(self, x):
        pass


class AmericanSt(AmericanOptionsBase):
    input_count = 2

    def __init__(self, strike_price, rf_rate, div_yield, sigma, maturity, option_type, network,
                 calculation_date=ql.Date(8, 5, 2015), steps=2000):
        self._strike_price = float(strike_price)
        self._rf_rate = float(rf_rate)
        self._div_yield = float(div_yield)
        self._sigma = float(sigma)
        self._maturity = float(maturity)
        self.option_type = OptionType.of(option_type)
        self.calculation_date = calculation_date

        domain = [(0, 8 * self._strike_price), (0, self._maturity)]
        binomial_pricer = BinomialAmericanPricer(use_tqdm=True, calculation_date=self.calculation_date,
                                                 option_type=self.option_type, steps=steps)
        AmericanOptionsBase.__init__(self, domain, binomial_pricer, self.option_type, network)

    def underlying(self, x):
        return x[0]

    def real_time(self, x):
        return x[1]

    def strike_price(self, x):
        return self._strike_price

    def rf_rate(self, x):
        return self._rf_rate

    def div_yield(self, x):
        return self._div_yield

    def sigma(self, x):
        return self._sigma

    def maturity(self, x):
        return self._maturity


class AmericanOptionsSigmaSt(AmericanOptionsBase):
    input_count = 3

    def __init__(self, strike_price, rf_rate, div_yield, maturity, option_type, network,
                 calculation_date=ql.Date(8, 5, 2015), steps=2000):
        self._strike_price = float(strike_price)
        self._rf_rate = float(rf_rate)
        self._div_yield = float(div_yield)
        self._maturity = float(maturity)
        self.option_type = OptionType.of(option_type)
        self.calculation_date = calculation_date

        domain = [(0, 4 * self._strike_price), (0, self._maturity), (0.05, 0.5)]
        binomial_pricer = BinomialAmericanPricer(use_tqdm=True, calculation_date=self.calculation_date,
                                                 option_type=self.option_type, steps=steps)
        AmericanOptionsBase.__init__(self, domain, binomial_pricer, self.option_type, network)

    def underlying(self, x):
        return x[0]

    def real_time(self, x):
        return x[1]

    def sigma(self, x):
        return x[2]

    def strike_price(self, x):
        return self._strike_price

    def rf_rate(self, x):
        return self._rf_rate

    def div_yield(self, x):
        return self._div_yield

    def maturity(self, x):
        return self._maturity


class AmericanOptionsStrikeSt(AmericanOptionsBase):
    input_count = 3

    def __init__(self, sigma, rf_rate, div_yield, maturity, option_type, network,
                 calculation_date=ql.Date(8, 5, 2015), steps=2000):
        self._sigma = float(sigma)
        self._rf_rate = float(rf_rate)
        self._div_yield = float(div_yield)
        self._maturity = float(maturity)
        self.option_type = OptionType.of(option_type)
        self.calculation_date = calculation_date

        domain = [(0, 4 * 100), (0, self._maturity), (0, 100)]
        binomial_pricer = BinomialAmericanPricer(use_tqdm=True, calculation_date=self.calculation_date,
                                                 option_type=self.option_type, steps=steps)
        AmericanOptionsBase.__init__(self, domain, binomial_pricer, self.option_type, network)

    def underlying(self, x):
        return x[0]

    def real_time(self, x):
        return x[1]

    def sigma(self, x):
        return self._sigma

    def strike_price(self, x):
        return x[2]

    def rf_rate(self, x):
        return self._rf_rate

    def div_yield(self, x):
        return self._div_yield

    def maturity(self, x):
        return self._maturity


class AmericanOptionsSigmaStrikeSt(AmericanOptionsBase):
    input_count = 4

    def __init__(self, rf_rate, div_yield, maturity, option_type, network,
                 calculation_date=ql.Date(8, 5, 2015), steps=2000):
        self._rf_rate = float(rf_rate)
        self._div_yield = float(div_yield)
        self._maturity = float(maturity)
        self.option_type = OptionType.of(option_type)
        self.calculation_date = calculation_date

        domain = [(0, 200), (0, self._maturity), (0.1, 0.4), (0, 50)]
        binomial_pricer = BinomialAmericanPricer(use_tqdm=True, calculation_date=self.calculation_date,
                                                 option_type=self.option_type, steps=steps)
        AmericanOptionsBase.__init__(self, domain, binomial_pricer, self.option_type, network)

    def underlying(self, x):
        return x[0]

    def real_time(self, x):
        return x[1]

    def sigma(self, x):
        return x[2]

    def strike_price(self, x):
        return x[3]

    def rf_rate(self, x):
        return self._rf_rate

    def div_yield(self, x):
        return self._div_yield

    def maturity(self, x):
        return self._maturity
