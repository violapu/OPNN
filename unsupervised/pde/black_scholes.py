from abc import abstractmethod

import tensorflow.compat.v1 as tf
import numpy as np
from unsupervised.pde.PDE import PDENeuralNetwork
from tensorflow_probability import distributions as tfd
from enums import OptionType


class BlackScholesBase(PDENeuralNetwork):
    def __init__(self, domain, option_type, network=None):
        self.option_type = option_type

        PDENeuralNetwork.__init__(self, domain, option_type, network)

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
        self.magnitude_loss_validate = loss_int_validate / magnitude_int_validate + \
                                       loss_bound_validate / magnitude_bound_validate

        # Create fetch lists
        self.fetch_list = [loss_int, loss_bound]
        self.fetch_list_validate = [loss_int, loss_bound, loss_int_validate, loss_bound_validate]

        # Analytical Solution
        self.analytical_interior = self.analytical_solution(self.network.x_int)
        self.analytical_boundary = self.analytical_solution(self.network.x_bound)

        # Analytical magnitudes
        dvds = self.partial_derivative(self.analytical_interior, self.underlying(self.network.x_int), 1)
        dvdt = self.partial_derivative(self.analytical_interior, self.real_time(self.network.x_int), 1)
        d2vds2 = self.partial_derivative(self.analytical_interior, self.underlying(self.network.x_int),
                                         2)
        self.analytical_interior_magnitude = self.interior_domain_size * \
                                             tf.reduce_mean(tf.square(dvds + dvdt + d2vds2))
        self.analytical_boundary_magnitude = self.total_boundary_domain_size * \
                                             tf.reduce_mean(tf.square(self.network.boundary_condition))

    def compute_loss_terms(self, x_int, y_int, y_bound, boundary_condition):
        dvds = PDENeuralNetwork.partial_derivative(y_int, self.underlying(x_int), 1)
        dvdt = PDENeuralNetwork.partial_derivative(y_int, self.real_time(x_int), 1)
        d2vds2 = PDENeuralNetwork.partial_derivative(y_int, self.underlying(x_int), 2)
        loss_int = tf.reduce_mean(
            tf.square(
                dvdt + 0.5 * (self.sigma(x_int) ** 2) * tf.square(self.underlying(x_int)) * d2vds2 +
                (self.rf_rate(x_int) - self.div_yield(x_int)) * self.underlying(
                    x_int) * dvds - self.rf_rate(x_int) * y_int))
        magnitude_int = tf.reduce_mean(tf.square(
            tf.abs(dvdt) + tf.abs(
                0.5 * (self.sigma(x_int) ** 2) * tf.square(self.underlying(x_int)) * d2vds2) + tf.abs(
                (self.rf_rate(x_int) - self.div_yield(x_int)) * self.underlying(x_int) * dvds) + tf.abs(
                self.rf_rate(x_int) * y_int)))
        loss_bound = tf.reduce_mean(tf.square(y_bound - boundary_condition))
        magnitude_bound = tf.reduce_mean(tf.square(boundary_condition))

        return loss_int, magnitude_int, loss_bound, magnitude_bound

    def analytical_solution(self, x):
        d1 = (tf.log(self.underlying(x) / self.strike_price(x)) + (
                self.rf_rate(x) - self.div_yield(x) + 0.5 * self.sigma(x) ** 2) * (
                      self.maturity(x) - self.real_time(x))) / (
                     self.sigma(x) * tf.sqrt(self.maturity(x) - self.real_time(x)))
        d2 = d1 - self.sigma(x) * tf.sqrt(self.maturity(x) - self.real_time(x))

        n = tfd.Normal(loc=0., scale=1.)
        cdf_d1 = tf.cast(n.cdf(tf.cast(d1, tf.float32)), tf.float64)
        cdf_d2 = tf.cast(n.cdf(tf.cast(d2, tf.float32)), tf.float64)

        if self.option_type == OptionType.Call:
            return self.underlying(x) * tf.exp(-self.div_yield(x) * (
                    self.maturity(x) - self.real_time(x))) * cdf_d1 - self.strike_price(x) * tf.exp(
                -self.rf_rate(x) * (self.maturity(x) - self.real_time(x))) * cdf_d2
        elif self.option_type == OptionType.Put:
            return self.strike_price(x) * tf.exp(
                -self.rf_rate(x) * (self.maturity(x) - self.real_time(x))) * (
                           1.0 - cdf_d2) - self.underlying(x) * tf.exp(
                -self.div_yield(x) * (self.maturity(x) - self.real_time(x))) * (1.0 - cdf_d1)

    def boundary_condition(self, x):
        if self.option_type == OptionType.Call:
            return np.maximum(self.underlying(x) - self.strike_price(x) * np.exp(
                -self.rf_rate(x) * (self.maturity(x) - self.real_time(x))), 0)
        elif self.option_type == OptionType.Put:
            return np.maximum(self.strike_price(x) * np.exp(
                -self.rf_rate(x) * (self.maturity(x) - self.real_time(x))) - self.underlying(x), 0)

    # Sample uniform collocation points on the boundary of the domain
    def sample_boundary_x(self, point_count):
        x_bound = super().sample_boundary_x(point_count)
        return np.delete(x_bound, np.where(self.real_time(x_bound) == 0)[0], axis=1)

    def get_analytical_plot_data(self, x):
        new_x = None
        inputs = []
        for xx in x:
            if isinstance(xx, tuple):
                if new_x is None:
                    new_x = np.linspace(xx[0], xx[1])
                    inputs.append(tf.constant(new_x))
                else:
                    raise ValueError('Can only provide one range in the domain.')
            elif isinstance(xx, int) or isinstance(xx, float):
                inputs.append(tf.constant(np.linspace(xx, xx)))
        result = self.analytical_solution(inputs)
        return new_x, result.eval(session=tf.Session())

    def get_predicted_plot_data(self, x):
        new_x = None
        inputs = []
        for xx in x:
            if isinstance(xx, tuple):
                if new_x is None:
                    new_x = np.array([np.linspace(xx[0], xx[1])]).transpose()
                    inputs.append(new_x)
                else:
                    raise ValueError('Can only provide one range in the domain.')
            elif isinstance(xx, int) or isinstance(xx, float):
                inputs.append(np.array([np.linspace(xx, xx)]).transpose())
        feed_dict = self.get_feed_dict(x_int=inputs)
        y = self.network.session.run(self.network.y_int, feed_dict)
        return new_x.transpose()[0], y.transpose()[0]

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


class BSSt(BlackScholesBase):
    input_count = 2

    def __init__(self, strike_price, rf_rate, div_yield, sigma, maturity, option_type, network):
        self._strike_price = float(strike_price)
        self._rf_rate = float(rf_rate)
        self._div_yield = float(div_yield)
        self._sigma = float(sigma)
        self._maturity = float(maturity)
        self.option_type = OptionType.of(option_type)

        domain = [(0, 4 * self._strike_price), (0, self._maturity)]
        BlackScholesBase.__init__(self, domain, self.option_type, network)

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


class BSSigmaSt(BlackScholesBase):
    input_count = 3

    def __init__(self, strike_price, rf_rate, div_yield, maturity, option_type, network):
        self._strike_price = float(strike_price)
        self._rf_rate = float(rf_rate)
        self._div_yield = float(div_yield)
        self._maturity = float(maturity)
        self.option_type = OptionType.of(option_type)

        domain = [(0, 4 * self._strike_price), (0, self._maturity), (0.05, 0.5)]
        BlackScholesBase.__init__(self, domain, self.option_type, network)

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
        return x[2]

    def maturity(self, x):
        return self._maturity


class BSStrikeSt(BlackScholesBase):
    input_count = 3

    def __init__(self, rf_rate, div_yield, sigma, maturity, option_type, network):
        self._rf_rate = float(rf_rate)
        self._div_yield = float(div_yield)
        self._sigma = float(sigma)
        self._maturity = float(maturity)
        self.option_type = OptionType.of(option_type)

        domain = [(0, 4 * 100), (0, self._maturity), (0, 100)]
        BlackScholesBase.__init__(self, domain, self.option_type, network)

    def underlying(self, x):
        return x[0]

    def real_time(self, x):
        return x[1]

    def strike_price(self, x):
        return x[2]

    def rf_rate(self, x):
        return self._rf_rate

    def div_yield(self, x):
        return self._div_yield

    def sigma(self, x):
        return self._sigma

    def maturity(self, x):
        return self._maturity


class BSStrikeSigmaSt(BlackScholesBase):
    input_count = 4

    def __init__(self, rf_rate, div_yield, maturity, option_type, network):
        self.option_type = OptionType.of(option_type)
        self._rf_rate = float(rf_rate)
        self._div_yield = float(div_yield)
        self._maturity = float(maturity)

        domain = [(0, 200), (0, self._maturity), (0, 50), (0.10, 0.40)]
        BlackScholesBase.__init__(self, domain, self.option_type, network)

    def underlying(self, x):
        return x[0]

    def real_time(self, x):
        return x[1]

    def strike_price(self, x):
        return x[2]

    def rf_rate(self, x):
        return self._rf_rate

    def div_yield(self, x):
        return self._div_yield

    def sigma(self, x):
        return x[3]

    def maturity(self, x):
        return self._maturity


class BSAll(BlackScholesBase):
    input_count = 6

    def __init__(self, maturity, option_type, network):
        self._maturity = float(maturity)
        self.option_type = OptionType.of(option_type)

        domain = [(0, 400), (0, self._maturity), (0, 100), (-0.02, 0.08), (0, 0.08), (0.05, 0.5)]
        BlackScholesBase.__init__(self, domain, self.option_type, network)

    def underlying(self, x):
        return x[0]

    def real_time(self, x):
        return x[1]

    def strike_price(self, x):
        return x[2]

    def rf_rate(self, x):
        return x[3]

    def div_yield(self, x):
        return x[4]

    def sigma(self, x):
        return x[5]

    def maturity(self, x):
        return self._maturity

