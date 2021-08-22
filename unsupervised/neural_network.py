import tensorflow.compat.v1 as tf
import numpy as np
from enums import ActivationFunction
from unsupervised.tensorflow1.external_optimizer import ScipyOptimizerInterface


# Neural network class
class NeuralNetwork:
    def __init__(self, input_dimension, hidden_layers, activation_function=ActivationFunction.Tanh,
                 seed=None):
        self.layers = [input_dimension] + hidden_layers + [1]
        self.activation_function = activation_function

        if seed is not None:
            tf.set_random_seed(seed)
            np.random.seed(seed)

        # Network parameters
        self.weights, self.biases = self.create_network_parameters()

        # Input placeholders
        self.x_int = []
        self.x_bound = []
        self.x_int_validate = []
        self.x_bound_validate = []
        for i in range(input_dimension):
            self.x_int.append(tf.placeholder(tf.float64, shape=[None, 1], name="xInt" + str(i)))
            self.x_bound.append(tf.placeholder(tf.float64, shape=[None, 1], name="xBound" + str(i)))
            self.x_int_validate.append(
                tf.placeholder(tf.float64, shape=[None, 1], name="xIntValidate" + str(i)))
            self.x_bound_validate.append(
                tf.placeholder(tf.float64, shape=[None, 1], name="xBoundValidate" + str(i)))

        # Outputs
        self.y_int = self.create_graph(self.x_int, activation_function)
        self.y_bound = self.create_graph(self.x_bound, activation_function)
        self.y_int_validate = self.create_graph(self.x_int_validate, activation_function)
        self.y_bound_validate = self.create_graph(self.x_bound_validate, activation_function)

        # Boundary condition & Source functions
        self.boundary_condition = tf.placeholder(tf.float64, shape=[None, 1], name="BoundaryCondition")
        self.boundary_condition_validate = tf.placeholder(tf.float64, shape=[None, 1],
                                                          name="BoundaryConditionValidate")
        self.source_function = tf.placeholder(tf.float64, shape=[None, 1], name="SourceFunction")
        self.source_function_validate = tf.placeholder(tf.float64, shape=[None, 1],
                                                       name="SourceFunctionValidate")

        self.saver = tf.train.Saver()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_network_parameters(self):
        weights = []
        biases = []

        initializer = tf.initializers.he_normal()

        for i in range(len(self.layers) - 1):
            # w = tf.Variable(NeuralNetwork.glorot_initializer(self.layers[i], self.layers[i + 1]),
            #                 dtype=tf.float64)
            x = tf.cast(initializer((self.layers[i], self.layers[i + 1])), tf.float64)
            print(x)
            w = tf.Variable(x, dtype=tf.float64)
            b = tf.Variable(tf.zeros([1, self.layers[i + 1]], dtype=tf.float64), dtype=tf.float64)

            weights.append(w)
            biases.append(b)

        return weights, biases

    # @staticmethod
    # def glorot_initializer(dim1, dim2):
    #     return tf.random_uniform([dim1, dim2],
    #                              minval=- np.sqrt(6 / (dim1 + dim2)),
    #                              maxval=np.sqrt(6 / (dim1 + dim2)), dtype=tf.float64)

    def create_graph(self, x, activation_function):
        y = tf.concat(x, axis=1)
        for i in range(len(self.layers) - 2):
            w = self.weights[i]
            b = self.biases[i]

            if activation_function == ActivationFunction.Tanh:
                y = tf.nn.tanh(tf.add(tf.matmul(y, w), b))

            elif activation_function == ActivationFunction.Sigmoid:
                y = tf.nn.sigmoid(tf.add(tf.matmul(y, w), b))

            elif activation_function == ActivationFunction.Sin:
                y = tf.sin(tf.add(tf.matmul(y, w), b))

            elif activation_function == ActivationFunction.Cos:
                y = tf.cos(tf.add(tf.matmul(y, w), b))

            elif activation_function == ActivationFunction.Atan:
                y = tf.atan(tf.add(tf.matmul(y, w), b))

            elif activation_function == ActivationFunction.Relu:
                y = tf.nn.relu(tf.add(tf.matmul(y, w), b))

            elif activation_function == ActivationFunction.Softplus:
                y = tf.nn.softplus(tf.add(tf.matmul(y, w), b))

        w = self.weights[-1]
        b = self.biases[-1]
        return tf.add(tf.matmul(y, w), b)

    def train(self, loss_function, iterations, feed_dict, fetch_list, callback):
        optimizer = ScipyOptimizerInterface(tf.log(loss_function),
                                            method='L-BFGS-B',
                                            options={'maxiter': iterations,
                                                     'maxfun': iterations,
                                                     'maxcor': 50,
                                                     'maxls': 50,
                                                     'ftol': 1.0 * np.finfo(
                                                         np.float64).eps,
                                                     'gtol': 0.000001})

        optimizer.minimize(self.session, feed_dict=feed_dict, fetches=fetch_list, loss_callback=callback)

    def predict(self, x):
        feed_dict = dict()
        for i in range(len(x)):
            feed_dict[self.x_int[i]] = x[i]
        return self.session.run(self.y_int, feed_dict=feed_dict)

    def save_weights(self, path="Autosave"):
        self.saver.save(self.session, path)

    def load_weights(self, path="Autosave"):
        self.saver.restore(self.session, path)

    def cleanup(self):
        self.session.close()
        tf.reset_default_graph()
