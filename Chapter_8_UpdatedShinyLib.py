from typing import Union, List
import numpy as np
from time import time
from functools import reduce


class ShinyActivations:
    class Activation:
        delta = None
        sending_delta = None
        learning_rate = ""
        output_size = ""
        weights = ""

        def __repr__(self):
            return f"{self.__class__.__name__} Activation"

        def forward(self, x):
            raise NotImplementedError

        def backward(self, x):
            raise NotImplementedError

    class Relu(Activation):
        def forward(self, x):
            return (x > 0) * x

        def backward(self, x):
            return x > 0

    class Sigmoid(Activation):
        def forward(self, x):
            return 1 / (1 + np.exp(-x))

        def backward(self, x):
            return x * (1 - x)


class ShinyLayers:
    default_learning_rate = 1e-4

    class Layer:
        output_size = ""
        learning_rate = ""
        weights = ""

        def __init__(self):
            if type(self) == ShinyLayers.Layer:
                raise ShinyExceptions.Instantiation(
                    f"You can not instantiate {self.__class__.__name__} class, please be careful :)")

    class Hidden(Layer):
        def __init__(self):
            super().__init__()
            if type(self) == ShinyLayers.Hidden:
                raise ShinyExceptions.Instantiation(
                    f"You can not instantiate {self.__class__.__name__} class, please be careful :)")

        def get_output_size(self):
            return self.output_size

        def __repr__(self):
            return f"{self.__class__.__name__} Layer: output size = {self.output_size}," \
                   f" learning rate: {self.learning_rate}"

        def forward(self, x: np.array):
            raise NotImplementedError

        def backward(self, x: np.array):
            raise NotImplementedError

    class FullyConnected(Hidden):
        def __init__(self, output_size: int, learning_rate: float = None):
            super().__init__()
            self.output_size = output_size
            self.learning_rate = ShinyLayers.default_learning_rate if learning_rate is None else learning_rate
            self.weights: np.array = None

        def forward(self, x: np.array):
            return np.dot(x, self.weights)

        def backward(self, x: np.array):
            return np.dot(x, self.weights.T)

    class Dropout(Hidden):
        def __init__(self, p):
            super().__init__()
            self.p = p
            self.mask: np.array = None

        def forward(self, x: np.array):
            self.mask = np.ones(x.shape)
            self.mask[:, np.random.choice(np.arange(0, len(x)), int(len(x) * self.p), replace=False)] = 0
            return x * self.mask * (1 / (1 - self.p))

        def backward(self, x: np.array):
            return x * self.mask

    class Input(Layer):
        def __init__(self, size=0):
            super().__init__()
            self.output_size = size

        def set_size(self, size):
            self.output_size = size

        def __repr__(self):
            return f"Input Layer: size = {self.output_size}"


class ShinyErrorMeasures:
    class ErrorMeasure:

        def measure(self, x, y):
            raise NotImplementedError

    class SquaredError(ErrorMeasure):
        def measure(self, x, y):
            return sum((x - y) ** 2) ** (1 / 2)


class ShinyExceptions:
    class SetData(Exception):
        ...

    class DataShape(Exception):
        ...

    class ModelLayer(Exception):
        ...

    class Instantiation(Exception):
        ...


class ShinyInformation:
    class Information:
        def __init__(self):
            self.error: np.array = np.array([])
            self.weights: np.array = np.array([])
            self.delta: np.array = np.array([])
            self.iteration: int = 0
            self.epoch: int = 0
            self.start_time: float = 0
            self.end_time: float = 0
            self.step_time: float = 0
            self.total_time: float = 0
            self.layers: List[Union[ShinyLayers.Hidden, ShinyLayers.Input, ShinyActivations.Activation]] = []
            self.test_X: np.array = None
            self.test_Y: np.array = None
            self.network = None
            self.test_error: float = float("inf")
            self.layer_outputs: List[ShinyLayerOutput] = []
            self.early_stopping = False
            self.additional_data: dict = {}  # user can save his/her arbitrary data


class ShinyRepresentations:
    class Representation:
        @staticmethod
        def display(information: ShinyInformation.Information):
            raise NotImplementedError

        @staticmethod
        def stop_condition(information: ShinyInformation.Information):
            return False and information

    class StandardRepresentation(Representation):
        @staticmethod
        def display(information: ShinyInformation.Information):
            print(
                f"Time:{information.total_time}\n epoch: {information.epoch} iteration: {information.iteration} error:"
                f" {information.error}")
            # print(
            #     f"""Test Error: {
            #     ShinyTestErrorMeasurements.MeanSquaredError.measure(information.network(information.test_X),
            #                                                         information.test_Y)
            #     }""")

        @staticmethod
        def stop_condition(information: ShinyInformation.Information):
            return information.test_error < 100


class ShinyLayerOutput:
    def __init__(self):
        self.value = None
        self.delta = None
        self.sending_delta = None

    def __call__(self, *args, **kwargs):
        return self.value


class ShinyTestErrorMeasurements:
    class TestErrorMeasurement:
        @staticmethod
        def measure(predict, goal):
            raise NotImplementedError

    class MeanSquaredError(TestErrorMeasurement):
        @staticmethod
        def measure(predict, goal, in_string=False):
            value = np.sum((predict - goal) ** 2) / len(predict)
            return f"{value:,}" if in_string else value

    class MeanAbsoluteError(TestErrorMeasurement):
        @staticmethod
        def measure(predict, goal, in_string=False):
            value = np.sum(abs(predict - goal)) / len(predict)
            return f"{value:,}" if in_string else value

    class CrossEntropy(TestErrorMeasurement):
        @staticmethod
        def measure(predict, goal):
            ...

    class BinaryCrossEntropy(TestErrorMeasurement):
        @staticmethod
        def measure(predict, goal):
            ...


class Shiny:
    def __init__(self,
                 error_measurement=ShinyErrorMeasures.SquaredError,
                 representation=ShinyRepresentations.StandardRepresentation,
                 early_stopping=False):
        self.train_X: np.array = None
        self.train_y: np.array = None
        self.error_measurement = error_measurement()
        self.information = ShinyInformation.Information()
        self.representation = representation
        self.information.early_stopping = early_stopping

    def add(self, *layers: Union[ShinyLayers, ShinyActivations]):
        if len(self.information.layers) == 0:
            self.information.layers.append(ShinyLayers.Input(self.train_X.shape[1] if self.train_X is not None else ""))
        layers = list(
            filter(lambda layer: isinstance(layer, ShinyLayers.Hidden) and type(layer) is not ShinyLayers.Hidden,
                   layers))
        if len(layers) == 0:
            raise ShinyExceptions.ModelLayer("No hidden layer is available :(")
        self.information.layers = [*self.information.layers, *layers]
        return self

    def set_data(self, train_x: np.array, train_y: np.array, test_x: np.array = None, test_y: np.array = None):
        if len(train_X_shape := train_x.shape) != 2:
            raise ShinyExceptions.DataShape(
                f"your train data is {train_X_shape}D! it must be 2D bro! please be careful with your training data,"
                f" second dimension must be FEATURES!!!")
        if len(train_y_shape := train_y.shape) != 1:
            raise ShinyExceptions.DataShape(f"your goal data is {train_y_shape}D! it must be 1D bro!")
        self.train_X = train_x
        self.train_y = train_y
        if test_x is not None:
            if test_y is None:
                raise ShinyExceptions.SetData("you can not set test_X and ignore test_y!")
            self.information.test_X = test_x
            self.information.test_Y = test_y
        else:
            if test_y is not None:
                raise ShinyExceptions.SetData("you can not set test_X and ignore test_y and VICE VERSA!")

        return self

    def represent(self):
        self.representation.display(self.information)

    def __configuration(self):
        if self.train_X is None or self.train_y is None:
            raise ShinyExceptions.SetData("you did not set train data! please do it by set_data method")
        if not self.information.layers:
            raise ShinyExceptions.ModelLayer("your model sequential is full of empty! please at least add one layer")
        input_size = self.train_X.shape[1]
        self.information.layers[0].set_size(input_size)
        for layer in filter(lambda l: isinstance(l, ShinyLayers.FullyConnected), self.information.layers):
            layer.weights = 2 * np.random.random((input_size, (input_size := layer.output_size))) - 1
        self.information.layer_outputs = [ShinyLayerOutput() for _ in range(len(self.information.layers))]

    def train(self, epoch_num=1, batch_size=1):
        self.__configuration()
        for self.information.epoch in range(epoch_num):
            for self.information.iteration in range(int(len(self.train_X) / batch_size)):
                self.information.start_time = time()
                self.information.layer_outputs[0].value, goal = \
                    self.train_X[
                    self.information.iteration * batch_size:(self.information.iteration + 1) * batch_size], \
                    self.train_y[
                    self.information.iteration * batch_size:(self.information.iteration + 1) * batch_size].reshape(-1,
                                                                                                                   1)
                for index, layer in enumerate(self.information.layers[1:], 1):
                    self.information.layer_outputs[index].value = layer.forward(
                        self.information.layer_outputs[index - 1].value)
                self.information.error = self.error_measurement.measure(self.information.layer_outputs[-1].value, goal)
                self.information.delta = self.information.layer_outputs[-1].delta = (self.information.layer_outputs[
                                                                                         -1].value - goal) / batch_size
                self.information.layer_outputs[-1].sending_delta = self.information.layers[-1].backward(
                    self.information.layer_outputs[-1].delta)
                for layer_output_index in range(len(self.information.layer_outputs) - 2, 0, -1):
                    if isinstance((layer := self.information.layers[layer_output_index]), ShinyActivations.Activation):
                        self.information.layer_outputs[layer_output_index].sending_delta = \
                            self.information.layer_outputs[layer_output_index].delta = \
                            self.information.layer_outputs[layer_output_index + 1].sending_delta * layer.backward(
                                self.information.layer_outputs[layer_output_index].value)
                    elif isinstance((layer := self.information.layers[layer_output_index]), ShinyLayers.Dropout):
                        self.information.layer_outputs[layer_output_index].sending_delta = \
                            self.information.layer_outputs[layer_output_index].delta = layer.backward(
                            self.information.layer_outputs[layer_output_index + 1].sending_delta)
                    elif isinstance(self.information.layers[layer_output_index], ShinyLayers.FullyConnected):
                        self.information.layer_outputs[layer_output_index].delta = self.information.layer_outputs[
                            layer_output_index + 1].sending_delta
                        self.information.layer_outputs[layer_output_index].sending_delta = self.information.layers[
                            layer_output_index].backward(
                            self.information.layer_outputs[layer_output_index].delta)
                for layer_index in range(1, len(self.information.layers)):
                    if isinstance((layer := self.information.layers[layer_index]), ShinyLayers.FullyConnected):
                        layer.weights -= layer.learning_rate * \
                                         self.information.layer_outputs[
                                             layer_index - 1].value.T.dot(
                                             self.information.layer_outputs[
                                                 layer_index].delta)
                self.information.end_time = time()
                self.information.step_time = self.information.end_time - self.information.start_time
                self.information.total_time += self.information.step_time
                self.information.network = self.predict
                self.represent()
                if self.representation.stop_condition(self.information):
                    return self
        return self

    def __predict(self, value):
        return reduce(lambda x, y: y.forward(x), [value, *list(
            filter(lambda layer: isinstance(layer, (ShinyActivations.Activation, ShinyLayers.FullyConnected)),
                   self.information.layers))])

    def predict(self, test_x):
        return np.array(list(map(lambda data: self.__predict(data), test_x))).reshape(len(test_x, ))

    def info(self):
        attrs = ["output_size", "learning_rate"]
        layers = list(map(lambda l: [l.__class__.__name__, *list(map(lambda attr: str(getattr(l, attr)), attrs))],
                          self.information.layers))
        titles = ["Layer",
                  *list(map(lambda a: " ".join(list(map(lambda s: s.capitalize(), a.split("_")))), attrs))]
        max_string_lengths = np.vectorize(lambda x: len(x))(np.array([titles, *layers])).max(axis=0)
        print('─' * (np.sum(max_string_lengths) + len(titles) - 1),
              " ".join(list(map(lambda x: x[0].ljust(x[1]), list(zip(titles, max_string_lengths))))),
              '=' * (np.sum(max_string_lengths) + len(titles) - 1),
              sep="\n"
              )
        print(*list(
            map(lambda layer: " ".join(list(map(lambda x: x[0].ljust(x[1]), list(zip(layer, max_string_lengths))))),
                layers)), sep="\n" + '─' * (np.sum(max_string_lengths) + len(titles) - 1) + "\n")
