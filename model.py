"""
Volatility prediction model and scenarios.
"""

import arch
import collections
import db
import itertools
import logging
import math
import numpy
import portfolio
import random
import tensorflow
import tensorflow.models.rnn
import util
import sys

class Pattern:
    """A training pattern."""

    def __init__(self, past, future, features = None):
        """Create a training pattern.

        Parameters:
        past -- past feature vectors as a tensor of shape [P, V]
            where P is past days and V is the vectors/day
        future -- future feature vectors as a tensor of [F, V]
            where F is future days and V is the vectors/day
        features -- a sequence of feature names to use
            where None means use all features
        """

        # calculate training input from past features
        past_subfeatures = [[self._subfeatures(vector, features)
            for vector in vectors]
                for vectors in past]
        self._input = numpy.array(
            [list(util.flatten(vectors)) for vectors in past_subfeatures])

        # calculate training output from future volatility
        future_returns = numpy.log1p(
            [[vector.ret for vector in vectors] for vectors in future])
        self._output = numpy.std(future_returns, axis = 0, ddof = 1)\
            * numpy.sqrt(252)

        # calculate past returns for forecasts
        self._past_returns = numpy.log1p(
            [[vector.ret for vector in vectors] for vectors in past])

    @property
    def input(self):
        """Return the training input as a tensor of shape [P, V*K]
            where P is past days, V is vectors/day and K is features/vector.
        """

        return self._input

    @property
    def output(self):
        """Return the training output as a tensor of shape [V]
            where V is vectors/day."""

        return self._output

    def simple_forecast(self):
        """Return historical volatility as a tensor of shape [V]
            where V is vectors/day."""

        try:
            return self._simple_forecast
        except AttributeError:
            self._simple_forecast = numpy.std(
                self._past_returns, axis = 0, ddof = 1) * numpy.sqrt(252)
            return self._simple_forecast

    def ewma_forecast(self, decay = 0.94):
        """Return EWMA volatility as a tensor of shape [V]
            where V is vectors/day.

        Parameters:
            decay -- lambda decay factor
        """

        try:
            return self._ewma_forecast
        except AttributeError:
            variances = self._past_returns[0] ** 2
            for returns in self._past_returns[1:]:
                variances = decay * variances + (1 - decay) * returns ** 2
            self._ewma_forecast = numpy.sqrt(variances) * numpy.sqrt(252)
            return self._ewma_forecast

    def garch_forecast(self):
        """Return GARCH(1,1) volatility forecast as a tensor of shape[V]
            where V is vectors/day.
        """

        # short circuit for speeding up testing
        #return numpy.array([0] * len(self._past_returns[0]))

        try:
            return self._garch_forecast
        except AttributeError:
            logging.info("runing garch forecast")
            variances = []
            for values in numpy.transpose(self._past_returns):
                garch = arch.arch_model(values)
                results = garch.fit(disp = "off")
                omega = results.params["omega"]
                alpha1 = results.params["alpha[1]"]
                beta1 = results.params["beta[1]"]
                forecast = omega\
                    + alpha1 * results.resid[-1] ** 2\
                    + beta1 * results.conditional_volatility[-1] ** 2
                if numpy.isnan(forecast):
                    forecast = 0
                forecast = max(forecast, 0) # ignore negative variance
                forecast = min(forecast, 0.04) # limit to trading halt trigger
                variances.append(forecast)
            self._garch_forecast = numpy.sqrt(variances) * numpy.sqrt(252)
            return self._garch_forecast

    def _subfeatures(self, vector, features):
        """Select named features from a feature vector.

        Parameters:
        vector -- a fecture vector
        features -- a sequence of feature names
            where None means select all features
        """

        return [getattr(vector, feature) for feature in features]\
            if features else vector

class Patterns(list):
    """A list of training patterns."""

    def __init__(self, timeseries,
            history = 30, forecast = 10, stride = 1, features = None):
        """Create a list of training patterns.

        Parameters:
        timeseries -- timeseries of feature vector tensors of of shape [V]
            where V is the vectors/day
        history -- days to treat as past in each data window
        forecast -- days to treat as future in each data window
        stride -- days move sliding window 
        features -- a sequence of feature names to use
            where None means use all features
        """

        super().__init__()

        # strip date column
        vectors = map(lambda x: x[1:], timeseries)

        # apply a sliding window to rows of features
        for window in util.slidingwindow(vectors, history + forecast, stride):
            self.append(Pattern(window[:history], window[history:], features))

class Model:
    """The forecasting model."""

    def __init__(self, batchsize, steps, inputs, outputs, hidden, layers):
        """Create a forecasting model.

        Parameters:
        batchsize -- size of learning batch
        steps -- number of input time steps
        inputs -- number of inputs
        outputs -- number of outputs
        hidden -- number of hidden nodes
        layers -- number of layers
        """

        self._batchsize = batchsize
        self._training_batches = 0

        # create input pattern placeholders
        self._x = tensorflow.placeholder(tensorflow.float32,
            [None, steps, inputs])
        self._y = tensorflow.placeholder(tensorflow.float32,
            [None, outputs])

        # transpose input for rnn core
        # this technique adopted from aymericdamien
        # https://github.com/aymericdamien/TensorFlow-Examples/
        x = tensorflow.transpose(self._x, [1, 0, 2])

        # create weights and biases for inputs
        input_weights = tensorflow.Variable(
            tensorflow.random_normal([inputs, hidden]))
        input_biases = tensorflow.Variable(
            tensorflow.random_normal([hidden]))

        # apply input linear transform
        # this technique adopted from aymericdamien
        # https://github.com/aymericdamien/TensorFlow-Examples/
        x = tensorflow.reshape(x, [-1, inputs])
        x = tensorflow.matmul(x, input_weights) + input_biases

        # create a multilayered stack ef basic LSTM cells
        cell = tensorflow.models.rnn.rnn_cell.BasicLSTMCell(hidden)
        stack = tensorflow.models.rnn.rnn_cell.MultiRNNCell([cell] * layers)

        # create RNN network
        x = tensorflow.split(0, steps, x)
        initial = stack.zero_state(batchsize, tensorflow.float32)
        out, states = tensorflow.models.rnn.rnn.rnn(stack, x,
            initial_state = initial)

        # create weights and biases for outputs from the LSTM stack
        output_weights = tensorflow.Variable(
            tensorflow.random_normal([hidden, outputs]))
        output_biases = tensorflow.Variable(
            tensorflow.random_normal([outputs]))

        # apply output linear transform
        # this technique adopted from aymericdamien
        # https://github.com/aymericdamien/TensorFlow-Examples/
        self._output = tensorflow.matmul(out[-1], output_weights)\
            + output_biases

        # define loss function and track summary
        self._loss = tensorflow.reduce_mean(
            tensorflow.square(self._output - self._y))
        tensorflow.scalar_summary('loss', self._loss)

        # define optimizer
        self._optimizer = tensorflow.train.AdamOptimizer(
            learning_rate = 0.001).minimize(self._loss)

        # merge summary nodes
        self._merged = tensorflow.merge_all_summaries()

    def train(self, session, patterns, writer = None):
        """Train the model.

        Parameters:
        session -- tensorflow session
        patterns -- training patterns
        writer -- tensorflow summary writer
        """

        # train with each batch
        for index, batch in zip(
                itertools.count(0), util.batches(patterns, self._batchsize)):
            logging.debug("batch {}".format(index))

            # run a training step
            x = [pattern.input for pattern in batch]
            y = [pattern.output for pattern in batch]
            summary, _ = session.run([self._merged, self._optimizer],
                feed_dict = {self._x: x, self._y: y})

            # write summary data
            if writer:
                writer.add_summary(summary, self._training_batches)
            self._training_batches += 1

    def test(self, session, patterns):
        """Test the model.

        Parameters:
        session -- tensorflow session
        patterns -- training patterns
        """

        # test with each batch
        for index, batch in zip(
                itertools.count(0), util.batches(patterns, self._batchsize)):

            # run a forecast step
            x = [pattern.input for pattern in batch]
            y = [pattern.output for pattern in batch]
            model_forecast = session.run(self._output,
                feed_dict = {self._x: x, self._y: y})

            # calculate simple, ewma and garch historical forecasts
            simple_forecast = [pattern.simple_forecast() for pattern in batch]
            ewma_forecast = [pattern.ewma_forecast() for pattern in batch]
            garch_forecast = [pattern.garch_forecast() for pattern in batch]

            # create result vector
            results = [index]
            for forecast in (model_forecast,
                    simple_forecast, ewma_forecast, garch_forecast):
                # add RMS and max error to results
                results.extend([self._rms_error(forecast, y),
                    self._max_error(forecast, y)])
            yield results

    def _rms_error(self, forecast, actual):
        """Return RMS error.

        Parameters:
        forecast -- forecast tensor
        actual -- actual tensor
        """

        return numpy.sqrt(numpy.mean(numpy.subtract(forecast, actual) ** 2))

    def _max_error(self, forecast, actual):
        """Return maximum error.

        Parameters:
        forecast -- forecast tensor
        actual -- actual tensor
        """

        return numpy.max(numpy.abs(numpy.subtract(forecast, actual)))

class Scenario:
    """Model scenario."""

    def __init__(self, dbpath, portfolio, start = None, end = None,
            history = 30, forecast = 10, stride = 1, features = None,
            batchsize = 1, hidden = 0, layers = 1, epochs = 1):
        """Create a model scenario.

        Parameters:
        dbpath -- database path name
        portfolio -- list of stock symbols
        start -- start date (None = start of time)
        end -- end date (None = end of time)
        history -- days to treat as past in each data window
        forecast -- days to treat as future in each data window
        stride -- days move sliding window 
        features -- a sequence of feature names to use (None = all)
        batchsize -- size of learning batch
        hidden -- number of hidden nodes (0 = model input nodes)
        layers -- number of layers
        epochs -- number of training epochs
        """

        self._dbpath = dbpath
        self._portfolio = portfolio
        self._start = start
        self._end = end
        self._history = history
        self._forecast = forecast
        self._stride = stride
        self._features = features
        self._batchsize = batchsize
        self._hidden = hidden
        self._batchsize = batchsize
        self._layers = layers
        self._epochs = epochs

    def run(self):
        """Run the scenario."""

        # read data from database
        logging.info("reading data")
        stocks = db.Stocks(self._dbpath)
        permnos = stocks.permnos(self._portfolio)
        multiseries = stocks.multiseries(permnos,
            start = self._start, end = self._end)

        # build training patterns
        logging.info("generating patterns")
        patterns = Patterns(multiseries,
            self._history, self._forecast, self._stride, self._features)
        steps, inputs = patterns[0].input.shape
        outputs, = patterns[0].output.shape

        # split patterns into random training and testing sets
        logging.info("selecting training and testing sets")
        random.shuffle(patterns)
        length = int(len(patterns) * 0.9)
        training_patterns = patterns[:length]
        testing_patterns = patterns[length:]

        # build the model
        logging.info("constructing model")
        model = Model(self._batchsize, steps, inputs, outputs,
            self._hidden or inputs, self._layers) 

        # print CSV header
        print(",".join(["Epoch", "Batch",
            "Model_RMS_Error", "Model_Max_Error",
            "Simple_RMS_Error", "Simple_Max_Error",
            "EWMA_RMS_Error", "EWMA_Max_Error",
            "GARCH_RMS_Error", "GARCH_Max_Error"]))

        with tensorflow.Session() as session:
            # initialize
            writer = tensorflow.train.SummaryWriter("summary", session.graph)
            session.run(tensorflow.initialize_all_variables())

            # run each epoch
            for epoch in range(self._epochs):
                # train with batches in random order
                logging.info("epoch {}".format(epoch))
                random.shuffle(training_patterns)
                model.train(session, training_patterns, writer)

                # test model
                results = list(model.test(session, testing_patterns))

                # print epoch summary line
                xresults = numpy.transpose(results)
                summary = [epoch, "", 
                    numpy.mean(xresults[1]), numpy.max(xresults[2]),
                    numpy.mean(xresults[3]), numpy.max(xresults[4]),
                    numpy.mean(xresults[5]), numpy.max(xresults[6]),
                    numpy.mean(xresults[7]), numpy.max(xresults[8])]

                # print batch results
                print(",".join(map(str, summary)))
                for row in results:
                    print(",".join(map(str, [epoch] + row)))

class Scenarios:
    """Prebuilt model scenarios."""

    alpha = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["ret"],
        batchsize = 20,
        hidden = 0,
        layers = 1,
        epochs = 15)

    beta = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol"],
        batchsize = 20,
        hidden = 0,
        layers = 1,
        epochs = 15)

    gamma = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol"],
        batchsize = 20,
        hidden = 0,
        layers = 2,
        epochs = 15)

    delta = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol"],
        batchsize = 20,
        hidden = 0,
        layers = 4,
        epochs = 15)

    epsilon = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx"],
        batchsize = 20,
        hidden = 0,
        layers = 4,
        epochs = 15)

    zeta = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx"],
        batchsize = 20,
        hidden = 0,
        layers = 6, # 8 is too large for gpu
        epochs = 15)

    eta = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx"],
        batchsize = 10,
        hidden = 0,
        layers = 4,
        epochs = 15)

    theta = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx"],
        batchsize = 40,
        hidden = 0,
        layers = 4,
        epochs = 15)

    iota = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx"],
        batchsize = 20,
        hidden = len(portfolio.sp100) * 8, # more inputs, 10 too big for gpu
        layers = 4,
        epochs = 15)

    kappa = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx", "trend"],
        batchsize = 20,
        hidden = 0,
        layers = 4,
        epochs = 15)

    mu = Scenario(
        "stocks.db",
        portfolio.nasdaq100,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx"],
        batchsize = 20,
        hidden = 0,
        layers = 4,
        epochs = 15)

    nu = Scenario( # too large for gpu
        "stocks.db",
        portfolio.sp500,
        start = "2005-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx"],
        batchsize = 20,
        hidden = 0,
        layers = 4,
        epochs = 15)

    xi = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2014-10-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx"],
        batchsize = 5,
        hidden = 0,
        layers = 4,
        epochs = 15)

    omnicron = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "2005-10-01",
        end = None,
        history = 60,
        forecast = 20,
        stride = 1,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx"],
        batchsize = 20,
        hidden = 0,
        layers = 4,
        epochs = 15)

    pi = Scenario(
        "stocks.db",
        portfolio.sp100,
        start = "1990-01-01",
        end = None,
        history = 30,
        forecast = 10,
        stride = 5,
        features = ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx"],
        batchsize = 20,
        hidden = 0,
        layers = 4,
        epochs = 15)

    default = xi

def main(argv):
    """Main entry point."""

    logging.basicConfig(level = logging.INFO)

    if len(argv) < 2:
        # run default scenario
        getattr(Scenarios, "default").run()
    else:
        # run requested scenarios
        for name in argv[1:]:
            getattr(Scenarios, name).run()

if __name__ == "__main__":
    main(sys.argv)
