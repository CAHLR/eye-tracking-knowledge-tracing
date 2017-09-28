from keras.layers import SimpleRNN
import numpy as np
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.legacy import interfaces
from keras.layers import Recurrent


# https://github.com/braingineer/ikelos/blob/master/ikelos/layers/cwrnn.py
# modified for keras 2.0
        # self.U is now self.recurrent_kernel
        # self.W is now self.kernel
        # self.b is now self.bias
class ClockworkRNN(SimpleRNN):
    '''
        Clockwork Recurrent Unit - Koutnik et al. 2014

        Clockwork RNN splits simple RNN neurons into groups of equal sizes.
        Each group is activated every specified period. As a result, fast
        groups capture short-term input features while slow groups capture
        long-term input features.

        References:
            A Clockwork RNN
                http://arxiv.org/abs/1402.3511
    '''
    def __init__(self, units, period_spec=[1], **kwargs):
        self.units = units
        assert units % len(period_spec) == 0, ("ClockworkRNN requires the units to be " +
                                                "a multiple of the number of periods; " +
                                                "units %% len(period_spec) failed.")
        self.period_spec = np.asarray(sorted(period_spec, reverse=True))

        super(ClockworkRNN, self).__init__(units, **kwargs)

    def build(self, input_shape):

        ### construct the clockwork structures
        ### basically: every n units the period changes;
        ### `period` is for flaggin this; `mask` is for enforcing it
        n = self.units // len(self.period_spec)
        mask = np.zeros((self.units, self.units), K.floatx())
        period = np.zeros((self.units,), np.int16)
        for i, t in enumerate(self.period_spec):
            mask[i*n:(i+1)*n, i*n:] = 1
            period[i*n:(i+1)*n] = t
        self.mask = K.variable(mask, name='clockword_mask')
        self.period = K.variable(period, dtype='int16', name='clockwork_period')

        super(ClockworkRNN, self).build(input_shape)

        # self.U is now self.recurrent_kernel
        # self.W is now self.kernel
        # self.b is now self.bias
        self.recurrent_kernel = self.recurrent_kernel * self.mask  ### old implementation did this at run time...

        ### simple rnn initializes the wrong size self.states
        ### we want to also keep the time step in the state.
        if self.stateful:
            self.reset_states()
        else:
            self.states = [None, None]



    def get_initial_states(self, x):
        initial_states = super(ClockworkRNN, self).get_initial_states(x)
        if self.go_backwards:
            input_length = self.input_spec[0].shape[1]
            initial_states[-1] = float(input_length)
        else:
            initial_states[-1] = K.variable(0.)
        return initial_states


    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')

        if self.go_backwards:
            initial_time = self.input_spec[0].shape[1]
        else:
            initial_time = 0.

        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1], initial_time)
        else:
            self.states = [K.zeros((input_shape[0], self.units)), K.variable(initial_time)]

    def get_constants(self, x, training = None):
        consts = super(ClockworkRNN, self).get_constants(x, training = None)
        consts.append(self.period)
        return consts

    def step(self, x, states):
        prev_output = states[0]
        time_step = states[1]
        B_U = states[2]
        B_W = states[3]
        period = states[4]

        # if self.consume_less == 'cpu':
        #     h = x
        # else:
        #     h = K.dot(x * B_W, self.kernel) + self.bias

        h = x

        output = self.activation(h + K.dot(prev_output * B_U, self.recurrent_kernel))

        if (K.backend() == 'tensorflow'):
            import tensorflow as tf

            output = tf.where(K.equal(tf.cast(time_step, tf.float32) % tf.cast(period, tf.float32), 0.), output, prev_output)
        else:
            output = K.switch(K.equal(time_step % period, 0.), output, prev_output)


        return output, [output, time_step+1]


    def get_config(self):
        config = {"periods": self.periods}
        base_config = super(CWRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


'''
#old implementation:

class ClockworkRNN(Recurrent):
    \'''
        Clockwork Recurrent Unit - Koutnik et al. 2014

        Clockwork RNN splits simple RNN neurons into groups of equal sizes.
        Each group is activated every specified period. As a result, fast
        groups capture short-term input features while slow groups capture
        long-term input features.

        References:
            A Clockwork RNN
                http://arxiv.org/abs/1402.3511
    \'''
    def __init__(self, output_dim, periods=[1],
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        assert output_dim % len(periods) == 0
        self.periods = np.asarray(sorted(periods, reverse=True))
        self.init = initializers.get(init)
        self.inner_init = initializers.get(inner_init)

        self.activation = activations.get(activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(ClockworkRNN, self).__init__(**kwargs)


    def build(self):
        input_dim = self.input_shape[2]
        n = self.output_dim // len(self.periods)
        self.mask = np.zeros((self.output_dim, self.output_dim), theano.config.floatX)
        self.period = np.zeros((self.output_dim,), 'i')
        for i, t in enumerate(self.periods):
            self.mask[i*n:(i+1)*n, i*n:] = 1
            self.period[i*n:(i+1)*n] = t
        self.input = T.tensor3()
        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.params = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, t, x_t,
              y_tm1,
              u, period):
        y = x_t + T.dot(y_tm1, u)
        y_t = T.switch(T.eq(t % period, 0), self.activation(y), y_tm1)
        return y_t

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle(1, 0, 2)
        x = T.dot(X, self.W) + self.b
        outputs, updates = theano.scan(
            self._step,
            sequences=[T.arange(x.shape[0]), x],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U * self.mask, self.period],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "periods": self.periods,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(ClockworkRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
'''
