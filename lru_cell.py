import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops, rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear


class LRUCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, initializer=None, forget_bias=1.0,
                 activation=math_ops.tanh, reuse=None):
        """Initialize the parameters for an LRU cell.
        Args:
          num_units: int, The number of units in the LRU cell
          initializer: (optional) The initializer to use for the weight matrices.
          forget_bias: (optional) float, default 1.0, The initial bias of the
            forget gate, used to reduce the scale of forgetting at the beginning
            of the training.
          activation: (optional) Activation function of the inner states.
            Default is `tf.tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        """
        super().__init__(_reuse=reuse)
        self._num_units = num_units
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse
        self._linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Run one step of LRU.
        Args:
            inputs: input Tensor, 2D, batch x input size.
            state: state Tensor, 2D, batch x num units.
        Returns:
            new_output: batch x num units, Tensor representing the output of the LRU
                after reading `inputs` when previous state was `state`. Identical to
                `new_state`.
            new_state: batch x num units, Tensor representing the state of the LRU
                after reading `inputs` when previous state was `state`.
        Raises:
            ValueError: If input size cannot be inferred from inputs via
                static shape inference.
        """
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError('Could not infer input size from inputs.get_shape()[-1]')

        with vs.variable_scope(vs.get_variable_scope(), initializer=self._initializer):
            cell_inputs = array_ops.concat([inputs, state], axis=1)

            if self._linear is None:
                self._linear = _Linear(cell_inputs, self._num_units, build_bias=True)

            delta_act = self._linear(cell_inputs)
            delta = math_ops.maximum(0.0, delta_act - self._forget_bias)

            new_state = state + delta
            new_output = new_state

        return new_output, new_state


'''
class AddressableMemoryArray(LRUCell):
    @property
    def state_size(self):
        return self._num_units**2

    def call(self, inputs, state):
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError('Could not infer input size from inputs.get_shape()[-1]')

        with vs.variable_scope(vs.get_variable_scope(), initializer=self._initializer):
            cell_inputs = array_ops.concat([inputs, state], 1)

            if self._state is None:
                pass

            if self._linear is None:
                self._linear = _Linear(cell_inputs, 2 * self._num_units, build_bias=True)

            rnn_matrix = self._linear(cell_inputs)
            [w_act, r_act] = array_ops.split(axis=1, num_or_size_splits=2, value=rnn_matrix)

            w = math_ops.sigmoid(w_act + self._forget_bias)
            r = math_ops.softmax(r_act)

            new_state = g * state + (1.0 - g) * c
            new_output = new_state

        return new_output, new_state
'''
