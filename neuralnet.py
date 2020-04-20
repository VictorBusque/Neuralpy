import numpy as np


class Activation:
    def softmax(self, x):
        sum_exp = sum([np.exp(i) for i in x])
        return [ np.exp(x_i)/sum_exp for x_i in x ]

    def linear(self, x):
        return x

    def sigmoid_derivative(self, x):
        f = 1/(1 + np.exp( np.negative(x) ))
        df = f * (1 - f)
        return df

    def apply_activation(self, x, activation):
        if activation == "linear": return self.linear(x)
        elif activation == "softmax": return self.softmax(x)

class Loss:
    # L(y,x) = −∑(c=1 -> M) y_o,c * log(po,c)
    def categorical_crossentropy(self, p, y):
        M = len(y)
        CE = 0
        for c in range(M):
            CE += y[c] * np.log(p[c])
        return -CE


class Layer(object):
    def __init__(self, n_neurons, activation="linear"):
        self.n_neurons = n_neurons
        self.activation = activation


class NeuralNet(object):
    def __init__(self):
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.w_matrices = []
        self.bias_vectors = []
        self.activations = []

    def add_input_layer(self, n_neurons):
        self.input_layer = Layer(n_neurons)

    def add_hidden_layer(self, n_neurons, activation="linear"):
        self.hidden_layers.append( Layer(n_neurons, activation) )

    def add_output_layer(self, n_neurons, activation=None):
        self.output_layer = Layer(n_neurons, activation=activation)

    def compile(self):
        matrix_shape = (self.input_layer.n_neurons, self.hidden_layers[0].n_neurons)
        self.w_matrices.append( np.random.rand( matrix_shape[0], matrix_shape[1] ) )
        self.bias_vectors.append( np.random.rand(self.hidden_layers[0].n_neurons) )
        self.activations.append(self.hidden_layers[0].activation)
        for i, layer in enumerate(self.hidden_layers[:-1]):
            matrix_shape = (layer.n_neurons, self.hidden_layers[i+1].n_neurons)
            matrix = np.random.rand( matrix_shape[0], matrix_shape[1] )
            self.w_matrices.append( matrix )
            self.bias_vectors.append( np.random.rand(self.hidden_layers[i+1].n_neurons) )
            self.activations.append(self.hidden_layers[i+1].activation)
        matrix_shape = (self.hidden_layers[-1].n_neurons, self.output_layer.n_neurons)
        matrix = np.random.rand( matrix_shape[0], matrix_shape[1] )
        self.w_matrices.append( matrix )
        self.bias_vectors.append( np.random.rand(self.output_layer.n_neurons) )
        self.activations.append(self.output_layer.activation)

    def feed_forward(self, x):
        intermediate_outputs = []
        for w_matrix, bias, activation in zip(self.w_matrices, self.bias_vectors, self.activations):
            x = np.dot(x, w_matrix) + bias
            x = Activation().apply_activation(x, activation)
            intermediate_outputs.append(x)
        return x, np.array(intermediate_outputs)

    def back_propagate(self, x, y, p, i_outs):
        error = y-p
        delta = error*Activation().sigmoid_prime(p)
        for i in range(len(self.w_matrices)-1):
            ii = len(self.w_matrices)-1-i
            i_out = np.array(i_outs[ii])
            self.w_matrices[ii] += np.dot(i_out.transpose(), delta)

            error = np.dot(delta, self.w_matrices[ii].transpose())
            delta = error*Activation().sigmoid_prime(np.array(i_outs[ii-1]))


    def train(self, x, y, epochs=100):
        x = np.array(x)
        y = np.array(y)
        for i in range(epochs):
            print(f"===== epoch {i} =====")
            p, i_outs = self.feed_forward(x)
            print("Prediction: {}".format(p))
            loss = Loss().categorical_crossentropy(p, y)
            print("Loss: {}".format(loss))
            self.back_propagate(x, y, p, i_outs)


if __name__ == "__main__":
    nn = NeuralNet()
    nn.add_input_layer(10)
    nn.add_hidden_layer(25)
    nn.add_hidden_layer(12)
    nn.add_output_layer(8, activation="softmax")
    nn.compile()
    x = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9]
    y = [0,0,1,0,0,0,0,0]
    nn.train(x,y)