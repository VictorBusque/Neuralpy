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
        return -np.sum( y * np.log(p) )


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
        intermediate_outputs = [x]
        for w_matrix, bias, activation in zip(self.w_matrices, self.bias_vectors, self.activations):
            x = np.dot(x, w_matrix) + bias
            x = Activation().apply_activation(x, activation)
            intermediate_outputs.append(x)
        return x, np.array(intermediate_outputs)

    def back_propagate(self, x, y, p, i_outs, lr=10e-4):
        ### Calculate changes in output 
        p = np.array(p).reshape((1,len(p)))
        y = np.array(y).reshape((1,len(y)))
        i_outs = np.array(i_outs)
        dcost_dzo = p - y
        dzo_dwo = np.array(i_outs[-2]).reshape( (1, len(i_outs[-2])) )

        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)
        dcost_bo = dcost_dzo

        self.w_matrices[-1] -= lr * dcost_wo
        self.bias_vectors[-1] -= lr * dcost_bo.reshape(dcost_bo.shape[1])

        ### Rest of the layers
        # for i in reversed( range(1, len(self.w_matrices)) ):
        for i in range(1, len(self.w_matrices)-1 ):
            dzo_dah = self.w_matrices[-i]
            dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
            dah_dzh = Activation().sigmoid_derivative(i_outs[-i-1])
            dzh_dwh = np.array(i_outs[-i-2]).reshape(1,len(i_outs[-i-2]))
            dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
            dcost_bh = dcost_dah * dah_dzh

            self.w_matrices[-i-1] -= lr * dcost_wh
            self.bias_vectors[-i-1] -= lr * dcost_bh.reshape(dcost_bh.shape[1])

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
    np.random.seed(22)
    nn = NeuralNet()
    nn.add_input_layer(10)
    nn.add_hidden_layer(12)
    nn.add_hidden_layer(6)
    nn.add_output_layer(8, activation="softmax")
    nn.compile()
    x = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9]
    y = [0,0,0,0,0,0,0,1]
    nn.train(x,y)