import numpy as np
import json
from time import time

###################################################################
#                           ACTIVATIONS
###################################################################

class Activation:
    def __init__(self, function):
        self.function_name = function
        if function == "softmax": self.function = Softmax()
        if function == "sigmoid": self.function = Sigmoid()
        if function == "linear": self.function = Linear()
        if function == "relu": self.function = ReLU()
        if function == "sigmoid_derivative": self.function = SigmoidDerivative()

    def __apply__(self, x):
        return self.function.__apply__(x)

class Softmax:
    def __apply__(self, x):
        x = np.array(x)
        x = x - max(x)
        x_exp = np.exp(x)
        result = x_exp/np.sum(x_exp)
        return result

class Sigmoid:
    def __apply__(self, x):
        return 1/(1+np.exp(-x))

class Linear:
    def __apply__(self, x):
        return x

class ReLU:
    def __apply__(self, x):
        return np.maximum(0, x)

class SigmoidDerivative:
    def __apply__(self, x):
        f = 1/(1 + np.exp( np.negative(x) ))
        df = f * (1 - f)
        return df

###################################################################
#                           LOSSES
###################################################################


class Loss:
    def __init__(self, function):
        if function == "categorical_crossentropy": self.function = CategoricalCrossEntropy()
        if function == "mse": self.function = MeanSquaredError()

    def __apply__(self, p, y):
        return self.function.__apply__(p, y)

class CategoricalCrossEntropy:
    # L(y,x) = −∑(c=1 -> M) y_o,c * log(po,c)
    def __apply__(self, p, y):
        return -np.sum( y * np.log(p+1e-9) )

class MeanSquaredError:
    def __apply__(self, p, y):
        return np.sum( (p-y)**2 )/len(p)


###################################################################
#                           LAYERS
###################################################################


class Layer(object):
    def __init__(self, n_neurons, activation="linear"):
        self.n_neurons = n_neurons
        self.activation = activation


###################################################################
#                        NEURAL NETWORK
###################################################################


class NeuralNet(object):
    def __init__(self):
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.w_matrices = []
        self.bias_vectors = []
        self.activations = []
        self.loss = None
        self.backprop_activation = Activation("sigmoid_derivative")

    def add_input_layer(self, n_neurons):
        self.input_layer = Layer(n_neurons)

    def add_hidden_layer(self, n_neurons, activation="linear"):
        self.hidden_layers.append( Layer(n_neurons, activation) )

    def add_output_layer(self, n_neurons, activation=None):
        self.output_layer = Layer(n_neurons, activation=activation)

    def compile(self, loss):
        matrix_shape = (self.input_layer.n_neurons, self.hidden_layers[0].n_neurons)
        self.w_matrices.append( np.random.rand( matrix_shape[0], matrix_shape[1] ) )
        self.bias_vectors.append( np.random.rand(self.hidden_layers[0].n_neurons) )
        self.activations.append( Activation(self.hidden_layers[0].activation) )
        for i, layer in enumerate(self.hidden_layers[:-1]):
            matrix_shape = (layer.n_neurons, self.hidden_layers[i+1].n_neurons)
            matrix = np.random.rand( matrix_shape[0], matrix_shape[1] )
            self.w_matrices.append( matrix )
            self.bias_vectors.append( np.random.rand(self.hidden_layers[i+1].n_neurons) )
            self.activations.append( Activation(self.hidden_layers[i+1].activation) )
        matrix_shape = (self.hidden_layers[-1].n_neurons, self.output_layer.n_neurons)
        matrix = np.random.rand( matrix_shape[0], matrix_shape[1] )
        self.w_matrices.append( matrix )
        self.bias_vectors.append( np.random.rand(self.output_layer.n_neurons) )
        self.activations.append( Activation(self.output_layer.activation) )

        total_params = 0
        for i, w_matrix in enumerate(self.w_matrices):
            print("=====================")
            params = w_matrix.shape[0]*w_matrix.shape[1] + len(self.bias_vectors[i])
            total_params += params
            print(f"Layer {i}:\n\tshape: {w_matrix.shape}\n\tn_params: {params}")
        print("================================")
        print(f"Network has a total of {total_params} parameters.")
        self.loss = Loss(loss)

    def feed_forward(self, x):
        intermediate_outputs = [x]
        for w_matrix, bias, activation in zip(self.w_matrices, self.bias_vectors, self.activations):
            x = np.dot(x, w_matrix) + bias
            x = activation.__apply__(x)
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
            dah_dzh = self.backprop_activation.__apply__(i_outs[-i-1])
            dzh_dwh = np.array(i_outs[-i-2]).reshape(1,len(i_outs[-i-2]))
            dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
            dcost_bh = dcost_dah * dah_dzh

            self.w_matrices[-i-1] -= lr * dcost_wh
            self.bias_vectors[-i-1] -= lr * dcost_bh.reshape(dcost_bh.shape[1])
            dcost_dzo = dcost_dah

    def train(self, X, Y, epochs=100):
        X = np.array(X)
        Y = np.array(Y)
        n_samples = len(X)
        y_loss = Y.reshape(n_samples)

        for i in range(epochs):
            p_loss = []
            print(f"===== epoch {i+1}/{epochs} =====")
            for x, y in zip(X,Y):
                t = time()
                p, i_outs = self.feed_forward(x)
                p_loss.append(p[0])
                self.back_propagate(x, y, p, i_outs)
            loss = self.loss.__apply__(p_loss, y_loss)
            print(f"loss = {loss}")
            print(f"epoch {i} took {round(time()-t, 4)} seconds.")

    def save(self, filename):
        np.save(f'{filename}_weights.npy', self.w_matrices)
        np.save(f'{filename}_bias.npy', self.bias_vectors)
        with open(f"{filename}_activations.json", "w", encoding="utf8") as f: 
            activations = [activation.function_name for activation in self.activations]
            json.dump(activations, f, indent=4)

    def load(self, filename):
        self.w_matrices = np.load(f'{filename}_weights.npy')
        self.bias_vectors = np.load(f'{filename}_bias.npy')
        with open(f"{filename}_activations.json", "r", encoding="utf8") as f: activations = json.load(f)
        self.activations = list([ Activation(function) for function in activations ])


if __name__ == "__main__":
    np.random.seed(22)

    # num_samples = 100

    # nn = NeuralNet()
    # nn.add_input_layer(1)
    # nn.add_hidden_layer(32, activation="relu")
    # nn.add_hidden_layer(16, activation="relu")
    # nn.add_output_layer(1, activation="linear")
    # nn.compile(loss="mse")
    # x = [ [np.random.randint(0,10)/10] for _ in range(num_samples) ]
    # y = [ [x_val[0]+1] for x_val in x ]
    # nn.train(x, y)
    # nn.save("models/sample_model")

    nn = NeuralNet()
    nn.load("models/sample_model")
    p, _ = nn.feed_forward([0.5])
    print(p)