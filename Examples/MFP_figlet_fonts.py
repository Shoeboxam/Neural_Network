import numpy as np
import pyfiglet

from MFP import *

# Two learning modes:
#   1. Autoencoder for font denoising
#   2. Classifier for character identification

"""     #######
        #       #  ####  #      ###### #####
        #       # #    # #      #        #
        #####   # #      #      #####    #
        #       # #  ### #      #        #
        #       # #    # #      #        #
        #       #  ####  ###### ######   #      """

# http://www.figlet.org/


class FigletFonts:

    def __init__(self, font='banner', noise=0., autoencoder=False, ascii_vals=None):
        self.noise = noise
        self.autoencoder = autoencoder

        self.stimuli = []
        self.character_set = set()

        self.max_width = 0

        if not ascii_vals:
            ascii_vals = [i for i in range(32, 127)]
        self.ascii_vals = ascii_vals

        # Initial pass to determine letter width
        for letter in ascii_vals:
            raw = pyfiglet.figlet_format(chr(letter), font=font)

            width = len(raw.splitlines()[0])
            self.max_width = self.max_width if self.max_width > width else width

            numerical = []
            for line in raw.splitlines():
                numerical.append([ord(cha) for cha in line])

            processed = np.array(numerical)
            self.character_set.update(np.unique(processed))
            self.stimuli.append(np.array(processed))

        self.character_set = {idx: char for idx, char in enumerate(self.character_set)}

        # Convert stimuli to padded categorical flattened arrays
        for idx in range(len(self.stimuli)):
            for classidx, char_class in self.character_set.items():
                self.stimuli[idx][np.where(self.stimuli[idx] == char_class)] = classidx

            self.stimuli[idx] = np.pad(self.stimuli[idx], ((0, 0), (0, self.max_width - self.stimuli[idx].shape[1])), 'constant')
            self.stimuli[idx] = self.stimuli[idx].flatten()

        self.stimuli = np.array(self.stimuli)
        self.expected = np.eye(len(ascii_vals))

    def sample(self, quantity=1):
        x = np.random.randint(len(self.ascii_vals), size=quantity)

        if self.noise:
            generated_noise = np.random.normal(0., scale=len(self.character_set) // 2, size=self.stimuli[x].shape).astype(int)
            mask = np.random.binomial(1, self.noise, size=self.stimuli[x].shape)
            stimuli = np.mod(self.stimuli[x] + generated_noise * mask, len(self.character_set))
        else:
            stimuli = self.stimuli[x]

        if self.autoencoder:
            return [stimuli.T, self.stimuli[x].T]
        else:
            return [stimuli.T, self.expected[x].T]

    def survey(self, quantity=None):

        if not quantity:
            quantity = len(self.ascii_vals)
        x = np.linspace(0, len(self.ascii_vals) - 1, quantity).astype(int)  # Size changes error granularity

        if self.autoencoder:
            return [self.stimuli[x].T, self.stimuli[x].T]
        else:
            return [self.stimuli[x].T, self.expected[x].T]

    def size_input(self):
        return np.size(self.stimuli[0])

    def size_output(self):
        if self.autoencoder:
            return np.size(self.stimuli[0])
        else:
            return np.size(self.expected[0])

    def plot(self, plt, predict):
        # Do not attempt to plot an image
        pass

    def error(self, expect, predict):
        if self.autoencoder:
            x = np.random.randint(0, len(self.ascii_vals))
            print(self.reformat(predict[:, x]))
            print(self.reformat(expect[:, x]))
            return np.linalg.norm(expect - predict)

        predict_id = np.argmax(predict, axis=0)
        expect_id = np.argmax(expect, axis=0)

        return int((1.0 - np.mean((predict_id == expect_id).astype(float))) * 100)

    def reformat(self, data):
        data = np.round(np.clip(data.reshape((-1, self.max_width)), 0, len(self.character_set) - 1)).astype(int)
        ascii_valued = np.zeros(data.shape)
        for classidx, char_class in self.character_set.items():
            ascii_valued[np.where(data == classidx)] = char_class

        output = ''
        for line in ascii_valued:
            output += ''.join([chr(int(round(symbol))) for symbol in line]) + '\n'
        return output


ascii_vals = [i for i in range(33, 126)]
environment = FigletFonts('banner3', noise=0.0, autoencoder=True, ascii_vals=ascii_vals)

# ~~~ Create the network ~~~
network_params = {
    # Shape of network
    "units": [environment.size_input(), 150, environment.size_output()],

    # Basis function(s) from network's Function.py
    "basis": basis_bent,
    "basis_final": basis_logistic,

    # Distribution to use for weight initialization
    "distribute": dist_normal
    }

network = MFP(**network_params)

# ~~~ Train the network ~~~
optimizer_params = {
    # Source of stimuli
    "batch_size": 5,

    "cost": cost_cross_entropy,
    "normalize": False,

    # Learning rate
    "learn_step": 0.01,
    "learn_anneal": anneal_fixed,

    # "batch_norm_step": 0.0001,
    "batch_norm_decay": 0.0,

    "epsilon": 0.0,           # error allowance
    "iteration_limit": None,  # limit on number of iterations to run

    "debug_frequency": 20,
    "debug": True,
    "graph": True
    }

Adagrad(network, environment, **optimizer_params).minimize()

# ~~~ Test the network ~~~
[stimuli, expectation] = environment.survey()
print(network.predict(stimuli))
