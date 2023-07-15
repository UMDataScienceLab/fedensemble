# Importing libraries
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# Defining method to encode one hot labels
# arr is an array that may contain multiple words
def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    # the i-th input corresponds to the i-th output
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


# Defining method to make mini-batches for training
def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''

    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr) // batch_size_total

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n + seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y



# Defining a method to generate the next character
def predict(net, char, h=None, top_k=None):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
    '''

    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if (train_on_gpu):
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    p = F.softmax(out, dim=1).data
    if (train_on_gpu):
        p = p.cpu()  # move to cpu

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p / p.sum())

    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h


# Declaring a method to generate new text
def sample(net, size, prime='The', top_k=None):
    if (train_on_gpu):
        net.cuda()
    else:
        net.cpu()

    net.eval()  # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

def prepare_dataset_from_txt(filename):
    with open(filename, 'r') as f:
        text = f.read()

    # Showing the first 100 characters
    print(text[:100])

    # encoding the text and map each character to an integer and vice versa

    # We create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # Encode the text
    encoded = np.array([char2int[ch] for ch in text])
    return chars, encoded

if __name__ == "__main__":

    encoded = prepare_dataset_from_txt('data/shakespeare.txt')

    # Define and print the net
    n_hidden = 512
    n_layers = 2
    drop_prob = 0.5
    lr = 0.001

    # Check if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        print('Training on GPU!')
    else:
        print('No GPU available, training on CPU; consider making n_epochs very small.')

    net = CharRNN(chars, n_hidden, n_layers, drop_prob, lr, train_on_gpu)
    print(net)

    # Declaring the hyperparameters
    batch_size = 128
    seq_length = 100
    n_epochs = 20  # start smaller if you are just testing initial behavior

    # train the model
    train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=50)

    # Saving the model
    model_name = 'rnn_20_epoch.net'

    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'tokens': net.chars}

    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)
    # Showing the first 100 encoded characters
    encoded[:100]

    # Generating new text
    print(sample(net, 1000, prime='A', top_k=5))