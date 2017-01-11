import numpy as np
import collections
import os

def read_words(conf):
    words = []
    for file in os.listdir(conf.data_dir):
        with open(os.path.join(conf.data_dir, file), 'r') as f:
            for line in f.readlines():
                tokens = line.split()
                # NOTE Currently, only sentences with a fixed size are chosen
                # to account for fixed convolutional layer size.
                if len(tokens) == conf.context_size-2:
                    words.extend((['<pad>']*(conf.filter_h/2)) + ['<s>'] + tokens + ['</s>'])
    return words

def index_words(words, conf):
    word_counter = collections.Counter(words).most_common(conf.vocab_size-1)
    word_to_idx = {'<unk>': 0}
    idx_to_word = {0: '<unk>'}
    for i,_ in enumerate(word_counter):
        word_to_idx[_[0]] = i+1
        idx_to_word[i+1] = _[0]
    data = []
    for word in words:
        idx = word_to_idx.get(word)
        idx = idx if idx else word_to_idx['<unk>']
        data.append(idx)
    return np.array(data), word_to_idx, idx_to_word

def create_batches(data, conf):
    conf.num_batches = int(len(data) / (conf.batch_size * conf.context_size))
    data = data[:conf.num_batches * conf.batch_size * conf.context_size]
    xdata = data
    ydata = np.copy(data)

    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    x_batches = np.split(xdata.reshape(conf.batch_size, -1), conf.num_batches, 1)
    y_batches = np.split(ydata.reshape(conf.batch_size, -1), conf.num_batches, 1)

    for i in xrange(conf.num_batches):
        x_batches[i] = x_batches[i][:,:-1]
        y_batches[i] = y_batches[i][:,:-1]
    return x_batches, y_batches, conf

def get_batch(x_batches, y_batches, batch_idx):
    x, y = x_batches[batch_idx], y_batches[batch_idx]
    batch_idx += 1
    if batch_idx >= len(x_batches):
        batch_idx = 0
    return x, y.reshape(-1,1), batch_idx


def prepare_data(conf):
	words = read_words(conf)
	data, word_to_idx, idx_to_word = index_words(words, conf)
	x_batches, y_batches, conf = create_batches(data, conf)

	del words
	del data

	return x_batches, y_batches
