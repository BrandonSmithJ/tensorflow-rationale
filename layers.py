import numpy as np
import tensorflow as tf 
import gzip

from utils_ import create_variables


class EmbeddingLayer(object):

    def __init__(self, path):
        vecs = []
        stop = ['<pad>', '<unk>']

        stop_idx = {}
        word_ids = {}
        id_words = {}
        with gzip.open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.decode('utf-8').split()
                    word = parts[0]
                    vals = np.array([ float(x) for x in parts[1:] ])
                    vecs.append(vals)
                    word_ids[word] = len(vecs)
                    id_words[len(vecs)] = word 

        for word in stop:
            if 'pad' in word:
                vec = np.zeros(vecs[-1].shape)
                stop_idx[word] = 0
                vecs = [vec] + vecs

            else:
                vec = np.mean(vecs, axis=0)
                stop_idx[word] = len(vecs)
                vecs.append(vec)

        vecs = np.array(vecs)
        self.embeddings = tf.get_variable('Embeddings', 
                                        shape      = vecs.shape, 
                                        initializer= tf.constant_initializer(vecs),
                                        trainable  = False)
        for word in stop_idx:
            word_ids[word] = stop_idx[word]
            id_words[stop_idx[word]] = word 

        self.word_ids = word_ids
        self.id_words = id_words
        self.pad_id = stop_idx['<pad>']
        self.unk_id = stop_idx['<unk>']
        self.shape  = vecs.shape

    def forward(self, ids):
        return tf.nn.embedding_lookup(self.embeddings, ids)

    def words_to_ids(self, words):
        return [self.word_ids[word] if word in self.word_ids else self.unk_id for word in words]

    def ids_to_words(self, ids):
        return [self.id_words[id_] for id_ in ids]


class Layer(object):

    def __init__(self, in_size, out_size, activation='tanh', name='', **kwargs):
        self.in_size, self.out_size = in_size, out_size
        self.W, self.b  = create_variables([in_size, out_size], name)
        self.activation = tf.nn.softmax#getattr(tf, activation) 

    def forward(self, x):    
        vals  = tf.nn.xw_plus_b(x, self.W, self.b, name='Wx_b')
        return self.activation(vals)

class Layer2(object):

    def __init__(self, in_size, out_size, activation='tanh', name='', **kwargs):
        self.in_size, self.out_size = in_size, out_size
        self.W, self.b  = create_variables([in_size, out_size], name)
        self.activation = getattr(tf, activation) 

    def forward(self, x):    
        vals = tf.map_fn(lambda z: tf.nn.xw_plus_b(z, self.W, self.b), x, dtype=tf.float32)
        return self.activation(vals)


class LSTM(object):
    def __init__(self, depth=1, batch_size=100, name='', **kwargs):
        self.in_size  = kwargs['in_size']
        self.out_size = kwargs['out_size']
        self.name     = name

        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.in_size)
        self.cell  = tf.nn.rnn_cell.MultiRNNCell([lstm] * depth)
        self.state = None
        self.batch = batch_size

    def forward(self, x, mask, length=None):        
        x.set_shape([None, self.batch, self.in_size])
        vals, self.state = tf.nn.dynamic_rnn(self.cell, x, 
            time_major     = True, 
            dtype          = tf.float32,
            sequence_length= length,
            # initial_state  = self.state,
        )
        return vals


class BiDirLSTM(object):
    def __init__(self, depth=1, batch_size=100, name='', **kwargs):
        self.in_size  = kwargs['in_size']
        self.out_size = kwargs['out_size']
        self.name     = name

        self.fcell = tf.nn.rnn_cell.BasicLSTMCell(self.in_size)
        self.bcell = tf.nn.rnn_cell.BasicLSTMCell(self.in_size)
        self.state = [None] * 2
        self.batch = batch_size

    def forward(self, x, length=None):        
        used   = tf.sign(tf.reduce_max(tf.abs(x), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=0)
        self.l = length
        vals, self.state = tf.nn.bidirectional_dynamic_rnn(self.fcell, self.bcell, x, 
            time_major     = True, 
            dtype          = tf.float32,
            sequence_length= tf.to_int32(length),
            # initial_state_fw = self.state[0],
            # initial_state_bw = self.state[1],
        )
        return tf.concat_v2(vals, 2)

