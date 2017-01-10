import numpy as np
import tensorflow as tf 
import gzip

from utils import create_variables


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
        self.activation = getattr(tf, activation) 

    def forward(self, x):    
        # Reshape to two dimensions, multiply, reshape back to n-dimensional tensor  
        # x2d   = tf.reshape(x, [-1, self.in_size])
        vals  = tf.nn.xw_plus_b(x, self.W, self.b, name='Wx_b')
        # shape = tf.shape(x)
        # shape = tf.slice(shape, [0], [tf.shape(shape)[0]-1])
        # shape = tf.concat_v2([shape, [self.out_size]], 0)
        # vals  = tf.reshape(vals, shape)
        return self.activation(vals)

class Layer2(object):

    def __init__(self, in_size, out_size, activation='tanh', name='', **kwargs):
        self.in_size, self.out_size = in_size, out_size
        self.W, self.b  = create_variables([in_size, out_size], name)
        self.activation = getattr(tf, activation) 

    def forward(self, x):    
        # Reshape to two dimensions, multiply, reshape back to n-dimensional tensor  
        vals = tf.map_fn(lambda z: tf.nn.xw_plus_b(z, self.W, self.b), x, dtype=tf.float32)
        # vals  = tf.nn.xw_plus_b(x2d, self.W, self.b, name='Wx_b')
        # shape = tf.shape(x)
        # shape = tf.slice(shape, [0], [tf.shape(shape)[0]-1])
        # shape = tf.concat_v2([shape, [self.out_size]], 0)
        # vals  = tf.reshape(vals, shape)
        return self.activation(vals)


class LSTM(object):
    def __init__(self, depth=1, batch_size=100, name='', **kwargs):
        self.in_size  = kwargs['in_size']
        self.out_size = kwargs['out_size']
        self.name     = name

        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.in_size)
        self.cell  = tf.nn.rnn_cell.MultiRNNCell([lstm] * depth)
        self.state = None#self.cell.zero_state(batch_size, tf.float32)
        self.batch = batch_size

    def forward(self, x, length=None):        
        x.set_shape([None, self.batch, self.in_size])
        vals, self.state = tf.nn.dynamic_rnn(self.cell, x, 
            time_major     = True, 
            dtype          = tf.float32,
            sequence_length= length,
            # initial_state  = self.state,
        )
        # shape = tf.shape(x)
        # max_time = shape[0]
        # batch_size = self.batch#shape[1]
        # input_depth = self.in_size#shape[2]
        # inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        # inputs_ta = inputs_ta.unpack(x)
        # length = tf.to_int32(length)
        # self.prior_state = self.cell.zero_state(batch_size, tf.float32)
        # mask = tf.to_float(mask)
        # def loop_fn(time, cell_output, cell_state, loop_state):
        #   emit_output = cell_output  # == None for time == 0
        #   if cell_output is None:  # time == 0
        #     next_cell_state = self.cell.zero_state(batch_size, tf.float32)
        #   else:
        #     emit_output *= mask[time]
        #     next_cell_state = tf.cond(
        #         tf.cast(mask[time], tf.bool),
        #         lambda:cell_state,
        #         lambda:self.prior_state)
        #     self.prior_state = next_cell_state
        #   elements_finished = (time >= length)
        #   finished = tf.reduce_all(elements_finished)
        #   next_input = tf.cond(
        #       tf.cast(tf.to_float(finished) * mask[time], tf.bool),
        #       lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
        #       lambda: inputs_ta.read(time))
        #   next_loop_state = None
        #   return (elements_finished, next_input, next_cell_state,
        #           emit_output, next_loop_state)
        # vals, self.state, _ = tf.nn.raw_rnn(self.cell, loop_fn)
        return vals#.pack()


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
        if length is None:
            used   = tf.sign(tf.reduce_max(tf.abs(x), reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=0)
        x.set_shape([None, None, self.in_size])       
        self.l = length
        vals, self.state = tf.nn.bidirectional_dynamic_rnn(self.fcell, self.bcell, x, 
            time_major     = True, 
            dtype          = tf.float32,
            sequence_length= tf.to_int32(length),
            # initial_state_fw = self.state[0],
            # initial_state_bw = self.state[1],
        )
        return tf.concat_v2(vals, 2)#[vals[0], tf.reverse_v2(vals[1], 2)], 2)

