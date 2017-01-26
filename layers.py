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
    ''' batch independence '''
    def __init__(self, in_size, out_size, activation='tanh', name='', **kwargs):
        self.in_size, self.out_size = in_size, out_size
        self.W, self.b  = create_variables([in_size, out_size], name)
        self.activation = getattr(tf, activation) 

    def forward(self, x, length=None):    
        if length is not None:
            vals = tf.map_fn(lambda z:tf.pad(tf.nn.xw_plus_b(z[0][:z[1]], self.W, self.b), [[0,tf.shape(z[0])[0] - z[1]], [0,0]]), (x,length), dtype=tf.float32)        
        else:
            vals = tf.map_fn(lambda z: tf.nn.xw_plus_b(z, self.W, self.b), x, dtype=tf.float32)
        return self.activation(vals)


class Layer3(object):
    ''' no bias '''
    def __init__(self, in_size, out_size, activation='tanh', name='', **kwargs):
        self.in_size, self.out_size = in_size, out_size
        self.W = tf.get_variable(name+'_W', [in_size, out_size], initializer=tf.truncated_normal_initializer(stddev=0.001))

    def forward(self, x):    
        return tf.matmul(x, self.W)


class Recurrent(object):
    def __init__(self, in_size, out_size, activation='tanh', name='', **kwargs):
        self.in_size, self.out_size = in_size, out_size
        self.W, self.b  = create_variables([in_size+out_size, out_size], name)
        self.activation = getattr(tf, activation) 

    def forward(self, x, h):
        return self.activation(
            tf.matmul(x, self.W[:self.in_size]) + tf.matmul(h, self.W[self.in_size:]) + self.b 
        )

class RCNN(object):
    ''' N-gram modified LSTM '''

    def __init__(self, in_size, out_size, activation='tanh', name='', order=2, **kwargs):
        self.out_size = out_size
        self.in_size  = in_size
        self.order    = order 
        self.activation = getattr(tf, activation)
        self.name = name
        self.layers = layers = []
        for _ in range(order):
            layers.append(Layer3(in_size, out_size, name=name+'_L3_%i'%_))
        self.forget = Recurrent(in_size, out_size, activation='sigmoid', name=name+'_rnn')#tf.nn.rnn_cell.BasicRNNCell(out_size, activation=tf.sigmoid)
        self.bias = tf.get_variable(name+'_rcnn_b', [out_size], initializer=tf.random_uniform_initializer(-0.05,.05))

    def forward(self, x):
        def act(last, batch):
            batch
            no, ni = self.out_size, self.in_size

            h_tm1 = last[:, no*self.order:]
            f_t = self.forget.forward(batch, h_tm1)#, scope=self.name)

            v = []
            for i, layer in enumerate(self.layers):
                c_i_tm1 = last[:, no*i:no*i+no]
                in_i_t  = layer.forward(batch)
                if not i:
                    c_i_t = f_t * c_i_tm1 + (1-f_t) * in_i_t
                else:
                    c_i_t = f_t * c_i_tm1 + (1-f_t) * (in_i_t + c_im1_tm1)
                v.append(c_i_t)
                c_im1_tm1 = c_i_tm1
                c_im1_t = c_i_t
            h_t = self.activation(c_i_t + self.bias)
            v.append(h_t)
            h = tf.concat_v2(v, 1)
            return h 


        h0 = tf.zeros((tf.shape(x)[1], self.out_size*(self.order+1)))
        h = tf.scan(act, x, h0)
        return h[:,:,self.out_size*self.order:]

    def forward2(self, x, mask):
        def act(last, batch):
            xx, mask = batch
            no, ni = self.out_size, self.in_size

            h_tm1 = last[:, no*self.order:]
            f_t = self.forget.forward(xx, h_tm1)#, scope=self.name)

            v = []
            for i, layer in enumerate(self.layers):
                c_i_tm1 = last[:, no*i:no*i+no]
                in_i_t  = layer.forward(xx)
                if not i:
                    c_i_t = f_t * c_i_tm1 + (1-f_t) * in_i_t
                else:
                    c_i_t = f_t * c_i_tm1 + (1-f_t) * (in_i_t + c_im1_tm1)
                v.append(c_i_t)
                c_im1_tm1 = c_i_tm1
                c_im1_t = c_i_t
            h_t = self.activation(c_i_t + self.bias)
            v.append(h_t)
            h = tf.concat_v2(v, 1)
            return h * mask + (1-mask) * last


        h0 = tf.zeros((tf.shape(x)[1], self.out_size*(self.order+1)))
        h = tf.scan(act, (x,mask), h0)
        return h[:,:,self.out_size*self.order:]

    def fwd(self, x, hc):
        no, ni = self.out_size, self.in_size

        h_tm1 = hc[:, no*self.order:]
        f_t = self.forget.forward(x, h_tm1)#, scope=self.name)

        v = []
        for i, layer in enumerate(self.layers):
            c_i_tm1 = hc[:, no*i:no*i+no]
            in_i_t  = layer.forward(x)
            if not i:
                c_i_t = f_t * c_i_tm1 + (1-f_t) * in_i_t
            else:
                c_i_t = f_t * c_i_tm1 + (1-f_t) * (in_i_t + c_im1_tm1)
            v.append(c_i_t)
            c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t
        h_t = self.activation(c_i_t + self.bias)
        v.append(h_t)
        h = tf.concat_v2(v, 1)
        return h

class LayerZ(object):
    ''' Context dependent probabilistic selection '''

    def __init__(self, in_size, out_size, activation='tanh', name='', order=2, **kwargs):
        self.in_size = in_size
        self.h_size  = out_size
        self.activation = getattr(tf, activation)

        self.w1 = tf.get_variable(name+'_zlayer_w1', [in_size,1], initializer=tf.random_uniform_initializer(-0.05,.05))
        self.w2 = tf.get_variable(name+'_zlayer_w2', [out_size,1], initializer=tf.random_uniform_initializer(-0.05,.05))
        self.bias = tf.get_variable(name+'_zlayer_b', [1], initializer=tf.random_uniform_initializer(-0.05,.05))
        self.rcnn = RCNN(in_size+1, out_size, activation=activation)

    def forward(self, x, z):#, h_tm1, pz_tm1):
        xz = tf.concat_v2([x, tf.expand_dims(z, -1)], 2)
        h0 = tf.zeros((1, tf.shape(x)[1], self.h_size))
        h  = self.rcnn.forward(xz)
        h_prev = tf.concat_v2([h0, h[:-1]], 0)
        pz = tf.nn.sigmoid( tf.matmul(tf.reshape(x, [-1, self.in_size]), self.w1) +
                            tf.matmul(tf.reshape(h_prev, [-1, self.h_size]), self.w2) +
                            self.bias)
        return tf.reshape(pz, [tf.shape(x)[0], tf.shape(x)[1]])

    def sample(self, x):
        h0 = tf.zeros((tf.shape(x)[1], self.h_size * (self.rcnn.order + 1)))
        z0 = tf.zeros((tf.shape(x)[1],))

        def act(last, x_t):
            z_tm1, h_tm1 = last 
            pz_t = tf.nn.sigmoid(tf.matmul(x_t, self.w1) +
                                 tf.matmul(h_tm1[:, -self.h_size:], self.w2) +
                                 self.bias)
            pz_t = tf.reshape(pz_t, [-1])

            uniform = tf.contrib.distributions.Uniform()
            samples = uniform.sample(tf.shape(pz_t))   
            z_t  = tf.to_float(tf.less(samples, pz_t))
            xz_t = tf.concat_v2([x_t, tf.reshape(z_t, [-1,1])], 1)
            h_t  = self.rcnn.fwd(xz_t, h_tm1)
            return z_t, h_t

        (z, h) = tf.scan(act, x, (z0,h0))
        return z, None


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

