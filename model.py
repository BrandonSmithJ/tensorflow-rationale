from layers import LSTM, Layer, EmbeddingLayer, BiDirLSTM
from utils  import VarScopeClass, NameScopeClass, preprocess, read_data

import tensorflow as tf
import numpy as np
import time, os, json

flags = tf.app.flags
flags.DEFINE_integer("aspect", 0, "Class to predict [0]")
flags.DEFINE_integer("hidden", 200, "Number of hidden nodes [200]")
flags.DEFINE_integer("depth", 2, "Depth of the Encoder network [2]")
flags.DEFINE_integer("batch", 256, "Number of elements per batch [128]")
flags.DEFINE_integer("maxlen", 256, "Maximum text length allowed [256]")
flags.DEFINE_integer("epochs", 50, "Number of training epochs [50]")
flags.DEFINE_integer("seed", None, "Random seed [None]")
flags.DEFINE_float("keep_prob", 0.9, "Probability to keep during dropout [0.9]")
flags.DEFINE_float("l2_reg", 1e-6, "L2 regularization parameter [1e-6]")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for adam optimization [5e-4]")
flags.DEFINE_float("sparsity", 3e-4, "Sparsity coefficient (higher=fewer Generator selections)")
flags.DEFINE_float("coherency", 6e-4, "Coherency coefficient (higher=longer sequences)")
flags.DEFINE_string("embedding", "", "Path to word embeddings")
flags.DEFINE_string("training", "", "Path to training data")
flags.DEFINE_string("testing", "", "Path to testing data")
flags.DEFINE_string("output", "", "Path to save output file")
flags.DEFINE_string("checkpoint", "checkpoint", "Path to save network checkpoints")
flags.DEFINE_string("log", "log", "Path to save log files")
FLAGS = flags.FLAGS


class Generator(object, metaclass=VarScopeClass):
	
	def __init__(self, embeddings, pad_mask, placeholders, net_kwargs,
					dropout, loss_f):
		self.placeholders = placeholders

		out_kwargs = dict(net_kwargs)
		out_kwargs['in_size'], out_kwargs['out_size'] = out_kwargs['out_size']*2, 1
		out_kwargs['activation'] = 'sigmoid'
		output  = Layer(name='Generator_Output', **out_kwargs)
		network = BiDirLSTM(**net_kwargs)

		# Pass sequence through the network, and bound outputs at (0,1)
		h = dropout( network.forward(embeddings) ) 
		probs = output.forward(h)
		shape = tf.shape(embeddings)[:2]
		probs = tf.reshape(probs, shape)
		probs = tf.maximum(tf.minimum(probs, 1-1e-8), 1e-8)

		# Binomial sampling not yet implemented in tensorflow.... 
		# binomial   = tf.contrib.distributions.Binomial(n=1, p=self.probs)
		# z = binomial.sample(probs.get_shape())

		# Instead, sample the uniform distribution and construct it manually
		uniform = tf.contrib.distributions.Uniform()
		samples = uniform.sample(tf.shape(probs))	
		z = tf.to_float(tf.less(samples, probs))	

		# Calculate sample loss statistics
		sparsity   = tf.reduce_sum(z, 0) 		
		# selection  = tf.to_float(shape[0]) / (sparsity + 1)
		self.zsum  = sparsity# + selection					  # Sparsity			 	
		self.zdiff = tf.reduce_sum(tf.abs(z[1:] - z[:-1]), 0) # Coherency

		# Maximize entropy wrt. binomial sampling (i.e. push probabilities toward [0,1])
		self.loss = -loss_f(probs, z) * pad_mask		
		self.z 	  = tf.stop_gradient(z) 					
		
		self.regularize = self.zsum * placeholders['sparsity'] + \
					 	  self.zdiff* placeholders['coherency']

		tf.summary.histogram('zsum',self.zsum)
		tf.summary.histogram('zdiff', self.zdiff)
		tf.summary.histogram('probs', tf.reduce_mean(probs, 0))
		# tf.summary.scalar('sparse selection penalty', tf.reduce_mean(selection))
		tf.summary.scalar('zsum', tf.reduce_mean(self.zsum))
		tf.summary.scalar('zdiff', tf.reduce_mean(self.zdiff))


	def create_minimization(self, loss_vec, step):
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Generator')
		l2_cost   = tf.add_n([tf.nn.l2_loss(v) for v in variables])

		cost_vec = loss_vec + self.regularize + l2_cost*self.placeholders['lambda']
		cost_gen = tf.reduce_mean(cost_vec * tf.reduce_sum(self.loss, 0))
		self.obj = tf.reduce_mean(loss_vec + self.regularize)
		self.reg = tf.reduce_mean(self.regularize)

		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
		self.train_g = optimizer.minimize(cost_gen, var_list=variables, global_step=step)
		
		tf.summary.scalar('Generator_Cost', cost_gen)
		tf.summary.scalar('Objective', self.obj)



class Encoder(object, metaclass=VarScopeClass):

	def __init__(self, embeddings, pad_mask, placeholders, net_kwargs,
					dropout, loss_f):

		out_kwargs = dict(net_kwargs)
		out_kwargs['in_size'], out_kwargs['out_size'] = out_kwargs['out_size'], \
			placeholders['y'].get_shape().as_list()[1]
		out_kwargs['activation'] = 'sigmoid'
		self.output_layer = Layer(name='Encoder_Output', **out_kwargs)
		self.network 	  = LSTM(**net_kwargs)
		self.embeddings   = embeddings
		self.placeholders = placeholders
		self.pad_mask 	  = pad_mask
		self.dropout 	  = dropout
		self.loss_f 	  = loss_f


	def create_minimization(self, samples):
		# Due to sparse nature of samples, the tensor needs to be reordered
		# with non-zero frames first. This ensures RNN sequences are calculated
		# over only the valid frames in the current sampling
		mask   = tf.to_int32(1 - samples * self.pad_mask)
		order  = lambda v: tf.concat_v2(tf.dynamic_partition(v[0], v[1], 2), 0)
		inds   = tf.transpose(mask,[1,0])
		length = tf.reduce_sum(1-mask, 0)
		tf.summary.histogram('length',length)
		# Map the reordering function over all batch samples, changing the 
		# time / batch -major ordering of the tensor appropriately
		signal  = tf.expand_dims(samples, -1) * self.embeddings
		b_major = tf.transpose(signal, [1,0,2])
		ordered = tf.map_fn(order, (b_major, inds), dtype=tf.float32)
		t_major = tf.transpose(ordered, [1,0,2])
		
		# Pass the samples through the network, averaging over all RNN timesteps
		# for the final activation. (Also possible to use gather with lengths,
		# but currently requires sparse operations which is difficult for TF to
		# automatically apply a gradient to)
		h_all  = self.network.forward(t_major, length)
		h_last = tf.reduce_sum(h_all, 0) / tf.expand_dims(tf.maximum(tf.to_float(length),1),-1)
		preds  = self.preds = self.output_layer.forward( self.dropout(h_last) )

		loss_mat = self.loss_f(preds, self.placeholders['y'])
		if FLAGS.aspect == -1:  self.loss_vec = tf.reduce_mean(loss_mat, 1)
		else:			 		self.loss_vec = loss_mat[:, FLAGS.aspect]

		self.loss = tf.reduce_mean(self.loss_vec)
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encoder')
		l2_cost   = tf.add_n([tf.nn.l2_loss(v) for v in variables])
		cost_enc  = self.loss + l2_cost * self.placeholders['lambda']
		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate) 
		self.train_e = optimizer.minimize(cost_enc,	var_list=variables)

		tf.summary.histogram('Loss_Vec',self.loss_vec)
		tf.summary.histogram('Predictions', preds[:,FLAGS.aspect])
		tf.summary.histogram('Y', self.placeholders['y'][:,FLAGS.aspect])
		tf.summary.scalar('Encoder Cost', cost_enc)



class Model(object):

	def __init__(self):	
		self.session = tf.InteractiveSession()
		tf.set_random_seed(FLAGS.seed)
		np.random.seed(FLAGS.seed)

		print('Reading embeddings...', end='', flush=True)
		self.embedding = EmbeddingLayer(FLAGS.embedding)
		print('done')

		# Define network parameters
		nkwargs = { 
			'in_size' 	: self.embedding.shape[1],
			'out_size'	: FLAGS.hidden,
			'depth'   	: FLAGS.depth,
			'batch_size': FLAGS.batch,
		}

		# Define the inputs, and their respective type & size
		inputs = {  
			'x'   		: [tf.int32,   [None, FLAGS.batch]],                
			'y'   		: [tf.float32, [FLAGS.batch, 5]],    
			'kp'  		: [tf.float32, []], 
			'lambda'	: [tf.float32, []],
			'sparsity' 	: [tf.float32, []],
			'coherency'	: [tf.float32, []],
		}

		# Create placeholders
		with tf.name_scope('Placeholders'):
			p = { name: tf.placeholder(*args, name=name) 
				  for name, args in inputs.items() }

		self.train_fd = lambda x,y, kp=FLAGS.keep_prob: {
			p['x'] 			: x,
			p['y'] 			: y,
			p['kp'] 		: kp,
			p['lambda'] 	: FLAGS.l2_reg,
			p['sparsity'] 	: FLAGS.sparsity,
			p['coherency'] 	: FLAGS.coherency,
		}

		dropout   = lambda x: tf.nn.dropout(x, p['kp'])
		bxentropy = lambda x,y: -(y * tf.log(x) + (1. - y) * tf.log(1. - x))
		sq_err    = lambda x,y: (x - y) ** 2

		pad_mask  = tf.to_float(tf.not_equal(p['x'], self.embedding.pad_id))
		embedding = dropout( self.embedding.forward(p['x']) ) 

		print('Creating model...', end='', flush=True)
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		self.generator = Generator(embedding, pad_mask, p, nkwargs, dropout, bxentropy)
		self.encoder   = Encoder(embedding, pad_mask, p, nkwargs, dropout, sq_err)
		self.encoder.create_minimization(self.generator.z)
		self.generator.create_minimization(self.encoder.loss_vec, self.global_step)
		print('done')


	def train(self):
		print('Initializing variables...', end='', flush=True)
		logdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log')
		writer = tf.summary.FileWriter(logdir, self.session.graph)
		saver  = tf.train.Saver()
		self.session.run(tf.global_variables_initializer())		
	
		checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint)
		if checkpoint and checkpoint.model_checkpoint_path:
			print('restoring previous checkpoint...', end='', flush=True)
			name = os.path.basename(checkpoint.model_checkpoint_path)
			saver.restore(self.session, os.path.join(FLAGS.checkpoint, name))
		merger = tf.summary.merge_all()
		print('done')

		print('Fetching data...', end='', flush=True)
		x,y   = read_data(FLAGS.training)
		train = ([self.embedding.words_to_ids(s) for s in x], y)
		x,y  = read_data(FLAGS.testing)
		test = ([self.embedding.words_to_ids(s) for s in x], y)
		print('done')
		
		for epoch in range(FLAGS.epochs):
			start_time = time.time()
			train_x, train_y = preprocess(train, FLAGS.batch, self.embedding.pad_id, 
										  FLAGS.maxlen)

			scost = ocost = tcost = p_one = 0
			for bx,by in zip(train_x, train_y):
				result = self.session.run([merger, 
									self.generator.train_g, self.encoder.train_e, 
									self.generator.reg, self.generator.obj, 
									self.encoder.loss, self.generator.z, 
									self.global_step],
									feed_dict=self.train_fd(bx, by))
				writer.add_summary(result[0], result[7])
				
				scost += result[3]
				ocost += result[4]
				tcost += result[5]
				p_one += np.sum(result[6]) / FLAGS.batch / len(bx[0])

			print('Regularization: ', scost / float(len(train_x)))
			print('Objective: ', ocost / float(len(train_x)))
			print('Prediction loss: ', tcost / float(len(train_x)))
			print('Generator Selection %: ', p_one / float(len(train_x)))

			if not epoch % 1:
				results = []
				ocost = tcost = 0
				test_x, test_y = preprocess(test, FLAGS.batch, self.embedding.pad_id, 
										    FLAGS.maxlen)
				for bx,by in zip(test_x, test_y):
					preds, bz, gobj, eloss = self.session.run([
												self.encoder.preds, 
												self.generator.z, 
												self.generator.obj,
												self.encoder.loss],
												feed_dict=self.train_fd(bx, by, 1.))
					ocost += gobj
					tcost += eloss
					for p, x, y, z in zip(preds, bx.T, by, bz.T):
						w = self.embedding.ids_to_words(x)
						w = [u.replace('<pad>', '_') for u in w]
						r = [u if v == 1 else '_' for u,v in zip(w,z)]
						results.append((p, r, w, y))
					
				print('Test Objective: ', ocost / float(len(test_x)))
				print('Test Prediction loss: ',tcost / float(len(test_x)))

				with open(FLAGS.output, 'w+') as f:
					for p, r, w, y in results:
						f.write(json.dumps({
							'rationale' : ' '.join(r),
							'original'  : ' '.join(w),
							'y' : str(list(y)),
							'p' : str(list(p)),
						}) + '\n')

				saver.save(self.session, 
							os.path.join(FLAGS.checkpoint, 'GEN.model'), 
							global_step=self.global_step)

			print('Finished epoch %s in %.2f seconds\n' % (epoch, time.time() - start_time))



def main(_): 
	assert(FLAGS.embedding and FLAGS.training), \
		'Both embedding and training data must be specified'
	if not os.path.exists(FLAGS.checkpoint):
		os.makedirs(FLAGS.checkpoint)
	Model().train()
if __name__ == '__main__':
	tf.app.run()

