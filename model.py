from layers import LSTM, Layer, EmbeddingLayer, BiDirLSTM, Layer2, RCNN
from utils_  import VarScopeClass, NameScopeClass, preprocess, read_data, create_gradients

import tensorflow as tf
import numpy as np
import time, os, json

''' check spanish review after convergence: search tiempo '''
flags = tf.app.flags
flags.DEFINE_integer("aspect", 3, "Class to predict [0]")
flags.DEFINE_integer("hidden", 200, "Number of hidden nodes [200]")
flags.DEFINE_integer("depth", 2, "Depth of the Encoder network [2]")
flags.DEFINE_integer("batch", 200, "Number of elements per batch [128]")
flags.DEFINE_integer("maxlen", 256, "Maximum text length allowed [256]")
flags.DEFINE_integer("epochs", 500, "Number of training epochs [50]")
flags.DEFINE_integer("seed", None, "Random seed [None]")
flags.DEFINE_float("keep_prob", 0.9, "Probability to keep during dropout [0.9]")
flags.DEFINE_float("l2_reg", 1e-6, "L2 regularization parameter [1e-6]")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for adam optimization [5e-4]")
flags.DEFINE_float("sparsity", 5e-5, "Sparsity coefficient (higher=fewer Generator selections)")
flags.DEFINE_float("coherency", 1e-4, "Coherency coefficient (higher=longer sequences)")
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
		output = Layer2(name='Generator_Output', **out_kwargs)
		length = tf.reduce_sum(pad_mask, 0)
		ilength= tf.to_int32(length)

		# Pass sequence through the network
		fwd = RCNN(name='fwd',**net_kwargs)
		bwd = RCNN(name='bwd',**net_kwargs)

		bembed = tf.reverse_sequence(embeddings, ilength, 0, 1)
		h1 = fwd.forward(embeddings)
		h2 = bwd.forward(bembed)
		h2 = tf.reverse_sequence(h2, ilength, 0, 1)
		h = tf.concat_v2([h1,h2], 2)
		h = dropout(h)

		# Pass the values through the output to get selection probabilities
		b_major = tf.transpose(h, [1,0,2])
		probs = output.forward( b_major, ilength )
		probs = tf.transpose(probs, [1,0,2])
		shape = tf.shape(embeddings)[:2]
		probs = tf.reshape(probs, shape)

		# Binomial sampling not yet implemented in tensorflow.... 
		# binomial   = tf.contrib.distributions.Binomial(n=1, p=self.probs)
		# z = binomial.sample(probs.get_shape())

		# Instead, sample the uniform distribution and construct it manually
		uniform = tf.contrib.distributions.Uniform()
		samples = uniform.sample(tf.shape(probs))	
		z = tf.to_float(tf.less(samples, probs)) * pad_mask	

		# Calculate sample loss statistics
		sparsity   = tf.reduce_sum(z, 0)	
		begin_cost = ((length/2) / (sparsity + 1)) ** 1
		end_cost   =((length*2) / (length - sparsity + 1)) ** 1
		zsum  = sparsity +end_cost + begin_cost  # Sparsity - x + len / (x+1) + (len / (len-x+1)) ** 2	
		# begin/end costs create smooth increases in cost at the extremes:
		# google.com/search?q=100+%2F+(x%2B1)+%2B+x+%2B+(100+%2F+(100-x%2B1))**2%2C+x
		# (50 / (x+1))**2 + x + (200 / (100-x+1))**2, x
		
		# Penalize the number of text->no text swaps
		zpad  = tf.pad(z, [[1,1],[0,0]]) 
		zdiff = tf.reduce_sum(tf.abs(zpad[1:] - zpad[:-1]), 0)

		# Penalize segments inversely proportional to their lengths
		# padded  = tf.pad(z, [[1,0],[0,0]])
		# shiftmul= (1-padded)[1:] * padded[:-1]
		# indices = tf.cumsum(shiftmul, axis=0)
		# segments= tf.to_int32(indices - tf.reduce_min(indices, axis=0))
		# zdiff   = tf.map_fn(lambda q: #tf.reduce_logsumexp(q[2] / tf.maximum(1., tf.segment_sum(q[0], q[1]))),
		# 			# tf.reduce_max(tf.log1p(q[2] / tf.maximum(1., tf.segment_sum(q[0], q[1])))), 
		# 			# (lambda qq: tf.reduce_max(qq)+tf.reduce_mean(qq))(tf.exp(1. / tf.maximum(tf.segment_sum(q[0], q[1]), 1.))), 
		# 			(lambda segs: tf.reduce_logsumexp(tf.to_float(tf.shape(segs)[0]) / segs))(tf.maximum(tf.segment_sum(q[0], q[1]), 1.)),
		# 			(tf.transpose(z, [1,0]), tf.transpose(segments, [1,0]), length), 
		# 			dtype=tf.float32)
		# tf.summary.image('segments', tf.expand_dims(tf.expand_dims(tf.transpose(
		# 			tf.map_fn(lambda q: tf.pad(tf.segment_sum(q[0], q[1]), [[0,tf.shape(q[0])[0]-tf.reduce_max(q[1])]]),
		# 			#tf.reduce_sum(tf.log(q[2] / tf.segment_sum(q[0], q[1]))), 
		# 			(tf.transpose(z, [1,0]), tf.transpose(segments, [1,0]), length), 
		# 			dtype=tf.float32),[1,0]), -1), 0))

		# Penalize probabilities directly. Generalization of zdiff -> probs
		# pm = probs * pad_mask
		# values = tf.transpose(pm, [1,0])
		# padded = tf.pad(tf.pad(values, [[0,0], [1,1]], 'SYMMETRIC'), [[0,0], [1,1]], 'SYMMETRIC')
		# expad  = tf.expand_dims(padded, -1)
		# filtr  = tf.reshape(tf.constant([0.5,0.5, 0, 0.5,0.5]), [-1,1,1])
		# conv   = tf.nn.conv1d(expad, filtr, 1, 'VALID')[:,:,0]
		# zdiff  = tf.reduce_sum(tf.abs(conv/2 - values)**2, 1)
		
		self.zsum  = zsum  # Sparsity
		self.zdiff = zdiff # Coherency

		# Push probabilities toward [0,1] values via negative xentropy
		self.loss = -loss_f(probs, z) * pad_mask		
		self.z 	  = tf.stop_gradient(z) 					

		self.regularize =(self.zsum  * placeholders['sparsity'] + \
						  self.zdiff * placeholders['coherency'])
		color = lambda v:tf.tile(tf.expand_dims((1-pad_mask),-1), [1,1,3])*[[[0,0,.4 * tf.reduce_max(v)-tf.reduce_min(v)]]]
		to_img = lambda v: tf.expand_dims(tf.tile(tf.expand_dims(v, -1), [1,1,3]) + color(v), 0)

		tf.summary.histogram('samples', tf.reduce_mean(samples, 1))
		tf.summary.histogram('zsum',self.zsum)
		tf.summary.histogram('zdiff', self.zdiff)
		tf.summary.histogram('sample_probability', tf.reduce_mean(probs*pad_mask, 0))
		tf.summary.histogram('sample_distribution', tf.reduce_mean(probs*pad_mask, 1))
		tf.summary.scalar('zsum', tf.reduce_mean(self.zsum) * placeholders['sparsity'])
		tf.summary.scalar('zdiff', tf.reduce_mean(self.zdiff) * placeholders['coherency'])
		tf.summary.image('probabilities', to_img(probs*pad_mask))
		tf.summary.image('normed probabilities', to_img((probs*pad_mask) / (tf.expand_dims(tf.reduce_max(probs*pad_mask, 1)+1e-8, 1))))
		tf.summary.image('normed probabilities2', to_img((probs*pad_mask) / (tf.reduce_max(probs*pad_mask, 0)+1e-8)))
		tf.summary.image('samples', to_img(z))
		tf.summary.scalar('max_prob', tf.reduce_max(probs*pad_mask))
		tf.summary.scalar('mean_prob', tf.reduce_mean(tf.reduce_sum(probs*pad_mask,0)/length))


	def create_minimization(self, loss_vec, step):
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Generator')
		l2_cost   = tf.add_n([tf.nn.l2_loss(v) for v in variables])

		cost_vec = loss_vec  + self.regularize
		cost_gen = tf.reduce_mean(cost_vec * tf.reduce_sum(self.loss, 0))+ l2_cost*self.placeholders['lambda']
		self.obj = tf.reduce_mean(loss_vec + self.regularize)
		self.reg = tf.reduce_mean(self.regularize)

		self.train_g = create_gradients(cost_gen, variables, FLAGS.learning_rate, step)

		tf.summary.scalar('Generator_Cost', cost_gen)
		tf.summary.scalar('Objective', self.obj)



class Encoder(object, metaclass=VarScopeClass):

	def __init__(self, embeddings, pad_mask, placeholders, net_kwargs,
					dropout, loss_f):

		out_kwargs = dict(net_kwargs)
		out_kwargs['in_size'], out_kwargs['out_size'] = out_kwargs['out_size']*2, \
			placeholders['y'].get_shape().as_list()[1]
		out_kwargs['activation'] = 'sigmoid'
		self.output_layer = Layer(name='Encoder_Output', **out_kwargs)
		self.net_kwargs   = net_kwargs
		self.embeddings   = embeddings
		self.placeholders = placeholders
		self.pad_mask 	  = pad_mask
		self.dropout 	  = dropout
		self.loss_f 	  = loss_f


	def create_minimization(self, samples):

		# Create networks and initial states
		layers = [RCNN(name=str(_)+'ERCNN', **self.net_kwargs) for _ in range(self.net_kwargs['depth'])]
		h_prev = self.embeddings
		h_prev.set_shape([None, FLAGS.batch, self.net_kwargs['in_size']]) 

		z = tf.expand_dims(samples, -1)
		z.set_shape([None, FLAGS.batch, 1])

		# Pass samples through all networks
		states = []
		for l in layers:
			h_next = l.forward2(h_prev, z)
			states+= [h_next[-1]]
			h_prev = self.dropout(h_next)

		# Use values to get a final prediction on the text
		preds = self.preds = self.output_layer.forward(self.dropout(tf.concat_v2(states, 1)))

		loss_mat = self.loss_f(preds, self.placeholders['y'])
		if FLAGS.aspect == -1:  self.loss_vec = tf.reduce_mean(loss_mat, 1)
		else:			 		self.loss_vec = loss_mat[:, FLAGS.aspect]

		self.loss = tf.reduce_mean(self.loss_vec)
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encoder')
		l2_cost   = tf.add_n([tf.nn.l2_loss(v) for v in variables])
		cost_enc  = self.loss + l2_cost * self.placeholders['lambda']

		self.train_e = create_gradients(cost_enc, variables, FLAGS.learning_rate)

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
		bxentropy = lambda x,y: -(y * tf.log(x + 1e-8) + (1. - y) * tf.log(1. - x + 1e-8))
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

