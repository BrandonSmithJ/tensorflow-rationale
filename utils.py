import tensorflow as tf 
import numpy as np
import gzip 

def create_variables(shape, name=''): 
	''' Creates both weight and bias Variables '''

	W_init = tf.truncated_normal_initializer(stddev=0.01)
	b_init = tf.constant_initializer(0.0)

	W = tf.get_variable(name+'_W', shape,       initializer=W_init)
	b = tf.get_variable(name+'_b', [shape[-1]], initializer=b_init)

	scope = W.name.split('/')[-2] if not name else name
	tf.summary.histogram(scope + '_Weights', W)
	tf.summary.histogram(scope + '_Bias', b) 
	return W, b


def scope(name, f_scope=tf.name_scope):
	''' Decorator for easier variable scoping on functions '''

	def wrapper(f):
		def new_f(*args, **kwargs):
			with f_scope(name + '/'):
				return f(*args, **kwargs)
		return new_f
	return wrapper


class VarScopeClass(type):
	''' Meta class which applies variable scoping to an entire class '''

	def __new__(cls, name, bases, local):
		for attr in local:
			value = local[attr]
			if callable(value):
				local[attr] = scope(name, tf.variable_scope)(value)
		return type.__new__(cls, name, bases, local)


class NameScopeClass(type):
	''' Meta class which applies name scoping to an entire class '''

	def __new__(cls, name, bases, local):
		for attr in local:
			value = local[attr]
			if callable(value):
				local[attr] = scope(name)(value)
		return type.__new__(cls, name, bases, local)


def preprocess(data, batch_size, pad_id, max_len):
	''' Put the data into batches and ensure batch samples are the same shape '''
	x,y = data

	perm = list(range(len(x)))
	perm = sorted(perm, key=lambda i: len(x[i]))
	
	x = [ x[i] for i in perm ]
	y = [ y[i] for i in perm ]

	num_batches = (len(x)-1)//batch_size + 1
	batches_x, batches_y = [], []

	assert(min(len(s) for s in x)), 'Can\'t have 0 length samples'
	normalize_length = lambda s,m: np.pad(s, (0, m-len(s)), 'constant',
											constant_values=pad_id)
	for i in range(num_batches):
		bx = x[i*batch_size:(i+1)*batch_size] + x[:max((i+1)*batch_size - len(x), 0)]
		by = y[i*batch_size:(i+1)*batch_size] + y[:max((i+1)*batch_size - len(y), 0)]

		largest_x = max(len(s) for s in bx)
		m    	  = min(largest_x, max_len)
		batches_x.append( np.column_stack([normalize_length(s[:m], m) for s in bx]) )
		batches_y.append( np.vstack(by) )    

	perm = list(range(len(batches_x)))
	np.random.shuffle(perm)
	batches_x = [batches_x[i] for i in perm]
	batches_y = [batches_y[i] for i in perm]
	return batches_x, batches_y


def read_data(path):
	data_x, data_y = [ ], [ ]
	fopen = gzip.open if path.endswith(".gz") else open
	with fopen(path) as fin:
		for line in fin.readlines():
			y, sep, x = line.strip().partition(bytes("\t", 'utf-8'))
			x, y = x.decode('utf-8').split(), y.decode('utf-8').split()
			if len(x) == 0: continue
			y = np.asarray([ float(v) for v in y ])
			data_x.append(x)
			data_y.append(y)

	return data_x, data_y