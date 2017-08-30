import time
import numpy as np
import tensorflow as tf
import os

class Config():
	def __init__(self, batch_size=60, num_steps=100, hidden_size=512, lstm_layers=2,
					learning_rate=1e-3, keep_prob=0.5, num_classes=None, grad_clip=5,
					sampling=False, use_embedding=False, embedding_size=None):
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.hidden_size = hidden_size
		self.lstm_layers = lstm_layers
		self.learning_rate = learning_rate
		self.num_classes = num_classes
		self.grad_clip = grad_clip
		self.sampling = sampling
		self.keep_prob = keep_prob
		self.use_embedding = use_embedding
		self.embedding_size = embedding_size

class CharRnn(object):
	def __init__(self, config):
		self.sampling = config.sampling
		self.batch_size = config.batch_size
		self.num_steps = config.num_steps
		self.hidden_size = config.hidden_size
		self.lstm_layers = config.lstm_layers
		self.learning_rate = config.learning_rate
		self.grad_clip = config.grad_clip
		self.num_classes = config.num_classes
		self.use_embedding = config.use_embedding
		self.embedding_size = config.embedding_size
		
		if self.sampling == True:
			self.batch_size, self.num_steps = 1, 1
		
		tf.reset_default_graph()
		
		self.build_inputs()
		self.build_lstm()
		self.build_loss()
		self.build_optimizer()
		self.saver = tf.train.Saver()
		
	
	def build_inputs(self):
		"""
		build the input layer
		"""
		with tf.variable_scope("inputs"):
			self.inputs = tf.placeholder(dtype=tf.int32,
				shape=(self.batch_size, self.num_steps), name="inputs")
			self.targets = tf.placeholder(dtype=tf.int32,
				shape=(self.batch_size, self.num_steps), name="targets")
		
			#for drop out
			self.keep_prob = tf.placeholder(dtype=tf.float32,
				name="keep_prob")
			
			if self.use_embedding == False:
				self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
			#embedding only supports cpu operation
			else:
				#with tf.device("/cpu:0"):
				embedding = tf.get_variable("embedding", [self.num_classes, self.embedding_size])
				self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
	
	
	def build_lstm(self):
		"""
		build the lstm layers
		note: need to deal with the first layer and remaining layers seperately,
		due to different input sizes!
		"""
		def get_a_cell(hidden_size, keep_prob):
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
			drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
			return drop
		
		with tf.variable_scope("lstm"):
			cell = tf.contrib.rnn.MultiRNNCell(
				[get_a_cell(self.hidden_size, self.keep_prob) for _ in range(self.lstm_layers)])
			
			self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
			
			self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
												cell, self.lstm_inputs, initial_state=self.initial_state)
			
			seq_output = tf.concat(self.lstm_outputs, axis=1)
			x = tf.reshape(seq_output, [-1, self.hidden_size])
			
			with tf.variable_scope("softmax"):
				softmax_w = tf.Variable(tf.truncated_normal([self.hidden_size, self.num_classes], stddev=0.1))
				softmax_b = tf.Variable(tf.zeros(self.num_classes))
		
			#logits -> ((batch_size*num_steps)*out_size)
			self.logits = tf.matmul(x, softmax_w) + softmax_b
		
			#out -> ((batch_size*num_steps)*out_size)
			self.prediction = tf.nn.softmax(self.logits, name="prediction")
		
	
	def build_loss(self):
		"""
		build the loss
		"""
		with tf.variable_scope('loss'):
			y_one_hot = tf.one_hot(self.targets, self.num_classes)
			y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
		
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
												logits=self.logits, labels=y_reshaped))
		
	def build_optimizer(self):
		"""
		build the optimizer
		solve the gradients explosion problem
		"""
		tvars = tf.trainable_variables()
		#threshold the gradients
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
		train_op = tf.train.AdamOptimizer(self.learning_rate)
		self.optimizer = train_op.apply_gradients(zip(grads, tvars))
		
	
	def train(self, batch_generator, num_batches, keep_prob=1.0, epochs=20, 
			save_path="checkpoints/", save_every_n=1000, log_every_n=100):
		self.session = tf.Session()
		with self.session as sess:
			sess.run(tf.global_variables_initializer())
			
			step = 0
			
			for e in range(epochs):
				loss = 0
				#reset the initial state for every epoch
				new_state = sess.run(self.initial_state)
				for x,y in batch_generator:
					step += 1
					start = time.time()
					feed_dict = {self.inputs:x,
								self.targets:y,
								self.keep_prob:keep_prob,
								self.initial_state:new_state}
					batch_loss, new_state, _ = sess.run([self.loss,
													self.final_state,
													self.optimizer],
													feed_dict=feed_dict)
					end = time.time()
			
					if step % log_every_n == 0:
						print("epoch: {}/{}... ".format(e+1, epochs),
							"steps: {}...".format(step),
							"loss: {:.4f}... ".format(batch_loss),
							"{:.4f} sec/batch".format(end-start))
					
					if step % save_every_n == 0:
						self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
					
					if step % num_batches == 0:
						break
			
			
			self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
			
	
	
	def sample(self, n_samples, prime, vocab_size):
		
		def pick_top_n(preds, vocab_size, top_n=5):
			"""
			randomly pick the output from top_n candidates
	
			preds: predictions
			vocab_size:
			top_n:
			"""
			p = np.squeeze(preds)
	
			p[np.argsort(p)[:-top_n]] = 0
	
			p = p / np.sum(p)
	
			c = np.random.choice(vocab_size, 1, p=p)[0]
			return c
	
		samples = [c for c in prime]
		
		sess = self.session
		new_state = sess.run(self.initial_state)
			
		for c in prime:
			x = np.zeros((1,1))
			x[0,0] = c
			feed_dict = {self.inputs:x,
						self.keep_prob:1.0,
						self.initial_state:new_state}
			preds, new_state = sess.run([self.prediction, self.final_state],
										feed_dict=feed_dict)
		
		c = pick_top_n(preds, vocab_size)
		
		samples.append(c)
		
		
		for i in range(n_samples):
			x[0,0] = c
			feed_dict = {self.inputs:x,
						self.keep_prob:1.0,
						self.initial_state:new_state}
			preds, new_state = sess.run([self.prediction, self.final_state],
										feed_dict=feed_dict)
			c = pick_top_n(preds, vocab_size)
		
			samples.append(c)
	
		return np.array(samples)
		
	def load(self, ckpt):
		#restoring and sampling must be in the same session!
		self.session = tf.Session()
		self.saver.restore(self.session, ckpt)
		print("Restored from: {}".format(ckpt))