import time
import numpy as np
import tensorflow as tf
import pickle
import collections

class TextConverter(object):
	
	def __init__(self, text=None, max_vocab=5000, filename=None):
		if filename is not None:
			with open(filename, 'rb') as f:
				self.vocab = pickle.load(f)
		
		else:
			data = list(text)
			counter = collections.Counter(data)
			count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	
			vocab, _  = list(zip(*count_pairs))
			if len(vocab) > max_vocab:
				vocab = vocab[:max_vocab]
			self.vocab = vocab
			
		self.word_to_int_table = dict(zip(self.vocab, range(len(self.vocab))))
		self.int_to_word_table = dict(enumerate(self.vocab))
	
	@property
	def vocab_size(self):
		return len(self.vocab) + 1 #extra 1 slot for unidentified chars
	
	def word_to_int(self, word):
		if word in self.word_to_int_table:
			return self.word_to_int_table[word]
		else:
			return len(self.vocab) #unidentified chars
	
	def int_to_word(self, index):
		if index == len(self.vocab):
			return '<unk>' #the unknown token for unidentified chars
		elif index < len(self.vocab):
			return self.int_to_word_table[index]
		else:
			raise Exception("Unknown Index Error")
	
	def text_to_arr(self, text):
		arr = []
		for word in text:
			arr.append(self.word_to_int(word))
		return np.array(arr)
	
	def arr_to_text(self, arr):
		words = []
		for index in arr:
			words.append(self.int_to_word(index))
		return "".join(words)
	
	def save_to_file(self, filename):
		with open(filename, "wb") as f:
			pickle.dump(self.vocab, f)
	
	
def get_batches(arr, batch_size, num_steps):
	"""
	arr: input
	batch_size: # of sequence in a batch
	num_steps: # of steps
	"""
	data_len = batch_size * num_steps
	num_batches = int(len(arr) / data_len)
	
	arr = arr[:data_len*num_batches]
	
	arr = arr.reshape((batch_size, -1))
	
	#define a generator
	while True:
		for n in range(0, arr.shape[1], num_steps):
			x = arr[:, n:n+num_steps]
			y = np.zeros_like(x)
			try:
				y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+num_steps]
			except:
				y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
			yield x, y