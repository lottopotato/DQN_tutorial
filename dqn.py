"""
DQN Class
reference by https://github.com/hunkim/ReinforcementZeroToAll/ 
It is also reference by here. 
DQN(NIPS-2013)
"Playing Atari with Deep Reinforcement Learning"
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
DQN(Nature-2015)
"Human-level control through deep reinforcement learning"
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
"""

import numpy as np
import tensorflow as tf

class DQN:
	def __init__(self, session: tf.Session, input_size: int, output_size: int,
		name:str = "main"):
	
		self.session = session
		self.input_size = input_size
		self.output_size = output_size
		self.net_name = name
		self.network()

	def network(self, hidden_size=16, learning_rate = 1e-3):
		with tf.variable_scope(self.net_name):
			self._X = tf.placeholder(tf.float32, [None, self.input_size], name = "input_x")
			net = self._X

			#fully connected
			net = tf.layers.dense(net, hidden_size, activation = tf.nn.relu)
			net = tf.layers.dense(net, self.output_size)
			self._Qpred = net

			self._Y = tf.placeholder(tf.float32, [None, self.output_size])
			self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

			train_op = tf.train.AdamOptimizer(learning_rate = learning_rate)
			self._train_op = train_op.minimize(self._loss)

	def predict(self, state: np.ndarray):
		x = np.reshape(state, [-1, self.input_size])
		return self.session.run(self._Qpred, feed_dict={self._X:x})

	def update(self, x_stack:np.ndarray, y_stack:np.ndarray):
		feed = {
		self._X: x_stack,
		self._Y: y_stack
		}
		return self.session.run([self._loss, self._train_op], feed_dict = feed)
