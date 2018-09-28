"""
openAI gym 'cart pole-v0'
"""

import numpy as np
import tensorflow as tf
import random
import dqn
import gym
import matplotlib.pyplot as plt

# define environment
env = gym.make('CartPole-v0')

# define parameters
INPUT_SIZE = env.observiation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

# DISCOUNT_RATE : y = (1-dr)x + dr(r+f(x+1))
# REPLAY_MEMORY : memory size
# BATCH_SIZE : BATCH- training
# TARGET_UPDATE_FREQUENCY : targetW <- mainW each n
# MAX_EPISODE : n of trainning epoch
DISCOUNT_RATE = 0.9
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODE = 5000

# copy targetW from mainW values
def get_copy_var_ops(src_scope_name:str, dest_scope_name:str)->List[tf.Operlation]:
	holder = []
	src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
		scope = src_scope_name)
	dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
		scope = dest_scope_name)
	for src_var, dest_var in zip(src_vars, dest_vars):
		holder.append(dest_var.assign(src_var.value()))
	return holder

def replay_train(mainDQN:dqn.DQN, targetDQN:dqn.DQN, train_batch:list)->float:
	states = np.vstack([x[0] for x in train_batch])
	actions = np.array([x[1] for x in train_batch])
	rewards = np.array([x[2] for x in train_batch])
	next_states = np.vstack([x[3] for x in train_batch])
	done = np.array([x[4] for x in train_batch])

	Q_target = rewards + DISCOUNT_RATE*np.max(targetDQN.predict(next_states), axis=1)*~done

	y = mainDQN.predict(states)
	y[np.arange(len(states)), actions]

	return mainDQN.update(X,y)

def bot_play(mainDQN:dqn.DQN, env:gym.Env)->None:
	state = env.reset()
	reward_sum = 0

	while True:
		env.render()
		action = np.argmax(mainDQN.predict(state))
		state, reward, done, _ = env.step(action)
		reward_sum += reward

		if done:
			print("\n Total Score : {}".format(reward_sum))
			break

def main():
	replay_buffer = deque(maxlen=REPLAY_MEMORY)
	last_100 = deque(maxlen=100)

	with tf.Session as sess:
		mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
		targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
		sess.run(tf.global_variables_initializer())

		copy_ops = get_copy_var_ops("main","target")
		sess.run(copy_ops)

		for episode in range(MAX_EPISODE):
			e = 1./ ((episode/10)+1)
			done = False
			step_count = 0
			state = env.reset()

			while not done:
				if np.random.rand() < e:
					action = env.action_space.sample()
				else:
					action = np.argmax(mainDQN.predict(state))

				next_states, reward, done, _ = env.step(action)

				if done:
					reward = -1
				replay_buffer.append((state, action, reward, next_states, done))

				if len(replay_buffer) > BATCH_SIZE:
					minibatch = random.sample(replay_buffer, BATCH_SIZE)
					loss, _ = replay_train(mainDQN, targetDQN, minibatch)

				if step_count % TARGET_UPDATE_FREQUENCY == 0:
					sess.run(copy_ops)

				state = next_state
				step_count += 1

			print(" EP : {} | steps : {}".format(episode+1, step_count))

			last_100.append(step_count)

			if len(last_100) == last_100.maxlen:
				avg_reward = np.mean(last_100)
				if avg_reward>199:
					break

if __name__ == "__main__":
	main()
