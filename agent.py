import tensorflow as tf
import tensorflow.contrib as tc


#Create Actor for DDPG
class Actor(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, hidden_size):
        #parameters defination
        self.sess = sess
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # create actor and target
        self.inputs, self.outputs, self.scaled_outputs = self.create()
        self.actor_weights = tf.compat.v1.trainable_variables()
        self.target_inputs, self.target_outputs, self.target_scaled_outputs = self.create()
        self.target_actor_weights = tf.compat.v1.trainable_variables()[len(self.actor_weights):]

        # set target weights to be actor weights
        self.update_target_weights = \
            [self.target_actor_weights[i].assign(tf.multiply(self.actor_weights[i], self.tau) +
                                                 tf.multiply(self.target_actor_weights[i], 1. - self.tau))
             for i in range(len(self.target_actor_weights))]

        # placeholder for gradient feed from critic
        self.c_gradients = tf.compat.v1.placeholder(tf.float32, [None, self.action_dim])

        # combine actor gradients and critic gradients, then normalize
        self.unm_actor_gradients = tf.gradients(self.scaled_outputs, self.actor_weights, -self.c_gradients)
        self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), self.unm_actor_gradients))

        # Adam optimizer
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.actor_weights))

        # count of weights
        self.n_actor_vars = len(self.actor_weights) + len(self.target_actor_weights)

    # function to create agent actor network
    def create(self):
        #normal layer
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, self.state_dim])
        layerA = tc.layers.layer_norm(inputs, center=True, scale=True, begin_norm_axis=0)
        layerA = tf.compat.v1.layers.Dense(self.hidden_size).apply(layerA)
        layerA = tc.layers.layer_norm(layerA, center=True, scale=True)
        layerA = tf.nn.relu(layerA)
        layerA = tf.compat.v1.layers.Dense(self.hidden_size).apply(layerA)
        layerA = tc.layers.layer_norm(layerA, center=True, scale=True)
        layerA = tf.nn.relu(layerA)

        # activation layer
        k_init = tf.random_uniform_initializer(minval=-0.004, maxval=0.004)
        outputs = tf.compat.v1.layers.Dense(self.action_dim, kernel_initializer=k_init).apply(layerA)
        outputs = tf.nn.tanh(outputs)

        # scale output fit action_bound
        scaled_outputs = tf.multiply(outputs, self.action_bound)
        return inputs, outputs, scaled_outputs

    # adding predict,predict_target,optimize and gradients,update
    def predict(self, inputs):
        return self.sess.run(self.scaled_outputs, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_outputs, feed_dict={
            self.target_inputs: inputs
        })

    def train(self, inputs, grad):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.c_gradients: grad
        })

    def update(self):
        self.sess.run(self.update_target_weights)


#Create Critic for DDPG
class Critic(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, n_actor_vars, hidden_size):
        # parameters defination
        self.sess = sess
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        # create critic and target
        self.inputs, self.actions, self.outputs = self.create()
        self.critic_weights = tf.compat.v1.trainable_variables()[n_actor_vars:]
        self.target_inputs, self.target_actions, self.target_outputs = self.create()
        self.target_critic_weights = tf.compat.v1.trainable_variables()[(len(self.critic_weights) + n_actor_vars):]

        # set target weights to be critic weights
        self.update_target_weights = \
            [self.target_critic_weights[i].assign(tf.multiply(self.critic_weights[i], self.tau)
                                                  + tf.multiply(self.target_critic_weights[i], 1. - self.tau))
             for i in range(len(self.target_critic_weights))]

        # placeholder for predicted q
        self.pred_q = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # Adam optimizer
        self.loss = tf.reduce_mean(tf.square(self.pred_q - self.outputs))
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # set comment gradients to feed actor
        self.c_gradients = tf.gradients(self.outputs, self.actions)

    # function to create agent critic network
    def create(self):
        # state branch
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, self.state_dim])
        LayerC = tc.layers.layer_norm(inputs, center=True, scale=True, begin_norm_axis=0)
        LayerC = tf.compat.v1.layers.Dense(self.hidden_size).apply(LayerC)
        LayerC = tc.layers.layer_norm(LayerC, center=True, scale=True)
        LayerC = tf.nn.relu(LayerC)

        # action branch
        actions = tf.compat.v1.placeholder(tf.float32, shape=[None, self.action_dim])

        # merge
        LayerC = tf.concat([LayerC, actions], axis=1)
        LayerC = tf.compat.v1.layers.Dense(self.hidden_size).apply(LayerC)
        LayerC = tc.layers.layer_norm(LayerC, center=True, scale=True)
        LayerC = tf.nn.relu(LayerC)

        # activation layer
        k_init = tf.random_uniform_initializer(minval=-0.004, maxval=0.004)
        outputs = tf.compat.v1.layers.Dense(1, kernel_initializer=k_init).apply(LayerC)
        return inputs, actions, outputs

    # function to train by adding states, actions, and q values
    def train(self, inputs, actions, pred_q):
        return self.sess.run([self.outputs, self.optimize], feed_dict={
            self.inputs: inputs,
            self.actions: actions,
            self.pred_q: pred_q
        })

    # function to compute gradients to feed actor
    def get_comment_gradients(self, inputs, actions):
        return self.sess.run(self.c_gradients, feed_dict={
            self.inputs: inputs,
            self.actions: actions
        })

    # defination of predict,predict_target,update
    def predict(self, inputs, actions):
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: inputs,
            self.actions: actions
        })

    def predict_target(self, inputs, actions):
        return self.sess.run(self.target_outputs, feed_dict={
            self.target_inputs: inputs,
            self.target_actions: actions
        })

    def update(self):
        self.sess.run(self.update_target_weights)
