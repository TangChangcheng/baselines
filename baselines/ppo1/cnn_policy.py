import baselines.common.tf_util as U
import tensorflow as tf
import gym
import re
import os, sys
from baselines.hyperparams import *
from baselines.common.distributions import make_pdtype

class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, hid_size=32, num_hid_layers=2, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, hid_size, num_hid_layers, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size=32, num_hid_layers=2, kind='large'):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(nb_group * ac_space.shape[0])
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        # x = ob / 255.0
        x = ob
        for i in range(num_hid_layers):
            x = tf.nn.relu(U.conv2d(x, hid_size, 'conv%d'%i, [1, 3 * ob_space.shape[1]], pad='SAME'))
        x = tf.nn.relu(U.conv2d(x, 1, 'conv%d' % num_hid_layers, [1, 3 * ob_space.shape[1]], pad='SAME'))
          
        # if kind == 'small': # from A3C paper
        #     x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
        #     x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
        #     x = U.flattenallbut0(x)
        #     x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        # elif kind == 'large': # Nature DQN
        #     x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
        #     x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
        #     x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
        #     x = U.flattenallbut0(x)
        #     x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        # else:
        #     raise NotImplementedError
        x_shape = x.shape
        x = tf.reshape(x, [-1, ob_space.shape[1]])
        

        logits = tf.layers.dense(x, 2 * ac_space.shape[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
        logits = tf.reshape(logits, (-1, nb_group, 2 * ac_space.shape[0]))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=U.normc_initializer(1.0))[:,0]
        self.vpred = tf.reshape(self.vpred, (-1, ob_space.shape[0]))
        self.vpred = tf.reduce_mean(self.vpred, axis=-1)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, ob], [ac, self.vpred])
      
        self.saver = tf.train.Saver()

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
    
    def save(self, dir):
      # name = self.scope
      # os.makedirs('{dir}/{name}'.format(dir=dir, name=name),exist_ok=True)
      self.saver.save(tf.get_default_session(), os.path.join(dir, env_id, 'model.ckpt'), global_step=self.step)
      print('save model successful.')

    def load(self, dir):
        # name = self.scope
        try:
          checkpoint = tf.train.latest_checkpoint(os.path.join(dir, env_id))
          self.saver.restore(tf.get_default_session(), checkpoint)
          self.step = int(re.findall(r'\d+', checkpoint)[0])
          print('load model successful.')
        except Exception as e:
          print('Loading model error: ', e, file=sys.stderr)

