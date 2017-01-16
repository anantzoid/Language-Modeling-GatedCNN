import numpy as np
import tensorflow as tf

class GatedCNN(object):

    def __init__(self, conf):
        tf.reset_default_graph()
            
        self.X = tf.placeholder(shape=[conf.batch_size, conf.context_size-1], dtype=tf.int32, name="X")
        self.y = tf.placeholder(shape=[conf.batch_size, conf.context_size-1], dtype=tf.int32, name="y")

        embed = self.create_embeddings(self.X, conf)
        h, res_input = embed, embed

        for i in range(conf.num_layers):
            fanin_depth = h.get_shape()[-1]
            filter_size = conf.filter_size if i < conf.num_layers-1 else 1
            shape = (conf.filter_h, conf.filter_w, fanin_depth, filter_size)
            
            with tf.variable_scope("layer_%d"%i):
                conv_w = self.conv_op(h, shape, "linear")
                conv_v = self.conv_op(h, shape, "gated")
                h = conv_w * tf.sigmoid(conv_v)
                if i % conf.block_size == 0:
                    h += res_input
                    res_input = h
        h = tf.reshape(h, (-1, conf.embedding_size))
        y_shape = self.y.get_shape().as_list()
        self.y = tf.reshape(self.y, (y_shape[0] * y_shape[1], 1))

        softmax_w = tf.get_variable("softmax_w", [conf.vocab_size, conf.embedding_size], tf.float32, 
                                    tf.random_normal_initializer(0.0, 0.1))
        softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], tf.float32, tf.constant_initializer(1.0))

        #Preferance: NCE Loss, heirarchial softmax, adaptive softmax
        self.loss = tf.reduce_mean(tf.nn.nce_loss(softmax_w, softmax_b, h, self.y, conf.num_sampled, conf.vocab_size))

        trainer = tf.train.MomentumOptimizer(conf.learning_rate, conf.momentum)
        gradients = trainer.compute_gradients(self.loss)
        clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
        self.optimizer = trainer.apply_gradients(clipped_gradients)
        self.perplexity = tf.exp(self.loss)

        self.create_summaries()

    def create_embeddings(self, X, conf):

        embeddings = tf.get_variable("embeds",(conf.vocab_size, conf.embedding_size), tf.float32, tf.random_uniform_initializer(-1.0,1.0))
        embed = tf.nn.embedding_lookup(embeddings, X)
        mask_layer = np.ones((conf.batch_size, conf.context_size-1, conf.embedding_size))
        mask_layer[:,0:conf.filter_h/2,:] = 0
        embed *= mask_layer
        
        embed_shape = embed.get_shape().as_list()
        embed = tf.reshape(embed, (embed_shape[0], embed_shape[1], embed_shape[2], 1))
        return embed


    def conv_op(self, fan_in, shape, name):
        W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
        b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
        return tf.add(tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME'), b)
    
    def create_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("perplexity", self.perplexity)
        self.merged_summary_op = tf.summary.merge_all()
