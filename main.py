import numpy as np
import tensorflow as tf
import collections
import math
import os
import time

from model import *
from data_utils import *


flags = tf.app.flags
flags.DEFINE_integer("vocab_size", 500, "")
flags.DEFINE_integer("embedding_size", 50, "")
flags.DEFINE_integer("filter_size", 1, "")
flags.DEFINE_integer("batch_size", 64, "")
flags.DEFINE_integer("epochs", 5, "")
flags.DEFINE_integer("num_layers", 4, "")
flags.DEFINE_integer("block_size", 2, "")
flags.DEFINE_integer("filter_h", 5, "")
flags.DEFINE_integer("context_size", 20, "")
flags.DEFINE_string("data_dir", 'data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled', "")
flags.DEFINE_integer("num_sampled", 1, "")
flags.DEFINE_integer("learning_rate", 1.0, "")
flags.DEFINE_integer("momentum", 0.99, "")
flags.DEFINE_integer("grad_clip", 0.1, "")
flags.DEFINE_integer("num_batches", 0, "")
flags.DEFINE_string("ckpt_path", "ckpt", "")
flags.DEFINE_string("summary_path", "logs", "")


def main(_):
    conf = prepare_conf(flags.FLAGS)
    
    x_batches, y_batches = prepare_data(conf)
    model = GatedCNN(conf)

    saver = tf.train.Saver(tf.trainable_variables())
    print "Started Model Training..."
    
    batch_idx = 0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(conf.summary_path, graph_def=sess.graph)

        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print "Model Restored"
       
        for i in xrange(conf.epochs):
            start = time.time()
            for j in xrange(conf.num_batches):
                inputs, labels, batch_idx = get_batch(x_batches, y_batches, batch_idx)
                _, l = sess.run([model.optimizer, model.loss], feed_dict={model.X:inputs, model.y:labels})
            end = time.time()
            
            print "Epoch: %.2f, Time: %.2f,  Loss: %.2f"%(i, end-start, l)
            if l <= 380:
                perp = sess.run(model.perplexity, feed_dict={model.X:inputs, model.y:labels})
                print "Perplexity: %.2f"%perp
   
            summaries = sess.run(model.merged_summary_op, feed_dict={model.X:inputs, model.y:labels})
            summary_writer.add_summary(summaries, i)
            saver.save(sess, conf.ckpt_file)
            
            


if __name__ == '__main__':
    tf.app.run()