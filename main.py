import tensorflow as tf
import os
import time

from model import *
from data_utils import *
from conf_utils import *


flags = tf.app.flags
flags.DEFINE_integer("vocab_size", 2000, "Maximum size of vocabulary")
flags.DEFINE_integer("embedding_size", 200, "Embedding size of each token")
flags.DEFINE_integer("filter_size", 64, "Depth of each CNN layer")
flags.DEFINE_integer("num_layers", 10, "Number of CNN layers")
flags.DEFINE_integer("block_size", 5, "Size of each residual block")
flags.DEFINE_integer("filter_h", 5, "Height of the CNN filter")
flags.DEFINE_integer("context_size", 20, "Length of sentence/context")
flags.DEFINE_integer("batch_size", 64, "Batch size of data while training")
flags.DEFINE_integer("epochs", 50, "Number of epochs")
flags.DEFINE_integer("num_sampled", 1, "Sampling value for NCE loss")
flags.DEFINE_integer("learning_rate", 1.0, "Learning rate for training")
flags.DEFINE_integer("momentum", 0.99, "Nestrov Momentum value")
flags.DEFINE_integer("grad_clip", 0.1, "Gradient Clipping limit")
flags.DEFINE_integer("num_batches", 0, "Predefined: to be calculated")
flags.DEFINE_string("ckpt_path", "ckpt", "Path to store checkpoints")
flags.DEFINE_string("summary_path", "logs", "Path to store summaries")
flags.DEFINE_string("data_dir", "data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled", "Path to store data")


def main(_):
    conf = prepare_conf(flags.FLAGS)
    
    x_batches, y_batches = prepare_data(conf)
    model = GatedCNN(conf)

    saver = tf.train.Saver(tf.trainable_variables())
    print "Started Model Training..."
    
    batch_idx = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(conf.summary_path, graph=sess.graph)

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

            if i % 10 == 0:
                perp = sess.run(model.perplexity, feed_dict={model.X:inputs, model.y:labels})
                print "Perplexity: %.2f"%perp
                saver.save(sess, conf.ckpt_file)
   
            summaries = sess.run(model.merged_summary_op, feed_dict={model.X:inputs, model.y:labels})
            summary_writer.add_summary(summaries, i)
            
            


if __name__ == '__main__':
    tf.app.run()
