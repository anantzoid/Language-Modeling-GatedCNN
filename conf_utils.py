import tensorflow as tf
import os

def prepare_conf(conf):
    conf.filter_w = conf.embedding_size
    conf.context_size += conf.filter_h/2

    ckpt_model_path = 'vocab:%d_embed:%d_filters:%d_batch:%d_layers:%d_block:%d_fdim:%d'%(conf.vocab_size, conf.embedding_size, 
            conf.filter_size, conf.batch_size, conf.num_layers, conf.block_size, conf.filter_h)

    conf.ckpt_full_path = os.path.join(conf.ckpt_path, ckpt_model_path)

    if not os.path.exists(conf.ckpt_path):
        os.makedirs(conf.ckpt_path)

    conf.ckpt_file = os.path.join(conf.ckpt_path, "model.ckpt")

    if tf.gfile.Exists(conf.summary_path):
        tf.gfile.DeleteRecursively(conf.summary_path)
    tf.gfile.MakeDirs(conf.summary_path)

    return conf 