#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('rm -r out2/')

# USE IN COLAB
#%tensorflow_version 1.x
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# In[2]:


# directory for plots of generated images
if not os.path.exists('out2/'):
    os.makedirs('out2/')

def sample_Z(m, n):
    #return np.random.uniform(-1., 1., size=[m, n])
    return np.random.normal(0,1,(m,n))

Z_dim=20

def plot(samples):
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(6, 6)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig



with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('pretrained115/my_test_model_115.meta')
    saver.restore(sess,tf.train.latest_checkpoint('pretrained115/'))
    graph = tf.get_default_graph()

    Z = graph.get_tensor_by_name("Z:0")
    G_W1 = sess.run("G_W1:0")
    G_W2 = sess.run("G_W2:0")
    G_b1 = sess.run("G_b1:0")
    G_b2 = sess.run("G_b2:0")
    
    def generator(z):
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob


    #for tensor in graph.get_operations():
    #  print(tensor.name)
 
    G_sample = generator(Z)
    samples = sess.run(G_sample, feed_dict={Z: sample_Z(36, Z_dim)})

    fig = plot(samples)
    plt.savefig('out2/{}.png'.format(str(0)), bbox_inches='tight')
        
    plt.close(fig)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




