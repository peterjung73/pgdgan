#!/usr/bin/env python
# coding: utf-8

# In[1]:


#USE IN COLAB
# %tensorflow_version 1.x
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

get_ipython().system('rm -r out/')

# xavier initialization
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Input image for discriminator
X = tf.placeholder(tf.float32, shape=[None, 784],name="X")

# initialize weights and biases for discriminator
D_W1 = tf.Variable(xavier_init([784, 128]),name="D_W1")
D_b1 = tf.Variable(tf.zeros(shape=[128]),name="D_b1")

D_W2 = tf.Variable(xavier_init([128, 1]),name="D_W2")
D_b2 = tf.Variable(tf.zeros(shape=[1]),name="D_b2")

theta_D = [D_W1, D_W2, D_b1, D_b2]


# Input noise for generator
Z = tf.placeholder(tf.float32, shape=[None, 20],name="Z")

# initialize weights and biases for generator
G_W1 = tf.Variable(xavier_init([20, 200]),name="G_W1")
G_b1 = tf.Variable(tf.zeros(shape=[200]),name="G_b1")

G_W2 = tf.Variable(xavier_init([200, 784]),name="G_W2")
G_b2 = tf.Variable(tf.zeros(shape=[784]),name="G_b2")

theta_G = [G_W1, G_W2, G_b1, G_b2]

########################################################################
def sample_Z(m, n):
    #return np.random.uniform(-1., 1., size=[m, n])
    return np.random.normal(0,1,(m,n))


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

# plot images
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


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# default learning rate =0.001
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# mini batch size
mb_size = 128
# latent space dim
Z_dim = 20

# store losses
with open("loss_logs.csv","w") as f:
    f.write('Iteration,Discriminator Loss,Generator Loss\n')



# USE THIS ONE COLAB
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

saver = tf.train.Saver(max_to_keep=150)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

if not os.path.exists('pretrained/'):
    os.makedirs('pretrained/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(36, Z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        
        with open("loss_logs.csv","a") as f:
            f.write("%d,%f,%f\n"%(it,D_loss_curr,G_loss_curr))
        
        print('Iter: {}'.format(it))
        print('D_loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
        #if abs(D_loss_curr - 0.5)< 0.1:
         #   saver.save(sess, 'my_test_model')
          #  break
        #if it >60000:
            #saver.save(sess, 'pretrained/my_test_model'+'_'+str(i))


# In[2]:


# plot discriminator and generator loss
def plot_losses():
    with open("loss_logs.csv", "r") as f:
        x = []
        y1 = []
        y2 = []
        # skip header
        i=0
        for line in f:
            if i>0:
                row = line.split(",")
                x.append(float(row[0]))
                y1.append(float(row[1]))
                y2.append(float(row[2]))
            i=i+1

    plt.plot(x, y1,label='Discriminator Loss',c='b')
    plt.plot(x, y2,label='Generator Loss',c='g')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend(loc='upper right')
    
    plt.savefig("loss_logs.png", dpi=300)


# In[3]:


plot_losses()


# In[ ]:




