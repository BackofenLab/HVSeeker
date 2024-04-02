
# coding: utf-8
import os 
import numpy as np
import tensorflow as tf
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from sklearn import metrics
# os.chdir('dir') 

tf.compat.v1.disable_eager_execution()  # Disable eager execution to use placeholders

X_train=np.loadtxt(open("encoded_x_train.csv","rb"),delimiter=",",skiprows=0)
print(X_train.shape)
y_train=np.loadtxt(open("Y_train_corrected.csv","rb"),delimiter=",",skiprows=0)
print(y_train.shape)


def next_batch(train_data, train_target, batch_size):  
    index = [ i for i in range(0,len(train_target)) ]  
    np.random.shuffle(index);  
    batch_data = []; 
    batch_target = []; 
    batch_seqlen = [];
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]);  
        batch_target.append(train_target[index[i]])  
        batch_seqlen.append(len(train_data[index[i]])/2)
    return batch_data, batch_target, batch_seqlen

# f=open('unpadding_file.csv','r')
# test_seqlen = []
# lines=f.readlines()
# for line in lines:
#     a=len(line)
#     b=a/2
#     test_seqlen.append(int(b))
# f.close()


learning_rate = 0.00001 
training_iters = 180000 
batch_size = 256 
display_step = 10 
seq_max_len = 500 
n_hidden = 256 
n_classes = 2 


x = tf.compat.v1.placeholder("float", [None, seq_max_len,1], name='x')
y = tf.compat.v1.placeholder("float", [None, n_classes], name='y')
seqlen = tf.compat.v1.placeholder(tf.int32, [None])

weights = {
    'out': tf.Variable(tf.random.normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random.normal([n_classes]))
}

def dynamicRNN(x, seqlen, weights, biases):
    
    lstm_layer = tf.keras.layers.LSTM(n_hidden, return_sequences=True, return_state=True)
    # Apply the layer to your input (assuming 'x' is your input tensor and 'seqlen' your sequence lengths)
    outputs, last_state_h, last_state_c = lstm_layer(x, mask=tf.sequence_mask(seqlen))
    value = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)

    return tf.matmul(last, weights['out']) + biases['out']
    

pred = dynamicRNN(x, seqlen, weights, biases)
preds = tf.reshape(pred, [-1, 256, 2])
# tf.add_to_collection('network-output', pred)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    saver=tf.train.Saver(max_to_keep=1)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = next_batch(X_train,y_train,batch_size)
        batch_x = np.expand_dims(batch_x,2)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y ,seqlen: batch_seqlen})
        if step % display_step == 0:
            
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss))
        step += 1
        
    print("Optimization Finished!")

