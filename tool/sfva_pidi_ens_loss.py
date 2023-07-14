"""Implementation of attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time
import utils
import os
from scipy import misc
from scipy import ndimage
import PIL
import io
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

slim = tf.contrib.slim

tf.flags.DEFINE_string('model_name', 'inception_resnet_v2', 'The Model used to generate adv.')
#  inception_v3   inception_v4   resnet_v2_152   inception_resnet_v2

tf.flags.DEFINE_string('attack_method', 'SI_PF_PIDI', 'The name of attack method.')

tf.flags.DEFINE_string('layer_name','InceptionResnetV2/InceptionResnetV2/Mixed_5b/concat','The layer to be attacked.')
# 'InceptionV3/InceptionV3/Mixed_5b/concat'
# 'InceptionV4/InceptionV4/Mixed_5e/concat'
# 'resnet_v2_152/block2/unit_8/bottleneck_v2/add'
# 'InceptionResnetV2/InceptionResnetV2/Mixed_5b/concat'
tf.flags.DEFINE_float('a', 0.5, 'uniform limit')


tf.flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './adv/sfva_pidi_ens/', 'Output directory with images.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_float('alpha', 1.6, 'Step size.')

tf.flags.DEFINE_integer('batch_size', 8, 'How many images process at one time.') # 20

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_string('GPU_ID', '0', 'which GPU to use.')

"""parameter for DIM"""
tf.flags.DEFINE_integer('image_size', 299, 'size of each input images.')

tf.flags.DEFINE_integer('image_resize', 331, 'size of each diverse images.')

tf.flags.DEFINE_float('prob', 0.7, 'Probability of using diverse inputs.')

"""parameter for TIM"""
tf.flags.DEFINE_integer('Tkern_size', 15, 'Kernel size of TIM.')

"""parameter for PIM"""
tf.flags.DEFINE_float('amplification_factor', 2.5, 'To amplifythe step size.')

tf.flags.DEFINE_float('gamma', 0.5, 'The gamma parameter.')

tf.flags.DEFINE_integer('Pkern_size', 3, 'Kernel size of PIM.')

"""parameter for NAA"""
tf.flags.DEFINE_float('ens', 25.0, 'Aggregated N for NAA or Mask number for FIA.') # 30
"""parameter for FIA"""
tf.flags.DEFINE_float('probb', 0.9, 'keep probability = 1 - drop probability.')

FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID

import random
np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

"""obtain the feature map of the target layer"""
def get_opt_layers(layer_name):
    opt_operations = []
    #shape=[FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,3]
    operations = tf.get_default_graph().get_operations()
    for op in operations:
        if layer_name == op.name:
            opt_operations.append(op.outputs[0])
            shape=op.outputs[0][:FLAGS.batch_size].shape
            break
    return opt_operations,shape

"""the loss function for FDA"""
def get_fda_loss(opt_operations):
    loss = 0
    for layer in opt_operations:
        batch_size = FLAGS.batch_size
        tensor = layer[:batch_size]
        mean_tensor = tf.stack([tf.reduce_mean(tensor, -1), ] * tensor.shape[-1], -1)
        wts_good = tensor < mean_tensor
        wts_good = tf.to_float(wts_good)
        wts_bad = tensor >= mean_tensor
        wts_bad = tf.to_float(wts_bad)
        loss += tf.log(tf.nn.l2_loss(wts_good * (layer[batch_size:]) / tf.cast(tf.size(layer),tf.float32)))
        loss -= tf.log(tf.nn.l2_loss(wts_bad * (layer[batch_size:]) / tf.cast(tf.size(layer),tf.float32)))
    loss = loss / len(opt_operations)
    return loss

"""the loss function for NRDM"""
def get_nrdm_loss(opt_operations):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:]
        loss+=tf.norm(ori_tensor-adv_tensor)/tf.cast(tf.size(layer),tf.float32)
    loss = loss / len(opt_operations)
    return loss

# 权重貌似不变，变的只有特征图
"""the loss function for FIA"""
def get_fia_loss(opt_operations,weights):
    loss = 0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size] # 这行没用
        adv_tensor = layer[FLAGS.batch_size:]
        loss += tf.reduce_sum(adv_tensor*weights) / tf.cast(tf.size(layer), tf.float32) 

    loss = loss / len(opt_operations)
    return loss

"""the loss function for NAA"""
def get_NAA_loss(opt_operations,weights,base_feature):
    loss = 0
    gamma = 1.0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]  # 前一半是干净样本
        adv_tensor = layer[FLAGS.batch_size:]  # 后一半是对抗样本
        attribution = (adv_tensor-base_feature)*weights
        #attribution = (adv_tensor)*weights
        blank = tf.zeros_like(attribution) # 基准图片 0像素，纯黑色
        positive = tf.where(attribution >= 0, attribution, blank)
        negative = tf.where(attribution < 0, attribution, blank)
        ## Transformation: Linear transformation performs the best
        positive = positive
        negative = negative
        ##
        balance_attribution = positive + gamma*negative # shape = attribution
        loss += tf.reduce_sum(balance_attribution) / tf.cast(tf.size(layer), tf.float32) # 一个batch所有神经元的平均贡献？
        
    loss = loss / len(opt_operations)
    return loss

def get_patch_loss(opt_operations,weights):
    loss = 0

    for layer in opt_operations:  # opt_operations [batchsize,299,299,3 ]
        ori_tensor = layer[:FLAGS.batch_size]  # 前一半是干净样本，没用上
        adv_tensor = layer[FLAGS.batch_size:]  # 后一半是对抗样本
        clean_attribution = ori_tensor*weights 
        adv_attribution = adv_tensor*weights
        
        blank = tf.zeros_like(clean_attribution) 
        
        pclean = tf.where(clean_attribution >= 0, clean_attribution, blank)
        padv = tf.where(clean_attribution >= 0, adv_attribution, blank)
        
        nclean = tf.where(clean_attribution < 0, clean_attribution, blank)
        nadv = tf.where(clean_attribution < 0, adv_attribution, blank)
        
        p_attribution = pclean - padv
        n_attribution = nclean - nadv
        
        loss += (tf.reduce_sum(p_attribution) + tf.reduce_sum(n_attribution)) / tf.cast(tf.size(layer), tf.float32)
  
    
    loss = loss / len(opt_operations)
    return loss


def advanced_mask(img,p):
    # get width and height of the image
    s = img.shape
    b = s[0]
    wd = s[1]
    ht = s[2]
   
    img_copy = np.copy(img)
    np.random.shuffle(img_copy)

    # possible grid size, 0 means no hiding
    grid_sizes=[0,20,40,60,80] #掩码块的大小

    # hiding probability
    hide_prob = p
 
    # randomly choose one grid size
    grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]

    # hide the patches
    if(grid_size!=0): # 每个batch的掩码块大小给定就不变了
         for x in range(0,wd,grid_size):
             for y in range(0,ht,grid_size):
                 x_end = min(wd, x+grid_size)  
                 y_end = min(ht, y+grid_size)
                 if(random.random() <=  hide_prob):
                    #img[:,x:x_end,y:y_end,:]= 255  
                    
                    img[:,x:x_end,y:y_end,:]= np.random.uniform(low=0, high=255, size = np.shape(img[:,x:x_end,y:y_end,:]))                      
                        
                     #for i in range(0,b):
                        #img[i,x:x_end,y:y_end,:]= np.mean(img[i,:,:,:]) 
                        
                        #img[i,x:x_end,y:y_end,:]= img[i,x:x_end,y:y_end,:] * 0.1      ## scale mask
                        #img[i,x:x_end,y:y_end,:]= img_copy[i,x:x_end,y:y_end,:]       ## image mask
                        
                        #img[i,x:x_end,y:y_end,:]= np.mean(img[i,x:x_end,y:y_end,:]) 
                        
                        #img[i,x:x_end,y:y_end,0]=np.mean(img[i,x:x_end,y:y_end,0])
                        #img[i,x:x_end,y:y_end,1]=np.mean(img[i,x:x_end,y:y_end,1])
                        #img[i,x:x_end,y:y_end,2]=np.mean(img[i,x:x_end,y:y_end,2])
                      
                                    
    return img


def patch_gridfullcolor(img,ens,l): # 完全填补
    # get width and height of the image
    s = img.shape
    wd = s[1]
    ht = s[2]


    colorpool = ['r','g','b','bk','w']
    col =  colorpool[random.randint(0,len(colorpool)-1)] # 这个函数可以取到末尾
    
    
    n = int(ens**0.5)    
    grid_size = wd//n   
    
    rx = l % n
    x = rx * grid_size   # 
    xend = (rx+1) * grid_size  # 

    
    ry = l // n 
    y = ry * grid_size
    yend = (ry + 1) * grid_size    
    if col == 'r':
        img[:,y:yend,x:xend,0]=255 
        img[:,y:yend,x:xend,1]=0
        img[:,y:yend,x:xend,2]=0 
    elif col == 'g':
        img[:,y:yend,x:xend,0]=0 
        img[:,y:yend,x:xend,1]=255
        img[:,y:yend,x:xend,2]=0 
    elif col == 'b':
        img[:,y:yend,x:xend,0]=0 
        img[:,y:yend,x:xend,1]=0
        img[:,y:yend,x:xend,2]=255 
    elif col == 'bk':
        img[:,y:yend,x:xend,:]=0

    elif col == 'w':
        img[:,y:yend,x:xend,:]=255

    return img

def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1: # 1范数：除以绝对值之和
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square) # 开根号
    return nor_grad

def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3) # 第四个位置增加1维,也就是 channel_multiplier=1
    return stack_kern, kern_size // 2 # 向下取整

def project_noise(x, stack_kern, kern_size):
    x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT") # mode=“CONSTANT” 填充0
    x = tf.nn.depthwise_conv2d(x, stack_kern, strides=[1, 1, 1, 1], padding='VALID') 
    # 卷积核 要求是一个4维Tensor，具有[filter_height, filter_width, in_channels, channel_multiplier]
    return x

def gkern(kernlen=21, nsig=3): # 用不上，应该是Ti里面的
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    return stack_kernel

def input_diversity(input_tensor):
    """Input diversity: https://arxiv.org/abs/1803.06978"""
    rnd = tf.random_uniform((), FLAGS.image_size, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret=tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_size, FLAGS.image_size],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret

P_kern, kern_size = project_kern(FLAGS.Pkern_size)


def main(_):

    if FLAGS.model_name in ['vgg_16','vgg_19', 'resnet_v1_50','resnet_v1_152']:
        eps = FLAGS.max_epsilon # 16 
        alpha = FLAGS.alpha # 步长
    else:
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        alpha = FLAGS.alpha * 2.0 / 255.0

    num_iter = FLAGS.num_iter
    momentum = FLAGS.momentum

    image_preprocessing_fn = utils.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    
    # inception_v3   inception_v4   resnet_v2_152   inception_resnet_v2
    
    checkpoint_path1 = utils.checkpoint_paths['inception_v3']
    layer_name1='InceptionV3/InceptionV3/Mixed_5b/concat'
    checkpoint_path2 = utils.checkpoint_paths['inception_v4']
    layer_name2='InceptionV4/InceptionV4/Mixed_5e/concat'
    checkpoint_path3 = utils.checkpoint_paths['resnet_v2_152']
    layer_name3='resnet_v2_152/block2/unit_8/bottleneck_v2/add'
    checkpoint_path4 = utils.checkpoint_paths['inception_resnet_v2']
    layer_name4='InceptionResnetV2/InceptionResnetV2/Mixed_5b/concat'


    with tf.Graph().as_default():
        # Prepare graph
        ori_input  = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input = tf.placeholder(tf.float32, shape=batch_shape)
        num_classes = 1000 + utils.offset[FLAGS.model_name]
        label_ph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size*2,num_classes])
        accumulated_grad_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)
        amplification_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)

        network_fn1 = utils.nets_factory.get_network_fn('inception_v3', num_classes=num_classes, is_training=False)
        network_fn2 = utils.nets_factory.get_network_fn('inception_v4', num_classes=num_classes, is_training=False)
        network_fn3 = utils.nets_factory.get_network_fn('resnet_v2_152', num_classes=num_classes, is_training=False)        
        network_fn4 = utils.nets_factory.get_network_fn('inception_resnet_v2', num_classes=num_classes, is_training=False)
        
        
        # 模型框架实例化的地方。factory其实是个编写好的函数方便调用模型
        
        x=tf.concat([ori_input,adv_input],axis=0)

        # with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        #     logits1, end_points1 = inception_v3.inception_v3(
        #         x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

        # with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        #     logits2, end_points2 = inception_v4.inception_v4(
        #         x, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)

        # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        #     logits3, end_points3 = inception_resnet_v2.inception_resnet_v2(
        #         x, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)

        # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        #     logits4, end_points4 = resnet_v2.resnet_v2_152(
        #         x, num_classes=1001, is_training=False, scope='resnet_v2_152', reuse=tf.AUTO_REUSE)


        # whether using DIM or not
        if 'DI' in FLAGS.attack_method:
            logits1, end_points1 = network_fn1(input_diversity(x))
            logits2, end_points2 = network_fn2(input_diversity(x))
            logits3, end_points3 = network_fn3(input_diversity(x))
            logits4, end_points4 = network_fn4(input_diversity(x))
                     
        else:
            logits1, end_points1 = network_fn1(x)
            logits2, end_points2 = network_fn2(x)
            logits3, end_points3 = network_fn3(x)
            logits4, end_points4 = network_fn4(x)
         
            
         
            
            
            
        logits = (logits1 + logits2 + logits3 + logits4) * 0.25   
        # problity=tf.nn.softmax(logits,axis=1) # 用不上
        pred = tf.argmax(logits1, axis=1)
        one_hot = tf.one_hot(pred, num_classes) # 这部分在迭代过程中不变，模型预测干净图片制作的one_hot

        entropy_loss = tf.losses.softmax_cross_entropy(one_hot[:FLAGS.batch_size], logits[FLAGS.batch_size:])
        # 每次迭代都会到这，干净图片部分（前一半）是不动的 

        opt_operations1,shape1 = get_opt_layers(layer_name1)
        opt_operations2,shape2 = get_opt_layers(layer_name2)
        opt_operations3,shape3 = get_opt_layers(layer_name3)
        opt_operations4,shape4 = get_opt_layers(layer_name4)
        
        weights_ph1 = tf.placeholder(dtype=tf.float32, shape=shape1)
        weights_ph2 = tf.placeholder(dtype=tf.float32, shape=shape2)
        weights_ph3 = tf.placeholder(dtype=tf.float32, shape=shape3)
        weights_ph4 = tf.placeholder(dtype=tf.float32, shape=shape4)        
        base_feature1 = tf.placeholder(dtype=tf.float32, shape=shape1)
        base_feature2 = tf.placeholder(dtype=tf.float32, shape=shape2)
        base_feature3 = tf.placeholder(dtype=tf.float32, shape=shape3)        
        base_feature4 = tf.placeholder(dtype=tf.float32, shape=shape4)
        
        # select the loss function
        if 'FDA' in FLAGS.attack_method:
            loss = get_fda_loss(opt_operations)
        elif 'NRDM' in FLAGS.attack_method:
            loss = get_nrdm_loss(opt_operations)
        elif 'FIA' in FLAGS.attack_method:
            weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
            loss = get_fia_loss(opt_operations,weights_ph)
        elif 'NAA' in FLAGS.attack_method:
            weights_tensor1 = tf.gradients(tf.nn.softmax(logits1) * label_ph, opt_operations1[0])[0]
            loss1 = get_NAA_loss(opt_operations1,weights_ph1,base_feature1)
            
            weights_tensor2 = tf.gradients(tf.nn.softmax(logits2) * label_ph, opt_operations2[0])[0]
            loss2 = get_NAA_loss(opt_operations2,weights_ph2,base_feature2)       
            
            weights_tensor3 = tf.gradients(tf.nn.softmax(logits3) * label_ph, opt_operations3[0])[0]
            loss3 = get_NAA_loss(opt_operations3,weights_ph3,base_feature3)    
            
            weights_tensor4 = tf.gradients(tf.nn.softmax(logits4) * label_ph, opt_operations4[0])[0]
            loss4 = get_NAA_loss(opt_operations4,weights_ph4,base_feature4)  
            
            loss = (loss1 + loss2 + loss3 + loss4) * 0.25
        elif 'SI_PF' in FLAGS.attack_method:
            # weights_tensor = tf.gradients(tf.nn.softmax(logits) * label_ph, opt_operations[0])[0]
            # loss = get_patch_loss(opt_operations,weights_ph)    
            
            
            weights_tensor1 = tf.gradients(tf.nn.softmax(logits1) * label_ph, opt_operations1[0])[0]
            loss1 = get_patch_loss(opt_operations1,weights_ph1)
            
            weights_tensor2 = tf.gradients(tf.nn.softmax(logits2) * label_ph, opt_operations2[0])[0]
            loss2 = get_patch_loss(opt_operations2,weights_ph2)       
            
            weights_tensor3 = tf.gradients(tf.nn.softmax(logits3) * label_ph, opt_operations3[0])[0]
            loss3 = get_patch_loss(opt_operations3,weights_ph3)
            
            weights_tensor4 = tf.gradients(tf.nn.softmax(logits4) * label_ph, opt_operations4[0])[0]
            loss4 = get_patch_loss(opt_operations4,weights_ph4) 
            
            loss = (loss1 + loss2 + loss3 + loss4) * 0.25
        else:
            loss = entropy_loss

        gradient=tf.gradients(loss,adv_input)[0]

        noise = gradient
        adv_input_update = adv_input
        amplification_update = amplification_ph # pi

        # the default optimization process with momentum
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True) # 动量法中的除以1范数
        noise = momentum * accumulated_grad_ph + noise
        # accumulated_grad_ph 就是历史累计梯度 —— 动量项

        # whether using PIM or not
        if 'PI' in FLAGS.attack_method:
            # amplification factor
            alpha_beta = alpha * FLAGS.amplification_factor
            gamma = FLAGS.gamma * alpha_beta

            # Project cut noise
            amplification_update += alpha_beta * tf.sign(noise)
            cut_noise = tf.clip_by_value(abs(amplification_update) - eps, 0.0, 10000.0) * tf.sign(amplification_update)
            projection = gamma * tf.sign(project_noise(cut_noise, P_kern, kern_size))

            amplification_update += projection

            adv_input_update = adv_input_update + alpha_beta * tf.sign(noise) + projection
        else:
            adv_input_update = adv_input_update + alpha * tf.sign(noise)


        # saver=tf.train.Saver()
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))        
        s4 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))                
        with tf.Session() as sess:
            
            # 载入预训练模型节点
            # saver.restore(sess,checkpoint_path) 
            s1.restore(sess, checkpoint_path1)
            s2.restore(sess, checkpoint_path2)
            s3.restore(sess, checkpoint_path3)
            s4.restore(sess, checkpoint_path4)
            
            count=0
            print('****begin****')
            for images,names,labels in utils.load_image(FLAGS.input_dir, FLAGS.image_size,FLAGS.batch_size):
                count+=FLAGS.batch_size
                if count%20==0:
                    print("Generating:",count)

                images_tmp=image_preprocessing_fn(np.copy(images))
                if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                    labels=labels-1

                # obtain true label
                labels= to_categorical(np.concatenate([labels,labels],axis=-1),num_classes)
                #labels = sess.run(one_hot, feed_dict={ori_input: images_tmp, adv_input: images_tmp})

                #add some noise to avoid F_{k}(x)-F_{k}(x')=0
                if 'NRDM' in FLAGS.attack_method:
                    images_adv=images+np.random.normal(0,0.1,size=np.shape(images))
                else:
                    images_adv=images

                images_adv=image_preprocessing_fn(np.copy(images_adv))

                grad_np=np.zeros(shape=batch_shape)
                amplification_np=np.zeros(shape=batch_shape)
                
                # weight_np = np.zeros(shape=shape)
                weight_np1 = np.zeros(shape=shape1)
                weight_np2 = np.zeros(shape=shape2)
                weight_np3 = np.zeros(shape=shape3)
                weight_np4 = np.zeros(shape=shape4)

                for i in range(num_iter):
                    if i == 0:
                        images_base = np.zeros_like(images)
                        images_base = image_preprocessing_fn(images_base)
                            
                        feature_base1 = sess.run([opt_operations1[0]],
                                            feed_dict={ori_input: images_base, adv_input: images_base,label_ph: labels})
                        feature_base1 = feature_base1[0][:FLAGS.batch_size]

                        feature_base2 = sess.run([opt_operations2[0]],
                                            feed_dict={ori_input: images_base, adv_input: images_base,label_ph: labels})
                        feature_base2 = feature_base2[0][:FLAGS.batch_size]

                        feature_base3 = sess.run([opt_operations3[0]],
                                            feed_dict={ori_input: images_base, adv_input: images_base,label_ph: labels})
                        feature_base3 = feature_base3[0][:FLAGS.batch_size]

                        feature_base4 = sess.run([opt_operations4[0]],
                                            feed_dict={ori_input: images_base, adv_input: images_base,label_ph: labels})
                        feature_base4 = feature_base4[0][:FLAGS.batch_size]                        
                        
                        if 'FIA' in FLAGS.attack_method:
                            for l in range(int(FLAGS.ens)):
                                mask = np.random.binomial(1, FLAGS.probb, size=(batch_shape[0],batch_shape[1],batch_shape[2],batch_shape[3]))
                                images_tmp2 = images * mask
                                images_tmp2 = image_preprocessing_fn(np.copy(images_tmp2))
                                w, feature = sess.run([weights_tensor, opt_operations[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                weight_np = weight_np + w[:FLAGS.batch_size]

                            weight_np = -normalize(weight_np, 2)

                        if 'NAA' in FLAGS.attack_method:
                            for l in range(int(FLAGS.ens)):
                                x_base = np.array([0.0,0.0,0.0])
                                x_base = image_preprocessing_fn(x_base)
                                images_tmp2 = image_preprocessing_fn(np.copy(images))
                                # np.copy) 属于深拷贝,拷贝前的地址和拷贝后的地址不一样. 而 " = " 属于浅拷贝,拷贝后的地址和拷贝前的地址一样
                                
                                images_tmp2 += np.random.normal(size = images.shape, loc=0.0, scale=0.2)
                                # 给每张图片加点符合正态分布的随机噪声？
                                
                                images_tmp2 = images_tmp2*(1 - l/FLAGS.ens)+ (l/FLAGS.ens)*x_base
                                # 公式中的像素比例
                                
                                w1, feature1 = sess.run([weights_tensor1, opt_operations1[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                weight_np1 = weight_np1 + w1[:FLAGS.batch_size] # 干净图片部分

                                w2, feature2 = sess.run([weights_tensor2, opt_operations2[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                weight_np2 = weight_np2 + w2[:FLAGS.batch_size] # 干净图片部分

                                w3, feature3 = sess.run([weights_tensor3, opt_operations3[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                weight_np3 = weight_np3 + w3[:FLAGS.batch_size] # 干净图片部分

                                w4, feature4 = sess.run([weights_tensor4, opt_operations4[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                weight_np4 = weight_np4 + w4[:FLAGS.batch_size] # 干净图片部分

                            # normalize the weights
                            weight_np1 = -normalize(weight_np1, 2)
                            weight_np2 = -normalize(weight_np2, 2)
                            weight_np3 = -normalize(weight_np3, 2)
                            weight_np4 = -normalize(weight_np4, 2)

                        if 'SI_PF' in FLAGS.attack_method:
                            for l in range(int(FLAGS.ens)):

                                pima = advanced_mask(np.copy(images),0.1)
                                images_tmp2 = image_preprocessing_fn(pima)
                                # np.copy) 属于深拷贝,拷贝前的地址和拷贝后的地址不一样. 而 " = " 属于浅拷贝,拷贝后的地址和拷贝前的地址一样
                                
                                images_tmp2 += np.random.uniform(low=-FLAGS.a, high=FLAGS.a, size = images.shape)
                                # images_tmp2 += np.random.normal(size = images.shape, loc=0.0, scale=0.2)
                                # 给每张图片加点符合正态分布的随机噪声？
                                
                                images_tmp2 = images_tmp2*(1 - l/FLAGS.ens)
                                # 公式中的像素比例
                                
                                # w, feature = sess.run([weights_tensor, opt_operations[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                # weight_np = weight_np + w[:FLAGS.batch_size] # 干净图片部分，一直在累加，并没有除以平均
                                
                                
                                w1, feature1 = sess.run([weights_tensor1, opt_operations1[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                weight_np1 = weight_np1 + w1[:FLAGS.batch_size] # 干净图片部分

                                w2, feature2 = sess.run([weights_tensor2, opt_operations2[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                weight_np2 = weight_np2 + w2[:FLAGS.batch_size] # 干净图片部分

                                w3, feature3 = sess.run([weights_tensor3, opt_operations3[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                weight_np3 = weight_np3 + w3[:FLAGS.batch_size] # 干净图片部分

                                w4, feature4 = sess.run([weights_tensor4, opt_operations4[0]],feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})
                                weight_np4 = weight_np4 + w4[:FLAGS.batch_size] # 干净图片部分
                            # normalize the weights
                            # weight_np = normalize(weight_np, 2) # 除以2范数 # 去掉负号
                            weight_np1 = normalize(weight_np1, 2)
                            weight_np2 = normalize(weight_np2, 2)
                            weight_np3 = normalize(weight_np3, 2)
                            weight_np4 = normalize(weight_np4, 2)

                    # optimization 与 if i == 0：平级
                    # images_adv, grad_np, amplification_np=sess.run([adv_input_update, noise, amplification_update],
                    #                           feed_dict={ori_input:images_tmp,adv_input:images_adv,weights_ph:weight_np, base_feature:feature_base,
                    #                                      label_ph:labels,accumulated_grad_ph:grad_np,amplification_ph:amplification_np})
                    # images_adv, grad_np, amplification_np 这三个是会参与循环的，所以需要不断更新
                    images_adv, grad_np, amplification_np=sess.run([adv_input_update, noise, amplification_update],
                                              feed_dict={ori_input:images_tmp,adv_input:images_adv,weights_ph1:weight_np1,weights_ph2:weight_np2,weights_ph3:weight_np3,weights_ph4:weight_np4, base_feature1:feature_base1,base_feature2:feature_base2,base_feature3:feature_base3,base_feature4:feature_base4,
                                                         label_ph:labels,accumulated_grad_ph:grad_np,amplification_ph:amplification_np})

                    
                    images_adv = np.clip(images_adv, images_tmp - eps, images_tmp + eps)

                images_adv = inv_image_preprocessing_fn(images_adv)
                utils.save_image(images_adv, names, FLAGS.output_dir)

if __name__ == '__main__':
    tf.app.run()
