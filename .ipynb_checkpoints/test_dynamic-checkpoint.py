from config import *
from util import *
from image_sparse_warp import *


class PrecomputeStepTest(unittest.TestCase):
    def test_load_dataset(self):
        self.assertEqual(len(ele['das'].shape), 2)
        self.assertEqual(len(ele['dtce'].shape), 2)
        
        
    def test_res_image_shape(self): 
        image = ele['dtce']
        image_dim = tf.shape(image) # Image dimensions [height, width]
        image_height, image_width = tf.gather(image_dim, 0), tf.gather(image_dim, 1)
        a1 = tf.math.multiply(ele['final_radius'], tf.sin(ele['final_angle']))
        a2 = tf.cast(tf.math.divide(image_width, 2), tf.float32)
        horizontal_pad = tf.cast(tf.round(tf.math.subtract(a1, a2)), tf.int32)
        vertical_pad = tf.cast(tf.round(ele['initial_radius']), tf.int32)
        res_height = image_height + vertical_pad
        res_width = image_width + 2 * horizontal_pad

        empty_res_image_true = tf.zeros([res_height+1, res_width+1], tf.float32)
        
        self.assertEqual(empty_res_image_true.shape, empty_res_image.shape)
        
    
    def test_weights(self):
        summed_weights = tf.reduce_sum(val_weights, axis=2)
        self.assertEqual(tf.ones(shape=tf.shape(summed_weights)), summed_weights)

    

ds = tfds.load('duke_ultrasound', data_dir='gs://tfds-data/datasets')
test_dataset = ds['MARK'].map(process)
test_iter = iter(test_dataset)
ele = next(test_iter)

empty_res_image, points_xy, val_rtheta, val_weights = image_sparse_warp_precompute(
    ele['dtce'], ele, ele['dtce'].shape[0], ele['dtce'].shape[1])
res = image_sparse_warp(ele['dtce'], empty_res_image, ele, points_xy, val_rtheta, val_weights)


unittest.main()

