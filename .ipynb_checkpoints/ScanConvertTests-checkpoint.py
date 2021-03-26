from config import *
from util import *
from scan_convert import *


class ScanConvertUnitTest(unittest.TestCase):
    def test_load_dataset(self):
        self.assertEqual(len(ele['das'].shape), 2)
        self.assertEqual(len(ele['dtce'].shape), 2)
        
    def test_shapes(self):
        self.assertEqual(len(empty_res_image.shape), 2)
        self.assertEqual(len(points_xy.shape), 2)
        self.assertEqual(len(val_rtheta.shape), 3)
        self.assertEqual(len(val_weights.shape), 2)
        self.assertEqual(len(res.shape), 2)
        
    def test_res_image_dimension(self): 
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
        summed_weights = tf.reduce_sum(val_weights, axis=1)
        min_pix = tf.reduce_min(summed_weights).numpy()
        max_pix = tf.reduce_max(summed_weights).numpy()
        self.assertAlmostEqual(min_pix, 1.0, places=5)
        self.assertAlmostEqual(max_pix, 1.0, places=5)
        
    def test_res_bounds(self):
        min_pix = tf.reduce_min(res)
        max_pix = tf.reduce_max(res)
        self.assertLessEqual(max_pix, 1.0)
        self.assertGreaterEqual(min_pix, 0.0)

    

ds = tfds.load('duke_ultrasound', data_dir='gs://tfds-data/datasets')
test_dataset = ds['MARK'].map(process)
test_iter = iter(test_dataset)
ele = next(test_iter)

empty_res_image, points_xy, val_rtheta, val_weights = scan_convert_precompute(
    ele['dtce'], ele, ele['dtce'].shape[0], ele['dtce'].shape[1])
res = scan_convert_scratch(ele['dtce'], empty_res_image, ele, points_xy, val_rtheta, val_weights)

unittest.main()

