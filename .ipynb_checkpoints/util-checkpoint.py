from config import *

def process(ele):
    """Cuts to -80 dB and normalizes images from 0 to 1"""
    ele['das'] = tf.reshape(ele['das']['dB'], [ele['height'], ele['width']])
    ele['das'] = tf.clip_by_value(ele['das'], -80, 0)
    ele['das'] = (ele['das'] - tf.reduce_min(ele['das']))/(tf.reduce_max(ele['das']) - tf.reduce_min(ele['das']))
    ele['dtce'] = tf.reshape(ele['dtce'], [ele['height'], ele['width']])
    ele['dtce'] = (ele['dtce'] - tf.reduce_min(ele['dtce']))/(tf.reduce_max(ele['dtce']) - tf.reduce_min(ele['dtce']))
    return ele


