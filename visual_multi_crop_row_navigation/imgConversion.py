#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# numpy-only replacement for the parts of cv_bridge used by this package.
# The Humble cv_bridge is built against numpy 1.x and crashes with numpy 2,
# which the ZED SDK (pyzed) requires.
import numpy as np
from sensor_msgs.msg import Image


class CvBridgeError(TypeError):
    pass


_CHANNELS = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1,
             'mono16': 1, '16UC1': 1, '32FC1': 1, '8UC1': 1, '8UC3': 3}
_DTYPES = {'mono16': np.uint16, '16UC1': np.uint16, '32FC1': np.float32}


class CvBridge:

    def imgmsg_to_cv2(self, msg, desired_encoding='passthrough'):
        if msg.encoding not in _CHANNELS:
            raise CvBridgeError(f'unsupported encoding {msg.encoding}')
        dtype = np.dtype(_DTYPES.get(msg.encoding, np.uint8))
        if msg.is_bigendian:
            dtype = dtype.newbyteorder('>')
        channels = _CHANNELS[msg.encoding]
        img = np.ndarray((msg.height, msg.width, channels), dtype=dtype, buffer=bytes(msg.data),
                         strides=(msg.step, dtype.itemsize * channels, dtype.itemsize))
        img = img[:, :, 0] if channels == 1 else img
        if desired_encoding == 'passthrough' or desired_encoding == msg.encoding:
            return img.copy()
        if desired_encoding == 'bgr8':
            if msg.encoding in ('rgb8', 'rgba8'):
                return np.ascontiguousarray(img[:, :, 2::-1])
            if msg.encoding in ('bgr8', 'bgra8'):
                return np.ascontiguousarray(img[:, :, :3])
            if msg.encoding == 'mono8':
                return np.ascontiguousarray(np.repeat(img[:, :, None], 3, axis=2))
        raise CvBridgeError(f'cannot convert {msg.encoding} to {desired_encoding}')

    def cv2_to_imgmsg(self, img, encoding='passthrough'):
        img = np.ascontiguousarray(img)
        if encoding == 'passthrough':
            encoding = {np.dtype(np.uint8): 'mono8', np.dtype(np.uint16): 'mono16',
                        np.dtype(np.float32): '32FC1'}.get(img.dtype) if img.ndim == 2 else 'bgr8'
            if encoding is None:
                raise CvBridgeError(f'unsupported image type {img.dtype}')
        msg = Image()
        msg.height, msg.width = img.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = img.strides[0]
        msg.data = img.tobytes()
        return msg
