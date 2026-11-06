#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import signal
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
from rclpy.time import Time
import math
import cv2 as cv
import numpy as np
import time

from geometry_msgs.msg import Twist
import itertools

import tf2_ros
import tf2_geometry_msgs
import image_geometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, Bool, String
from std_srvs.srv import Trigger
from rclpy.qos import QoSDurabilityPolicy
from visual_multi_crop_row_navigation.imgConversion import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

# Fixed imports - add package name
from visual_multi_crop_row_navigation import Camera as cam
from visual_multi_crop_row_navigation import imageProc as imc
from visual_multi_crop_row_navigation import controller as visualServoingCtl
from visual_multi_crop_row_navigation import groundTracker as gt
from visual_multi_crop_row_navigation.featureMatching import *

class vs_nodeHandler(Node):
    
    def __init__(self):
        super().__init__('vs_node_handler')
        # print the per-image debug output (costs CPU)
        self.verbose = self.declare_parameter('verbose', False).value
        
        self.dbg("\n" + "="*60)
        self.dbg(">>> VS_NODE_HANDLER INITIALIZING <<<")
        self.dbg("="*60 + "\n")
        
        # cv bridge
        self.bridge = CvBridge()
        self.queue_size = 1

        # Declare parameters with SIMULATION defaults
        self.declare_parameter('frontColor_topic', 'front/rgb/image_raw')
        self.declare_parameter('frontDepth_topic', 'front/depth/image_raw')
        self.declare_parameter('frontCameraInfo_topic', 'front/camera_info')
        self.declare_parameter('backColor_topic', 'back/rgb/image_raw')
        self.declare_parameter('backDepth_topic', 'back/depth/image_raw')
        self.declare_parameter('backCameraInfo_topic', 'back/camera_info')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('navigationMode', 1)
        self.declare_parameter('stationaryDebug', False)
        self.declare_parameter('scanSteps', 45)
        self.declare_parameter('scanEndPoint', 1620)
        self.declare_parameter('scanStartPoint', 300)
        self.declare_parameter('scanWindowWidth', 180)
        self.declare_parameter('enable_roi', True)
        self.declare_parameter('p1', [1400, 0])
        self.declare_parameter('p2', [1920, 0])
        self.declare_parameter('p3', [1920, 1080])
        self.declare_parameter('p4', [1400, 1080])
        self.declare_parameter('p5', [0, 0])
        self.declare_parameter('p6', [520, 0])
        self.declare_parameter('p7', [520, 1080])
        self.declare_parameter('p8', [0, 1080])
        self.declare_parameter('imgResizeRatio', 100)
        self.declare_parameter('minContourArea', 100)
        self.declare_parameter('linesToPass', 1)
        self.declare_parameter('minKeypointNum', 10)
        self.declare_parameter('maxMatchingDifference', 100)
        self.declare_parameter('minMatchingDifference', 0)
        self.declare_parameter('sacleRatio', 0.8)
        self.declare_parameter('topOffset', 150)
        self.declare_parameter('bottomOffset', 150)
        self.declare_parameter('trackingBoxWidth', 250)
        self.declare_parameter('omegaScaler', 0.1)
        self.declare_parameter('maxOmega', 0.05)
        self.declare_parameter('minOmega', 0.01)
        self.declare_parameter('maxLinearVel', 0.5)
        self.declare_parameter('minLinearVel', 0.01)
        # forward speed while tracking a row, and while no row is detected
        self.declare_parameter('linearVel', 0.2)
        self.declare_parameter('exitLinearVel', 0.05)
        # camera mounting w.r.t. the robot (defaults: original phenobot values)
        self.declare_parameter('camera_height', 1.0)            # m above ground
        self.declare_parameter('camera_tilt', np.deg2rad(-80))  # rad, negative = pitched down
        self.declare_parameter('camera_forward_offset', 0.0)    # m, camera ahead of the rotation centre
        # false: stop at the end of the row instead of the lateral lane switch,
        # for robots that cannot drive sideways (no linear.y)
        self.declare_parameter('enable_row_switching', True)
        self.declare_parameter('camera_lateral_offset', 0.0)    # m, camera left of the robot center
        # height of the detected plant centers (canopy), the plane they are projected on
        self.declare_parameter('crop_plane_height', 0.0)
        # camera mounting errors (rad): yaw > 0 turned left, roll > 0 right side down
        self.declare_parameter('camera_yaw', 0.0)
        self.declare_parameter('camera_roll', 0.0)
        # images wider than this are downscaled before processing (0: never). All pixel
        # parameters (scan range, offsets, box width, contour area) refer to the processed image
        self.declare_parameter('processing_width', 0)
        # vegetation segmentation and row points
        self.declare_parameter('point_extraction', 'bands')     # 'bands' or 'contours' (previous)
        self.declare_parameter('band_height', 12)               # px, one row point per band
        self.declare_parameter('exg_normalized', True)          # brightness independent green index
        self.declare_parameter('exg_min', 0.1)                  # minimal green level (0..1), bare soil
        self.declare_parameter('exg_blur', 5)                   # px, gaussian blur before the threshold (0: none)
        self.declare_parameter('min_run_width', 3)              # px, narrowest vegetation run giving a point
        self.declare_parameter('band_fill', 0.3)                # share of a band's height a column must be green
        # steering: 'pure_pursuit' on the lane center (metres) or 'visual_servo' (previous)
        self.declare_parameter('controller', 'pure_pursuit')
        self.declare_parameter('lookahead_distance', 2.5)       # m, pure pursuit
        self.declare_parameter('lane_offset', 0.0)              # m, robot target left of the lane center

        # ground-plane row tracking (groundTracker.RowTracker); false: image only (previous tracker)
        self.declare_parameter('ground_tracking', True)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('min_rows', 2)                   # rows needed for a lane center
        self.declare_parameter('furrow_width', 0.0)             # m between rows, 0: measured at start
        self.declare_parameter('furrow_tolerance', 0.25)        # allowed row spacing error, fraction
        self.declare_parameter('min_furrow_width', 0.15)        # m, smaller spacing = wrong detection
        self.declare_parameter('max_row_angle_spread', 10.0)    # deg between rows at initialization
        self.declare_parameter('max_furrow_width', 1.2)         # m, longest row spacing searched
        self.declare_parameter('lane_rows', 0)                  # rows of the lane (bed), 0: all evenly spaced
        self.declare_parameter('init_confirm_frames', 3)        # same lane found in N images to start
        # end of the row: no plant ahead in row_end_frames images, then drive on (odometry)
        # until the robot is row_end_offset past the last plant
        self.declare_parameter('row_end_frames', 5)
        self.declare_parameter('row_end_distance', 0.5)         # m driven without plants ahead too
        self.declare_parameter('row_end_offset', 0.0)           # m, base_link past the last plant
        self.declare_parameter('history_size', 100)             # plant centers kept per row
        self.declare_parameter('history_length', 3.0)           # m kept behind the camera view
        self.declare_parameter('merge_radius', 0.04)            # m, same plant seen again
        self.declare_parameter('gate_ratio', 0.25)              # max plant distance to its row, x furrow
        self.declare_parameter('min_gate', 0.05)                # m
        self.declare_parameter('max_lateral_jump', 0.05)        # m between two images
        self.declare_parameter('max_lateral_error', 0.0)        # m, 0: furrow_width / 2
        self.declare_parameter('max_heading_error', 15.0)       # deg
        self.declare_parameter('max_bad_frames', 5)             # rejected images in a row before lost
        self.declare_parameter('max_row_gap', 1.0)              # m driven without seeing a row
        self.declare_parameter('max_curvature', 0.2)            # 1/m, sharpest row bend fitted (0: straight lines)
        self.declare_parameter('row_consistency', 0.35)         # furrow, a row further from its place in the lane is not used
        self.declare_parameter('min_row_points', 4)
        # recovery when the rows are lost: stop, scan, creep forward, give up
        self.declare_parameter('recovery_scan_time', 1.0)       # s scanning while stopped
        self.declare_parameter('recovery_step', 0.10)           # m driven between two scans
        self.declare_parameter('recovery_max_distance', 1.0)    # m driven before giving up
        # watchdogs
        self.declare_parameter('image_timeout', 0.5)            # s without a new image: stop
        self.declare_parameter('max_omega_rate', 0.5)           # rad/s^2

        # Get parameter values
        self.frontColor_topic = self.get_parameter('frontColor_topic').value
        self.frontDepth_topic = self.get_parameter('frontDepth_topic').value
        self.frontCameraInfo_topic = self.get_parameter('frontCameraInfo_topic').value
        self.backColor_topic = self.get_parameter('backColor_topic').value
        self.backDepth_topic = self.get_parameter('backDepth_topic').value
        self.backCameraInfo_topic = self.get_parameter('backCameraInfo_topic').value
        self.stationaryDebug = self.get_parameter('stationaryDebug').value

        self.dbg(f"[DEBUG] Subscribed to topics:")
        self.dbg(f"  frontColor: {self.frontColor_topic}")
        self.dbg(f"  frontDepth: {self.frontDepth_topic}")
        self.dbg(f"  backColor: {self.backColor_topic}")
        self.dbg(f"  backDepth: {self.backDepth_topic}")

        # Create QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.queue_size
        )

        # Subscribers with synchronisation (using TimeSynchronizer pattern)
        # Front camera
        self.frontColor_sub = self.create_subscription(
            Image, self.frontColor_topic, self.frontColor_callback, sensor_qos)
        self.frontDepth_sub = self.create_subscription(
            Image, self.frontDepth_topic, self.frontDepth_callback, sensor_qos)
        self.frontCameraInfo_sub = self.create_subscription(
            CameraInfo, self.frontCameraInfo_topic, self.frontCameraInfo_callback, sensor_qos)
        
        # Back camera
        self.backColor_sub = self.create_subscription(
            Image, self.backColor_topic, self.backColor_callback, sensor_qos)
        self.backDepth_sub = self.create_subscription(
            Image, self.backDepth_topic, self.backDepth_callback, sensor_qos)
        self.backCameraInfo_sub = self.create_subscription(
            CameraInfo, self.backCameraInfo_topic, self.backCameraInfo_callback, sensor_qos)
        
        # Store latest messages for synchronisation
        self.front_rgb = None
        self.front_depth = None
        self.front_camera_info = None
        self.back_rgb = None
        self.back_depth = None
        self.back_camera_info = None

        # Initialize ros publisher, ros subscriber, topics we publish
        self.graphic_pub = self.create_publisher(Image, 'vs_nav/graphic', 1)
        self.mask_pub = self.create_publisher(Image, 'vs_nav/mask', 1)
        self.exg_pub = self.create_publisher(Image, 'vs_nav/ExG', 1)

        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.velocity_pub = self.create_publisher(Twist, cmd_vel_topic, 1)
        self.dbg(f"[DEBUG] Publishing velocity to: {cmd_vel_topic}")

        # hand-over to the headland / turning navigation: true once the row is finished
        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.row_end_pub = self.create_publisher(Bool, 'vs_nav/row_end', latched_qos)
        self.row_end_pub.publish(Bool(data=False))
        self.create_service(Trigger, 'vs_nav/start', self.start_callback)
        # state and reason, e.g. "SCAN: rows lost (...)", "TRACKING: 3/3 rows, lateral +2 cm"
        self.status_pub = self.create_publisher(String, 'vs_nav/status', latched_qos)

        # settings
        self.navigationMode = self.get_parameter('navigationMode').value                   
        self.rotationDir = 1
        
        # direction of motion 1: forward, -1:backward
        if self.navigationMode == 1 or self.navigationMode == 2:
            self.linearMotionDir = 1
        else:
            self.linearMotionDir = -1
            
        # true: front camera, False: back camera
        if self.isUsingFrontCamera(): 
            self.primaryCamera = True
        else:
            self.primaryCamera = False
            
        self.omegaBuffer = list()

        self.scannerParams = {
            "scanSteps": self.get_parameter('scanSteps').value,
            "scanEndPoint": self.get_parameter('scanEndPoint').value,
            "scanStartPoint": self.get_parameter('scanStartPoint').value,
            "scanWindowWidth": self.get_parameter('scanWindowWidth').value
        }
        
        self.rioParams = {
            "enable_roi": self.get_parameter('enable_roi').value,
            "p1": self.get_parameter('p1').value,
            "p2": self.get_parameter('p2').value,
            "p3": self.get_parameter('p3').value,
            "p4": self.get_parameter('p4').value,
            "p5": self.get_parameter('p5').value,
            "p6": self.get_parameter('p6').value,
            "p7": self.get_parameter('p7').value,
            "p8": self.get_parameter('p8').value
        }

        self.contourParams = {
            "imgResizeRatio": self.get_parameter('imgResizeRatio').value,
            "minContourArea": self.get_parameter('minContourArea').value,
            "pointExtraction": self.get_parameter('point_extraction').value,
            "bandHeight": self.get_parameter('band_height').value,
            "exgNormalized": self.get_parameter('exg_normalized').value,
            "exgMin": self.get_parameter('exg_min').value,
            "exgBlur": self.get_parameter('exg_blur').value,
            "minRunWidth": self.get_parameter('min_run_width').value,
            "bandFill": self.get_parameter('band_fill').value,
        }

        self.featureParams = {
            "linesToPass": self.get_parameter('linesToPass').value,
            "minKeypointNum": self.get_parameter('minKeypointNum').value,
            "maxMatchingDifference": self.get_parameter('maxMatchingDifference').value,
            "minMatchingDifference": self.get_parameter('minMatchingDifference').value,
        }

        self.trackerParams = {
            "sacleRatio": self.get_parameter('sacleRatio').value,
            "topOffset": self.get_parameter('topOffset').value,
            "bottomOffset": self.get_parameter('bottomOffset').value,
            "trackingBoxWidth": self.get_parameter('trackingBoxWidth').value,
            "minRows": self.get_parameter('min_rows').value,
        }
        self.groundParams = {
            "minRows": self.get_parameter('min_rows').value,
            "minPoints": self.get_parameter('min_row_points').value,
            "furrowWidth": self.get_parameter('furrow_width').value,
            "furrowTolerance": self.get_parameter('furrow_tolerance').value,
            "minFurrowWidth": self.get_parameter('min_furrow_width').value,
            "maxRowAngleSpread": self.get_parameter('max_row_angle_spread').value,
            "historySize": self.get_parameter('history_size').value,
            "historyLength": self.get_parameter('history_length').value,
            "mergeRadius": self.get_parameter('merge_radius').value,
            "maxPointWeight": 10.0,
            "gateRatio": self.get_parameter('gate_ratio').value,
            "minGate": self.get_parameter('min_gate').value,
            "maxLateralJump": self.get_parameter('max_lateral_jump').value,
            "maxLateralError": self.get_parameter('max_lateral_error').value,
            "maxHeadingError": self.get_parameter('max_heading_error').value,
            "maxBadFrames": self.get_parameter('max_bad_frames').value,
            "maxRowGap": self.get_parameter('max_row_gap').value,
            "maxCurvature": self.get_parameter('max_curvature').value,
            "rowConsistency": self.get_parameter('row_consistency').value,
            "maxFurrowWidth": self.get_parameter('max_furrow_width').value,
            "laneRows": self.get_parameter('lane_rows').value,
            "initConfirmFrames": self.get_parameter('init_confirm_frames').value,
            "rowEndFrames": self.get_parameter('row_end_frames').value,
            "rowEndDistance": self.get_parameter('row_end_distance').value,
            # lane center position in the robot frame when the robot is on target: pure
            # pursuit drives it to -lane_offset, the visual servoing brings the camera above it
            "targetOffset": (-self.get_parameter('lane_offset').value
                             if self.get_parameter('controller').value == 'pure_pursuit'
                             else self.get_parameter('camera_lateral_offset').value),
        }

        self.omegaScaler = self.get_parameter('omegaScaler').value
        self.maxOmega = self.get_parameter('maxOmega').value
        self.minOmega = self.get_parameter('minOmega').value
        self.maxLinearVel = self.get_parameter('maxLinearVel').value
        self.minLinearVel = self.get_parameter('minLinearVel').value
        self.linearVel = self.get_parameter('linearVel').value
        self.exitLinearVel = self.get_parameter('exitLinearVel').value
        self.enableRowSwitching = self.get_parameter('enable_row_switching').value
        # false once the row is finished: no more velocity commands until vs_nav/start
        self.active = True
        self.odomFrame = self.get_parameter('odom_frame').value
        self.baseFrame = self.get_parameter('base_frame').value
        self.recoveryScanTime = self.get_parameter('recovery_scan_time').value
        self.recoveryStep = self.get_parameter('recovery_step').value
        self.recoveryMaxDistance = self.get_parameter('recovery_max_distance').value
        self.imageTimeout = self.get_parameter('image_timeout').value
        self.processingWidth = self.get_parameter('processing_width').value
        self.controller = self.get_parameter('controller').value
        self.lookahead = self.get_parameter('lookahead_distance').value
        self.laneOffset = self.get_parameter('lane_offset').value
        self.rowEndOffset = self.get_parameter('row_end_offset').value
        self.exitRef = None
        self.exitTarget = None
        self.cameraK = None
        self.imageScale = None
        self.processingMs = 0.0
        self.maxOmegaRate = self.get_parameter('max_omega_rate').value
        # SCAN (stopped, looking for rows) -> TRACKING; rows lost -> SCAN / ADVANCE (10 cm)
        # -> SCAN ... ; nothing after recovery_max_distance -> FAILED
        self.state = 'SCAN'
        self.statusText = ''
        self.scanStart = None
        self.recoveryOrigin = None
        self.advanceOrigin = None
        self.advanceStart = None
        self.lastCmd = (0.0, 0.0)
        self.lastCmdTime = None
        self.lastHold = None
        # images: arrival time (ROS clock) and a counter, to process each image once
        self.frontImgTime = None
        self.backImgTime = None
        self.frontImgSeq = 0
        self.backImgSeq = 0
        self.frontImgStamp = None
        self.backImgStamp = None
        self.processedSeq = None

        # images
        self.frontImg = None
        self.frontDepth = None
        self.backImg = None
        self.backDepth = None
        self.frontCameraInfo = None
        self.backCameraInfo = None

        self.velocityMsg = Twist()
        self.enoughPoints = True
        self.samplingDone = False
        self.imgHeight = 1080
        self.imgWidth = 1920

        # camera
        self.groundProjector = gt.GroundProjector(self.get_parameter('camera_forward_offset').value,
                                                  self.get_parameter('camera_lateral_offset').value,
                                                  self.get_parameter('camera_height').value
                                                  - self.get_parameter('crop_plane_height').value,
                                                  self.get_parameter('camera_tilt').value,
                                                  self.get_parameter('camera_yaw').value,
                                                  self.get_parameter('camera_roll').value)
        self.camera = cam.Camera(1, 1.2,
                                 self.get_parameter('camera_forward_offset').value,
                                 self.get_parameter('camera_height').value,
                                 self.get_parameter('camera_tilt').value,
                                 0.96, 0, 0, 1)
        self.imageProcessor = imc.imageProc(self.scannerParams,
                                            self.contourParams,
                                            self.rioParams,
                                            self.trackerParams)
        self.imageProcessor.verbose = self.verbose

        self.cameraModel = image_geometry.PinholeCameraModel()
        
        # Initialize TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.featureMatcher = featureMatching(self.featureParams)

        # Timer for main loop (similar to ROS1's spin)
        self.create_timer(0.033, self.navigate)  # ~30 Hz
        
        self.get_logger().info('Detection Camera initialised..')
        self.get_logger().info('#[VS] navigator initialied ... ')
        self.dbg("[DEBUG] Node initialization complete! Timer started.\n")
    
    # Callbacks for front camera
    def frontColor_callback(self, msg):
        try:
            self.frontImg = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.frontImgTime = self.nowSec()
            self.frontImgStamp = msg.header.stamp
            self.frontImgSeq += 1
            self.dbg(f"[DEBUG] Front RGB image received - shape: {self.frontImg.shape}")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
    
    def frontDepth_callback(self, msg):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.frontDepth = np.array(cv_depth, dtype=np.float32)
            cv.normalize(self.frontDepth, self.frontDepth, 0.0, 1.0, cv.NORM_MINMAX)
            self.dbg(f"[DEBUG] Front Depth image received - shape: {self.frontDepth.shape}")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
    
    def frontCameraInfo_callback(self, msg):
        self.frontCameraInfo = msg
        self.cameraModel.fromCameraInfo(msg)
        if self.imageProcessor.groundTracker is None and self.get_parameter('ground_tracking').value:
            # intrinsics known: the rows can be tracked on the ground
            self.cameraK = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.groundProjector.setIntrinsics(*self.cameraK)
            self.imageProcessor.groundTracker = gt.RowTracker(self.groundProjector, self.groundParams)
        self.dbg("[DEBUG] Front CameraInfo received")
    
    # Callbacks for back camera
    def backColor_callback(self, msg):
        try:
            self.backImg = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.backImgTime = self.nowSec()
            self.backImgStamp = msg.header.stamp
            self.backImgSeq += 1
            self.dbg(f"[DEBUG] Back RGB image received - shape: {self.backImg.shape}")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
    
    def backDepth_callback(self, msg):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.backDepth = np.array(cv_depth, dtype=np.float32)
            cv.normalize(self.backDepth, self.backDepth, 0.0, 1.0, cv.NORM_MINMAX)
            self.dbg(f"[DEBUG] Back Depth image received - shape: {self.backDepth.shape}")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
    
    def backCameraInfo_callback(self, msg):
        self.backCameraInfo = msg
        self.dbg("[DEBUG] Back CameraInfo received")
        
    # main function to guide the robot through crop rows
    def navigate(self):
        # Counter to reduce print spam
        if not hasattr(self, 'nav_counter'):
            self.nav_counter = 0
        self.nav_counter += 1

        if not self.active:
            return

        # Check if we have all required data (NO DEPTH REQUIRED!)
        if self.primaryCamera:
            primaryRGB, info, imgTime, imgSeq, imgStamp = (self.frontImg, self.frontCameraInfo, self.frontImgTime,
                                                           self.frontImgSeq, self.frontImgStamp)
        else:
            primaryRGB, info, imgTime, imgSeq, imgStamp = (self.backImg, self.backCameraInfo, self.backImgTime,
                                                           self.backImgSeq, self.backImgStamp)
        if primaryRGB is None or info is None or \
                (self.imageProcessor.groundTracker is None and self.get_parameter('ground_tracking').value):
            if self.nav_counter % 100 == 1:
                self.get_logger().info(f"Waiting for camera data - RGB: {primaryRGB is not None}, Info: {info is not None}")
            return
        primaryRGB = self.scaleImage(primaryRGB)
        self.imgHeight, self.imgWidth = primaryRGB.shape[:2]

        # watchdogs: never drive on an old image or without odometry
        now = self.nowSec()
        if now - imgTime > self.imageTimeout:
            self.holdStop("no camera image for %.1f s" % (now - imgTime))
            return
        # pose when the image was taken: the robot may have turned since
        pose = self.lookupPose(imgStamp)
        if pose is None:
            self.holdStop("no %s -> %s transform" % (self.odomFrame, self.baseFrame))
            return
        self.lastHold = None
        self.imageProcessor.robotPose = pose

        if self.state == 'ADVANCE':
            self.advance(pose, now)
            return
        if imgSeq == self.processedSeq:
            # no new image yet: repeat the last command, the drive controller drops stale ones
            if self.state in ('TRACKING', 'EXIT'):
                self.setRobotVelocities(self.lastCmd[0], 0.0, self.lastCmd[1])
            else:
                self.setRobotVelocities(0.0, 0.0, 0.0)
            return
        self.processedSeq = imgSeq

        # scans for the rows if not initialized, otherwise only detects the plants
        t0 = time.monotonic()
        found = self.imageProcessor.findCropLane(primaryRGB, None, mode='RGB')
        if self.state == 'SCAN':
            self.scan(found, pose, now)
        else:
            self.track(found, pose, now)
        self.processingMs = 1000.0 * (time.monotonic() - t0)
        self.publishImageTopics()

    def scan(self, found, pose, now):
        """robot stopped, looking for at least min_rows evenly spaced rows"""
        self.setRobotVelocities(0.0, 0.0, 0.0)
        if self.recoveryOrigin is None:
            self.recoveryOrigin = pose
        if self.scanStart is None:
            self.scanStart = now
        if found:
            # first image after the scan: initializes the ground tracker (spacing, parallel rows)
            self.imageProcessor.trackCropLane(self.navigationMode)
            tracker = self.imageProcessor.groundTracker
            if (tracker is None or tracker.initialized) and self.imageProcessor.cropLaneFound:
                self.recoveryOrigin = None
                self.scanStart = None
                self.lastCmd = (0.0, 0.0)
                self.setState('TRACKING', self.imageProcessor.trackStatus)
                return
        moved = self.distance(pose, self.recoveryOrigin)
        self.setState('SCAN', "%s, %.0f cm driven since the rows were lost"
                      % (self.imageProcessor.trackStatus, 100 * moved))
        if now - self.scanStart < self.recoveryScanTime:
            return
        if moved >= self.recoveryMaxDistance - self.recoveryStep / 2:
            self.fail("no %d crop rows found within %.1f m"
                      % (self.imageProcessor.minRows, self.recoveryMaxDistance))
            return
        self.advanceOrigin = pose
        self.advanceStart = now
        self.setState('ADVANCE', "no rows found, driving %.0f cm forward (%.0f cm driven so far)"
                      % (100 * self.recoveryStep, 100 * moved))

    def advance(self, pose, now):
        """creep straight forward by recovery_step, then scan again"""
        if self.distance(pose, self.advanceOrigin) >= self.recoveryStep:
            self.setRobotVelocities(0.0, 0.0, 0.0)
            self.scanStart = now
            self.setState('SCAN', "scanning after a %.0f cm step" % (100 * self.recoveryStep))
            return
        timeout = 3.0 * self.recoveryStep / max(self.exitLinearVel, 1e-3) + 2.0
        if now - self.advanceStart > timeout:
            self.fail("the robot did not drive %.0f cm in %.0f s" % (100 * self.recoveryStep, timeout))
            return
        self.setRobotVelocities(self.exitLinearVel * self.linearMotionDir, 0.0, 0.0)

    def track(self, found, pose, now):
        if not found:
            if self.isExitingLane():
                # no rows left while exiting: the row is done
                if not self.enableRowSwitching:
                    self.finishRow("crop rows lost")
            elif not self.isSwitchingLane():
                self.startRecovery(pose, now, self.imageProcessor.trackStatus or "crop rows lost")
            return

        # if the robot is currently following a line and is not turning just compute the controls
        if self.isFollowingLane():
            tracker = self.imageProcessor.groundTracker
            self.imageProcessor.exiting = (self.state == 'EXIT')
            self.imageProcessor.trackCropLane(self.navigationMode)
            if self.imageProcessor.trackingLost and not self.isExitingLane():
                self.startRecovery(pose, now, self.imageProcessor.trackStatus)
                return
            # ground tracker, no lane switch: the row ends where its last plant is
            groundRowEnd = tracker is not None and not self.enableRowSwitching and not self.isExitingLane()
            if groundRowEnd:
                if self.state == 'TRACKING' and self.imageProcessor.cropRowEnd:
                    self.startExit(pose)
                elif self.state == 'EXIT' and tracker.rowContinues():
                    # only a gap in the rows: back to normal tracking
                    self.imageProcessor.exiting = False
                    self.setState('TRACKING', 'rows go on after a gap')
                if self.state == 'EXIT' and tracker.farthestOdom is not None:
                    # last plant seen again, from closer: better position (the canopy is above
                    # the projection plane, far plants look further away than they are)
                    last = self.alongExit(tracker.farthestOdom)
                    self.exitTarget = min(self.exitTarget, last + self.rowEndOffset)
                if self.state == 'EXIT':
                    # the last plants are still in the camera view: they must be seen, otherwise
                    # the camera is blocked or blinded, it is not the end of the row
                    lastPlant = self.exitTarget - self.rowEndOffset - self.alongExit(pose)
                    xNear = tracker.viewRange(self.imgHeight, self.trackerParams["topOffset"])[0]
                    if lastPlant > xNear + 0.3 and tracker.blind():
                        self.imageProcessor.exiting = False
                        self.startRecovery(pose, now, "no crop row in view although the last plants are %.1f m ahead"
                                           % lastPlant)
                        return
                    remaining = self.exitTarget - self.alongExit(pose)
                    # plants ahead again (end of a gap, the rows are being found): not the end
                    if remaining <= 0.0 and tracker.rowEnd():
                        self.finishRow("end of row, %.2f m past the last plant" % self.rowEndOffset)
                        return
                    self.setState('EXIT', "no plants ahead, %.2f m to go (%s)"
                                  % (remaining, self.imageProcessor.trackStatus))
            ctlCommands = self.computeControls(self.imageProcessor.cropLaneFound,
                                            self.imageProcessor.P,
                                            self.imageProcessor.ang)

            if groundRowEnd or not self.imageProcessor.cropRowEnd:
                # limit the change of the rotational velocity (steering lag, no jerks)
                omega = ctlCommands[1]
                if self.lastCmdTime is not None:
                    dw = self.maxOmegaRate * min(max(now - self.lastCmdTime, 0.0), 0.5)
                    omega = float(np.clip(omega, self.lastCmd[1] - dw, self.lastCmd[1] + dw))
                self.lastCmd = (ctlCommands[0], omega)
                self.lastCmdTime = now
                self.setRobotVelocities(ctlCommands[0], 0.0, omega)
                if self.state == 'TRACKING':
                    self.setState('TRACKING', self.imageProcessor.trackStatus)

            elif self.imageProcessor.cropRowEnd:
                self.get_logger().info("End of Lane detected ...")

                if self.isExitingLane():
                    self.updateNavigationStage()
                else:
                    self.updateNavigationStage()
                    self.imageProcessor.reset()
                    self.imageProcessor.isInitialized = False
                    self.imageProcessor.cropRowEnd = False
                    self.switchCamera()

        elif self.isSwitchingLane() and not self.enableRowSwitching:
            # the lane switch drives sideways, which this robot cannot do
            self.finishRow("end of row")

        elif self.isSwitchingLane() and not self.samplingDone:
            self.get_logger().info("Sampling the Lane!!")
            self.stopRobot(2.0)
            # Compute the features for the turning and stop the movement
            self.featureMatcher.sampleCropRowFeatures(self.navigationMode,
                                                        self.imageProcessor.primaryRGBImg,
                                                        self.imageProcessor.greenIDX,
                                                        self.imageProcessor.mask,
                                                        self.imageProcessor.rowTrackingBoxes)
            self.samplingDone = True
            self.stopRobot(2.0)
        else:
            # test if the condition for the row switching is fulfilled
            foundNewCropLane = self.featureMatcher.detectNewCropLane(self.navigationMode,
                                                            self.imageProcessor.primaryRGBImg,
                                                            self.imageProcessor.greenIDX,
                                                            self.imageProcessor.mask,
                                                            self.imageProcessor.rowTrackingBoxes,
                                                            self.imageProcessor.numOfCropRows)
            if foundNewCropLane:
                self.stopRobot(2.0)
                self.get_logger().info("following new Lane !!")
                # the turn is completed and the new lines to follow are computed
                self.switchDirection()
                self.get_logger().info("Turning Mode disabled, Entering next lane")
                self.updateNavigationStage()
                self.imageProcessor.reset()
                self.imageProcessor.isInitialized = False
                self.imageProcessor.cropRowEnd = False
            else:
                # if the condition is not fulfilled the robot moves continuously sidewards
                self.setRobotVelocities(0.0, -0.05, 0.0)
                if self.nav_counter % 100 == 1:
                    self.get_logger().info("Side motion to find New Lane ...")

    def startExit(self, pose):
        """no plants ahead: the last plant is known (odom), drive on until past it"""
        tracker = self.imageProcessor.groundTracker
        last = tracker.lastPlantAlong()
        self.exitRef = tracker.centerOdom
        self.exitTarget = (last if last is not None else 0.0) + self.rowEndOffset
        togo = self.exitTarget - self.alongExit(pose)
        self.get_logger().info("No plants ahead: end of the row in %.2f m" % togo)
        self.setState('EXIT', "no plants ahead, end of the row in %.2f m" % togo)

    def alongExit(self, pose):
        """robot (base_link) position along the lane, from the reference taken at the row end"""
        ox, oy, direction = self.exitRef
        return (pose[0] - ox) * math.cos(direction) + (pose[1] - oy) * math.sin(direction)

    def scaleImage(self, img):
        """downscale to processing_width, with the intrinsics of the ground projector"""
        scale = 1.0
        if self.processingWidth and img.shape[1] > self.processingWidth:
            scale = self.processingWidth / img.shape[1]
            img = cv.resize(img, (self.processingWidth, int(round(img.shape[0] * scale))),
                            interpolation=cv.INTER_AREA)
        if scale != self.imageScale and self.cameraK is not None:
            fx, fy, cx, cy = self.cameraK
            self.groundProjector.setIntrinsics(fx * scale, fy * scale, cx * scale, cy * scale)
            self.imageScale = scale
        return img

    def startRecovery(self, pose, now, reason):
        """rows lost: stop and scan; the distance driven counts from the first loss"""
        self.setRobotVelocities(0.0, 0.0, 0.0)
        self.lastCmd = (0.0, 0.0)
        self.lastCmdTime = None
        if self.recoveryOrigin is None:
            self.recoveryOrigin = pose
        self.imageProcessor.reset()
        self.scanStart = now
        self.get_logger().warn(f"Crop rows lost ({reason}): robot stopped, scanning")
        self.setState('SCAN', "rows lost (%s)" % reason)

    def fail(self, reason):
        # stays stopped until vs_nav/start
        self.setRobotVelocities(0.0, 0.0, 0.0)
        self.active = False
        self.get_logger().error(f"Row following FAILED: {reason}. Robot stopped, waiting for vs_nav/start")
        self.setState('FAILED', reason)

    def holdStop(self, reason):
        """watchdog: stop without changing the navigation state"""
        self.setRobotVelocities(0.0, 0.0, 0.0)
        self.lastCmd = (0.0, 0.0)
        self.lastCmdTime = None
        if self.lastHold != reason.split(' for ')[0]:
            self.lastHold = reason.split(' for ')[0]
            self.get_logger().warn(f"Watchdog: {reason}, robot stopped")
        self.status_pub.publish(String(data=f"STOPPED: {reason} ({self.state})"))

    def setState(self, state, detail):
        if state != self.state:
            self.get_logger().info(f"{self.state} -> {state}: {detail}")
        self.state = state
        self.statusText = f"{state}: {detail}"
        self.status_pub.publish(String(data=f"{self.statusText} [{self.processingMs:.0f} ms]"))

    def lookupPose(self, stamp=None):
        """robot pose (x, y, yaw) in the odom frame, at the image time if available"""
        tf = None
        if stamp is not None:
            try:
                tf = self.tf_buffer.lookup_transform(self.odomFrame, self.baseFrame, Time.from_msg(stamp))
            except tf2_ros.TransformException:
                tf = None
        try:
            if tf is None:
                tf = self.tf_buffer.lookup_transform(self.odomFrame, self.baseFrame, Time())
        except tf2_ros.TransformException as e:
            if self.nav_counter % 100 == 1:
                self.get_logger().error(f"lookup_transform {self.odomFrame} -> {self.baseFrame} failed: {e}")
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, yaw)

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def nowSec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def dbg(self, *args):
        if self.verbose:
            print(*args)

    def publishImageTopics(self):
        now = self.get_clock().now().to_msg()
        frame = 'front_camera_link'

        # Publish the Graphics image
        self.imageProcessor.drawGraphics()
        if hasattr(self.imageProcessor, 'graphicsImg') and self.imageProcessor.graphicsImg is not None:
            # navigation state on top of the image
            for color, width in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv.putText(self.imageProcessor.graphicsImg, self.statusText[:90], (10, 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, color, width, cv.LINE_AA)
            graphic_img = self.bridge.cv2_to_imgmsg(self.imageProcessor.graphicsImg, encoding='bgr8')
            graphic_img.header.stamp = now
            graphic_img.header.frame_id = frame
            self.graphic_pub.publish(graphic_img)

        # publish predicted Mask
        if hasattr(self.imageProcessor, 'mask') and self.imageProcessor.mask is not None:
            mask_msg = self.bridge.cv2_to_imgmsg(self.imageProcessor.mask)
            mask_msg.header.stamp = now
            mask_msg.header.frame_id = frame
            self.mask_pub.publish(mask_msg)

        # publish Exg image
        if hasattr(self.imageProcessor, 'greenIDX') and self.imageProcessor.greenIDX is not None:
            exg_msg = self.bridge.cv2_to_imgmsg(self.imageProcessor.greenIDX)
            exg_msg.header.stamp = now
            exg_msg.header.frame_id = frame
            self.exg_pub.publish(exg_msg)

    def setRobotVelocities(self, x, y, omega):
        self.velocityMsg = Twist()
        self.velocityMsg.linear.x = float(x)
        self.velocityMsg.linear.y = float(y)
        self.velocityMsg.angular.z = float(omega)
        if not self.stationaryDebug:
            if self.velocityMsg is not None:
                self.velocity_pub.publish(self.velocityMsg)
                self.dbg(f"[DEBUG] Velocity published - linear.x={x}, angular.z={omega}")

    def stopRobot(self, delay):
        self.velocityMsg = Twist()
        self.velocityMsg.linear.x = 0.0
        self.velocityMsg.linear.y = 0.0
        self.velocityMsg.angular.z = 0.0
        time.sleep(delay)
        self.velocity_pub.publish(self.velocityMsg)
        self.dbg("[DEBUG] Robot stopped")

    def finishRow(self, reason):
        # stop once, then leave cmd_vel to the turning navigation
        self.setRobotVelocities(0.0, 0.0, 0.0)
        self.active = False
        self.row_end_pub.publish(Bool(data=True))
        self.get_logger().info(f"Row finished ({reason}): robot stopped, waiting for vs_nav/start")
        self.setState('ROW_END', reason)

    def start_callback(self, request, response):
        # follow the next row from the beginning: forward, front camera
        self.navigationMode = 1
        self.linearMotionDir = 1
        self.primaryCamera = True
        self.samplingDone = False
        self.imageProcessor.reset()
        self.imageProcessor.isInitialized = False
        self.imageProcessor.cropRowEnd = False
        if self.imageProcessor.groundTracker is not None:
            # new row: its lane is found from scratch
            self.imageProcessor.groundTracker.forgetLane()
            self.imageProcessor.groundTracker.reset()
        self.recoveryOrigin = None
        self.scanStart = None
        self.lastCmd = (0.0, 0.0)
        self.lastCmdTime = None
        self.imageProcessor.exiting = False
        self.exitRef = None
        self.active = True
        self.row_end_pub.publish(Bool(data=False))
        self.get_logger().info("Row following started")
        self.setState('SCAN', 'started, looking for the crop rows')
        response.success = True
        return response

    def updateNavigationStage(self):
        if self.imageProcessor.groundTracker is not None:
            # next stage (exit, other row): not the same lane
            self.imageProcessor.groundTracker.initialized = False
            self.imageProcessor.groundTracker.forgetLane()
        self.navigationMode += 1
        if self.navigationMode > 6:
            self.navigationMode = 1  
        self.get_logger().info(f"Switched to mode {self.navigationMode}")
    
    def isSwitchingLane(self):
        if self.navigationMode == 3 or self.navigationMode == 6:
            return True
        return False

    def isExitingLane(self):
        if self.navigationMode in [2 ,5]:
            return True
        return False
    
    def isFollowingLane(self):
        if self.navigationMode in [1,2,4,5]:
            return True
        return False
    
    def isUsingFrontCamera(self):
        if self.navigationMode == 1 or self.navigationMode == 5 or self.navigationMode == 6:
            return True
        return False
    
    def isUsingBackCamera(self):
        if self.navigationMode == 2 or self.navigationMode == 3 or self.navigationMode == 4:
            return True
        return False

    def switchDirection(self):
        self.linearMotionDir = -self.linearMotionDir
        self.get_logger().info(f"Switched Direction of Motion ... {self.linearMotionDir}")
    
    def switchRotationDir(self):
        self.rotationDir = -self.rotationDir
        self.get_logger().info(f"Switched Direction of Rotation ... {self.rotationDir}")

    def switchCamera(self):
        self.get_logger().info("Switch camera to the other ...")
        self.primaryCamera = not self.primaryCamera

    def getProcessingImage(self, frontRgb, frontDepth, backRgb, backDepth):
        if self.primaryCamera:
            primaryRgb = frontRgb
            primaryDepth = frontDepth
        else:
            primaryRgb = backRgb
            primaryDepth = backDepth
        return primaryRgb, primaryDepth
    
    def computeControls(self, LaneFound, P, Angle):
        tracker = self.imageProcessor.groundTracker
        if LaneFound and self.controller == 'pure_pursuit' and tracker is not None and tracker.center is not None:
            # lane center line in the robot frame (metres), target shifted by lane_offset
            xRef = tracker.viewRange(self.imgHeight, self.trackerParams["topOffset"])[0]
            a, b = tracker.center
            c = tracker.curvature
            # point of the target line lookahead_distance ahead (bent lane: on the curve),
            # circle arc through it
            yL = a + b * (self.lookahead - xRef) + c * (self.lookahead - xRef) ** 2 + self.laneOffset
            curvature = 2.0 * yL / (self.lookahead ** 2 + yL ** 2)
            rho = self.linearVel * self.linearMotionDir
            omega = float(np.clip(curvature * abs(rho), -self.maxOmega, self.maxOmega))
            return [rho, omega]
        if LaneFound:
            # define desired and actual feature vector
            desiredFeature = np.array([0.0, self.imgWidth/2, 0.0])
            # computeTheta measures from the image x axis (vertical row = -pi/2),
            # the control law expects the angle to the image y axis (vertical row = 0)
            actualFeature = np.array([P[0], P[1], visualServoingCtl.wrapToPi(Angle + np.pi/2)])
            # compute controls
            controls = visualServoingCtl.visualServoingCtl(self.camera,
                                                desiredFeature, 
                                                actualFeature, 
                                                self.maxLinearVel)
            if self.isExitingLane():
                self.rotationDir = -1
            else:
                self.rotationDir = 1
            # scale rotational velocity
            omega = float(np.clip(self.omegaScaler * np.squeeze(controls),
                                  -self.maxOmega, self.maxOmega))
            # set linear speed and direction
            rho = self.linearVel * self.linearMotionDir
            # store the command in a cache
            self.omegaBuffer.append(omega)
            
            return [rho, omega]  
        else:
            # using the last known control 
            if len(self.omegaBuffer) == 0:
                self.omegaBuffer.append(0.0)
            # straight exit
            omega = 0.0
            rho = self.exitLinearVel * self.linearMotionDir
            return [rho, omega] 

    def transformTargets(self, targets, frameName):
        cameraToOdomTF = None
        targetList = []
        try:
            cameraToOdomTF = self.tf_buffer.lookup_transform(
                frameName, self.cameraModel.tfFrame(), Time(seconds=0))
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"lookup_transform {frameName} to {self.cameraModel.tfFrame()} failed: {str(e)}")
        
        if cameraToOdomTF is not None:
            for i in range(len(targets)):
                pose_stamped = PoseStamped()
                pose_stamped.pose = targets[i]
                pose_stamped.header.frame_id = self.cameraModel.tfFrame()
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                stem = tf2_geometry_msgs.do_transform_pose(pose_stamped, cameraToOdomTF)
                targetList.append(stem.pose)
        return targetList

def main(args=None):
    # Ctrl+C / SIGTERM only set a flag: the loop below ends, the stop command is sent,
    # then rclpy shuts down (its own handler would shut down before the stop is sent)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(True))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    node = vs_nodeHandler()
    try:
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.1)
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        # never leave the robot driving on the last command
        node.velocity_pub.publish(Twist())
        node.get_logger().info("Stopped: zero velocity sent")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()