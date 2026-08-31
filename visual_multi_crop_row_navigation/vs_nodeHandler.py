#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
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
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

# Fixed imports - add package name
from visual_multi_crop_row_navigation import Camera as cam
from visual_multi_crop_row_navigation import imageProc as imc
from visual_multi_crop_row_navigation import controller as visualServoingCtl
from visual_multi_crop_row_navigation.featureMatching import *

class vs_nodeHandler(Node):
    
    def __init__(self):
        super().__init__('vs_node_handler')
        
        print("\n" + "="*60)
        print(">>> VS_NODE_HANDLER INITIALIZING <<<")
        print("="*60 + "\n")
        
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

        # Get parameter values
        self.frontColor_topic = self.get_parameter('frontColor_topic').value
        self.frontDepth_topic = self.get_parameter('frontDepth_topic').value
        self.frontCameraInfo_topic = self.get_parameter('frontCameraInfo_topic').value
        self.backColor_topic = self.get_parameter('backColor_topic').value
        self.backDepth_topic = self.get_parameter('backDepth_topic').value
        self.backCameraInfo_topic = self.get_parameter('backCameraInfo_topic').value
        self.stationaryDebug = self.get_parameter('stationaryDebug').value

        print(f"[DEBUG] Subscribed to topics:")
        print(f"  frontColor: {self.frontColor_topic}")
        print(f"  frontDepth: {self.frontDepth_topic}")
        print(f"  backColor: {self.backColor_topic}")
        print(f"  backDepth: {self.backDepth_topic}")

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
        print(f"[DEBUG] Publishing velocity to: {cmd_vel_topic}")

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
        }

        self.omegaScaler = self.get_parameter('omegaScaler').value
        self.maxOmega = self.get_parameter('maxOmega').value
        self.minOmega = self.get_parameter('minOmega').value
        self.maxLinearVel = self.get_parameter('maxLinearVel').value
        self.minLinearVel = self.get_parameter('minLinearVel').value
        
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
        self.camera = cam.Camera(1,1.2,0,1,np.deg2rad(-80),0.96,0,0,1)
        self.imageProcessor = imc.imageProc(self.scannerParams,
                                            self.contourParams,
                                            self.rioParams,
                                            self.trackerParams)

        self.cameraModel = image_geometry.PinholeCameraModel()
        
        # Initialize TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.featureMatcher = featureMatching(self.featureParams)

        # Timer for main loop (similar to ROS1's spin)
        self.create_timer(0.033, self.navigate)  # ~30 Hz
        
        self.get_logger().info('Detection Camera initialised..')
        self.get_logger().info('#[VS] navigator initialied ... ')
        print("[DEBUG] Node initialization complete! Timer started.\n")
    
    # Callbacks for front camera
    def frontColor_callback(self, msg):
        try:
            self.frontImg = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print(f"[DEBUG] Front RGB image received - shape: {self.frontImg.shape}")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
    
    def frontDepth_callback(self, msg):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.frontDepth = np.array(cv_depth, dtype=np.float32)
            cv.normalize(self.frontDepth, self.frontDepth, 0.0, 1.0, cv.NORM_MINMAX)
            print(f"[DEBUG] Front Depth image received - shape: {self.frontDepth.shape}")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
    
    def frontCameraInfo_callback(self, msg):
        self.frontCameraInfo = msg
        self.cameraModel.fromCameraInfo(msg)
        print("[DEBUG] Front CameraInfo received")
    
    # Callbacks for back camera
    def backColor_callback(self, msg):
        try:
            self.backImg = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print(f"[DEBUG] Back RGB image received - shape: {self.backImg.shape}")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
    
    def backDepth_callback(self, msg):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.backDepth = np.array(cv_depth, dtype=np.float32)
            cv.normalize(self.backDepth, self.backDepth, 0.0, 1.0, cv.NORM_MINMAX)
            print(f"[DEBUG] Back Depth image received - shape: {self.backDepth.shape}")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
    
    def backCameraInfo_callback(self, msg):
        self.backCameraInfo = msg
        print("[DEBUG] Back CameraInfo received")
        
    # main function to guide the robot through crop rows
    def navigate(self):
        # Counter to reduce print spam
        if not hasattr(self, 'nav_counter'):
            self.nav_counter = 0
        self.nav_counter += 1
        
        if self.nav_counter % 100 == 1:  # Print every ~100 iterations
            print(f"[DEBUG] navigate() called - iteration {self.nav_counter}")
        
        # Check if we have all required data (NO DEPTH REQUIRED!)
        if self.primaryCamera:
            # ONLY require RGB and CameraInfo, NOT depth
            if self.frontImg is None or self.frontCameraInfo is None:
                if self.nav_counter % 100 == 1:
                    print(f"[DEBUG] Waiting for front camera data - RGB: {self.frontImg is not None}, Info: {self.frontCameraInfo is not None}")
                return
            primaryRGB = self.frontImg
            primaryDepth = None  # No depth needed
            # Update image dimensions
            self.imgHeight, self.imgWidth = primaryRGB.shape[:2]
        else:
            # ONLY require RGB and CameraInfo, NOT depth
            if self.backImg is None or self.backCameraInfo is None:
                if self.nav_counter % 100 == 1:
                    print(f"[DEBUG] Waiting for back camera data - RGB: {self.backImg is not None}, Info: {self.backCameraInfo is not None}")
                return
            primaryRGB = self.backImg
            primaryDepth = None  # No depth needed
            # Update image dimensions
            self.imgHeight, self.imgWidth = primaryRGB.shape[:2]
        
        if self.nav_counter % 100 == 1:
            print(f"[DEBUG] Camera data ready - Image size: {self.imgWidth}x{self.imgHeight}")
        
        # If the feature extractor is not initialized yet, this has to be done
        # Pass None for depth since we don't have it
        if not self.imageProcessor.findCropLane(primaryRGB, None, mode='RGB'):
            # this is only False if the initialization was unsuccessful
            if self.nav_counter % 50 == 1:
                self.get_logger().warn("The initialization was unsuccessful!! ")
        else:
            if self.nav_counter % 100 == 1:
                print("[DEBUG] findCropLane succeeded!")
            
            # if the robot is currently following a line and is not turning just compute the controls
            if self.isFollowingLane():
                self.imageProcessor.trackCropLane(self.navigationMode)
                ctlCommands = self.computeControls(self.imageProcessor.cropLaneFound,
                                                self.imageProcessor.P,
                                                self.imageProcessor.ang)

                if not self.imageProcessor.cropRowEnd:
                    self.setRobotVelocities(ctlCommands[0], 0.0, ctlCommands[1])
                    if self.nav_counter % 100 == 1:
                        print(f"[DEBUG] Publishing velocities - linear: {ctlCommands[0]}, angular: {ctlCommands[1]}")

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

        self.publishImageTopics()

    def publishImageTopics(self):
        now = self.get_clock().now().to_msg()
        frame = 'front_camera_link'

        # Publish the Graphics image
        self.imageProcessor.drawGraphics()
        if hasattr(self.imageProcessor, 'graphicsImg') and self.imageProcessor.graphicsImg is not None:
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
                print(f"[DEBUG] Velocity published - linear.x={x}, angular.z={omega}")

    def stopRobot(self, delay):
        self.velocityMsg = Twist()
        self.velocityMsg.linear.x = 0.0
        self.velocityMsg.linear.y = 0.0
        self.velocityMsg.angular.z = 0.0
        time.sleep(delay)
        self.velocity_pub.publish(self.velocityMsg)
        print("[DEBUG] Robot stopped")

    def updateNavigationStage(self):
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
        if LaneFound:
            # define desired and actual feature vector
            desiredFeature = np.array([0.0, self.imgWidth/2, 0.0])
            actualFeature = np.array([P[0], P[1], Angle])
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
            omega = self.omegaScaler * controls 
            # set linear speed and direction
            rho = 0.2 * self.linearMotionDir
            # store the command in a cache
            self.omegaBuffer.append(omega)
            
            return [rho, omega]  
        else:
            # using the last known control 
            if len(self.omegaBuffer) == 0:
                self.omegaBuffer.append(0.0)
            # straight exit
            omega = 0.0
            rho = 0.05 * self.linearMotionDir
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
    rclpy.init(args=args)
    node = vs_nodeHandler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()