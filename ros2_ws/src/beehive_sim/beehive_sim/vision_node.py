"""
vision_node.py — live Gazebo camera -> existing pipeline -> 3D target pose.

Role in the robotics architecture (Stages 1-4):

    /beehive/camera/image (sensor_msgs/Image, from Gazebo)
        -> keep the latest frame
        -> on ~/analyze (std_srvs/Trigger): run the EXISTING vision pipeline
           (OpenCV -> Gemini -> ranking) on that live frame
        -> take the SELECTED cell's pixel
        -> Stage 4: project it through the camera intrinsics (camera_info) and
           the FIXED camera->base extrinsics (TF tree) onto the honeycomb plane
        -> publish a geometry_msgs/PoseStamped IN THE ROBOT BASE FRAME

The downstream stack (RViz, MoveIt, pick-and-place) consumes ~/target_pose
directly. An RViz Marker is published at the same pose BEFORE any motion so the
projection can be checked visually.

The vision pipeline (web/) and the projection math (coord_transform) are both
imported, unchanged/standalone. This node only wires them to ROS.

Published:
  ~/annotated           sensor_msgs/Image            live camera + result overlay
  ~/result              std_msgs/String              JSON: cells/candidates/selected
  /selected_pose        geometry_msgs/PoseStamped    selected cell, ROBOT BASE frame
  /selected_pose_marker visualization_msgs/MarkerArray  sphere+text at the pose
  /selected_pixel       geometry_msgs/PointStamped   raw pixel (debug)

The motion node subscribes ONLY to /selected_pose and performs no pixel->world
conversion of its own.
"""

import json
import math
import os

import numpy as np
import rclpy
from rclpy.node import Node

import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger

from beehive_sim import vision_core, coord_transform


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")

        p = self.declare_parameter
        # --- vision pipeline knobs (mirror web/app.py) ---
        self.image_topic = p("image_topic", "/beehive/camera/image").value
        self.camera_info_topic = p(
            "camera_info_topic", "/beehive/camera/camera_info").value
        self.web_dir = p("web_dir", os.environ.get("BEEHIVE_WEB_DIR", "")).value
        self.mock = p("mock", True).value
        self.detector = p("detector", "geometry").value
        self.model = p("model", os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")).value
        self.api_key = p("api_key", os.environ.get("GEMINI_API_KEY", "")).value
        self.confidence_threshold = p("confidence_threshold", 0.7).value
        self.max_cells = p("max_cells", 150).value
        self.batch_size = p("batch_size", 48).value
        self.reference_dir = p("reference_dir", "").value
        self.publish_rate = float(p("publish_rate", 10.0).value)
        self.auto_analyze = p("auto_analyze", False).value
        self.auto_period = float(p("auto_period", 5.0).value)

        # --- Stage 4: coordinate transform knobs ---
        self.base_frame = p("base_frame", "panda_link0").value
        # The comb is a VERTICAL WALL: a plane at x = plane_point_x facing the
        # robot (normal +X). A point on it and its normal, both in the base frame.
        # (For a flat tabletop instead, set plane_normal=[0,0,1], plane_point=[0,0,0.003].)
        self.plane_point = [float(x) for x in
                            p("plane_point", [0.5, 0.0, 0.0]).value]
        self.plane_normal = [float(x) for x in
                             p("plane_normal", [1.0, 0.0, 0.0]).value]
        # kept for reference / legacy horizontal mode (unused for the wall)
        self.plane_z = float(p("plane_z", 0.003).value)
        # wall-facing grasp: rpy = (0, pi/2, 0) -> tool Z points +X (into the wall)
        ar = p("approach_rpy", [0.0, math.pi / 2.0, 0.0]).value
        self.approach_quat = coord_transform.rpy_to_quat(ar[0], ar[1], ar[2])

        # --- cv_bridge / tf ---
        from cv_bridge import CvBridge
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- state ---
        self.latest_frame = None
        self.latest_header = None
        self.latest_result = None
        self.K = None                  # 3x3 intrinsics from camera_info
        self.optical_frame = None      # camera_info.header.frame_id
        self.pipeline = None

        self._build_pipeline()

        # --- pub/sub/srv ---
        self.sub_img = self.create_subscription(
            Image, self.image_topic, self._on_image, 10)
        self.sub_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_info, 10)

        self.pub_annotated = self.create_publisher(Image, "~/annotated", 10)
        self.pub_result = self.create_publisher(String, "~/result", 10)
        # Absolute topics so the motion node consumes them directly. The motion
        # node uses ONLY /selected_pose and does no pixel->world conversion.
        self.pub_pose = self.create_publisher(PoseStamped, "/selected_pose", 10)
        self.pub_marker = self.create_publisher(
            MarkerArray, "/selected_pose_marker", 10)
        self.pub_pixel = self.create_publisher(PointStamped, "/selected_pixel", 10)

        self.srv = self.create_service(Trigger, "~/analyze", self._on_analyze)

        if self.publish_rate > 0:
            self.create_timer(1.0 / self.publish_rate, self._publish_annotated)
        if self.auto_analyze and self.auto_period > 0:
            self.create_timer(self.auto_period, lambda: self._run_analysis())

        self.get_logger().info(
            f"vision_node up. image={self.image_topic} base_frame={self.base_frame} "
            f"plane_z={self.plane_z} detector={self.detector} mock={self.mock} "
            f"pipeline={'ready' if self.pipeline else 'UNAVAILABLE'}. "
            "Call ~/analyze to run the pipeline + project the selected cell.")

    # ------------------------------------------------------------------ build
    def _build_pipeline(self):
        if not self.web_dir:
            self.get_logger().error(
                "web_dir empty — set 'web_dir' param or BEEHIVE_WEB_DIR env. "
                "Live view still works; analysis disabled.")
            return
        try:
            self.pipeline = vision_core.VisionPipeline(
                self.web_dir, mock=self.mock, detector=self.detector,
                model=self.model, api_key=self.api_key,
                confidence_threshold=self.confidence_threshold,
                max_cells=self.max_cells, batch_size=self.batch_size,
                reference_dir=(self.reference_dir or None))
            self.get_logger().info(f"vision pipeline built from {self.web_dir}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"failed to build vision pipeline: {e}")
            self.pipeline = None

    # --------------------------------------------------------------- callbacks
    def _on_image(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_header = msg.header
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"could not convert image: {e}")

    def _on_info(self, msg: CameraInfo):
        k = msg.k  # row-major 3x3
        self.K = [[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]]
        self.optical_frame = msg.header.frame_id

    def _on_analyze(self, request, response):
        ok, msg = self._run_analysis()
        response.success = ok
        response.message = msg
        return response

    # ----------------------------------------------------------------- helpers
    def _run_analysis(self):
        if self.pipeline is None:
            return False, "vision pipeline unavailable (check web_dir)"
        if self.latest_frame is None:
            return False, f"no camera frame yet on {self.image_topic}"

        frame = self.latest_frame
        try:
            result = self.pipeline.analyze_bgr(frame)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"pipeline run failed: {e}")
            return False, f"pipeline error: {e}"
        self.latest_result = result

        # full result as JSON
        s = String()
        s.data = json.dumps({
            "selected": result.get("selected"),
            "image": result.get("image"),
            "candidates": result.get("candidates"),
            "cells": result.get("cells"),
        })
        self.pub_result.publish(s)

        px = vision_core.selected_pixel(result)
        if px is None:
            self._publish_annotated()
            return True, "no cell selected (nothing to project)"

        # debug: raw pixel
        pt = PointStamped()
        pt.header = self.latest_header or pt.header
        pt.point.x, pt.point.y, pt.point.z = float(px[0]), float(px[1]), 0.0
        self.pub_pixel.publish(pt)

        # Stage 4: pixel -> base-frame pose
        pose = self._project_to_base(px)
        self._publish_annotated()
        if pose is None:
            return True, (f"selected #{result.get('selected')} pixel={px}; "
                          "pose unavailable (no camera_info or TF yet)")

        self.pub_pose.publish(pose)
        self._publish_marker(pose, result.get("selected"))
        pos = pose.pose.position
        summary = (f"selected #{result.get('selected')} pixel={px} "
                   f"-> base pose ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) "
                   f"in {self.base_frame}")
        self.get_logger().info(f"analysis: {summary}")
        return True, summary

    def _project_to_base(self, px):
        """Pixel -> PoseStamped in base_frame via camera_info + TF. None on fail."""
        if self.K is None:
            self.get_logger().warn(
                f"no camera_info on {self.camera_info_topic} yet — cannot project")
            return None
        src = self.optical_frame or "honeycomb_camera_optical_frame"
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, src, rclpy.time.Time())
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f"TF {self.base_frame}<-{src} unavailable: {e}")
            return None

        tr = tf.transform.translation
        q = tf.transform.rotation
        R = coord_transform.quat_to_rotmat(q.x, q.y, q.z, q.w)  # base<-optical
        t = np.array([tr.x, tr.y, tr.z])

        point = coord_transform.pixel_to_base_point(
            self.K, px[0], px[1], R, t,
            plane_normal=self.plane_normal, plane_point=self.plane_point)
        if point is None:
            self.get_logger().warn("ray did not intersect the honeycomb wall")
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.position.z = float(point[2])
        qx, qy, qz, qw = self.approach_quat
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        return pose

    def _publish_marker(self, pose: PoseStamped, cell_id):
        arr = MarkerArray()

        sphere = Marker()
        sphere.header = pose.header
        sphere.ns = "beehive_target"
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose = pose.pose
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.012
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = \
            0.0, 1.0, 0.0, 0.9
        arr.markers.append(sphere)

        text = Marker()
        text.header = pose.header
        text.ns = "beehive_target"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = pose.pose.position.x
        text.pose.position.y = pose.pose.position.y
        text.pose.position.z = pose.pose.position.z + 0.03
        text.pose.orientation.w = 1.0
        text.scale.z = 0.02
        text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
        text.text = f"target #{cell_id}"
        arr.markers.append(text)

        self.pub_marker.publish(arr)

    def _publish_annotated(self):
        if self.latest_frame is None:
            return
        try:
            img = vision_core.annotate(self.latest_frame, self.latest_result)
            out = self.bridge.cv2_to_imgmsg(img, "bgr8")
            if self.latest_header is not None:
                out.header = self.latest_header
            self.pub_annotated.publish(out)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"annotate/publish failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
