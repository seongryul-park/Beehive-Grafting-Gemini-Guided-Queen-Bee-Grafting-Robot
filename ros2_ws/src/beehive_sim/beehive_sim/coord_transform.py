"""
coord_transform.py — ROS-free pixel -> 3D projection (Stage 4 math).

Given a selected image pixel, the pinhole intrinsics, the FIXED camera->base
extrinsics (from the TF tree), and the FIXED honeycomb plane, this returns the
target point in the robot base frame. Because every transform is fixed, no
online calibration is needed — the node just looks the extrinsics up once via TF
and calls into here.

Pipeline:
    pixel (u,v)
      -> back-project with K to a ray in the camera OPTICAL frame
      -> rotate/translate the ray into the base frame (R_base_opt, t_base_opt)
      -> intersect with the honeycomb plane (z = plane_z in the base frame)
      -> 3D point in the base frame

Kept ROS-free so it can be unit-tested without Gazebo/rclpy.
"""

import math

import numpy as np


def quat_to_rotmat(x, y, z, w):
    """Unit-quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def rpy_to_quat(roll, pitch, yaw):
    """ROS RPY (XYZ, applied Z*Y*X) -> quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def backproject_pixel(K, u, v):
    """Pixel (u, v) -> unit ray direction in the camera optical frame.

    K is the 3x3 intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]. Optical convention
    (REP 103): +Z forward (into the scene), +X right, +Y down.
    """
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    if fx == 0 or fy == 0:
        raise ValueError("invalid intrinsics: fx/fy is zero")
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return d / np.linalg.norm(d)


def ray_plane_intersection(origin, direction, plane_point, plane_normal):
    """Intersect a ray with a plane. Returns the 3D point, or None.

    None if the ray is parallel to the plane or the hit is behind the camera.
    """
    origin = np.asarray(origin, float)
    direction = np.asarray(direction, float)
    plane_point = np.asarray(plane_point, float)
    plane_normal = np.asarray(plane_normal, float)

    denom = float(np.dot(plane_normal, direction))
    if abs(denom) < 1e-9:
        return None
    s = float(np.dot(plane_normal, plane_point - origin)) / denom
    if s <= 0:
        return None
    return origin + s * direction


def pixel_to_base_point(K, u, v, R_base_opt, t_base_opt, plane_z=None,
                        plane_normal=(0.0, 0.0, 1.0), plane_point=None):
    """Project pixel (u, v) onto the honeycomb plane, in the base frame.

    R_base_opt, t_base_opt : rotation (3x3) and translation (3,) of the camera
        OPTICAL frame expressed in the base frame (i.e. base <- optical), as read
        from the TF tree.

    The target plane can be given two ways:
      - plane_point + plane_normal : a general plane (e.g. a VERTICAL comb wall
        at x = 0.5 -> plane_point=(0.5,0,0), plane_normal=(1,0,0)).
      - plane_z (legacy)           : a HORIZONTAL plane z = plane_z, i.e.
        plane_point=(0,0,plane_z), plane_normal=(0,0,1).
    plane_point takes precedence when supplied.

    Returns a (3,) np.ndarray point in the base frame, or None if no hit.
    """
    R = np.asarray(R_base_opt, float).reshape(3, 3)
    t = np.asarray(t_base_opt, float).reshape(3)
    d_opt = backproject_pixel(K, u, v)
    d_base = R @ d_opt
    o_base = t
    if plane_point is None:
        if plane_z is None:
            raise ValueError("provide either plane_point or plane_z")
        plane_point = np.array([0.0, 0.0, float(plane_z)])
    else:
        plane_point = np.asarray(plane_point, float)
    return ray_plane_intersection(o_base, d_base, plane_point,
                                  np.asarray(plane_normal, float))
