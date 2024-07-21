from math import cos, sin
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import torch

import numpy as np
import mediapipe as mp




def draw_pose_angles(img, yaw, pitch, roll, reference_angle=0, text_position=(10, 30), font_scale=0.8,
                     line_thickness=2):
    """
    Draw yaw, pitch, and roll angles on an image using OpenCV's cv2.putText, with continuous error color encoding.

    Args:
        img (numpy.ndarray): Input image.
        yaw (torch.Tensor or float): Yaw angle in degrees.
        pitch (torch.Tensor or float): Pitch angle in degrees.
        roll (torch.Tensor or float): Roll angle in degrees.
        reference_angle (float, optional): Reference angle for error calculation. Defaults to 0.
        text_position (tuple, optional): Position to display the text. Defaults to (10, 30).
        font_scale (float, optional): Font scale factor. Defaults to 0.8.
        line_thickness (int, optional): Thickness of the lines used for the text. Defaults to 2.

    Returns:
        numpy.ndarray: Image with pose angles drawn.
    """

    yaw_error = yaw - reference_angle
    pitch_error = pitch - reference_angle
    roll_error = roll - reference_angle

    # Define colors based on error values using a color map
    cmap = plt.cm.jet  # Choose a color map (e.g., jet)
    norm = plt.Normalize(vmin=-100, vmax=100)  # Normalize error values between -180 and 180 degrees
    yaw_color = (0, 0, 0)  #
    pitch_color = (0, 0, 0)  #
    roll_color = (0, 0, 0)

    yaw_str = f'Yaw: {yaw:.2f} (Error: {yaw_error:.2f})'
    pitch_str = f'Pitch: {pitch:.2f} (Error: {pitch_error:.2f})'
    roll_str = f'Roll: {roll:.2f} (Error: {roll_error:.2f})'

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Draw text on the image with colors based on errors
    cv2.putText(img, yaw_str, text_position, font, font_scale, yaw_color, line_thickness)
    cv2.putText(img, pitch_str, (text_position[0], text_position[1] + 30), font, font_scale, pitch_color,
                line_thickness)
    cv2.putText(img, roll_str, (text_position[0], text_position[1] + 60), font, font_scale, roll_color, line_thickness)

    return img


def draw_lines(img, start_points, end_points, color, thickness):
    for start, end in zip(start_points, end_points):
        cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, thickness)


# def plot_pose_cube(img, yaw, pitch, roll, bbox):
#     x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
#     face_center_x = x_min + int((x_max - x_min) / 2)
#     face_center_y = y_min + int((y_max - y_min) / 2)
#
#     p = pitch * np.pi / 180
#     y = -(yaw * np.pi / 180)
#     r = roll * np.pi / 180
#
#     bbox_width = abs(x_max - x_min)
#     face_top_left_x = face_center_x - 0.50 * bbox_width
#     face_top_left_y = face_center_y - 0.50 * bbox_width
#
#     base_x_end = bbox_width * (cos(y) * cos(r)) + face_top_left_x
#     base_y_end = bbox_width * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_top_left_y
#     base_x_start = bbox_width * (-cos(y) * sin(r)) + face_top_left_x
#     base_y_start = bbox_width * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_top_left_y
#     pillar_x_end = bbox_width * (sin(y)) + face_top_left_x
#     pillar_y_end = bbox_width * (-cos(y) * sin(p)) + face_top_left_y
#
#     # Draw base in red
#     cv2.line(img, (int(face_top_left_x), int(face_top_left_y)), (int(base_x_end), int(base_y_end)), (0, 0, 255), 3)
#     cv2.line(img, (int(face_top_left_x), int(face_top_left_y)), (int(base_x_start), int(base_y_start)), (0, 0, 255), 3)
#     cv2.line(img, (int(base_x_start), int(base_y_start)), (int(base_x_start + base_x_end - face_top_left_x), int(base_y_start + base_y_end - face_top_left_y)), (0, 0, 255), 3)
#     cv2.line(img, (int(base_x_end), int(base_y_end)), (int(base_x_end + base_x_start - face_top_left_x), int(base_y_end + base_y_start - face_top_left_y)), (0, 0, 255), 3)
#     # Draw pillars in blue
#     cv2.line(img, (int(face_top_left_x), int(face_top_left_y)), (int(pillar_x_end), int(pillar_y_end)), (255, 0, 0), 2)
#     cv2.line(img, (int(base_x_start), int(base_y_start)), (int(base_x_start + pillar_x_end - face_top_left_x), int(base_y_start + pillar_y_end - face_top_left_y)), (255, 0, 0), 2)
#     cv2.line(img, (int(base_x_end), int(base_y_end)), (int(base_x_end + pillar_x_end - face_top_left_x), int(base_y_end + pillar_y_end - face_top_left_y)), (255, 0, 0), 2)
#     cv2.line(img, (int(base_x_start + base_x_end - face_top_left_x), int(base_y_start + base_y_end - face_top_left_y)), (int(pillar_x_end + base_x_start + base_x_end - 2 * face_top_left_x), int(pillar_y_end + base_y_start + base_y_end - 2 * face_top_left_y)), (255, 0, 0), 2)
#     # Draw top in green
#     cv2.line(img, (int(pillar_x_end + base_x_start - face_top_left_x), int(pillar_y_end + base_y_start - face_top_left_y)), (int(pillar_x_end + base_x_start + base_x_end - 2 * face_top_left_x), int(pillar_y_end + base_y_start + base_y_end - 2 * face_top_left_y)), (0, 255, 0), 2)
#     cv2.line(img, (int(base_x_end + pillar_x_end - face_top_left_x), int(base_y_end + pillar_y_end - face_top_left_y)), (int(pillar_x_end + base_x_start + base_x_end - 2 * face_top_left_x), int(pillar_y_end + base_y_start + base_y_end - 2 * face_top_left_y)), (0, 255, 0), 2)
#     cv2.line(img, (int(pillar_x_end), int(pillar_y_end)), (int(pillar_x_end + base_x_start - face_top_left_x), int(pillar_y_end + base_y_start - face_top_left_y)), (0, 255, 0), 2)
#     cv2.line(img, (int(pillar_x_end), int(pillar_y_end)), (int(pillar_x_end + base_x_end - face_top_left_x), int(pillar_y_end + base_y_end - face_top_left_y)), (0, 255, 0), 2)
#
#     return img
def draw_base(img, start_point, end_point, color=(0, 0, 255), thickness=3):
    cv2.line(img, start_point, end_point, color, thickness)

def draw_pillars(img, start_point, end_point, color=(255, 0, 0), thickness=2):
    cv2.line(img, start_point, end_point, color, thickness)

def draw_top(img, start_point, end_point, color=(0, 255, 0), thickness=2):
    cv2.line(img, start_point, end_point, color, thickness)

def plot_pose_cube(img, yaw, pitch, roll, bbox):
    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
    face_center_x = x_min + int((x_max - x_min) / 2)
    face_center_y = y_min + int((y_max - y_min) / 2)

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180

    bbox_width = abs(x_max - x_min)
    face_top_left_x = face_center_x - 0.50 * bbox_width
    face_top_left_y = face_center_y - 0.50 * bbox_width

    base_x_end = bbox_width * (cos(y) * cos(r)) + face_top_left_x
    base_y_end = bbox_width * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_top_left_y
    base_x_start = bbox_width * (-cos(y) * sin(r)) + face_top_left_x
    base_y_start = bbox_width * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_top_left_y
    pillar_x_end = bbox_width * (sin(y)) + face_top_left_x
    pillar_y_end = bbox_width * (-cos(y) * sin(p)) + face_top_left_y

    draw_base(img, (int(face_top_left_x), int(face_top_left_y)), (int(base_x_end), int(base_y_end)))
    draw_base(img, (int(face_top_left_x), int(face_top_left_y)), (int(base_x_start), int(base_y_start)))
    draw_base(img, (int(base_x_start), int(base_y_start)), (int(base_x_start + base_x_end - face_top_left_x), int(base_y_start + base_y_end - face_top_left_y)))
    draw_base(img, (int(base_x_end), int(base_y_end)), (int(base_x_end + base_x_start - face_top_left_x), int(base_y_end + base_y_start - face_top_left_y)))
    # Draw pillars in blue
    draw_pillars(img, (int(face_top_left_x), int(face_top_left_y)), (int(pillar_x_end), int(pillar_y_end)))
    draw_pillars(img, (int(base_x_start), int(base_y_start)), (int(base_x_start + pillar_x_end - face_top_left_x), int(base_y_start + pillar_y_end - face_top_left_y)))
    draw_pillars(img, (int(base_x_end), int(base_y_end)), (int(base_x_end + pillar_x_end - face_top_left_x), int(base_y_end + pillar_y_end - face_top_left_y)))
    draw_pillars(img, (int(base_x_start + base_x_end - face_top_left_x), int(base_y_start + base_y_end - face_top_left_y)),
                 (int(pillar_x_end + base_x_start + base_x_end - 2 * face_top_left_x), int(pillar_y_end + base_y_start + base_y_end - 2 * face_top_left_y)))
    # Draw top in green
    draw_top(img, (int(pillar_x_end + base_x_start - face_top_left_x), int(pillar_y_end + base_y_start - face_top_left_y)),
             (int(pillar_x_end + base_x_start + base_x_end - 2 * face_top_left_x), int(pillar_y_end + base_y_start + base_y_end - 2 * face_top_left_y)))
    draw_top(img, (int(base_x_end + pillar_x_end - face_top_left_x), int(base_y_end + pillar_y_end - face_top_left_y)),
             (int(pillar_x_end + base_x_start + base_x_end - 2 * face_top_left_x), int(pillar_y_end + base_y_start + base_y_end - 2 * face_top_left_y)))
    draw_top(img, (int(pillar_x_end), int(pillar_y_end)), (int(pillar_x_end + base_x_start - face_top_left_x), int(pillar_y_end + base_y_start - face_top_left_y)))
    draw_top(img, (int(pillar_x_end), int(pillar_y_end)), (int(pillar_x_end + base_x_end - face_top_left_x), int(pillar_y_end + base_y_end - face_top_left_y)))

    return img

def draw_axis(img, yaw, pitch, roll, positions_x, positions_y, size=100):
    reference_angle = 0
    yaw_error = yaw - reference_angle
    pitch_error = pitch - reference_angle
    roll_error = roll - reference_angle
    cmap = plt.cm.jet  # Choose a color map (e.g., jet)
    norm = plt.Normalize(vmin=-60, vmax=60)  # Normalize error values between -180 and 180 degrees
    yaw_color = tuple(np.array(cmap(norm(yaw_error)))[:3] * 255)
    pitch_color = tuple(np.array(cmap(norm(pitch_error)))[:3] * 255)
    roll_color = tuple(np.array(cmap(norm(roll_error)))[:3] * 255)
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    x1 = size * (cos(yaw) * cos(roll)) + positions_x
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + positions_y

    x2 = size * (-cos(yaw) * sin(roll)) + positions_x
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + positions_y

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + positions_x
    y3 = size * (-cos(yaw) * sin(pitch)) + positions_y
    cv2.line(img, (int(positions_x), int(positions_y)), (int(x1), int(y1)), pitch_color, 4)
    cv2.line(img, (int(positions_x), int(positions_y)), (int(x2), int(y2)), yaw_color, 4)
    cv2.line(img, (int(positions_x), int(positions_y)), (int(x3), int(y3)), roll_color, 4)

    return img


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll face_center_x face_center_y tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, face_center_x, face_center_y]
    pose_params = pre_pose_params[:5]
    return pose_params


def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll face_center_x face_center_y tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params


def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d


# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


# input batch*4*4 or batch*3*3
# output torch batch*3 x, y, z in radiant
# the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0

    gpu = rotation_matrices.get_device()
    if gpu < 0:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3)).to(torch.device('cpu'))
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3)).to(torch.device('cuda:%d' % gpu))
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular

    return out_euler


def get_R(x, y, z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R
