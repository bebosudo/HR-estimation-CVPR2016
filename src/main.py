#!/usr/bin/env python3

from __future__ import division
import numpy as np
import cv2
import sys
import os
from collections import namedtuple

# Coord = namedtuple('coord', 'x y w h')

base_folder = os.path.dirname(os.path.abspath(__file__))

if sys.platform.startswith("linux"):
    path = "/usr/share/OpenCV/haarcascades/"
elif sys.platform == "darwin":  # OS/X
    path = ""       # TO-DO: Marti update this to reflect the OS/X setup.

face_cascade = cv2.CascadeClassifier(os.path.join(
    path, "haarcascade_frontalface_default.xml"))

# The eye_tree_eyeglasses seems to be more precise that the normal eye classifier:
#   $ cd /usr/share/OpenCV/
#   $ p samples/python/facedetect.py --cascade haarcascades/haarcascade_eye_tree_eyeglasses.xml
# eye_cascade = cv2.CascadeClassifier(os.path.join(path,
# "haarcascade_eye.xml"))
eye_cascade = cv2.CascadeClassifier(os.path.join(
    path, "haarcascade_eye_tree_eyeglasses.xml"))

mouth_cascade = cv2.CascadeClassifier(
    os.path.join(path, "haarcascade_smile.xml"))


x = y = w = h = None
white_color = np.empty(3, np.uint8)
white_color.fill(255)
black_color = white_color - white_color
num_corners = 50

if len(sys.argv) == 1:
    print(" ERROR: Insert the video to load as an argument after the script.",
          file=sys.stderr)
    exit(1)

input_file_name = sys.argv[1]
if not os.path.exists(input_file_name):
    print(" ERROR: The input file provided does not exist.",
          file=sys.stderr)
    exit(1)

cap = cv2.VideoCapture(input_file_name)

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 1:
        sys.stderr.write("More than one face detected ('{}'). "
                         "Please provide a video with a face only.\n".format(len(faces)))
        exit()
    elif len(faces) == 0:
        print("faces = ", len(faces)) if len(faces) != 2 else None
        continue

    # Keep track of the ROI of the face.
    fx, fy, fw, fh = faces[0]

    # Grep eyes.
    # eyes = eye_cascade.detectMultiScale(gray[fy:fy + fh, fx:fx + fw], 1.3, 1)
    eyes = eye_cascade.detectMultiScale(gray[fy:fy + fh, fx:fx + fw],
                                        scaleFactor=1.35,
                                        minNeighbors=10)

    if len(eyes) != 2:
        print("eyes = ", len(eyes))
        continue

    assert eyes.shape[0] == 2
    average_eye_height_reduced = int(eyes[:, 3].sum() * .8) // 2

    # The second and third element are the y pos and the height.
    # lowest_eye_height = eyes[0]
    lowest_eye_height = np.maximum(
        eyes[0][1] + eyes[0][3], eyes[1][1] + eyes[1][3])

    scaling_down = lowest_eye_height + average_eye_height_reduced

    # Region below the eyes.
    bex, bey = fx + int(fw // 6), fy + lowest_eye_height
    bew, beh = fx + int(5 * fw // 6) - bex, fy + scaling_down - bey
    below_eyes_region = (bex, bey, bew, beh)

    # Find the index of the leftmost eye, the one that has the lowest `x`
    # value.
    left_eye = np.argmin(eyes[:, 0])

    cheecks = np.empty_like(eyes)
    np.copyto(cheecks, eyes)

    # Move down the y values to the lowest margin of the region below the-eyes
    # (relatively to the face coords).
    cheecks[left_eye][1] = cheecks[1 - left_eye][1] = scaling_down

    # Change the height of the cheecks region to match the one below the eyes.
    cheecks[:, 3] = average_eye_height_reduced

    # Move the position of the cheecks a bit farther from the mouth.
    cheecks[left_eye][0] -= int(fw / 60)
    cheecks[left_eye][2] = int(cheecks[left_eye][2] * 4 // 5)
    cheecks[1 - left_eye][0] += int(fw / 20)
    cheecks[1 - left_eye][2] = int(cheecks[1 - left_eye][2] * 4 // 5)

    # Create a ROI as wide as the region below the eyes plus the two cheecks.
    ROI_array = np.empty((average_eye_height_reduced,
                          bew + cheecks[:, 2].sum(),
                          3),
                         np.uint8)

    # Copy the region below the eyes to the ROI, and copy the two cheecks on
    # the right, alongside the region below the eyes.
    np.copyto(ROI_array[:, :cheecks[left_eye][2]], frame[fy + cheecks[left_eye][1]:fy + cheecks[left_eye][
              1] + cheecks[left_eye][3], fx + cheecks[left_eye][0]:fx + cheecks[left_eye][0] + cheecks[left_eye][2]])
    np.copyto(ROI_array[:, cheecks[left_eye][2]:cheecks[left_eye][
              2] + bew], frame[bey:bey + beh, bex:bex + bew])
    np.copyto(ROI_array[:, cheecks[left_eye][2] + bew:], frame[fy + cheecks[1 - left_eye][1]:fy + cheecks[1 - left_eye][1] +
                                                               cheecks[1 - left_eye][3], fx + cheecks[1 - left_eye][0]:fx + cheecks[1 - left_eye][0] + cheecks[1 - left_eye][2]])

    # Draw the region below the eyes.
    cv2.rectangle(frame, (bex, bey), (bex + bew, bey + beh),
                  color=white_color.tolist(), thickness=3)

    # Draw the cheecks, which have the face as the base point.
    for cx, cy, cw, ch in cheecks:
        cv2.rectangle(frame, (fx + cx, fy + cy), (fx + cx + cw, fy + cy + ch),
                      color=white_color.tolist(), thickness=3)

    # Draw the eyes, which have the face as the base point.
    for ex, ey, ew, eh in eyes:
        lowest_eye_height = np.minimum(lowest_eye_height, np.sum([ey, eh]))
        cv2.rectangle(frame, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh),
                      color=white_color.tolist(), thickness=1)

    # Draw the face.
    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh),
                  color=white_color.tolist(), thickness=1)

    mask = np.zeros(gray.shape, np.uint8)
    mask[fy:fy + fh, fx:fx + fw] = gray[fy:fy + fh, fx:fx + fw]

    """
    # goodFeaturesToTrack returns an array with shape (num_corners, 1, 2)
    # made of floats.
    corners = cv2.goodFeaturesToTrack(gray,
                                      maxCorners=num_corners,
                                      qualityLevel=0.001,
                                      minDistance=5,
                                      mask=mask)
    corners = np.int0(corners)

    # To convert it from a 3D to a 2D matrix of shape (num_corners, 2).
    # corners = corners[:, 0]
    corners = corners.reshape(num_corners, 2)

    # Collect the maximum and minimum points from the ones tracked with
    # shi tomasi, so that we can use these to track the movements in the
    # following frames.
    # print(fx, fy, fw, fh)
    # fx, fy = corners[:, 0].min(), corners[:, 1].min()
    # fw, fh = corners[:, 0].max() - fx, corners[:, 1].max() - fy
    """

    # TO-DO: this could be optimized by reading the ROI just once
    # (and maybe without extracting the ROI at all?).
    blue_ROI = ROI_array[:, :, 0]
    green_ROI = ROI_array[:, :, 1]
    red_ROI = ROI_array[:, :, 2]

    # Calculate the arithmetic mean and std for each color for each frame.
    b_mean = blue_ROI.mean()
    g_mean = green_ROI.mean()
    r_mean = red_ROI.mean()

    b_std = blue_ROI.std()
    g_std = green_ROI.std()
    r_std = red_ROI.std()

    # Formulae from the paper "Self-adaptive Matrix Completion..."
    R_n = red_ROI * r_std / r_mean
    G_n = green_ROI * g_std / g_mean
    X = 3 * R_n - 2 * G_n
    Y = 1.5 * R_n + G_n - 1.5 * (blue_ROI * b_std / b_mean)

    # Apply the pass-banded filter.
    X_f, Y_f = X, Y

    alpha = X_f.std() / Y_f.std()
    crominance = X_f - alpha * Y_f

    print("C = {}".format(crominance.mean()))

    cv2.imshow('video', frame)
    cv2.imshow('ROI', ROI_array)

    # Close the video only when ESC (27) or 'q' are pressed.
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()


# Order: capture e poi tracking
#      * CAPTURE = [ ViolaJones (Cascade) + Shi-Tomasi==good features to track (per i facial landmarks da selezionare) ]
#      * TRACKING = KLT
# inizialmente trovare la faccia, poi applicare shi tomasi per trovare i punti interessanti da tracciare,
# ed infine KLT traccia gli spostamenti nel video, frame per frame
# while frames in video:
#     catturo il viso del soggetto
#     if numero punti catturati < N:
#         rieseguo la cattura dei punti (=good features to TRACK), che hanno poi da essere tracciati con KLT
#     continuo il tracciamento con KLT
# analisi delle ROI per calcolare e studiare la crominanza, etc..
# a questo punto si applica il paper sulla crominanza
