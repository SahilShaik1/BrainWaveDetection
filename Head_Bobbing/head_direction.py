import cv2
import mediapipe as mp
import numpy as np
import time


def fate(cam):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    ret, frame = cam.read()

    start = time.time()

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    frame.flags.writeable = False

    res = face_mesh.process(frame)

    frame.flags.writeable = True

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    h, w, c = frame.shape

    face3d = []
    face2d = []

    if res.multi_face_landmarks:
        for face_landmarks in res.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * w, lm.y * h)
                        nose_3d = (lm.x * w, lm.y * h, lm.z * 3000)
                    x, y = int(lm.x * w), int(lm.y * h)

                    face2d.append([x, y])
                    face3d.append([x, y, lm.z])

            face2d = np.array(face2d, dtype=np.float64)

            face3d = np.array(face3d, dtype = np.float64)

            focal_length = w

            cam_matrix = np.array([[focal_length, 0, h / 2],
                                   [0, focal_length, w / 2],
                                   [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face3d, face2d, cam_matrix, dist_matrix)

            rmat, jmac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if x < -10:
                return True
            else:
                return False
