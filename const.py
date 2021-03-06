from enum import Enum

# NOTE: All indices start from 1.

H = 480
W = 640

# TODO: This is screaming for bug issues!
NUM_KPTS = 17

#MASK = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
#    27, 28, 29, 30, 31, 32, 33, 38, 39, 40, 41, 42, 43, 44,
#    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 61,
#    62, 66, 67]

# NOTE: These keypoint indices start from 1.
KPTS_23 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 34,
    35, 36, 37, 58, 59, 60, 63, 64, 65]

KPTS_17 = [1, 5, 6, 10, 11, 12, 13, 34,
    35, 36, 37, 58, 59, 60, 63, 64, 65]

KPTS_15 = [
        1,      # hips
        5,      # neck
        6,      # head
        10,     # lshoulder
        11,     # larm
        13,     # lhand
        34,     # rshoulder
        35,     # rarm
        37,     # rhand
        58,     # rightUpLeg
        59,     # rleg
        60,     # rfoot
        63,     # leftUpLeg
        64,     # lleg
        65      # lfoot
    ]

PELVIS = 1
H36M_PELVIS = 0

BODY_PARTS_23 = [
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (7, 9),
    (5, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (5, 34),
    (34, 35),
    (35, 36),
    (36, 37),
    (1, 58),
    (58, 59),
    (59, 60),
    (1, 63),
    (63, 64),
    (64, 65)
]

BODY_PARTS_17 = [
    (1, 5),
    (5, 6),
    (5, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (5, 34),
    (34, 35),
    (35, 36),
    (36, 37),
    (1, 58),
    (58, 59),
    (59, 60),
    (1, 63),
    (63, 64),
    (64, 65)
]

H36M_KPTS_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, \
        18, 19, 25, 26, 27]

H36M_PARTS_17 = [
    (0, 1),
    (0, 6),
    (1, 2),
    (2, 3),
    (6, 7),
    (7, 8),
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (13, 17),
    (17, 18),
    (18, 19),
    (13, 25),
    (25, 26),
    (26, 27)
]

H36M_KPTS_15 = [0, 14, 15, 17, 18, 19, 25, 26, 27, 1, 2, 3, 6, 7, 8]

H36M_PARTS_15 = [
        (0, 1),
        (0, 14),
        (0, 6),
        (1, 2),
        (2, 3),
        (6, 7),
        (7, 8),
        (14, 15),
        (14, 17),
        (14, 25),
        (17, 18),
        (18, 19),
        (25, 26),
        (26, 27)
]

H36M_TRAIN = [1, 5, 6, 7, 8]
H36M_TEST  = [9, 11]

H36M_READ_ROOT  = '/h36m-fetch/processed/'
H36M_WRITE_ROOT = 'dataset/h36m/'

OPENPOSE_PARTS_15 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14)
]

#SMPL_KPTS_15 = [0, 15, 55, 16, 18, 20, 17, 19, 21, 1, 4, 7, 2, 5, 8]
SMPL_KPTS_15 = [0, 1, 2, 4, 5, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21]

SMPL_PARTS = [
    (0, 12),
    (12, 15),
    (12, 16),
    (16, 18),
    (18, 20),
    (12, 17),
    (17, 19),
    (19, 21),
    (0, 1),
    (1, 4),
    (4, 10),
    (0, 2),
    (2, 5),
    (5, 11)
]

#RADIUS = 54.67287
RADIUS = 4.67287

K = [[700.0, 0.0, 320.0],
    [0.0, 700.0, 240.0],
    [0.0, 0.0, 1.0]]
