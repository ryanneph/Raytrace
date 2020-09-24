import math

import numpy as np

def rotateAroundAxisRHS(vec, center, axis, angle):
    """
    ASSUMES r IS NORMALIZED ALREADY
    vec - vector to rotate
    center- center point
    axis - rotation axis
    angle - rotation angle
    """
    c, s = math.cos(angle), math.sin(angle)
    axsq = np.square(axis)
    return (
        (center[0]*(axsq[1]+axsq[2]) - axis[0]*(center[1]*axis[1] + center[2]*axis[2] - axis[0]*vec[0] - axis[1]*vec[1] - axis[2]*vec[2]))*(1-c) + vec[0]*c + (-center[2]*axis[1] + center[1]*axis[2] - axis[2]*vec[1] + axis[1]*vec[2])*s,
        (center[1]*(axsq[0]+axsq[2]) - axis[1]*(center[0]*axis[0] + center[2]*axis[2] - axis[0]*vec[0] - axis[1]*vec[1] - axis[2]*vec[2]))*(1-c) + vec[1]*c + ( center[2]*axis[0] - center[0]*axis[2] + axis[2]*vec[0] - axis[0]*vec[2])*s,
        (center[2]*(axsq[0]+axsq[1]) - axis[2]*(center[0]*axis[0] + center[1]*axis[1] - axis[0]*vec[0] - axis[1]*vec[1] - axis[2]*vec[2]))*(1-c) + vec[2]*c + (-center[1]*axis[0] + center[0]*axis[1] - axis[1]*vec[0] + axis[0]*vec[1])*s
    )

def rotateAroundAxisAtOriginRHS(vec, axis, angle):
    """
    ASSUMES r IS NORMALIZED ALREADY
    p - vector to rotate
    q - center point
    r - rotation axis
    t - rotation angle
    """
    c, s = math.cos(angle), math.sin(angle)
    return (
        (-axis[0]*(-axis[0]*vec[0] - axis[1]*vec[1] - axis[2]*vec[2]))*(1-c) + vec[0]*c + (-axis[2]*vec[1] + axis[1]*vec[2])*s,
        (-axis[1]*(-axis[0]*vec[0] - axis[1]*vec[1] - axis[2]*vec[2]))*(1-c) + vec[1]*c + (+axis[2]*vec[0] - axis[0]*vec[2])*s,
        (-axis[2]*(-axis[0]*vec[0] - axis[1]*vec[1] - axis[2]*vec[2]))*(1-c) + vec[2]*c + (-axis[1]*vec[0] + axis[0]*vec[1])*s
    )

def inverseRotateBeamAtOriginRHS(vec, theta, phi, coll):
    # invert what was done in forward rotation
    tmp = rotateAroundAxisAtOriginRHS(vec, (0.0, 1.0, 0.0), -(phi+coll)) # coll rotation + correction
    c, s = math.cos(-phi), math.sin(-phi)
    rotation_axis = (s, 0.0, c)                                          # couch rotation
    return rotateAroundAxisAtOriginRHS(tmp, rotation_axis, theta)        # gantry rotation
