import numpy as np


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion into a 3×3 rotation matrix.

    Args:
        q: Quaternion as [w, x, y, z].

    Returns:
        3×3 numpy array representing the rotation. If the input quaternion
        is near zero-norm, returns the identity matrix.
    """
    w, x, y, z = q
    norm_sq = np.dot(q, q)
    if norm_sq < np.finfo(float).eps:
        return np.eye(3)
    # Normalize quaternion
    w, x, y, z = q / np.sqrt(norm_sq)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])


def quaternion_from_two_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the quaternion that rotates vector u to vector v via the shortest path.

    Args:
        u: Source vector (3,).
        v: Destination vector (3,).

    Returns:
        Quaternion [w, x, y, z] representing the rotation.
    """
    # Normalize input vectors
    u_norm = u / np.linalg.norm(u)
    v_norm = v / np.linalg.norm(v)
    dot = np.dot(u_norm, v_norm)

    # If vectors are nearly opposite, pick an arbitrary orthogonal axis
    if dot < -0.999999:
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(u_norm[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = np.cross(u_norm, ortho)
        axis /= np.linalg.norm(axis)
        # 180° rotation quaternion has zero scalar part
        return np.hstack((0.0, axis))

    # General case: axis is cross product, scalar part from dot
    axis = np.cross(u_norm, v_norm)
    s = np.sqrt((1 + dot) * 2)
    w = s / 2
    xyz = axis / s
    return np.hstack((w, xyz))
