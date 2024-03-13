import numpy as np
from numba import njit, prange


def init_boids(boids: np.ndarray, asp: float, vrange: tuple[float, float]):
    """
    This function initializes the agents
    :param boids: An array of agents that is filled with coordinates and velocities
    :param asp: The upper bound for generating an array of x coordinates
    :param vrange: An array for generating numbers for calculating speeds
    :return: None
    """
    n = boids.shape[0]
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s


@njit(parallel=True)
def circle_init(circle_arr: np.ndarray, ci: int, asp: float):
    """
    This function generates obstacles from which agents will be repelled.
    Obstacles are generated without intersections
    :param circle_arr: An array that will be filled with coordinates and radii of circles
    :param ci: Number of obstacles
    :param asp: The upper bound for generating an array of x coordinates
    :return: None
    """
    rng = np.random
    circle_arr[0, 0] = rng.uniform(0., asp)
    circle_arr[0, 1] = rng.uniform(0., 1.)
    circle_arr[0, 2] = rng.uniform(0.05, 0.15)
    count = 1

    while count < ci:
        circle_arr[count, 0] = rng.uniform(0., asp)
        circle_arr[count, 1] = rng.uniform(0., 1.)
        circle_arr[count, 2] = rng.uniform(0.05, 0.1)

        tmp = 0
        for i in prange(count):
            if np.hypot(circle_arr[i, 0] - circle_arr[count, 0],
                        circle_arr[i, 1] - circle_arr[count, 1]) > \
                    circle_arr[count, 2] + circle_arr[i, 2]:
                tmp += 1
        if tmp == count:
            count += 1


@njit()
def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """
    This function calculates the direction of the agent
    :param boids: An array with information about agents
    :param dt: Time
    :return: array N x (x0, y0, x1, y1) for arrow painting
    """
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))


@njit()
def mean_arr(v: np.ndarray, num: int) -> np.ndarray:
    """
    This function calculates the average value
    in the array depending on the passed parameter num
    :param v: The array that the average value will be calculated from
    :param num: The value of the axis by which the average value will be calculated
    :return: An array with average values
    """
    if num == 0:
        res = np.empty(v.shape[1], dtype=float)
        for i in range(len(res)):
            res[i] = np.mean(v[:, i])
    else:
        res = np.empty(v.shape[0], dtype=float)
        for i in range(len(res)):
            res[i] = np.mean(v[i, :])

    return res


@njit()
def norma(v: np.ndarray, num: int) -> float:
    """
    This function calculates the norm of the vector
    :param v: The array for which the norm is calculated
    :param num: Axis value
    :return: The value of the norm
    """
    return np.sqrt(np.sum(v**2, axis=num))


@njit()
def vclip(v: np.ndarray, vrange: tuple[float, float]):
    """
    This function normalizes acceleration
    :param v: The array to be normalized for
    :param vrange: Boundaries for normalization
    :return: None
    """
    norm = norma(v, 1)
    mask = norm > vrange[1]
    if np.any(mask):
        v[mask] *= (vrange[1] / norm[mask]).reshape(-1, 1)


@njit()
def propagate(boids: np.ndarray,
              dt: float,
              vrange: tuple[float, float],
              arange: tuple[float, float]):
    """
    This function calculates the updated agent speeds
    :param boids: An array with information about agents
    :param dt: Time
    :param vrange: Boundaries for normalization
    :param arange: Boundaries for normalization
    :return: None
    """
    vclip(boids[:, 4:6], arange)
    boids[:, 2:4] += dt * boids[:, 4:6]
    vclip(boids[:, 2:4], vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]


@njit()
def distances(vecs: np.ndarray):
    """
    This function calculates the distance between the agents
    :param vecs: An array of agents between which distances are calculated
    :return: Distance
    """
    n, m = vecs.shape
    vecs = vecs.copy()
    delta = vecs.reshape((n, 1, m)) - vecs.reshape((1, n, m))
    d = norma(delta, 2)

    return d


@njit()
def distances_circ(vecs: np.ndarray, circs: np.ndarray):
    """
    This function calculates the distance between agents and circles
    :param vecs: Array of agents
    :param circs: An array of obstacles
    :return: Distance
    """
    n1, m1 = vecs.shape
    n2, m2 = circs.shape
    vecs, circs = vecs.copy(), circs.copy()
    delta = vecs.reshape((n1, 1, m1)) - circs.reshape((1, n2, m2))
    d = norma(delta, 2)

    return d


@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    """
    This function implements the acceleration cohesion component
    through the group's media center in the sector.
    :param boids: Array of agents
    :param idx: The number of a certain agent
    :param neigh_mask: Array mask
    :param perception: Error rate
    :return: Value of cohesion
    """
    center = mean_arr(boids[neigh_mask, :2], 0)
    a = (center - boids[idx, :2]) / perception

    return a


@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray) -> np.ndarray:
    """
    This function implements the process of separating
    the median acceleration within a group within a sector.
    :param boids: Array of agents
    :param idx: The number of a certain agent
    :param neigh_mask: Array mask
    :return: The average value of separation acceleration
    """
    neighbs = boids[neigh_mask, :2] - boids[idx, :2]
    norm = norma(neighbs, 1)
    mask = norm > 0
    if np.any(mask):
        neighbs[mask] /= norm[mask].reshape(-1, 1)
    d = mean_arr(neighbs, 0)
    norm_d = np.linalg.norm(d)
    if norm_d > 0:
        d /= norm_d

    return -d


@njit()
def check(boids: np.ndarray,
          idx: int,
          neigh_mask: np.ndarray,
          circle_arr: np.ndarray) -> np.ndarray:
    """
    This function simulates the repulsion of agents from obstacles
    :param boids: Array of agents
    :param idx: The number of a certain agent
    :param neigh_mask: Array mask
    :param circle_arr: An array of obstacles
    :return: The average value of separation acceleration
    """
    neighbs_ch = circle_arr[neigh_mask, :2] - boids[idx, :2]
    norm_ch = norma(neighbs_ch, 1)
    mask_ch = norm_ch > 0
    if np.any(mask_ch):
        neighbs_ch[mask_ch] /= norm_ch[mask_ch].reshape(-1, 1)
    d_ch = mean_arr(neighbs_ch, 0)
    norm_d_ch = np.linalg.norm(d_ch)
    if norm_d_ch > 0:
        d_ch /= norm_d_ch

    return -d_ch


@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:
    """
    This function implements acceleration equalization within the group sector
    :param boids: Array of agents
    :param idx: The number of a certain agent
    :param neigh_mask: Array mask
    :param vrange: Boundaries for normalization
    :return: Average acceleration by sector
    """
    v_mean = mean_arr(boids[neigh_mask, 2:4], 0)
    a = (v_mean - boids[idx, 2:4]) / (2 * vrange[1])

    return a


@njit()
def smoothstep(edge0: float,
               edge1: float,
               x: np.ndarray | float) -> np.ndarray | float:
    """
    This function smoothes the boundaries of the walls
    """
    x = np.clip((x - edge0) / (edge1 - edge0), 0., 1.)

    return x * x * (3.0 - 2.0 * x)


@njit()
def better_walls(boids: np.ndarray, asp: float, param: float):
    """
    This function calculates the boundaries of the screen
    :param boids: Array of agents
    :param asp: Constant value
    :param param: Wall width
    :return: Screen borders
    """
    x = boids[:, 0]
    y = boids[:, 1]
    w = param

    a_left = smoothstep(asp * w, 0.0, x)
    a_right = -smoothstep(asp * (1.0 - w), asp, x)

    a_bottom = smoothstep(w, 0.0, y)
    a_top = -smoothstep(1.0 - w, 1.0, y)

    return np.column_stack((a_left + a_right, a_bottom + a_top))


@njit()
def noise():
    """
    Noise reduction function
    :return: The value that will make the acceleration noise
    """
    v = np.random.rand(2)
    v[0] *= -1 if v[0] >= 1 / 2 else 1
    v[1] *= -1 if v[1] >= 1 / 2 else 1

    return v

@njit()
def mask_boids(boids: np.ndarray,
               perception: float):
    """
    This function applies a mask to the array
    :param boids: Array of agents
    :param perception: Error rate
    :return: An array of the boolean type
    """
    arr = distances(boids[:, :2])
    np.fill_diagonal(arr, perception + 1)

    return arr < perception


@njit()
def mask_circle(boids: np.ndarray,
                circle_arr: np.ndarray,
                perception: float):
    """
    This function applies a mask to the array
    :param boids: Array of agents
    :param circle_arr: An array of obstacles
    :param perception: Error rate
    :return: An array of the boolean type
    """
    arr = distances_circ(boids[:, :2], circle_arr[:, :2]) - circle_arr[:, 2]

    return arr < perception


@njit(parallel=True)
def flocking(boids: np.ndarray,
             circle_arr: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             vrange: tuple,
             order: float):
    """
    A function that simulates the interaction between agents
    :param boids: Array of agents
    :param circle_arr: An array of obstacles
    :param perception: Error rate
    :param coeffs: Coefficients of significance of various actions in calculating acceleration
    :param asp: Constant value
    :param vrange: Boundaries for normalization
    :param order: Wall width
    :return: None
    """
    size_arr = boids.shape[0]
    mask = mask_boids(boids, perception)
    mask_circ = mask_circle(boids, circle_arr, perception)
    wal = better_walls(boids, asp, order)
    for i in prange(size_arr):
        if not np.any(mask[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            alg = alignment(boids, i, mask[i], vrange)
            sep = separation(boids, i, mask[i])

        if not np.any(mask_circ[i]):
            ch = np.zeros(2)
        else:
            ch = check(boids,
                       i,
                       mask_circ[i],
                       circle_arr)

        ns = noise()
        a = coeffs[0] * coh + coeffs[1] * alg + \
            coeffs[2] * (sep + ch) + coeffs[3] * wal[i]
        'coeffs[4] * ns'

        boids[i, 4:6] = a
