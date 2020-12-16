import numpy as np

_inverse_direction_dict = {0: 1, # up -> down
                           1: 0, # down -> up
                           2: 3, # left -> right
                           3: 2} # right -> left


def argmax_k(k_t, t, t_, n, n_, g, phi):
    # k_ = 0
    val0 = g[k_t, 0] - phi[t][n][k_t] - phi[t_][n_][0]
    # k_ = 1
    val1 = g[k_t, 1] - phi[t][n][k_t] - phi[t_][n_][1]
    if val0 < val1:
        return 1
    else:
        return 0


def calculate_k(t, k_t, img_w, img_h, g, phi):
    # column index
    i = t // img_w
    # row index
    j = t % img_w

    dict_best_k = dict()
    # up
    if i > 0:
        t_ = t - img_w
        dict_best_k[0] = (argmax_k(k_t, t, t_, 0, _inverse_direction_dict[0], g, phi), t_)
    # down
    if i < img_h - 1:
        t_ = t + img_w
        dict_best_k[1] = (argmax_k(k_t, t, t_, 1, _inverse_direction_dict[1], g, phi), t_)
    # left
    if j > 0:
        t_ = t - 1
        dict_best_k[2] = (argmax_k(k_t, t, t_, 2, _inverse_direction_dict[2], g, phi), t_)
    # right
    if j < img_w - 1:
        t_ = t + 1
        dict_best_k[3] = (argmax_k(k_t, t, t_, 3, _inverse_direction_dict[3], g, phi), t_)
    return dict_best_k


def calculate_c(best_ks, t, k, g, phi, q):
    length = len(best_ks)
    c = 0
    for key_direction, (k_star, t_) in best_ks.items():
        c += g[k, k_star]
        c -= phi[t_][_inverse_direction_dict[key_direction]][k_star]
    c += q(k, t)
    c /= length
    return c


def update_phi(best_ks, c, t, k, g, phi):
    for key_direction, (k_star, t_) in best_ks.items():
        phi[t][key_direction][k] = g[k, k_star] - phi[t_][_inverse_direction_dict[key_direction]][k_star] - c



### prediction part ###
def calculate_max(k_t, t, t_, n, n_, g, phi):
    """
    >>> calculate_max(0, 0, 0, 0, 0, np.array([[0, 1],[0, 0]]), np.array([[[0, 1],[0, 1],[0, 0]]]))
    0
    >>> calculate_max(0, 0, 1, 0, 0, np.array([[0, 1],[0, 0]]), np.array([[[0, 1],[0, 1]],[[0, 1],[0, 1]]]))
    0
    >>> calculate_max(0, 1, 1, 0, 1, np.array([[1.7, 1],[2.5, 6.6]]), np.array([[[3.1, 1],[0, 1]],[[0, -1],[0, 1.4]]]))
    1.7
    >>> calculate_max(1, 1, 1, 0, 1, np.array([[1.7, 1],[2.5, 6.6]]), np.array([[[3.1, 1],[0, 1]],[[0, -1],[0, 1.4]]]))
    6.199999999999999
    >>> calculate_max(1, 0, 1, 0, 0, np.array([[1.7, 1],[2.5, 6.6]]), np.array([[[3.1, 1],[0, 1]],[[0, -1],[0, 1.4]]]))
    6.6
    >>> calculate_max(1, 0, 1, 0, 0, np.array([[1.7, 1],[-2.5, -6.6]]), np.array([[[3.1, 1],[0, 1]],[[0, -1],[0, 1.4]]]))
    -3.5
    >>> calculate_max(0, 0, 0, 0, 0, np.array([[1.7, 1],[-2.5, -6.6]]), np.array([[[3.1, 1],[0, 1]],[[0, -1],[0, 1.4]]]))
    -3.1
    >>> calculate_max(1, 1, 1, 1, 1, np.array([[1.7, 1],[-2.5, -6.6]]), np.array([[[3.1, 1],[0, 1]],[[0, -1],[0, 1.4]]]))
    -3.9
    >>> calculate_max(0, 1, 0, 1, 0, np.array([[1.7, 1],[25, -6.6]]), np.array([[[3.1, 1],[0, 1]],[[0, -1],[0, 1.4]]]))
    0.0
    >>> calculate_max(0, 1, 0, 1, 0, np.array([[1.7, 1],[25, -6.6]]), np.array([[[3.1, 1],[0, 12]],[[12, -1],[0, 13.4]]]))
    0.0
    """
    # k_ = 0
    val0 = g[k_t, 0] - phi[t][n][k_t] - phi[t_][n_][0]
    # k_ = 1
    val1 = g[k_t, 1] - phi[t][n][k_t] - phi[t_][n_][1]
    return max(val0, val1)


def predict(img_h, img_w, g, phi):
    mask = np.ones((img_h, img_w))
    for t in range(img_h * img_w):
        t_ = None
        max_k0 = None
        max_k1 = None
        # column index
        i = t // img_w
        j = t % img_w
        # down
        if i < img_h - 1:
            t_ = t + img_w
            max_k0 = calculate_max(0, t, t_, 1, _inverse_direction_dict[1], g, phi)
            max_k1 = calculate_max(1, t, t_, 1, _inverse_direction_dict[1], g, phi)
        else:
            t_ = t - img_w
            max_k0 = calculate_max(0, t, t_, 0, _inverse_direction_dict[0], g, phi)
            max_k1 = calculate_max(1, t, t_, 0, _inverse_direction_dict[0], g, phi)
        if max_k1 >= max_k0:
            mask[i][j] = 1
        else:
            mask[i][j] = 0
    return mask