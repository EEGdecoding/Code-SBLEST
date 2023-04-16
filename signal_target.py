import numpy as np
class SignalAndTarget(object):
    """
    Simple data container class.

    Parameters
    ----------
    X: 3darray or list of 2darrays
        The input signal per trial.
    y: 1darray or list
        Labels for each trial.
    """

    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y


def apply_to_X_y(fn, *sets):
    """
    Apply a function to all `X` and `y` attributes of all given sets.
    
    Applies function to list of X arrays and to list of y arrays separately.
    
    Parameters
    ----------
    fn: function
        Function to apply
    sets: :class:`.SignalAndTarget` objects

    Returns
    -------
    result_set: :class:`.SignalAndTarget`
        Dataset with X and y as the result of the
        application of the function.
    """
    X = fn(*[s.X for s in sets])
    y = fn(*[s.y for s in sets])
    return SignalAndTarget(X, y)

def convert_numbers_to_one_hot(arr):
    """
    将输入的一维数组中的数字1转换为[1, 0]，数字-1转换为[0, 1]，返回二维数组

    Args:
        arr (ndarray): 输入的一维数组

    Returns:
        ndarray: 转换后的二维数组
    """
    # 将输入数组转换为NumPy数组
    arr = np.array(arr)

    # 创建全零数组，形状为（arr长度，2）
    one_hot = np.zeros((arr.shape[0], 2))

    # 遍历输入数组
    for i, num in enumerate(arr):
        if num == 1:
            one_hot[i, 0] = 1
        elif num == -1:
            one_hot[i, 1] = 1

    return one_hot
