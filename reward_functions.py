import numpy as np


def goal_direction(dir_vector):
    last_pos = None

    def f(quat, pos):
        nonlocal last_pos
        if last_pos is None:
            last_pos = np.copy(pos)
            return 0.
        desired_pos = np.array(last_pos) + np.array(dir_vector)
        actual_pos = np.array(pos)
        squared_distance = np.linalg.norm(desired_pos - actual_pos) ** 2
        reward = -squared_distance
        last_pos = np.copy(pos)
        return reward

    return f


def cos_angle(dir_vector, limit_degrees=None):
    last_pos = None

    def f(quat, pos):
        nonlocal last_pos
        if last_pos is None:
            last_pos = np.copy(pos)
            return 0.
        vec = np.array(pos) - np.array(last_pos)
        cos = np.dot(vec, dir_vector) / (np.linalg.norm(vec) * np.linalg.norm(dir_vector))
        reward = cos
        if limit_degrees is not None:
            # Make sure we still penalise for going backward
            if reward > 0 and cos < np.cos(limit_degrees / 180 * np.pi):
                reward = 0.
        last_pos = np.copy(pos)
        return reward

    return f


reward_function_dict = {}

reward_function_dict['dummy'] = lambda quat, pos: 0.0

v = {
    'left': [0, -1, 0],
    'right': [0, 1, 0],
    'forward': [1, 0, 0],
    'backward': [-1, 0, 0],
    'up': [0, 0, 1],
    'down': [0, 0, -1]
}
# reward_function_dict['direction'] = {}
# reward_function_dict['goal'] = {}
reward_function_dict['cosangle'] = {}
reward_function_dict['cosanglelimit'] = {}
for dir_name, dir_vector in v.items():
    # reward_function_dict['direction'][dir_name] = lambda quat, pos: np.dot(pos, dir_vector)
    # reward_function_dict['goal'][dir_name] = goal_direction(dir_vector)
    reward_function_dict['cosangle'][dir_name] = cos_angle(dir_vector)
    reward_function_dict['cosanglelimit'][dir_name] = cos_angle(dir_vector,
                                                                limit_degrees=10)

# reward_function_dict['table_edges'] = {}
# above_table = lambda pos: pos[2] > 0.42
# e = 0.05
# reward_function_dict['table_edges']['left'] = lambda quat, pos: float(
#     above_table(pos) and pos[0] > 1.044 - e and pos[0] < 1.435 + e and pos[1] > 0.4 - e and pos[1] < 0.4 + e)
# reward_function_dict['table_edges']['right'] = lambda quat, pos: float(
#     above_table(pos) and pos[0] > 1.044 - e and pos[0] < 1.435 + e and pos[1] > 1.09 - e and pos[1] < 1.09 + e)
# reward_function_dict['table_edges']['back'] = lambda quat, pos: float(
#     above_table(pos) and pos[0] > 1.023 - e and pos[0] < 1.023 + e and pos[1] > 0.402 - e and pos[1] < 1.101 + e)
# reward_function_dict['table_edges']['front'] = lambda quat, pos: float(
#     above_table(pos) and pos[0] > 1.5 - e and pos[0] < 1.5 + e and pos[1] > 0.402 - e and pos[1] < 1.101 + e)
#
# reward_function_dict['orientation'] = lambda quat, pos: float(
#     all(np.isclose(quat, [np.sqrt(0.5), 0, np.sqrt(0.5), 0], atol=0.1)))
