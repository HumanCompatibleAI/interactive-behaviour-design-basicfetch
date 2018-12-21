import numpy as np

reward_function_dict = {}

reward_function_dict['dummy'] = lambda quat, pos: 0.0

reward_function_dict['direction'] = {}
reward_function_dict['direction']['left'] = lambda quat, pos: np.dot(pos, [0, -1, 0])
reward_function_dict['direction']['right'] = lambda quat, pos: np.dot(pos, [0, 1, 0])
reward_function_dict['direction']['forward'] = lambda quat, pos: np.dot(pos, [1, 0, 0])
reward_function_dict['direction']['backward'] = lambda quat, pos: np.dot(pos, [-1, 0, 0])
reward_function_dict['direction']['up'] = lambda quat, pos: np.dot(pos, [0, 0, 1])
reward_function_dict['direction']['down'] = lambda quat, pos: np.dot(pos, [0, 0, -1])

reward_function_dict['table_edges'] = {}
above_table = lambda pos: pos[2] > 0.42
e = 0.05
reward_function_dict['table_edges']['left'] = lambda quat, pos: float(above_table(pos) and pos[0] > 1.044 - e and pos[0] < 1.435 + e and pos[1] > 0.4 - e and pos[1] < 0.4 + e)
reward_function_dict['table_edges']['right'] = lambda quat, pos: float(above_table(pos) and pos[0] > 1.044 - e and pos[0] < 1.435 + e and pos[1] > 1.09 - e and pos[1] < 1.09 + e)
reward_function_dict['table_edges']['back'] = lambda quat, pos: float(above_table(pos) and pos[0] > 1.023 - e and pos[0] < 1.023 + e and pos[1] > 0.402 - e and pos[1] < 1.101 + e)
reward_function_dict['table_edges']['front'] = lambda quat, pos: float(above_table(pos) and pos[0] > 1.5 - e and pos[0] < 1.5 + e and pos[1] > 0.402 - e and pos[1] < 1.101 + e)

reward_function_dict['orientation_only'] = lambda quat, pos: float(all(np.isclose(quat, [np.sqrt(0.5), 0, np.sqrt(0.5), 0], atol=0.1)))
