import collections

# named tuple for data parameters
Data_params = collections.namedtuple("Data_params", "n_users, n_items, n_train, n_test")

# named tuple for user dictionary, including train and test
User_dict = collections.namedtuple("User_dict", "train_user_dict, test_user_dict")
