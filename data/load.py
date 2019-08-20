from data import utils

def load(name: str):
  return utils.load_train_valid_test_csvs(name, path='data/datasets/'+name)
