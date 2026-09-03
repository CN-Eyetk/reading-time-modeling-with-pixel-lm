import yaml
from fixations_to_plot import GazePlotter
def read_parameters(path_to_yaml):
    with open(path_to_yaml, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data

def load_plotter(path_to_yaml):
    paras = read_parameters(path_to_yaml)
    return GazePlotter(**paras)
if __name__ == "__main__":
    print(read_parameters("paras.yml"))