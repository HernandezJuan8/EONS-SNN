import configparser

def getHyperParameters(config: configparser)->dict:
    crossover_rate = float(config['Hyperparameters']['crossover_rate'])
    merge_rate = float(config['Hyperparameters']['merge_rate'])
    mutation_rate = float(config['Hyperparameters']['mutation_rate'])
    num_mutations = float(config['Hyperparameters']['num_mutations'])
    add_node_rate = float(config['Hyperparameters']['add_node_rate'])
    delete_node_rate = float(config['Hyperparameters']['delete_node_rate'])
    add_edge_rate = float(config['Hyperparameters']['add_edge_rate'])
    delete_edge_rate = float(config['Hyperparameters']['delete_edge_rate'])
    node_param_rate = float(config['Hyperparameters']['node_param_rate'])
    edge_param_rate = float(config['Hyperparameters']['edge_param_rate'])
    selection_type = config['Hyperparameters']['selection_type']
    random_factor = float(config['Hyperparameters']['random_factor'])
    num_best = float(config['Hyperparameters']['num_best'])
    population_size = float(config['Hyperparameters']['population_size'])
    num_generations = float(config['Hyperparameters']['num_generations'])
                         


    return {
            'crossover_rate': crossover_rate,
            'merge_rate': merge_rate,
            'mutation_rate': mutation_rate,
            'num_mutations': num_mutations,
            'add_node_rate': add_node_rate,
            'delete_node_rate': delete_node_rate,
            'add_edge_rate': add_edge_rate,
            'delete_edge_rate': delete_edge_rate,
            'node_param_rate': node_param_rate,
            'edge_param_rate': edge_param_rate,
            'selection_type': selection_type,
            'random_factor': random_factor,
            'num_best': num_best,
            'population_size': population_size,
            'num_generations': num_generations,       
    }



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    hyperparameters = getHyperParameters(config)
    print(hyperparameters)

