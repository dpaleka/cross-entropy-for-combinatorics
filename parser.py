import argparse

def get_args(): 
    parser = argparse.ArgumentParser(description='Reinforcement learning for graph counterexamples')
    args = vars(parser.parse_args())
    return args


