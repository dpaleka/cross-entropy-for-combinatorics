# Code to accompany the paper "Constructions in combinatorics via neural networks and LP solvers" by A Z Wagner
# Code for Conjecture 2.3, using numba for speed
#
# Please keep in mind that I am far from being an expert in reinforcement learning. 
# If you know what you are doing, you might be better off writing your own code.


import networkx as nx #for various graph parameters, such as eigenvalues, macthing number, etc. Does not work with numba (yet)
import random
import numpy as np
from statistics import mean
import pickle
import time
import math
import matplotlib.pyplot as plt 
import torch
from torch import nn

from models import DenseNet
from training import train_network
from score import score_graph


N = 20   #number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
MYN = int(N*(N-1)/2)  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)

LEARNING_RATE = 0.01 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions = 200 #number of new sessions per iteration
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration

FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

n_actions = 2 #The size of the alphabet. In this file we will assume this is 2. There are a few things we need to change when the alphabet size is larger,
              #such as one-hot encoding the input, and using categorical_crossentropy as a loss function.
              
observation_space = 2*MYN #Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
                          #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
                          #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.

len_game = MYN 
state_dim = (observation_space,)


#Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
#I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
#It is important that the loss is binary cross-entropy if alphabet size is 2.
model = DenseNet([2*MYN, FIRST_LAYER_NEURONS, SECOND_LAYER_NEURONS, THIRD_LAYER_NEURONS, 1])

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
#Adam optimizer also works well, with lower learning rate

def state_to_graph(state):
    #construct the graph G
    adjMatG = np.zeros((N,N),dtype=np.int8) #adjacency matrix determined by the state
    edgeListG = np.zeros((N,N),dtype=np.int8) #neighbor list
    Gdeg = np.zeros(N,dtype=np.int8) #degree sequence
    count = 0
    for i in range(N):
        for j in range(i+1,N):
            if state[count] == 1:
                adjMatG[i][j] = 1
                adjMatG[j][i] = 1
                edgeListG[i][Gdeg[i]] = j
                edgeListG[j][Gdeg[j]] = i
                Gdeg[i] += 1
                Gdeg[j] += 1
            count += 1

    return adjMatG, edgeListG, Gdeg


def score_state(state):
    return score_graph(*state_to_graph(state))

plt.ion()

def display_graph(adjMatG):
    print("Best adjacency matrix in current step:")
    print(adjMatG)

    G = nx.convert_matrix.from_numpy_array(adjMatG)

    plt.clf()
    nx.draw_circular(G)

    plt.axis('equal')
    plt.draw()
    plt.pause(0.001)
    plt.show()

def play_game(n_sessions, actions,state_next,states,prob, step, total_score):
    
    for i in range(n_sessions):
        if np.random.rand() < prob[i]:
            action = 1
        else:
            action = 0
        actions[i][step-1] = action
        state_next[i] = states[i,:,step-1]

        if (action > 0):
            state_next[i][step-1] = action
        state_next[i][MYN + step-1] = 0
        if (step < MYN):
            state_next[i][MYN + step] = 1            
        #calculate final score
        terminal = step == MYN
        if terminal:
            total_score[i] = score_state(state_next[i])
    
        # record sessions 
        if not terminal:
            states[i,:,step] = state_next[i]
        
    return actions, state_next,states, total_score, terminal    


def generate_session(agent, n_sessions, verbose = 1):    
    """
    Play n_session games using agent neural network.
    Terminate when games finish 
    
    Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
    """
    states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
    actions = np.zeros([n_sessions, len_game], dtype = int)
    state_next = np.zeros([n_sessions,observation_space], dtype = int)
    prob = np.zeros(n_sessions)
    states[:,MYN,0] = 1
    step = 0
    total_score = np.zeros([n_sessions])
    pred_time = 0
    play_time = 0
    
    while (True):
        step += 1        
        tic = time.time()
        prob = agent(torch.from_numpy(states[:,:,step-1]).to(torch.float))
        prob = prob.detach().cpu().numpy()

        pred_time += time.time()-tic
        tic = time.time()
        actions, state_next, states, total_score, terminal = play_game(
                n_sessions, actions,state_next, states,prob, step, total_score)
        play_time += time.time()-tic
        
        if terminal:
            break
    if (verbose):
        print("Predict: "+str(pred_time)+", play: " + str(play_time))
    return states, actions, total_score
    

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
    
    This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
    If this function is the bottleneck, it can easily be sped up using numba
    """
    counter = n_sessions * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch,percentile)

    elite_states = []
    elite_actions = []
    elite_rewards = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold-0.0000001:        
            if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                for item in states_batch[i]:
                    elite_states.append(item.tolist())
                for item in actions_batch[i]:
                    elite_actions.append(item)            
            counter -= 1
    elite_states = np.array(elite_states, dtype = int)    
    elite_actions = np.array(elite_actions, dtype = int)    
    return elite_states, elite_actions
    
def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    """
    Select all the sessions that will survive to the next generation
    Similar to select_elites function
    If this function is the bottleneck, it can easily be sped up using numba
    """
    
    counter = n_sessions * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch,percentile)

    super_states = []
    super_actions = []
    super_rewards = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold-0.0000001:
            if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                counter -= 1
    super_states = np.array(super_states, dtype = int)
    super_actions = np.array(super_actions, dtype = int)
    super_rewards = np.array(super_rewards)
    return super_states, super_actions, super_rewards
    

super_states =  np.empty((0,len_game,observation_space), dtype = int)
super_actions = np.array([], dtype = int)
super_rewards = np.array([])
sessgen_time = 0
fit_time = 0
score_time = 0


myRand = random.randint(0,1000) #used in the filename

for i in range(1000000): #1000000 generations should be plenty
    #generate new sessions
    #performance can be improved with joblib
    tic = time.time()
    sessions = generate_session(model,n_sessions,0) #change 0 to 1 to print out how much time each step in generate_session takes 
    sessgen_time = time.time()-tic
    tic = time.time()
    
    states_batch = np.array(sessions[0], dtype = int)
    actions_batch = np.array(sessions[1], dtype = int)
    rewards_batch = np.array(sessions[2])
    states_batch = np.transpose(states_batch,axes=[0,2,1])
    
    states_batch = np.append(states_batch,super_states,axis=0)

    if i>0:
        actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)    
    rewards_batch = np.append(rewards_batch,super_rewards)
        
    randomcomp_time = time.time()-tic 
    tic = time.time()

    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
    select1_time = time.time()-tic

    tic = time.time()
    super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
    select2_time = time.time()-tic
    
    tic = time.time()
    super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
    super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
    select3_time = time.time()-tic
    
    tic = time.time()
    
    train_data = torch.from_numpy(np.column_stack((elite_states, elite_actions)))
    train_data = train_data.to(torch.float)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)
    train_network(model, optimizer, train_loader)
    fit_time = time.time()-tic
    
    tic = time.time()
    
    super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
    super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
    super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
    
    rewards_batch.sort()
    mean_all_reward = np.mean(rewards_batch[-100:])    
    mean_best_reward = np.mean(super_rewards)    

    score_time = time.time()-tic
    
    print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
    
    #uncomment below line to print out how much time each step in this loop takes. 
    print(    "Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
    
    
    display_graph(state_to_graph(super_actions[0])[0])

    if (i%20 == 1): #Write all important info to files every 20 iterations
        with open('best_species_pickle_'+str(myRand)+'.txt', 'wb') as fp:
            pickle.dump(super_actions, fp)
        with open('best_species_txt_'+str(myRand)+'.txt', 'w') as f:
            for item in super_actions:
                f.write(str(item))
                f.write("\n")
        with open('best_species_rewards_'+str(myRand)+'.txt', 'w') as f:
            for item in super_rewards:
                f.write(str(item))
                f.write("\n")
        with open('best_100_rewards_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(mean_all_reward)+"\n")
        with open('best_elite_rewards_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(mean_best_reward)+"\n")
    if (i%200==2): # To create a timeline, like in Figure 3
        with open('best_species_timeline_txt_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(super_actions[0]))
            f.write("\n")
