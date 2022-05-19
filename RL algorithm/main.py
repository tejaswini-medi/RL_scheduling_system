from Instance import Instance
from Actor import ActorNetwork
from Critic import CriticNetwork
from Agent import Agent
import pandas as pd
import time as tm
import numpy as np
import torch
import sys
import os
import seaborn as sns
import time
import random
import matplotlib.pyplot as plt
import copy

#Instance to train on
path = 'Instances'
#train_nr = sys.argv[1] #INPUT1: personal count
train_nr = 4
#lb = 100 #lower bound from for this specific instance from literature, for graphs
#ub = 110 #upper bound from for this specific instance from literature, for graphs
RANDOM = False #Instanz fürs Training zufällig einlesen?

#Netzwerkparameter
hidden_nr = 2
TRAINING_EPOCHS = 5
BATCH_SIZE = 3
HORIZON = 5
#ITERATIONS = int(sys.argv[2]) #INPUT2: number of training iterations
ITERATIONS =5
HIDDEN_SIZE = 100
LR = 10**-3
C1 = 1  

#Learning params
GAMMA = 0.99
LAMBDA = 0.96
epsilon = 0.2
clip = 5

#Soll GPU-Beschleunigung CUDA verwendet werden?
#if sys.argv[3] == "True": #INPUT3
 #   use_cuda = True
#else:
 #   use_cuda = False

use_cuda = False
save_results_to = 'Figures/'

time_list = []
#Initialisiere Actor-Netzwerk
actor_network = ActorNetwork(6, HIDDEN_SIZE, use_cuda, LR, GAMMA, LAMBDA, epsilon, C1, clip)

#Initialisiere Critic-Network
critic_network = CriticNetwork(6, HIDDEN_SIZE, use_cuda, LR)

#Falls GPU-Beschleunigung CUDA verwendet werden soll
#########SSM: Do we need this if-else-statment? in a similar form it is part of all other scripts
if use_cuda == True:
    actor_network.cuda()
    critic_network.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type("torch.FloatTensor")


old_policy = copy.deepcopy(actor_network)
new_policy = copy.deepcopy(actor_network)


#Trainings-Algorithmus
def train(a_glob, b_glob):

    memory = []
    print("\n \n Training in Progress")
    global new_policy

    # Baue Instanz aus Pfad
    instance_class = Instance(path, inst, set_random=RANDOM)
    # Extrahiere Parameter aus Instanz
    instance = instance_class.instance

    nb_machines = instance_class.nb_machines
    # Initialisiere Agenten für jede Maschine
    machines = [Agent(
        i,instance_class,use_cuda) for i in range(1, nb_machines + 1)]

    #Setze Simulationszeit auf 0
    time = 0
    #Initialisiere Warteschlange der Agenten
    for machine in machines:
        machine.set_traveling()
        machine.set_buffer(time)
    #Führe Simulation aus
    while not instance["Done"].all():

        #Die Agenten werden nacheinander simuliert
        #Algorithmus wie in Masterarbeit
        for machine in machines:            
            if machine.processing.empty:
                machine.update()
                machine.set_buffer(time)

                if not machine.buffer.empty:

                    #Führt nur die entsprechende Vorhersage aus,
                    #wenn mehr als ein Arbeitsgang vor der Maschine wartet
                    if len(machine.buffer) > 1:

                        a_list, b_list = do_action_1(
                            time, machine, machines, new_policy, nb_machines)
                        a_glob.append(a_list)
                        b_glob.append(b_list)
                        step(machine, instance)
                        
                    elif len(machine.buffer) == 1:

                        job = machine.buffer["Job"].values[0]
                        operation = machine.buffer["Operation"].values[0]
                        machine.execute_action(
                            job, operation, time, 5, machines)
                        machine.number_of_actions += 1
                        step(machine, instance)

                        
                    
            elif not (
                machine.processing.empty and machine.buffer.empty) and (
                machine.processing["t_rest"].values[0] == 0):

                machine.set_done(time)
                machine.set_traveling()
                machine.set_buffer(time)
                machine.update()

                if len(machine.buffer) > 1:

                    a_list, b_list = do_action_1(
                        time, machine, machines, new_policy, nb_machines)
                    a_glob.append(a_list)
                    b_glob.append(b_list)
                    step(machine,instance)

                elif len(machine.buffer) == 1:

                    job = machine.buffer["Job"].values[0]
                    operation = machine.buffer["Operation"].values[0]

                    machine.execute_action(job, operation, time, 5, machines)
                    machine.update()
                    machine.number_of_actions += 1
                    step(machine, instance)

            elif not machine.processing.empty and (
                machine.processing["t_rest"].values[0] > 0):

                step(machine, instance)

        time += 1

    #Aktualisiere globale Strategie
    new_policy = copy.deepcopy(actor_network)
    makespan = time
    print("Training finished, your makespan is {}".format(time))
    glob_reward = []
    for i in machines:
        memory.append(i.machine_memory)
        x = sum(i.sumreward)/len(i.sumreward) if len(i.sumreward) > 0 else 0
        glob_reward.append(x)
        
    return instance, memory, time_list, makespan, np.mean(glob_reward), instance_class.opt, instance_class.instance_name, a_glob, b_glob

def do_action_1(time, machine, machines, new_policy, nb_machines):

    #Transformiere Zustand in Zustandsparameter
    state = machine.transform_state(nb_machines, time)
    #Transformiere Aktionen in Aktionsparameter
    actions = machine.transform_action()

    #Sage Aktion und Nutzen vorher
    predicted_action, predicted_value = machine.predict_action_and_value(
        state, new_policy, critic_network)

    #Optimale Aktion wird aus den Beta-Verteilungen gesamplet
    optimal_action, probs, a_list, b_list = machine.sample_optimal_action(
        predicted_action)

    #Beste Aktion wird durch euklidische Distanz ermittelt
    best_action, best_action_parameters = machine.search_best_fitting_action(
        optimal_action, actions, probs)  

    job, operation = best_action[0]

    #Die Aktion wird ausgeführt und die Beobachtungen gespeichert
    s_t, s_t_1, reward, action = machine.execute_action(
        job, operation, time, 5, machines)

    #Vorhersage V(S)
    v_state = critic_network.forward(torch.tensor(s_t))
    v_state = v_state.view([1, len(v_state)]).flatten()

    #Vorhersage V(S')
    v_state_1 = critic_network.forward(torch.tensor(s_t_1))
    v_state_1 = v_state_1.view([1, len(v_state_1)]).flatten()

    #Sichere alle Beobachtungen in Maschinenspeicher
    machine.save_to_machine_memory(
        machine.number_of_actions, reward, best_action_parameters, 
        predicted_value, optimal_action, s_t, s_t_1, v_state, v_state_1)

    machine.number_of_actions += 1

    return a_list, b_list

def do_action_2(
        time, machine, machines, new_policy, 
        nb_machines, job, operation):

    #Diese Aktion wird nur ausgeführt, wenn Warteschlange = 1
    s_t, s_t_1, reward, action = machine.execute_action(
        job, operation, time, 5, machines)

    v_state = critic_network.forward(torch.tensor(s_t))
    v_state = v_state.view([1, len(v_state)]).flatten()

    v_state_1 = critic_network.forward(torch.tensor(s_t_1))
    v_state_1 = v_state_1.view([1, len(v_state_1)]).flatten()

    machine.save_to_machine_memory(
        machine.number_of_actions, reward, best_action_parameters,
        predicted_value, optimal_action, s_t, 
        s_t_1, v_state, v_state_1)

    machine.number_of_actions += 1

def step(machine, instance):

    #Reduziert die Bearbeitungszeit auf der Maschine um 1
    machine.update()
    row=(
        "Job{}".format(machine.processing["Job"].values[0]),
        "Operation{}".format(machine.processing["Operation"].values[0]))

    instance.at[row, "t_rest"] -= 1

    machine.update_proc(
        machine.processing["Job"].values[0],
        machine.processing["Operation"].values[0])



def trajectories_to_sample(memory, makespan, opt):

    #Kombiniert alle Trajektorien der einzelnen Agenten zu einem Batch
    sample = [[], [], [], [], [], [], []]

    for i in memory:
        if len(i["State"]) > 0:
            #Berechnung der Vorteilsfunktion
            trajectory = actor_network.compute_advantage(
                i, HORIZON, makespan, opt, critic_network)

            for i in range(len(sample)):
                sample[i].append(trajectory[i])

    for i in range(len(sample)):
        sample[i] = torch.cat(sample[i])

    #Randomisiere Batch, sodass später random Mini-Batch gezogen werden kann
    a = random.sample(range(0, len(sample[0])), len(sample[0]))
    sample = [sample[i][a] for i in range(len(sample))]

    return sample

def sample_mini_batch(sample, begin, batchsize):
    #Ermittelt so viele Mini-Batches, bis Buffer leer ist
    end = begin + batchsize
    end = end if end < len(sample[0]) else len(sample[0])

    a = []
    a.extend(range(begin, end))

    mini_batch = [sample[i][a] for i in range(len(sample))]

    return mini_batch, end



def update_network(
        memory, new_policy, old_policy, loss_list, step_list, 
        step, critic_loss_list, critic_step_list, 
        critic_step_nb, makespan, opt):

    #Funktion um Actor und Critic zu updaten
    global actor_network  
    global critic_network

    sample = trajectories_to_sample(memory, makespan, opt)
    begin = 0

    while begin < len(sample[0]):
        #Ermittle Mini-Batch
        mini_batch, begin = sample_mini_batch(sample, begin, BATCH_SIZE)
        
        #Update Actor
        loss_list, step_list, step = actor_network.backwards(
            mini_batch, old_policy, loss_list, 
            step_list, step, TRAINING_EPOCHS, critic_network)

        #Update Critic
        critic_loss_list, critic_step_list, critic_step_nb = critic_network.backwards(
            mini_batch, critic_loss_list, 
            critic_step_list, critic_step_nb, TRAINING_EPOCHS)

    #Sichere alte Policy, aktualisiere neue Policy
    old_policy = copy.deepcopy(new_policy)
    new_policy = copy.deepcopy(actor_network)

    return old_policy, new_policy, loss_list, step_list, step, critic_loss_list, critic_step_list, critic_step_nb

makespan_list = []
performance_list = []
iteration_list = []  
loss_list = []
step_nb = 0
critic_loss_list = []
critic_step_list = []
critic_step_nb = 0
step_list = []  
cpu_time = time.time()
for i in os.listdir(str(path)):
    inst = str(i[:-4])
    #Simulationsepochen
    for i in range(ITERATIONS):
        a_glob = []
        b_glob = []
        #Train 
        instance, memory, tm_list, makespan, glob_reward, opt, name, a_glob, b_glob = train(a_glob, b_glob)
        
        old_policy,new_policy,loss_list,step_list,step_nb,critic_loss_list,critic_step_list,critic_step_nb=update_network(memory,new_policy,old_policy,loss_list,step_list,step_nb,critic_loss_list,critic_step_list,critic_step_nb,makespan,opt)
        #Tracke CPU-Zeit
        cpu_time_iteration = time.time() - cpu_time
        #Speichere erreichte Durchlaufzeit
        makespan_list.append(makespan)
        iteration_list.append(i)
        #Speichere erreichte Performance
        performance_list.append(opt/makespan)
        #Statusupdate
        msg = "Statusupdate\n Aktuelle Makespan= {}, Iterationen= {}, reward= {}, performance={}, name={}".format(makespan,i,glob_reward,opt/makespan,name)
        print(msg)
    
    # #Visualisiere alle 10 Epochen
    # if i % 10 == 0 or i == 999:
    #     sns.set(style="darkgrid")

    #     #Aktionsvorhersage
    #     plt.figure(figsize=(12, 5))
    #     x = plt.plot(a_glob, b_glob, '.')
    #     plt.legend(x, ('$p_1$', '$p_2$', '$p_3$', '$p_4$', '$p_5$', '$p_6$'))
    #     plt.xlabel("a")
    #     plt.ylabel("b")
    #     plt.savefig(save_results_to + 'actions_{}_.png'.format(train_nr), dpi=300)
    #     plt.close()

    #     #Durchlaufzeit
    #     plt.figure(figsize=(12, 5))
    #     x = pd.DataFrame(iteration_list)
    #     y = pd.DataFrame(makespan_list)

    #     #Gleitender Durchschnitt
    #     yhat = y.rolling(20).mean()
    #     #plt.plot(x, y, label='agent')

    #     #Bounds und Optima
    #     if RANDOM == False:
    #         plt.plot(x, x*0 + opt, color='green', label='theoretical opt')
    #         plt.plot(x, x*0 + lb, color='red', label='lower bound')
    #         plt.plot(x, x*0 + ub, color='indianred', label='upper bound')
    #         plt.plot(x, x*0 + min(makespan_list), color='darkmagenta', label='upper bound agent')
    #         plt.plot(x, yhat, color='olive',label='moving avg. (10)')
    #     plt.gcf().text(0,0.5, ' iterations:{}\n batchsize:{}\n hidden size:{}\n hidden layers:{}\n lr:{}\n gamma:{}\n lambda:{}\n epsilon:{}\n instance:{}\n rand={}\n C1={}\n cpu-time=\n{}\n min={}\n mean20={}\n opt{}'.format(
    #             ITERATIONS, BATCH_SIZE, HIDDEN_SIZE, hidden_nr, LR,
    #             GAMMA, LAMBDA, epsilon, inst, RANDOM, C1, 
    #             float('%.3f'%(cpu_time_iteration)), float('%.3f'%(min(makespan_list))),
    #             y.rolling(20).mean().tail(1), opt), {"fontsize": 10})

    #     plt.subplots_adjust(left=0.2)
    #     plt.title("makespan for instance {}".format(inst))
    #     plt.xlabel("epoch")
    #     plt.ylabel("makespan")
    #     plt.legend(loc='upper left')
    #     plt.savefig(
    #         save_results_to + 'makespan_{}_.png'.format(train_nr), dpi = 300)
    #     plt.close()

    #     #Actor-Loss
    #     plt.figure(figsize=(12, 5))
    #     x = pd.DataFrame(step_list)
    #     y = pd.DataFrame(loss_list)
    #     yhat = y.rolling(50).mean()
    #     plt.plot(x, y, label='agent')
    #     plt.plot(x, yhat, color='olive', label='moving avg. (50)')
    #     plt.gcf().text(0, 0.5, ' iterations:{}\n batchsize:{}\n hidden size:{}\n hidden layers:{}\n lr:{}\n gamma:{}\n lambda:{}\n epsilon:{}\n instance:{}\n rand={}\n C1={}\n cpu-time=\n{}  mean20={}\n'.format(
    #             ITERATIONS, BATCH_SIZE, HIDDEN_SIZE, hidden_nr, LR,
    #             GAMMA, LAMBDA, epsilon, inst, RANDOM, C1,
    #             float('%.3f'%(cpu_time_iteration)),  y.rolling(20).mean().tail(1)),  {"fontsize": 10})
    #     plt.subplots_adjust(left=0.2)
    #     plt.title("policy loss for instance {}".format(inst))
    #     plt.xlabel("iterations")
    #     plt.ylabel("policy loss")
    #     plt.legend(loc='upper left')
    #     plt.savefig(
    #         save_results_to + 'actor_loss_{}_.png'.format(train_nr), dpi = 300)
    #     plt.close()

    #     #Critic-Loss
    #     plt.figure(figsize=(12, 5))
    #     x = pd.DataFrame(critic_step_list)
    #     y = pd.DataFrame(critic_loss_list)
    #     yhat = y.rolling(50).mean()
    #     plt.plot(x, y, label='agent')
    #     plt.plot(x, yhat, color='olive', label='moving avg. (50)')
    #     plt.gcf().text(0, 0.5, ' iterations:{}\n batchsize:{}\n hidden size:{}\n hidden layers:{}\n lr:{}\n gamma:{}\n lambda:{}\n epsilon:{}\n instance:{}\n rand={}\n C1={}\n cpu-time=\n{} mean20={}\n'.format(
    #             ITERATIONS, BATCH_SIZE, HIDDEN_SIZE, hidden_nr,  LR,
    #             GAMMA, LAMBDA, epsilon, inst, RANDOM, C1, 
    #             float('%.3f'%(cpu_time_iteration)), y.rolling(20).mean().tail(1)), {"fontsize": 10})
    #     plt.subplots_adjust(left=0.2)
    #     plt.xlabel("iteration")
    #     plt.ylabel("value loss")
    #     plt.title("value loss for instance {}".format(inst))
    #     plt.legend(loc='upper left')
    #     plt.savefig(
    #         save_results_to + 'value_loss_{}_.png'.format(train_nr), dpi = 300)
    #     plt.close()

    #     #Performance
    #     plt.figure(figsize=(12, 5))
    #     x = pd.DataFrame(iteration_list)
    #     y = pd.DataFrame(performance_list)
    #     yhat = y.rolling(20).mean()

    #     plt.plot(x, y, label='agent')
    #     if RANDOM == False:
    #         plt.plot(x, yhat, color='olive', label='moving avg. (10)')
    #         plt.plot(x, x*0 + 1, color='green', label='theoretical opt.')
    #         plt.plot(x, x*0 + opt/lb, color='red', label='lower bound')
    #         plt.plot(x, x*0 + opt/ub, color='indianred', label='upper bound')
    #         plt.plot(x, x*0 + max(performance_list), color='darkmagenta', label='upper bound agent')
    #     plt.gcf().text(0, 0.5, ' iterations:{}\n batchsize:{}\n hidden size:{}\n hidden layers:{}\n lr:{}\n gamma:{}\n lambda:{}\n epsilon:{}\n instance:{}\n rand={}\n C1={}\n cpu-time=\n{}\n max={}\n mean20={}'.format(
    #             ITERATIONS, BATCH_SIZE, HIDDEN_SIZE, hidden_nr, LR,
    #             GAMMA, LAMBDA, epsilon, inst, RANDOM, C1,
    #             float('%.3f'%(cpu_time_iteration)), float('%.3f'%(max(performance_list))),
    #             y.rolling(20).mean().tail(1)), {"fontsize": 10})

    #     plt.subplots_adjust(left=0.2)
    #     plt.xlabel("epoch")
    #     plt.ylabel("performance")        
    #     plt.title("performance for instance {}".format(inst))
    #     plt.legend(loc='upper left')
    #     plt.savefig(
    #         save_results_to + 'performance_{}_.png'.format(train_nr), dpi = 300)
    #     plt.close()
    # #Speichere das Torch-Modell und die zugehörige Parametrierung
torch.save(new_policy,  "Models/model_{}_{}".format(inst, train_nr))
