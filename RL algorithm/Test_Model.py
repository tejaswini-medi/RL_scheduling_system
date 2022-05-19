from Instance import Instance
from Agent import Agent

import time as tm
import numpy as np
import torch
import sys
import plotly
import plotly.figure_factory as ff
import os
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio
import json
import matplotlib.pyplot as plt
import copy

##############Dieses Modul ermöglicht den Test der jeweiligen Actor-Modelle

# Verwendetes Modell
train_nr = 1
model_path = str(r"C:/Users/medit/OneDrive/Desktop/User Interface/Algorithm_Version_2021/Algorithm_Version_2021/Models/")
path = "Instances"  #all instances in this file will be tested
ITERATIONS = int(sys.argv[1])  #INPUT1: iterations to average the results?
use_cuda = False
save_results_to = 'Figures/'

time_list = []
ergebnisse = [[], []]


def test(rule):
    memory = []

    print("\n \n Testing in Progress")
    global new_policy
    # Initialisiere Instanz
    instance_class = Instance(path, inst)
    instance = instance_class.instance
    nb_machines = instance_class.nb_machines
    # Initialisiere Agenten für alle Maschinen
    machines = [Agent(i, instance_class, use_cuda) for i in range(1, nb_machines + 1)]

    time = 0
    for machine in machines:
        machine.set_traveling()
        machine.set_buffer(time)

    # Run with new policy
    while not instance["Done"].all():

        for machine in machines:
            if machine.processing.empty:
                machine.update()
                machine.set_buffer(time)
                if not machine.buffer.empty:
                    if len(machine.buffer) > 1:
                        do_action_1(time, machine, machines, new_policy, nb_machines, rule)
                        step(machine, instance)

                    elif len(machine.buffer) == 1:
                        job = machine.buffer["Job"].values[0]
                        operation = machine.buffer["Operation"].values[0]
                        machine.execute_action(job, operation, time, 5, machines)
                        machine.number_of_actions += 1
                        step(machine, instance)

            elif not (machine.processing.empty) and machine.processing["t_rest"].values[0] == 0:

                machine.set_done(time)
                machine.set_traveling()
                machine.set_buffer(time)
                machine.update()

                if len(machine.buffer) > 1:
                    do_action_1(time, machine, machines, new_policy, nb_machines, rule)
                    step(machine, instance)
                elif len(machine.buffer) == 1:
                    job = machine.buffer["Job"].values[0]
                    operation = machine.buffer["Operation"].values[0]
                    machine.execute_action(job, operation, time, 5, machines)
                    machine.update()
                    machine.number_of_actions += 1
                    step(machine, instance)

            elif not machine.processing.empty and machine.processing["t_rest"].values[0] > 0:
                step(machine, instance)
        time += 1
    new_policy = copy.deepcopy(global_network)
    makespan = time
    print("Testing finished, your makespan is {}".format(time))
    glob_reward = []
    for i in machines:
        memory.append(i.machine_memory)
        x = sum(i.sumreward) / len(i.sumreward) if len(i.sumreward) > 0 else 0
        glob_reward.append(x)

    print(instance)
    return instance, memory, time_list, makespan, np.mean(glob_reward), instance_class.opt, instance_class.instance_name


def do_action_1(time, machine, machines, new_policy, nb_machines, rule):
    state = machine.transform_state(nb_machines, time)
    actions = machine.transform_action()
    predicted_action = machine.predict_action(state, new_policy)

    optimal_action, feature_index, a_list, b_list = machine.sample_optimal_action(predicted_action)
    best_action, best_action_parameters = machine.search_best_fitting_action(
        optimal_action, actions, feature_index, rule)
    job, operation = best_action[0]
    s_t, s_t_1, reward, action = machine.execute_action(job, operation, time, 5, machines)


def do_action_2(time, machine, machines, new_policy, nb_machines, job, operation):
    s_t, s_t_1, reward, action = machine.execute_action(job, operation, time, 5, machines)


def step(machine, instance):
    machine.update()
    row = ("Job{}".format(machine.processing["Job"].values[0]),
           "Operation{}".format(machine.processing["Operation"].values[0]))
    instance.at[row, "t_rest"] -= 1
    machine.update_proc(machine.processing["Job"].values[0], machine.processing["Operation"].values[0])


def runner(rule):
    makespan_list = []
    performance_list = []
    iteration_list = []
    loss_list = []
    step_nb = 0
    step_list = []
    makespan_total = 0
    for i in range(ITERATIONS):
        instance, memory, tm_list, makespan, glob_reward, opt, name = test(rule)
        makespan_list.append(makespan)
        iteration_list.append(i)
        performance_list.append(opt / makespan)
        msg = "Statusupdate\n Aktuelle Makespan= {}, Iterationen= {}, reward= {}, performance={}, name={}".format(
            makespan, i, glob_reward, opt / makespan, name)
        print(msg)

        df = [dict(
            Task=instance["Machine_chosen"][i],
            Start=instance["T_start"][i],
            Finish=instance["T_finished"][i],
            Resource="Job {}".format(instance["Job"][i])) for i in range(len(instance))]
        # print(df)
        fig = ff.create_gantt(df, group_tasks=True, title="Makespan for {}".format(name))
        fig['layout']['xaxis'].update({'type': None})
        makespan_total = makespan_total + makespan

    return makespan_total / ITERATIONS, name, fig


# decision_rule = ["FIFO"] #KOZ #rnd
decision_rule = ["model"]

for k in decision_rule:
    dict_model = {}
    for i in os.listdir(str(path)):
        inst = str(i[:-4]) #instance is selected
        model_name = "model_{}_{}".format(inst, train_nr) # model for the instance is loaded
        model = model_path + model_name
        print(model)
        global_network = torch.load(model)
        old_policy = copy.deepcopy(global_network)
        new_policy = copy.deepcopy(global_network)
        makespan, name, fig = runner(k)
        dict_model.update({str(name): makespan})
        # static_image_bytes = pio.to_image(fig, format='png')
        # pio.write_image(fig, file='Gantt-Charts/{}_{}.png'.format(name, k), format='png')
        with open('Testresults/result_{}_{}.json'.format(k, model_name), 'w') as fp:
            json.dump(dict_model, fp)

# runner()
