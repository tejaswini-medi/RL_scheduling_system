import pandas as pd
import numpy as np
import torch
import math
import copy
import random
import time as tm

class Agent():
    
    #Initialisiere Agenten
    def __init__(self,machine,instance_class,use_cuda):

        #Falls GPU-Beschleunigung CUDA verwendet werden soll
        if use_cuda == True:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        self.machine_memory = {
                "Index":[],
                "State":[],
                "Action":[],
                "Optimal_Action":[],
                "Reward":[],
                "State_1":[],
                "Value":[],
                "delta":[],
                "advantage":[],
                "r_disc":[],
                "loss":[],
                "v_states":[],
                "v_states_1":[],
                }
        
        self.machine = machine
        
        self.instance = instance_class.instance
        self.opt = instance_class.opt
        
        
        self.sumreward = []
        self.loss = 0
        
        self.number_of_actions = 0
        
        self.buffer = pd.DataFrame()
        self.processing = pd.DataFrame()
        
        self.set_traveling()
        
        
    def set_traveling(self):
        self.instance.loc[(self.instance["Machine {}".format(self.machine)].notnull()) & (self.instance["Status"] == 0),["Status"]] = 1

    def set_buffer(self,time):
        #Buffer -> Status = 2
        self.instance.Pre_Status=self.instance.Status.shift(periods=1)
        row=self.instance.index[(((self.instance["Pre_Status"] == 4) & (
                self.instance["Operation"] != 0))|(self.instance["Operation"] == 0)) & (self.instance["Status"]==1) & (self.instance["Machine {}".format(self.machine)].notnull())]
        self.instance.at[row,["T_queue"]] = time
        self.instance.at[row,["Status"]] = 2
        self.buffer=self.instance.loc[(((self.instance["Pre_Status"] == 4) & (
                self.instance["Operation"] != 0))|(self.instance["Operation"] == 0)) & ((self.instance["Status"]==1)|(self.instance["Status"] == 2)) & (self.instance["Machine {}".format(self.machine)].notnull())]
        
    def set_processing(self,job,operation,time):
        #Processing -> Status = 3
        row = self.instance.index[(self.instance["Job"] == job) & (self.instance["Operation"] == operation)].tolist()
        self.instance.at[row,["Status"]] = 3

        self.instance.at[row,["T_start"]] = time

        self.instance.at[row,["t_rest"]] = self.instance.loc[(self.instance["Job"] == job) & (self.instance["Operation"] == operation),"Machine {}".format(self.machine)]
        self.instance.at[row,["Machine_chosen"]] = self.machine

        self.processing = self.instance.loc[(self.instance["Job"] == job) & (self.instance["Operation"] == operation)]
        
        
    def set_done(self,time):
        #Done -> Status = 4
        row=self.instance.index[(self.instance["Status"] == 3) & (self.instance["t_rest"] == 0)].tolist()
        self.instance.at[row, 'Done'] = True
        self.instance.at[row,"T_finished"] = time
        self.instance.at[row,"Status"] = 4
        
        self.processing = pd.DataFrame()
        
    def update_proc(self,job,operation):
        self.processing = self.instance.loc[(self.instance["Job"] == job) & (self.instance["Operation"] == operation)]
        
    def update(self):
        #Aktualisiere Warteschlange
        rows = self.instance.index[(self.instance["Machine {}".format(self.machine)].notnull()) & (
                (self.instance["Operation"] == 0) | ((self.instance["Operation"] != 0) & 
                    (self.instance["Pre_Status"] == 4))) & ((self.instance["Status"] == 1) | (self.instance["Status"] == 2))]
        self.buffer = self.instance.loc[rows]   
    
    
    def transform_state(self,nb_machines,time):
        #Diese Funktione dient dazu, den aktuellen Zustand der Maschine in eine Form
        #zu bringen, die von dem Neuronalen netz als Input verwendet werden kann.
        
        #Verfügbare Aktionen sind die Operationen die in der aktuellen maschine
        #in der Menge "Warteschlange" zu finden sind.
        actions = self.buffer
        #Faktoren:
        #Bearbeitungszeit der Operationen min,max
        #Bearbeitungszeit sämtlicher noch ausstehender Operationen in den jeweiligen Jobs min,max
        #Anzahl der Gesamtbearbeitungszeit im Job       
        
        #Bearbeitungszeit der Operationen
        proc_times = actions["Machine {}".format(self.machine)].values
        t_proc_min = min(proc_times)
        t_proc_max = max(proc_times)
        
        #Bearbeitungszeit sämtlicher noch ausstehender Operationen in den jeweiligen Jobs
        tj_rest = []
        #Anzahl der Gesamtbearbeitungszeit im Job
        tj_total = []
        
        rem_ops_total = []
        rem_ops = 0
        rem_times = 0
        t_waits = []
        #sämtliche Operationen im jeweiligen Job:
        total_times_machine = sum(self.instance.loc[self.instance["Machine {}".format(self.machine)].notnull()]["Machine {}".format(self.machine)].values)
        for index,row in actions.iterrows():
            job = row["Job"]
            operation = row["Operation"]
            total_times = sum(self.instance.loc[(self.instance["Job"] == job)]["Avg_Time"].values)
            rem_times = sum(self.instance.loc[(self.instance["Operation"] >= operation) & (self.instance["Job"] == job)]["Avg_Time"].values)
            rem_ops += len(self.instance.loc[(self.instance["Operation"] >= operation) & (self.instance["Job"] == job)]["Avg_Time"].values)
            #print(rem_ops)
            rem_ops_1 = len(self.instance.loc[(self.instance["Operation"] >= operation) & (self.instance["Job"] == job)]["Avg_Time"].values)
            tj_rest.append(rem_times)
            tj_total.append(total_times)
            rem_ops_total.append(rem_ops_1)
            t_wait = time - self.instance.loc[(self.instance["Operation"] == operation) & (self.instance["Job"] == job)]["T_queue"].values
            t_wait = t_wait.item()
            t_waits.append(t_wait)
        rem_times_machine = sum(self.instance.loc[(self.instance["Machine {}".format(self.machine)].notnull()) & (self.instance["Done"] == False)]["Machine {}".format(self.machine)].values)
        rem_ops_machine = len(self.instance.loc[(self.instance["Machine {}".format(self.machine)].notnull()) & (self.instance["Done"] == False)]["Machine {}".format(self.machine)].values)

        t_wait_min = min(t_waits)
        t_wait_max = max(t_waits)

        tj_rest_min = min(tj_rest)
        tj_rest_max = max(tj_rest)
        tj_total_min = min(tj_total)
        tj_total_max = max(tj_total)
        tj_rest_sum = sum(tj_rest)
        rem_ops_min = min(rem_ops_total)
        rem_ops_max = max(rem_ops_total)
        len(proc_times)
        return [len(proc_times) if len(proc_times) > 0 else 0, 
        tj_rest_min/self.opt, 
        tj_rest_max/self.opt, 
        t_proc_min/t_proc_max if t_proc_max>0 else 0, 
        tj_rest_min/tj_rest_max if tj_rest_max>0 else 0, 
        rem_ops_min/rem_ops_max]
   
    
    def transform_action(self,action=None):
        #Diese Transformation dient dazu, sämtliche Aktionen mithilfe von klaren
        #aussagekräftigen Skalarwerten darzustellen

        actions = self.buffer if action is None else action
        #Time of getting in Buffer = FIFO
        T_queue_total = actions["T_queue"].values
        T_queue_total_sum = sum(T_queue_total)
        #total processing time over all operations in Buffer = KOZ
        t_proc_total = actions["Machine {}".format(self.machine)].values
        #print(t_proc_total)
        t_proc_total_sum = sum(t_proc_total)
        #total remaining processing times over all jobs = KRZ
        tj_rest = []
        #total job processing times over all jobs = KGB
        tj_total = []
        #number of operations that need to be processed over all jobs = WAA
        j_wait = []
        
        #depending
        depending_list = []
        dependend_list = []
        
        #identifier (job,Operation)
        ident_list = []
        for index, row in actions.iterrows():
            ident = (int(row["Job"])), int(row["Operation"])
            rem_times = sum(self.instance.loc[(self.instance["Operation"]>=ident[1])&(self.instance["Job"]==ident[0])]["Avg_Time"].values)
            total_job_times = sum(self.instance.loc[(self.instance["Job"]==ident[0])]["Avg_Time"].values)
            j_waiting = len(self.instance.loc[(self.instance["Operation"]>ident[1])&(self.instance["Job"]==ident[0])]["Avg_Time"].values)
            depending = j_waiting
            dependend = ident[1] - 1 if ident[1] > 0 else 0
            total = depending + dependend + 1
            
            depending_list.append(depending/total)
            dependend_list.append(dependend/total)
            
            ident_list.append(ident)
            
            tj_total.append(total_job_times)
            tj_rest.append(rem_times)
            j_wait.append(j_waiting)
            
        tj_rest_total_sum=sum(tj_rest)
        tj_total_sum=sum(tj_total)
        nb_j_rest_total_sum=sum(j_wait)

        return [[ident_list[i],[T_queue_total[i]/T_queue_total_sum if T_queue_total_sum!=0 else 1,
                             t_proc_total[i]/t_proc_total_sum if t_proc_total_sum!=0 else 1,
                             tj_rest[i]/tj_rest_total_sum if tj_rest_total_sum!=0 else 1,
                             tj_total[i]/tj_total_sum if tj_total_sum !=0 else 1,
                             j_wait[i]/nb_j_rest_total_sum if nb_j_rest_total_sum!=0 else 1,
                             depending_list[i]
                             ]] for i in range(len(depending_list))]
    
    
    def predict_action(self,transformed_state,actor_policy):
        #Vorhersage konkreter Aktionswerte auf Basis eines Eingangswertes (transformierter Zustand)
        #und des aktuellen neuronalen Netzes
        model_input = torch.tensor(transformed_state)
        prediction_action = actor_policy.forward(model_input)        
        return prediction_action

    def predict_action_and_value(self,transformed_state,actor_policy,critic_policy):
        #Vorhersage konkreter Aktionswerte auf Basis eines Eingangswertes (transformierter Zustand)
        #und des aktuellen neuronalen Netzes
        model_input = torch.tensor(transformed_state)

        prediction_action = actor_policy.forward(model_input)
        prediction_value = critic_policy.forward(model_input)
        return prediction_action,prediction_value
    
    def sample_optimal_action(self,prediction_action):
        #Auf Basis der vorhergesagten Parameter (action)
        optimal_action = []
        probs = []
        a_list = []
        b_list = []
        for i in range(len(prediction_action[0])):
            #Für jede der oben angegebenen Parameter die die Aktionen beschreiben
            #Wird eine kontinuierliche Aktion auf Basis einer Beta-Verteilung,
            #die durch alpha und beta charakterisiert ist gesamplet.
            #print(prediction_action[1][i])
            alpha = prediction_action[0][i]
            beta = prediction_action[1][i]

            m = torch.distributions.beta.Beta(alpha,beta)

            optimal_action_parameter = m.sample()
            a_list.append(alpha.item())
            b_list.append(beta.item())

            optimal_action.append(optimal_action_parameter.item())
            probs.append(torch.exp(m.log_prob(optimal_action_parameter)))

        return optimal_action, probs, a_list, b_list
    
    def search_best_fitting_action(self, optimal_action, transformed_actions, probs, rule="model"):
        best_action = ["none",math.inf]

        for i in transformed_actions:
            name = i[0]
            val = i[1]  

            #Hier lässt sich die entsprechende Regel auswählen
            #-> Zu Testzwecken genutzt.  
            #FIFO       
            if rule == "FIFO":
                distance = val[0]
            #KOZ
            elif rule == "KOZ":
                distance = val[1]
            #LOZ
            elif rule == "LOZ":
                distance = 1 - val[1]
            #KRZ
            elif rule == "KRZ":
                distance = val[2]
                
            elif rule == "rnd":
                distance = random.randint(1,101)
            
            #Berechne euklidische Distanz zwischen optimalen Aktionsparametern und
            #verfügbaren Aktionsparametern
            else:
                distance = sum((abs(optimal_action[r] - val[r])) for r in range(len(val)))

            if best_action[1] > distance:
                best_action = [name,distance]
                action_parameters = val

        return best_action, action_parameters
    
    def execute_action(self, job, operation, time, nb_machines, machines):
        action = self.buffer.loc[(self.buffer["Job"] == job) & (self.buffer["Operation"] == operation)]
        action_trans = self.transform_action(action=action)

        state_t_trans = self.transform_state(nb_machines,time)
        state_t_trans = copy.deepcopy(state_t_trans)
        self.set_processing(job,operation,time)
        self.set_buffer(time)

        if not self.buffer.empty:
	        state_t_1 = copy.deepcopy(self.instance)

	        state_t_1_trans = self.transform_state(nb_machines,time)
	        state_t_1_trans = copy.deepcopy(state_t_1_trans)
	        reward = self.calculate_reward()        
	        action_trans = action_trans[0][1]
        	return state_t_trans, state_t_1_trans, reward, action_trans
    
    def calculate_reward(self):
        #Wird nicht mehr benötigt, da Reward nun am Ende der Episode ausgeschüttet wird.
        utilisation = -(len(self.buffer))
        reward = utilisation
        reward = torch.tensor(reward,dtype=torch.float)
        self.sumreward.append(reward)
        return reward
    
    def save_to_machine_memory(self, obs, reward, action, value, optimal_action, state, state_1, v_state, v_state_1):
        #Speichert alle relevanten Beobachtungen im Maschinenspeicher
        self.machine_memory["Index"].append(obs)
        self.machine_memory["Reward"].append(reward)
        self.machine_memory["Action"].append(action)
        self.machine_memory["Optimal_Action"].append(optimal_action)
        self.machine_memory["State"].append(state)
        self.machine_memory["State_1"].append(state_1)
        self.machine_memory["v_states"].append(v_state)
        self.machine_memory["v_states_1"].append(v_state_1)
        
        

        