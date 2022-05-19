import random,os
import pandas as pd
import numpy as np
from collections import defaultdict

#Dieses Modul ermöglicht die Transformation der 
#Benchmark-Instanzen in eine konkrete Simulationsumgebung
#Wird zu Beginn einer Epoche einmalig ausgeführt.

class Instance():
    
    
    def __init__(self, path, inst, set_random=False):
        
        self.path = path
        self.set_random = set_random
        #Falls eine zufällige Instanz ausgewählt werden soll
        if self.set_random == True:            
            self.instance_name = str(random.choice(os.listdir(self.path)))
            self.random_instance = "{}/{}".format(
                self.path, self.instance_name)
        else:
            self.instance_name = inst
            self.random_instance = "{}/{}.txt".format(
                self.path, self.instance_name)
        #Als Datenstruktur wird ein Dataframe verwendet
        self.instance_dataframe()
        self.opt = self.instance.sum().Avg_Time/self.nb_machines
        self.nb_operations = len(self.instance.index)
    
    def get_machine_dict(self,machine_times):

        #Erzeugt Wörterbuch für jede Maschine
        s = machine_times
        r = []       
        
        for i in range(1, self.nb_machines + 1):
            a = False
            for p in s:
                if "Machine {}".format(i) in p and not a:
                    r.append(p)
                    a = True
            if not a:
                r.append(("Machine {}".format(i), float('NaN')))   
        
        d = defaultdict(int)
        for k, v in r:
            d[k] = v
            
        return d
    
    def instance_dataframe(self):
        #Wandelt textDatei in dictionary um -> Importiert die Benchmark Instanz
        jobs = []
        pop = []

        with open(self.random_instance) as f:
            lines = f.readlines()
        
        first_line = lines[0].split()
        #Anzahl der Aufträge in der Instanz
        self.nb_jobs = int(first_line[0])
        #Anzahl der Maschinen in der Instanz
        self.nb_machines = int(first_line[1])
        
        for i in range(1, len(lines)):
            jobs.append(lines[i].split())
        #Durch die Text-Datei iterieren
        for i in range(len(jobs)):
            job = i
            op = 0
            pop_j = []
            j = 1

            while j <= len(jobs[i]) - 2:
                M = []
                T = []
                for k in range(1, int(jobs[i][j]) + 1):
                    M.append("Machine {}".format(
                        int(jobs[i][j + k + (k - 1)])))
                    T.append(int(jobs[i][j + k*2]))
                avg_time = sum(T)/float(len(T))
                list_1 = list(zip(M,T))
                machine_times = self.get_machine_dict(list_1)
                pop_j.append((int(job), int(op), avg_time, machine_times))
                j += (2*int(jobs[i][j]) + 1)
                op += 1
            
            pop.append(pop_j)
        #Beschreibung eines Agenten durch Job, Operatio, Status ...
        pop_dict = {
            "Job{0}".format(i):{
                "Operation{0}".format(j):{
                    "Job": pop[i][j][0],
                    "Operation":pop[i][j][1],
                    "Status": 0,
                    "t_rest":float('NaN'),
                    "T_start":float('NaN'),
                    "T_finished":float('NaN'),
                    "Done":False,
                    "T_queue":float('NaN'),
                    "Pre_Status":float('NaN'),
                    "Avg_Time":pop[i][j][2],
                    "Machine_chosen":float('NaN')
                } 
                for j in range(len(pop[i]))
            }
            for i in range(len(pop))
        } 
                
        for i in range(len(pop)):
            for j in range(len(pop[i])):
                pop_dict["Job{}".format(i)]["Operation{}".format(j)].update(
                    pop[i][j][-1])
                
        reform = {(
            outerKey, innerKey
            ): values for outerKey, innerDict in pop_dict.items(
            ) for innerKey, values in innerDict.items()}

        #Abschließend wird das Wörterbuch in einen Dataframe umgewandelt.
        self.instance = pd.DataFrame(reform).T   
        
        