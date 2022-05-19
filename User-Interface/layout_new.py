import sys
from Instance import *
from draw_graph import*
import re
import math
import csv
import copy
import time

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import layout_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    layout_support.set_Tk_var()
    top = Toplevel1 (root)
    layout_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Toplevel1(root, *args, **kwargs)' .'''
    global w, w_win, root
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    layout_support.set_Tk_var()
    top = Toplevel1 (w)
    layout_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('vista')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("607x450+461+138")
        top.minsize(120, 1)
        top.maxsize(1920, 1080)
        top.resizable(1,  1)
        top.title("User Interface for Job Shop Scheduling")
        top.configure(background="#d9d9d9")

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.0, rely=0.0, relheight=0.1, relwidth=1.002)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#c0c0c0")

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0, rely=0.222)
        self.Label2.configure(background="#c0c0c0")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font="-family {Segoe UI} -size 9 -weight bold")
        self.Label2.configure(foreground="#000000")

        self.Label1 = tk.Label(self.Frame1)
        self.Label1.place(relx=0.313, rely=0.222, height=15, width=84)
        self.Label1.configure(background="#c0c0c0")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font="-family {Segoe UI} -size 9 -weight bold")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Select Instance''')

        self.TCombobox1 = ttk.Combobox(self.Frame1)
        self.TCombobox1['values'] = (' la01 (10 Jobs, 5 Machines)',
                                          ' la02 (10 Jobs, 5 Machines)',
                                          ' la03 (10 Jobs, 5 Machines)',
                                          ' la04 (10 Jobs, 5 Machines)',
                                          ' la05 (10 Jobs, 5 Machines)',
                                          ' la06 (15 Jobs, 5 Machines)',
                                          ' la07 (15 Jobs, 5 Machines)',
                                          ' la08 (15 Jobs, 5 Machines)',
                                          ' la09 (15 Jobs, 5 Machines)',
                                          ' la10 (15 Jobs, 5 Machines)',
                                          ' la11 (20 Jobs, 5 Machines)',
                                          ' la12 (20 Jobs, 5 Machines)',
                                          ' la13 (20 Jobs, 5 Machines)',
                                          ' la14 (20 Jobs, 5 Machines)',
                                          ' la15 (20 Jobs, 5 Machines)',
                                          ' la16 (10 Jobs, 10 Machines)',
                                          ' la17 (10 Jobs, 10 Machines)',
                                          ' la18 (10 Jobs, 10 Machines)',
                                          ' la19 (10 Jobs, 10 Machines)',
                                          ' la20 (10 Jobs, 10 Machines)',
                                          ' la25 (15 Jobs, 10 Machines)',
                                          ' la30 (20 Jobs, 10 Machines)',
                                          ' la35 (30 Jobs, 10 Machines)',
                                          ' la40 (15 Jobs, 15 Machines)',
                                          ' Mk01 (10 Jobs, 6 Machines)',
                                          ' Mk02 (10 Jobs, 6 Machines)',
                                          ' Mk03 (15 Jobs, 8 Machines)',
                                          ' Mk04 (15 Jobs, 8 Machines)',
                                          ' Mk05 (15 Jobs, 4 Machines)',
                                          ' Mk06 (10 Jobs, 15 Machines)',
                                          ' Mk07 (20 Jobs, 5 Machines)',
                                          ' Mk08 (20 Jobs, 10 Machines)',
                                          ' Mk09 (20 Jobs, 10 Machines)',
                                          ' Mk10 (20 Jobs, 15 Machines)')
        self.TCombobox1.place(relx=0.464, rely=0.222, relheight=0.489
                , relwidth=0.186)
        self.TCombobox1.configure(textvariable=layout_support.combobox)
        self.TCombobox1.configure(takefocus="")
        self.TCombobox1.bind("<<ComboboxSelected>>", self.select_instance)

        self.Frame2 = tk.Frame(top)
        self.Frame2.place(relx=0.0, rely=0.778, relheight=0.211, relwidth=1.008)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#c0c0c0")

        self.Scrolledwindow2 = ScrolledWindow(self.Frame2)
        self.Scrolledwindow2.place(relx=0.0, rely=0.0, relheight=1.021
                , relwidth=0.84)
        self.Scrolledwindow2.configure(background="white")
        self.Scrolledwindow2.configure(borderwidth="2")
        self.Scrolledwindow2.configure(highlightbackground="#d9d9d9")
        self.Scrolledwindow2.configure(highlightcolor="black")
        self.Scrolledwindow2.configure(insertbackground="black")
        self.Scrolledwindow2.configure(relief="groove")
        self.Scrolledwindow2.configure(selectbackground="blue")
        self.Scrolledwindow2.configure(selectforeground="white")
        self.color = self.Scrolledwindow2.cget("background")
        self.Scrolledwindow2_f = tk.Frame(self.Scrolledwindow2,
                            background=self.color)
        self.Scrolledwindow2.create_window(0, 0, anchor='nw',
                                           window=self.Scrolledwindow2_f)

        self.TButton2 = ttk.Button(self.Frame2, command=self.calculate_processing_time)
        self.TButton2.place(relx=0.899, rely=0.411, height=25, width=76)
        self.TButton2.configure(takefocus="")
        self.TButton2.configure(text='''Download''')

        self.Frame3 = tk.Frame(top)
        self.Frame3.place(relx=0.0, rely=0.111, relheight=0.656, relwidth=0.557)
        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#d9d9d9")
        self.Frame3.configure(cursor="arrow")

        self.Scrolledwindow1 = ScrolledWindow(self.Frame3)
        self.Scrolledwindow1.place(relx=0.0, rely=0.0, relheight=1.007
                , relwidth=0.988)
        self.Scrolledwindow1.configure(background="white")
        self.Scrolledwindow1.configure(borderwidth="2")
        self.Scrolledwindow1.configure(highlightbackground="#d9d9d9")
        self.Scrolledwindow1.configure(highlightcolor="black")
        self.Scrolledwindow1.configure(insertbackground="black")
        self.Scrolledwindow1.configure(relief="groove")
        self.Scrolledwindow1.configure(selectbackground="blue")
        self.Scrolledwindow1.configure(selectforeground="white")
        self.color = self.Scrolledwindow1.cget("background")
        self.Scrolledwindow1_f = tk.Frame(self.Scrolledwindow1,
                            background=self.color)
        self.Scrolledwindow1.create_window(0, 0, anchor='nw',
                                           window=self.Scrolledwindow1_f)
        self.Frame4 = tk.Frame(top)
        self.Frame4.place(relx=0.567, rely=0.111, relheight=0.656
                , relwidth=0.437)
        self.Frame4.configure(relief='groove')
        self.Frame4.configure(borderwidth="2")
        self.Frame4.configure(relief="groove")
        self.Frame4.configure(background="#c0c0c0")

        self.Canvas1 = tk.Canvas(self.Frame4)
        self.Canvas1.place(relx=0.038, rely=0.068, relheight=0.858
                , relwidth=0.925)
        self.Canvas1.configure(background="#d9d9d9")
        self.Canvas1.configure(borderwidth="2")
        self.Canvas1.configure(insertbackground="black")
        self.Canvas1.configure(relief="ridge")
        self.Canvas1.configure(selectbackground="blue")
        self.Canvas1.configure(selectforeground="white")

        self.TButton1 = ttk.Button(self.Frame1, command=self.destroyFrame)
        self.TButton1.place(relx=0.663, rely=0.244, height=25, width=76)
        self.TButton1.configure(takefocus="")
        self.TButton1.configure(text='''Reset''')


        self.name_label = ttk.Label(self.Frame2, text = 'Username', font=('calibre',10, 'bold'))
        self.name_var=tk.StringVar()
        self.name_entry = ttk.Entry(self.Frame2,textvariable = self.name_var, font=('calibre',10,'normal'))
        self.name_label.place(relx=0.850, rely=0.111, height=25, width=76)
        self.name_entry.place(relx=0.900, rely=0.111, height=25, width=150)
        self.list_time = []

    def resetMaps(self):
        self.machine_UI = {}
        self.machine_list = []
        self.job_operation_map = {}
        self.job_operation_time_map = {}
        self.machines_map = {}
        self.machine_operations_assigned = {}
        self.job_map = {}
        self.no_of_jobs = 0
        self.machine_timelines = {}
        self.machine_job_colors = {}
        self.job_colors = {}
        self.color_mult = 0
        self.list_item = []
        time.sleep(3)


    def clearFrame(self, frame):
        # destroy all widgets from frame
        for widget in frame.winfo_children():
            widget.destroy()

    def destroyFrame(self):
        self.resetMaps()
        self.TCombobox1.set('')
        self.clearFrame(self.Scrolledwindow1_f)
        self.clearFrame(self.Scrolledwindow2_f)
        self.clearFrame(self.Canvas1)

    def populate_machines_jobs_map(self, dataframe):
        job_operation_map = {}
        job_operation_time_map = {}
        machines_map = {}
        machine_list = []
        job_map = {}
        no_of_jobs = 0
        column_names = dataframe.columns
        for column_name in column_names:
            if "Machine " in column_name:
                machine_list.append(column_name)
        job_operation_times = dataframe[machine_list]
        job_operation_indexes = job_operation_times.index.values
        for (job, operation) in job_operation_indexes:
            job_no = int(re.findall(r'\d+', job)[0])
            operation_no = int(re.findall(r'\d+', operation)[0])
            if job_no in job_map:
                job_map[job_no].append(operation_no)
            else:
                no_of_jobs += 1
                job_map[job_no] = [operation_no]
            for machine in machine_list:
                time = job_operation_times[machine].loc[[(job, operation)]].values[0]
                if machine.lower() not in machines_map:
                    machines_map[machine.lower()] = []
                if not math.isnan(time):
                    key = job+"_"+operation+"_"+str(time)
                    job_operation_key = job+"_"+operation
                    operation_no = int(re.findall(r'\d+', operation)[0])
                    if operation_no == 0:
                        if machine.lower() in machines_map:
                            machines_map[machine.lower()].append(key)
                        else:
                            machines_map[machine.lower()] = [key]
                    job_operation_time_map[job_operation_key] = time

                    if job_operation_key in job_operation_map:
                        job_operation_map[job_operation_key].append(machine.lower())
                    else:
                        job_operation_map[job_operation_key] = [machine.lower()]
        self.machine_list = machine_list
        self.job_operation_map = job_operation_map
        self.job_operation_time_map = job_operation_time_map
        self.machines_map = machines_map
        self.job_map = job_map
        self.no_of_jobs = no_of_jobs
        self.color_mult = len(colors) // (self.no_of_jobs)
        self.job_colors['None'] = colors[0]
        for job_no in range(no_of_jobs):
            self.job_colors['job'+str(job_no)] = colors[(job_no + 1) * self.color_mult]
        for machine in self.machine_list:
            self.machine_job_colors[machine.lower()] = ()
            self.machine_timelines[machine.lower()] = [(0, 0)]
            self.machine_job_colors[machine.lower()] = self.machine_job_colors[machine.lower()] + (colors[0],)

    def update_machine_jobs_list(self, eventObject):
        self.list_time = ['0']
        machine_name = eventObject.widget._name.split('_')[0]
        button_value = eventObject.widget.cget('text')
        if button_value == 'idle':
            self.machine_UI[machine_name]['text'].insert(tk.END, 'idle, ')
        else:
            job_operation_time = button_value.split('_')
            job_operation = job_operation_time[0]+'_'+job_operation_time[1]
            time = int(job_operation_time[2])
            self.machine_UI[machine_name]['text'].insert(tk.END, job_operation+', ')
            for machine in self.job_operation_map[job_operation]:
                for job_operation_of_machine in self.machines_map[machine]:
                    temp = job_operation_of_machine.split('_')
                    temp_job_operation = temp[0] + '_' + temp[1]
                    if temp_job_operation == job_operation:
                        self.machines_map[machine].remove(job_operation_of_machine)
            #job_no = int(re.findall(r'\d+', job_operation_time[0])[0])
            next_operation_no = int(re.findall(r'\d+', job_operation_time[1])[0]) + 1
            next_job_operation = job_operation_time[0] + '_' + 'Operation' + str(next_operation_no)
            if next_job_operation in self.job_operation_time_map:
                next_job_operation_time = next_job_operation + '_' + str(self.job_operation_time_map[next_job_operation])
                for machine in self.job_operation_map[next_job_operation]:
                    self.machines_map[machine].append(next_job_operation_time)
            for machine in self.machine_UI:
                for key, button in self.machine_UI[machine]['button'].items():
                    button.destroy()
                del(self.machine_UI[machine]['button'])
                self.machine_UI[machine]['button'] = {}
                self.machine_UI[machine]['label'] = {}
                # grid_forget() is used to unmap any widget from toplevel and can be retrieved back
                self.machine_UI[machine]['select_frame'].grid_forget()       
                self.machine_UI[machine]['selected_frame'].grid_forget()
            # The following lines of code is for obtaining machine with least processing time.
            if machine_name in self.machine_operations_assigned:
                self.machine_operations_assigned[machine_name].append((job_operation, time))
            else:
                self.machine_operations_assigned[machine_name] = [(job_operation, time)]
        self.calculate_processing_time(plotOnly=True)
        load_balance_dict = {}
        lowest_machine_time = 0
        for machine, timeline in self.machine_timelines.items():
            machine_time = 0
            for start, time in timeline:
                end = start + time
                if end > 0 and end > machine_time:
                    machine_time = end
            if machine_time > 0:
                if lowest_machine_time > 0:
                    if machine_time < lowest_machine_time :
                        lowest_machine_time = machine_time
                else:
                    lowest_machine_time = machine_time
            size_timeline = len(timeline)
            load_balance_dict[machine] =timeline[size_timeline-1][0] + timeline[size_timeline-1][1]
        sorted_keys = sorted(load_balance_dict, key=load_balance_dict.get)
        x= len(sorted_keys)
        for i in range(x):
            z= sorted_keys[:x-i]
        z = ','.join(z)
        machine = z # Here machine is the machine with least processing time 
        self.machine_list = [x.lower() for x in self.machine_list]
        if machine in self.machine_list:
             #displaying only least processing time machine frame on window
            self.machine_UI[machine]['select_frame'].grid(sticky = 'n')
        cpt = ['0']  # current processing time list for all job_operations of a machine
        progress = ['NA'] # progress list for progress of all operations of a machine
        if machine in self.machines_map.keys():
            job_operation_list = self.machines_map[machine]
            for i in job_operation_list:
                j = re.findall('\d+', i )
                j = list(map(int, j))
                progress.append(int(j[1]+1))
                q = i.rsplit('_', 1)
                cpt.append(str(q[1]))           
                if q[0] in self.job_operation_time_map:
                    values = self.job_operation_time_map.values()
                    values_list = list(values)
                    remaining = values_list[(j[0]*5)+j[1]+1:(j[0]+1)*5]
                    remaining_processing_time = sum(remaining)
                    rpt = str(remaining_processing_time)
                    self.list_time.append(rpt) # self.list_time is list of all remaining processing times of particular machine job_operations
            self.hide_all_frames(machine) 
            # label for displaying the machine with least processing time 
            ttk.Label(self.machine_UI[machine]['select_frame'], text='-----------------------')\
                                                                            .grid(column=0,row=0,sticky='n')
            ttk.Label(self.machine_UI[machine]['select_frame'], text=machine+': ').grid(column=0, row=0+1, sticky='n')
            
            # Displaying required cpt,rpt,progress of a machine on window    
            for k, value in enumerate(['idle'] + self.machines_map[machine]):
                self.machine_UI[machine]['button'][value] = ttk.Button(self.machine_UI[machine]['select_frame'], text=value,
                                                                                     name=machine + '_' + value.lower())
                self.machine_UI[machine]['button'][value].grid(column=(k%1)+1, row=self.machine_UI[machine]['row_count']+1+(int(k/1)), sticky='nesw',
                                                                             padx=3, pady=3)
                self.machine_UI[machine]['button'][value].bind("<Button-1>", self.update_machine_jobs_list)
                    
                
                button1 = ttk.Button(self.machine_UI[machine]['select_frame'],name=machine + '_1' + value.lower())
                button1.config(state=tk.NORMAL)
                button1.config(text = 'CPT:' + cpt[k])
                button1.grid(column=(k%1)+2, row=self.machine_UI[machine]['row_count']+1+(int(k/1)), sticky='nesw',
                                                       padx=5, pady=5)
                button2= ttk.Button(self.machine_UI[machine]['select_frame'],name=machine + '_2' + value.lower())
                button2.config(state=tk.NORMAL)
                button2.config(text='RPT:'+self.list_time[k])
                button2.grid(column=(k%1)+3, row=self.machine_UI[machine]['row_count']+1+(int(k/1)), sticky='nesw',
                                                                       padx=5, pady=5)
                button3 = ttk.Button(self.machine_UI[machine]['select_frame'],name=machine + '_3' + value.lower())
                button3.config(state=tk.NORMAL)
                button3.config(text = 'Progess:' + str(progress[k]) + ' /5')
                button3.grid(column=(k%1)+4, row=self.machine_UI[machine]['row_count']+1+(int(k/1)), sticky='nesw',
                                                                           padx=5, pady=5)
                
    def hide_all_frames(self,machine): # Destroy widgets of particular machine frame
        for widget in self.machine_UI[machine]['select_frame'].winfo_children():
            widget.destroy()

    def default_value(self):
        return 0

    def update_machine_graph_maps(self, machine, start, time, job_no):
        color_mult = len(colors) // self.no_of_jobs
        if machine in self.machine_timelines:
            self.machine_timelines[machine].append((start, time))
            self.machine_job_colors[machine] = self.machine_job_colors[machine] + (colors[(job_no+1) * color_mult],)
        else:
            self.machine_job_colors[machine] = ()
            self.machine_timelines[machine] = [(start, time)]
            self.machine_job_colors[machine] = self.machine_job_colors[machine] + (colors[(job_no+1) * color_mult],)

    def calculate_processing_time(self, plotOnly=False):
        # self.machine_operations_assigned = {'machine 1': [('Job0_Operation0', 12), ('Job1_Operation3', 21), ('Job1_Operation4', 87)], 'machine 2': [('Job0_Operation4', 7), ('Job1_Operation0', 19)], 'machine 3': [('Job0_Operation1', 94), ('Job0_Operation2', 92), ('Job1_Operation1', 11)], 'machine 5': [('Job0_Operation3', 91)], 'machine 4': [('Job1_Operation2', 66)]}
        # self.job_map = {0: [0, 1, 2, 3, 4], 1: [0, 1, 2, 3, 4]}
        # self.machine_list = ['machine 1', 'machine 2','machine 3','machine 4','machine 5' ]
        max_total_time = -1
        machine_operations = copy.deepcopy(self.machine_operations_assigned)
        job_map = copy.deepcopy(self.job_map)
        flag = True
        processing_time = defaultdict(self.default_value)
        idle_time = defaultdict(self.default_value)
        job_timestamp = {}
        machine_timestamp = {}
        # self.machine_timelines = {}
        # self.machine_job_colors = {}
        while(flag):
            machine_operation_left = True
            for machine, job_operation_arr in machine_operations.items():
                if len(job_operation_arr) > 0:
                    job_operation, time = job_operation_arr[0]
                    job, operation = job_operation.split('_')
                    job_no = int(re.findall(r'\d+', job)[0])
                    operation_no = int(re.findall(r'\d+', operation)[0])
                    if len(job_map[job_no]) > 0 and job_map[job_no][0] == operation_no:
                        machine_operation_left = False
                        processing_time[machine] += time
                        job_map[job_no].pop(0)
                        job_operation_arr.pop(0)
                        if machine in machine_timestamp:
                            if job in job_timestamp:
                                if machine_timestamp[machine] < job_timestamp[job]:
                                    idle_time[machine] += job_timestamp[job] - machine_timestamp[machine]
                                    job_timestamp[job] += time
                                    machine_timestamp[machine] = job_timestamp[job]
                                    self.update_machine_graph_maps(machine, machine_timestamp[machine] - time, time, job_no)
                                else:
                                    machine_timestamp[machine] += time
                                    job_timestamp[job] = machine_timestamp[machine]
                                    self.update_machine_graph_maps(machine, machine_timestamp[machine] - time, time,
                                                                   job_no)
                            else:
                                machine_timestamp[machine] += time
                                job_timestamp[job] = machine_timestamp[machine]
                                self.update_machine_graph_maps(machine, machine_timestamp[machine] - time, time, job_no)
                        else:
                            if job in job_timestamp:
                                idle_time[machine] += job_timestamp[job]
                                job_timestamp[job] += time
                                machine_timestamp[machine] = job_timestamp[job]
                                self.update_machine_graph_maps(machine, machine_timestamp[machine] - time, time, job_no)
                            else:
                                machine_timestamp[machine] = time
                                job_timestamp[job] = machine_timestamp[machine]
                                self.update_machine_graph_maps(machine, machine_timestamp[machine] - time, time, job_no)
                        break
            if machine_operation_left:
                flag = False
        with open(self.TCombobox1.get().strip() + '_Result' + '_' + str(self.name_entry.get()) + '.csv', 'w') as output:
            writer = csv.writer(output)
            for i, machine in enumerate(self.machine_list):
                machine_lowercase = machine.lower()
                if machine_lowercase in self.machine_operations_assigned:
                    if max_total_time < (processing_time[machine_lowercase]+idle_time[machine_lowercase]):
                        max_total_time = processing_time[machine_lowercase]+idle_time[machine_lowercase]
                    ttk.Label(self.result_frame, background="#ffffff", text=machine+': Processing Time - '+str(processing_time[machine_lowercase])+'; Idle Time - '+str(idle_time[machine_lowercase])+'; Total Time - '+str(processing_time[machine_lowercase]+idle_time[machine_lowercase])).grid(column=0, row=i+2, sticky='n')
                    if (not plotOnly):
                        result_row = [machine, processing_time[machine_lowercase], idle_time[machine_lowercase], processing_time[machine_lowercase]+idle_time[machine_lowercase]]
                        for job_operation, time in self.machine_operations_assigned[machine_lowercase]:
                            result_row.append(job_operation)
                            result_row.append(time)
                        writer.writerow(result_row)
        if (not plotOnly):
            plot_graph(self.job_colors,self.machine_timelines, self.machine_job_colors, max_total_time, self.TCombobox1.get().strip()+'_Result' + '_' + str(self.name_entry.get()) +'.png')
        else:
            plot_graph_live(self.Canvas1, self.Label2, self.job_colors, self.machine_timelines, self.machine_job_colors, max_total_time, self.TCombobox1.get().strip() +'_Result' + '_' + str(self.name_entry.get()) +'.png')

    def select_instance(self, eventObject):
        instance = Instance('./Instances', self.TCombobox1.get().strip(), False)
        instance.instance_dataframe()
        self.resetMaps()
        self.clearFrame(self.Scrolledwindow1_f)
        self.clearFrame(self.Scrolledwindow2_f)
        self.clearFrame(self.Canvas1)
        self.populate_machines_jobs_map(instance.instance)
        row_count = 0
        for i, machine in enumerate(self.machine_list):
            machine_lowercase = machine.lower()
            self.machine_UI[machine_lowercase] = {}
            self.machine_UI[machine_lowercase]['select_frame'] = ttk.Frame(master=self.Scrolledwindow1_f)
            self.machine_UI[machine_lowercase]['select_frame'].grid(sticky='nsew')
            self.machine_UI[machine_lowercase]['selected_frame'] = ttk.Frame(master=self.Scrolledwindow1_f)
            self.machine_UI[machine_lowercase]['selected_frame'].grid(sticky='nsew')
            ttk.Label(self.machine_UI[machine_lowercase]['select_frame'], text='-----------------------')\
                .grid(column=0,row=row_count,sticky='n')
            ttk.Label(self.machine_UI[machine_lowercase]['select_frame'], text=machine_lowercase+': ').grid(column=0, row=row_count+1, sticky='n')
            self.machine_UI[machine_lowercase]['row_count'] = row_count
            self.machine_UI[machine_lowercase]['button'] = {}
            for k, value in enumerate(['idle']+self.machines_map[machine_lowercase]):
                self.machine_UI[machine_lowercase]['button'][value] = ttk.Button(self.machine_UI[machine_lowercase]['select_frame'], text=value, name=machine_lowercase+'_'+value.lower())
                self.machine_UI[machine_lowercase]['button'][value].grid(column=(k % 5) + 1,
                                                               row=row_count+ 1+(int(k / 5)), sticky='nesw',padx=3, pady=3)
                self.machine_UI[machine_lowercase]['button'][value].bind("<Button-1>", self.update_machine_jobs_list)

            self.machine_UI[machine_lowercase]['text'] = tk.Text(self.machine_UI[machine_lowercase]['selected_frame'], height=3, width=100)
            self.machine_UI[machine_lowercase]['text'].grid(column=0, row=0)
            row_count += 2
        self.result_frame = ttk.Frame(master=self.Scrolledwindow2_f)
        self.result_frame.grid(sticky='nsew')
        

# The following code is added to facilitate the Scrolled widgets you specified.
class AutoScroll(object):
    '''Configure the scrollbars for a widget.'''
    def __init__(self, master):
        #  Rozen. Added the try-except clauses so that this class
        #  could be used for scrolled entry widget for which vertical
        #  scrolling is not supported. 5/7/14.
        try:
            vsb = ttk.Scrollbar(master, orient='vertical', command=self.yview)
        except:
            pass
        hsb = ttk.Scrollbar(master, orient='horizontal', command=self.xview)
        try:
            self.configure(yscrollcommand=self._autoscroll(vsb))
        except:
            pass
        self.configure(xscrollcommand=self._autoscroll(hsb))
        self.grid(column=0, row=0, sticky='nsew')
        try:
            vsb.grid(column=1, row=0, sticky='ns')
        except:
            pass
        hsb.grid(column=0, row=1, sticky='ew')
        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(0, weight=1)
        # Copy geometry methods of master  (taken from ScrolledText.py)
        if py3:
            methods = tk.Pack.__dict__.keys() | tk.Grid.__dict__.keys() \
                  | tk.Place.__dict__.keys()
        else:
            methods = tk.Pack.__dict__.keys() + tk.Grid.__dict__.keys() \
                  + tk.Place.__dict__.keys()
        for meth in methods:
            if meth[0] != '_' and meth not in ('config', 'configure'):
                setattr(self, meth, getattr(master, meth))

    @staticmethod
    def _autoscroll(sbar):
        '''Hide and show scrollbar as needed.'''
        def wrapped(first, last):
            first, last = float(first), float(last)
            if first <= 0 and last >= 1:
                sbar.grid_remove()
            else:
                sbar.grid()
            sbar.set(first, last)
        return wrapped

    def __str__(self):
        return str(self.master)

def _create_container(func):
    '''Creates a ttk Frame with a given master, and use this new frame to
    place the scrollbars and the widget.'''
    def wrapped(cls, master, **kw):
        container = ttk.Frame(master)
        container.bind('<Enter>', lambda e: _bound_to_mousewheel(e, container))
        container.bind('<Leave>', lambda e: _unbound_to_mousewheel(e, container))
        return func(cls, container, **kw)
    return wrapped

class ScrolledWindow(AutoScroll, tk.Canvas):
    '''A standard Tkinter Canvas widget with scrollbars that will
    automatically show/hide as needed.'''
    @_create_container
    def __init__(self, master, **kw):
        tk.Canvas.__init__(self, master, **kw)
        AutoScroll.__init__(self, master)

import platform
def _bound_to_mousewheel(event, widget):
    child = widget.winfo_children()[0]
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        child.bind_all('<MouseWheel>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-MouseWheel>', lambda e: _on_shiftmouse(e, child))
    else:
        child.bind_all('<Button-4>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Button-5>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-Button-4>', lambda e: _on_shiftmouse(e, child))
        child.bind_all('<Shift-Button-5>', lambda e: _on_shiftmouse(e, child))

def _unbound_to_mousewheel(event, widget):
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        widget.unbind_all('<MouseWheel>')
        widget.unbind_all('<Shift-MouseWheel>')
    else:
        widget.unbind_all('<Button-4>')
        widget.unbind_all('<Button-5>')
        widget.unbind_all('<Shift-Button-4>')
        widget.unbind_all('<Shift-Button-5>')

def _on_mousewheel(event, widget):
    if platform.system() == 'Windows':
        widget.yview_scroll(-1*int(event.delta/120),'units')
    elif platform.system() == 'Darwin':
        widget.yview_scroll(-1*int(event.delta),'units')
    else:
        if event.num == 4:
            widget.yview_scroll(-1, 'units')
        elif event.num == 5:
            widget.yview_scroll(1, 'units')

def _on_shiftmouse(event, widget):
    if platform.system() == 'Windows':
        widget.xview_scroll(-1*int(event.delta/120), 'units')
    elif platform.system() == 'Darwin':
        widget.xview_scroll(-1*int(event.delta), 'units')
    else:
        if event.num == 4:
            widget.xview_scroll(-1, 'units')
        elif event.num == 5:
            widget.xview_scroll(1, 'units')

if __name__ == '__main__':
    vp_start_gui()





