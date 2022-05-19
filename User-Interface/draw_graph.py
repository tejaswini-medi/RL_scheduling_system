import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib._color_data as mcd
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("WebAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

colors = [name for name in mcd.CSS4_COLORS
          if "xkcd:" + name in mcd.XKCD_COLORS and name != 'white' and name != 'ivory']
job_colors = {}

def get_job_list(machine_color_list):
    job_list = []
    for color in machine_color_list:
        for job, job_color in job_colors.items():  
            if color == job_color:
                job_list.append(job)
                break
    return job_list

def plot_graph(job_colors, machine_timelines, machine_job_colors, max_total_time, file_name):
    fig, ax = plt.subplots()
    yticks = []
    y_plot = 10
    for machine, timeline in machine_timelines.items():
        ax.broken_barh(timeline, (y_plot, 9), facecolors=machine_job_colors[machine])
        yticks.append(y_plot+5)
        y_plot += 10
    job_list = []
    for job, color in job_colors.items():
        if job != 'None':
            job_list.append(mpatches.Patch(color=color, label=job))
    plt.legend(handles=job_list)
    ax.set_ylim(5, y_plot +10)
    ax.set_xlim(0, max_total_time)
    ax.set_xlabel('Duration (in Seconds)')
    ax.set_yticks(yticks)
    ax.set_yticklabels(list(machine_timelines.keys()))
    ax.grid(True)
    plt.savefig(file_name)
    plt.close()
    
def plot_graph_live(root, label, job_colors, machine_timelines, machine_job_colors, max_total_time, file_name):

    figure, ax = plt.subplots()
    yticks = []
    y_plot = 10
    load_balance_dict = {}
    lowest_machine_time = 0
    #print(machine_timelines)
    for machine, timeline in machine_timelines.items():
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
        job_list = get_job_list(machine_job_colors[machine])
        ax.broken_barh(timeline, (y_plot, 9), facecolors=machine_job_colors[machine],label=job_list)
        yticks.append(y_plot+5)
        y_plot += 10
    ax.axvline(x=lowest_machine_time)
    sorted_keys = sorted(load_balance_dict, key=load_balance_dict.get)
    x= len(sorted_keys)
    job_list = []
    for job, color in job_colors.items():
        if job != 'None':
            job_list.append(mpatches.Patch(color=color, label=job))
    plt.legend(handles=job_list)

    for i in range(x):
        z= sorted_keys[:x-i]
    ax.set_ylim(5, y_plot +10)
    ax.set_xlim(0, max_total_time)
    ax.set_xlabel('Duration (in Seconds)')
    plt.title('Maximum Total Time :' +' ' + str(max_total_time),weight='bold')
    ax.set_yticks(yticks)
    ax.set_yticklabels(list(machine_timelines.keys()))
    ax.grid(True)
    plt.close()
    canvas = FigureCanvasTkAgg(figure, root)
    canvas.get_tk_widget().grid(row=0, column=0)