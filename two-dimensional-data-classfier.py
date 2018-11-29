# Venkatesh, Ravi Chandra Kumar

import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog 
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.backends.tkagg as tkagg

import random
import math

class MainWindow(tk.Tk):
    def __init__(self, debug_print_flag=True):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.rowconfigure(0, weight=1, minsize=500)
        self.columnconfigure(0, weight=1, minsize=500)
        
        # set the properties of the row and columns in the master frame
        self.master_frame.rowconfigure(2, weight=10, minsize=400, uniform='xx')
        self.master_frame.rowconfigure(3, weight=1, minsize=10, uniform='xx')
        self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')
        
        self.left_frame = tk.Frame(self.master_frame)
        
        # Arrange the widgets
        self.left_frame.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # Create an object for plotting graphs in the left frame
        self.display_activation_functions = LeftFrame(self, self.left_frame, debug_print_flag=self.debug_print_flag)

    
# This module calculates the activation function
def calculate_activation_function(weight1, weight2,bias,input_array,type='Symmetric Hard Limit'):
    net_value = weight1 * input_array[0] + weight2 * input_array[1] + bias
    if type == "Symmetric Hard Limit":
        if net_value >= 0:
            activation = 1
        else:
            activation = -1
            
    elif type == "Hyperbolic Tangent":
        activation = np.tanh(net_value)
    
    elif type == "Linear":
        activation = net_value
        
    return activation


class LeftFrame:
    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.input_weight1 = 1
        self.input_weight2 = 1
        self.bias = 0.0
        self.positives = np.array([[0,0],[0,0]])
        self.negatives = np.array([[0,0],[0,0]])
        self.activation_type = "Symmetric Hard Limit"
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=2)
        
        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
        
        #actual plotting of the graph
        self.figure = plt.figure(figsize=(10,10)) #setting the size of the plot 
        self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.6])      #
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input')    #setting label for x-axis
        self.axes.set_ylabel('Output')   #setting label for y-axis
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # Create a frame to contain all the controls such as sliders, buttons, ...
        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        #########################################################################
        #  Set up the control widgets such as sliders and selection boxes
        #########################################################################
        
        # input weight one slider
        self.input_weight1_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Input Weight 1",
                                            command=lambda event: self.input_weight1_slider_callback())
        self.input_weight1_slider.set(self.input_weight1)
        self.input_weight1_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        #input weight 2 slider
        self.input_weight2_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Input Weight 2",
                                            command=lambda event: self.input_weight2_slider_callback())
        self.input_weight2_slider.set(self.input_weight2)
        self.input_weight2_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        
        #bias slider
        self.bias_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL, from_=-10.0,
                                    to_=10.0, resolution=0.01, bg="#DDDDDD", activebackground="#FF0000",
                                    highlightcolor="#00FFFF", label="Bias",
                                    command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.label_for_activation_function = tk.Label(self.controls_frame, text="Activation Function Type:",
                                                  justify="center")
        self.label_for_activation_function.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
                                                  "Symmetric Hard Limit", "Hyperbolic Tangent", "Linear",
                                                 command=lambda event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetric Hard Limit")
        self.activation_function_dropdown.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # creating a button to Train and adjust the weights and bias for 100 steps using the learning rule
        self.train_button = tk.Button(self.controls_frame, text="Train", fg="red", command=self.train_data)
        self.train_button.grid(row=0, column=4)
        
        # creating a button to add randomly generated points
        self.random_data = tk.Button(self.controls_frame, text="Create Random data", fg="blue", command=self.generate_random_points)
        self.random_data.grid(row=0, column=6)

    def apply_learning_rule(self, target, ipt):
        # error = target - actual
        error = target - self.actual
        # wnew = wold + error * input
        self.input_weight1 = self.input_weight1 + error * ipt[0]
        self.input_weight2 = self.input_weight2 + error * ipt[1]
        #bias new = bias old + error
        self.bias = self.bias + error
        
    # self.focus_set()
    def training_using_activation_function(self):
        for x in range(0, 100):
            for ipt in self.positives:
                self.actual = calculate_activation_function(self.input_weight1, self.input_weight2, self.bias, ipt,
                                                                      self.activation_type)
                self.apply_learning_rule(1, ipt)
                
            for ipt in self.negatives:
                self.actual = calculate_activation_function(self.input_weight1, self.input_weight2, self.bias, ipt,
                                                                      self.activation_type)
                self.apply_learning_rule(-1, ipt)
              
            #display line after each epoc
            self.display_line(self.input_weight1, self.input_weight2, self.bias)
        
    def generate_4_points(self):
        class1_point1 = random.uniform(-10, +10) , random.uniform(-10, +10)
        class1_point2 = random.uniform(-10, +10) , random.uniform(-10, +10)
        
        class2_point1 = random.uniform(-10, +10) , random.uniform(-10, +10)
        class2_point2 = random.uniform(-10, +10) , random.uniform(-10, +10)
        
        self.positives = np.array([class1_point1, class1_point2])
        self.negatives = np.array([class2_point1, class2_point2])
        
    def display_points(self):
        self.axes.cla()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        
        #plot the  positive points as 'X' in red
        self.axes.plot(self.positives[:, 0], self.positives[:,1], 'rx', markersize=10)
        #plot the negative points as 'O' in yellow
        self.axes.plot(self.negatives[:, 0], self.negatives[:,1], 'yo', markersize=10)
        
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)

        self.canvas.draw()
        
    def display_line(self, weight1, weight2, bias):
        self.axes.cla()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        
        resolution=100
        xs = np.linspace(self.xmin, self.xmax, resolution)
        ys = np.linspace(self.ymin, self.ymax, resolution)
        xx, yy = np.meshgrid(xs, ys)
        zz= weight1 * xx + weight2 * yy + bias
        zz[zz<0]=-1
        zz[zz>0]=+1
        
        self.axes.plot(self.positives[:, 0], self.positives[:,1], 'rx', markersize=10)
        self.axes.plot(self.negatives[:, 0], self.negatives[:,1], 'yo', markersize=10)
        
        #plot green color for positive class and red color for negative class
        self.axes.pcolormesh(xx, yy, zz, cmap="RdYlGn")
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        
        self.canvas.draw()
        
    def generate_random_points(self):
        self.generate_4_points()
        self.display_points()
        self.reset_sliders()
        
    def train_data(self):
        print("Started training of the data set using ", self.activation_type)
        #line initially - before training
        self.display_line(self.input_weight1, self.input_weight2, self.bias)
        self.training_using_activation_function()
        #line after training
        self.display_line(self.input_weight1, self.input_weight2, self.bias)
        print("Finished training of the data set using ", self.activation_type)
        
    def input_weight1_slider_callback(self):
        self.input_weight1 = np.float(self.input_weight1_slider.get())
        if(self.input_weight1 != 1.0 and self.input_weight2 != 1.0 and self.bias != 0.0):
            self.display_line(self.input_weight1, self.input_weight2, self.bias)
        
    def input_weight2_slider_callback(self):
        self.input_weight2 = np.float(self.input_weight2_slider.get())
        if(self.input_weight1 != 1.0 and self.input_weight2 != 1.0 and self.bias != 0.0):
            self.display_line(self.input_weight1, self.input_weight2, self.bias)
            
    def bias_slider_callback(self):
        self.bias = np.float(self.bias_slider.get())
        if(self.input_weight1 != 1.0 and self.input_weight2 != 1.0 and self.bias != 0.0):
            self.display_line(self.input_weight1, self.input_weight2, self.bias)
            
    def activation_function_dropdown_callback(self):
        self.activation_type = self.activation_function_variable.get()
        #self.display_activation_function()
        
    def reset_sliders(self):
        self.input_weight1 = 1.0
        self.input_weight2 = 1.0
        self.bias = 0.0
        self.input_weight1_slider.set(self.input_weight1)
        self.input_weight2_slider.set(self.input_weight2)
        self.bias_slider.set(self.bias)
             
def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

main_window = MainWindow(debug_print_flag=False)
main_window.wm_state('zoomed')
main_window.title('Venkatesh')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()