# Venkatesh, Ravi Chandra Kumar
# 1001-581-994
# 2018-10-29
# Assignment-04-01

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

import numpy as np
from numpy.linalg import inv
import scipy.misc
import os
from sklearn.model_selection import train_test_split
import pandas as pd

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
        
class LeftFrame:
    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        self.xmin = 0
        self.xmax = 10
        self.ymin = 0
        self.ymax = 2
        
        #learning rate
        self.alpha = 0.1
        #number of delayed elements
        self.delay = 10
        #step size for stride
        self.stride = 1
        #Training Sample Size (Percentage)
        self.training_percent = 80
        #number of iterations
        self.epochs = 10
        
        #create input matrix from files nd weight matrix with random numbers between -0.001 to 0.001
        self.read_input_data_from_files()
        self.initialize_weights_and_bias_zero()
        
        # separate the input data into two sets. The first set should include 80% of your data set (randomly selected). This is your training set. The second set (the other 20%) is the test set.
        self.split_into_training_test_sets()
        
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=2)
        
        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
        
        #actual plotting of the graph
        self.figure = plt.figure(figsize=(20,10)) #setting the size of the plot 
        #self.figure = plt.figure("")
        self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.6], autoscaley_on=True)      #
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Epoch')    #setting label for x-axis
        self.axes.set_ylabel('Error in %')   #setting label for y-axis
        self.axes.set_title("Plot of MSE and MAE after each iteration")
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
        
        # slider 1 : Number of Delayed Elements
        self.delay_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Delay",
                                            command=lambda event: self.delay_slider_callback())
        self.delay_slider.set(self.delay)
        self.delay_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # slider 2 : learning rate - alpha slider
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Alpha(learning rate)",
                                            command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # slider 3 : Training Sample Size (Percentage)
        self.training_size_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Training size",
                                            command=lambda event: self.training_size_slider_callback())
        self.training_size_slider.set(self.training_percent)
        self.training_size_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # slider 4 : Stride slider
        self.stride_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Stride",
                                            command=lambda event: self.stride_slider_callback())
        self.stride_slider.set(self.stride)
        self.stride_slider.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # slider 5 : Number of Iterations - epoch slider
        self.epochs_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Iterations",
                                            command=lambda event: self.epochs_slider_callback())
        self.epochs_slider.set(self.epochs)
        self.epochs_slider.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)

        # button 1 - When this button is pushed all the weights and biases should be set to zero
        self.init_weight_button = tk.Button(self.controls_frame, text="Init Weights", fg="blue", command=self.initialize_weights_and_bias_zero)
        self.init_weight_button.grid(row=0, column=5)
        
        # button 2 - When this button is pressed the LMS algorithm should be applied and plots should be displayed
        self.adjust_weight_button_LMS = tk.Button(self.controls_frame, text="Adjust Weight(LMS)", fg="red", command=self.train_data_LMS)
        self.adjust_weight_button_LMS.grid(row=0, column=6)
        
        # button 3 - When this button is pressed the stationary point of the performance function should be found by minimizing the MSE for all the training samples directly. 
        self.adjust_weight_button_Direct = tk.Button(self.controls_frame, text="Adjust Weight(Direct)", fg="blue", command=self.train_data_Direct)
        self.adjust_weight_button_Direct.grid(row=0, column=7)
        
        
    def read_input_data_from_files(self):
        # Each row of data in the file becomes a row in the matrix
        # So the resulting matrix has dimension [num_samples x sample_dimension]
        data = np.loadtxt("./data/data_set_2.csv", skiprows=1, delimiter=',', dtype=np.float32)
        print("shape of data::: ", data.shape)
        price = data[:,:1]
        volume = data[:,1:]
        self.price = 2*(price - np.min(price))/np.ptp(price)-1
        self.volume = 2*(volume - np.min(volume))/np.ptp(volume)-1

        print("shape of price vextor ", self.price.shape)
        print("shape of volume vextor ", self.volume.shape)
        

    # separate the input data into two sets. The first set should include 80% of your data set (randomly selected). This is your training set. The second set (the other 20%) is the test set.
    def split_into_training_test_sets(self):
       # print("splitting the data into training and test :: ", self.training_percent, (100 - self.training_percent))
        training_set_size = (self.training_percent / 100)
        test_set_size = ((100 - self.training_percent) / 100 )
        self.price_train, self.price_test = train_test_split(self.price, stratify=None, train_size=training_set_size, test_size=test_set_size , shuffle=False)
        self.vol_train, self.vol_test = train_test_split(self.volume, stratify=None, train_size=training_set_size, test_size=test_set_size , shuffle=False)
        
    def initialize_weights_and_bias_zero(self):
        print("weights and bias initialized.")
        self.bias = 0
        w = np.zeros((self.delay*2) + 2)
        self.W = w.reshape(-1, (self.delay * 2) + 2)
        print("shape of W : ", self.W.shape, self.W)
        
    def train_data_LMS(self):
        print("shape of price_train ", self.price_train.shape)
        print("shape of price_test ", self.price_test.shape)
        print("shape of vol_train ", self.vol_train.shape)
        print("shape of vol_test ", self.vol_test.shape)

        print("Started learning of the data set using LMS")
        size_train = self.price_train.shape[0]
        
        # run the learning for number of epochs selected
        for x in range(0, self.epochs):
            count = 0
            rem = size_train
            
            while(rem > self.stride):
                end_index = (self.delay + 1) + (count * self.stride)
                start_index = end_index - (self.delay + 1)
                delay_current = self.price_train[start_index : end_index : 1, :]
                delay_current = np.vstack((delay_current, self.vol_train[start_index:end_index:1, :]))
                t = self.price_train[end_index:end_index+1, :]
                a = self.W.dot(delay_current) + self.bias
                e = t - a
                self.W = self.W + 2 * self.alpha * e * delay_current.T
                self.bias = self.bias + 2 * self.alpha * e
                rem = size_train - end_index
                count += 1
                
            size_test = self.price_test.shape[0]
            self.errors_array = np.empty(0)
            count = 0
            rem = size_test
            # run on test set to find MSE
            while(rem > self.stride):
                end_index = (self.delay + 1) + (count * self.stride)
                start_index = end_index - (self.delay + 1)
                delay_current = self.price_test[start_index : end_index : 1, :]
                delay_current = np.vstack((delay_current, self.vol_test[start_index:end_index:1, :]))
                t = self.price_test[end_index:end_index+1, :]
                a = self.W.dot(delay_current) + self.bias
                e = np.asscalar(t) - np.asscalar(a)
                self.errors_array = np.append(self.errors_array, e)
                rem = size_test - end_index
                count += 1
            
            squared_errors_array = np.square(self.errors_array)
            absolute_errors_array = np.absolute(self.errors_array)
            mse = np.mean(squared_errors_array)
            mae = np.mean(absolute_errors_array)
            if(x == 0):
                self.mse_array = np.array([x, mse])
                self.mae_array = np.array([x, mae])
            else:
                self.mse_array = np.vstack((self.mse_array, np.array([x, mse])))
                self.mae_array = np.vstack((self.mae_array, np.array([x, mae])))

        print("**** values of mse_array : ", self.mse_array.shape, self.mse_array)
        print("**** values of mae_array : ", self.mae_array.shape, self.mae_array)
        self.axes.cla()
        self.display_points(self.mse_array, 'rx')
        self.display_points(self.mae_array, 'bx')

            
    def train_data_Direct(self):
        print("shape of price_train ", self.price_train.shape)
        print("shape of price_test ", self.price_test.shape)
        print("shape of vol_train ", self.vol_train.shape)
        print("shape of vol_test ", self.vol_test.shape)

        print("Adjusting weights directly")
        size_train = self.price_train.shape[0]
        count = 0
        rem = size_train
        while(rem > self.stride):
            end_index = (self.delay + 1) + (count * self.stride)
            start_index = end_index - (self.delay + 1)
            p = self.price_train[start_index : end_index : 1, :]
            p = np.vstack((p, self.vol_train[start_index:end_index:1, :]))
            z = np.vstack((p, 1))
            t = self.price_train[end_index:end_index+1, :]
            t_z = np.asscalar(t) * z
            if(count == 0):
                Z = z.dot(z.T)
                h = t_z
            else:
                Z = Z + (z.dot(z.T))
                h = h + t_z
            rem = size_train - end_index
            count += 1

        print("shape of Z : ", Z.shape)
        R = ( Z / count )
        print("shape of R ::: ", R.shape)
        R_inverse = inv(R)
        print("shape of R inv ::: ", R_inverse.shape)
        h_final = (h / count)
        print("shape of h_final ::: ", h_final.shape)
        self.X = R_inverse.dot(h_final)
        print("shape of X ::: ", self.X.shape)
        
        size_test = self.price_test.shape[0]
        count = 0
        self.errors_array = np.empty(0)
        rem = size_test
        while(rem > self.stride):
            end_index = (self.delay + 1) + (count * self.stride)
            start_index = end_index - (self.delay + 1)
            p = self.price_test[start_index : end_index : 1, :]
            p = np.vstack((p, self.vol_test[start_index:end_index:1, :]))
            z = np.vstack((p, 1))
            t = np.asscalar(self.price_test[end_index])
            a = self.X.T.dot(z)
            e = t - np.asscalar(a)
            self.errors_array = np.append(self.errors_array, e)
            rem = size_test - end_index
            count += 1
        
        squared_errors_array = np.square(self.errors_array)
        absolute_errors_array = np.absolute(self.errors_array)
        mse = np.mean(squared_errors_array)
        mae = np.mean(absolute_errors_array)
        print("mse and mae ::", mse, mae)
        self.axes.cla()
        self.display_points(np.array([[1, mse]]), 'rx')
        self.display_points(np.array([[1, mae]]), 'bx')

    def display_points(self, points_array, color):
        self.axes.set_xlabel('Iterations')    #setting label for x-axis
        self.axes.set_ylabel('Error(MSE in red, MAE in blue)')   #setting label for y-axis
        self.axes.set_title("Plot of MSE(red) and MAE(blue) after each iteration")
        self.axes.plot(points_array[:,0], points_array[:,1], color, markersize=5)
        self.axes.xaxis.set_visible(True)
        self.canvas.draw()
        
    def alpha_slider_callback(self):
        self.alpha = np.float(self.alpha_slider.get())
    
    def delay_slider_callback(self):
        self.delay = np.int_(self.delay_slider.get())
        self.initialize_weights_and_bias_zero()

    def training_size_slider_callback(self):
        self.training_percent = np.int_(self.training_size_slider.get())
        self.split_into_training_test_sets()

    def epochs_slider_callback(self):
        self.epochs = np.int_(self.epochs_slider.get())

    def stride_slider_callback(self):
        self.stride = np.int_(self.stride_slider.get())


def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

main_window = MainWindow(debug_print_flag=False)
main_window.wm_state('zoomed')
main_window.title('Assignment_04 --  Venkatesh')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()