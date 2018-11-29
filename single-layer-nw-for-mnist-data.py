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

import numpy as np
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
        
# This module calculates the activation function
def calculate_activation_function(weight_matrix, input_vector, type='Symmetric Hard Limit'):
    
    net_value = weight_matrix.dot(input_vector)
    
    if type == "Symmetric Hard Limit":
        temp1 = np.ma.masked_where(net_value < 0, net_value, copy=False)
        temp2 = temp1.filled(-1)
        temp3 = np.ma.masked_where(temp2 > 0, temp2, copy=False)
        activation = temp3.filled(1)
            
    elif type == "Hyperbolic Tangent":
        activation = np.tanh(net_value)
    
    elif type == "Linear":
        activation = net_value
        
    return activation

class LeftFrame:
    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        self.xmin = 0
        self.xmax = 1000
        self.ymin = 0
        self.ymax = 100
        self.counter = 0
        self.error_points = np.array([0,100])
        
        #learning rate
        self.alpha = 0.1
        
        # default values for learning rule type AND Activation function type
        self.activation_type = "Symmetric Hard Limit"
        self.learning_type = "Delta Rule"
        
        #create input matrix from files nd weight matrix with random numbers between -0.001 to 0.001
        self.read_input_data_from_files()
        self.initialize_weights_random()
        
        # separate the input data into two sets. The first set should include 80% of your data set (randomly selected). This is your training set. The second set (the other 20%) is the test set.
        self.split_into_training_test_sets()
        
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=2)
        
        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
        
        #actual plotting of the graph
        self.figure = plt.figure(figsize=(20,10)) #setting the size of the plot 
        #self.figure = plt.figure("")
        self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.6])      #
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Epoch')    #setting label for x-axis
        self.axes.set_ylabel('Error in %')   #setting label for y-axis
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
        
        # learning rate - alpha slider
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Alpha",
                                            command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # creating a button - When this button is pressed the selected Hebbian learning rule should  be applied for 100 epochs.
        self.adjust_weight_button = tk.Button(self.controls_frame, text="Adjust Weight", fg="red", command=self.train_data)
        self.adjust_weight_button.grid(row=0, column=1)
        
        # creating a button - When this button is pressed weights and biases should be randomized
        # should be from -0.001 to 0.001.
        self.random_weight_button = tk.Button(self.controls_frame, text="Rondomize Weights", fg="blue", command=self.initialize_weights_random)
        self.random_weight_button.grid(row=0, column=2)
        
        # creating a button - When this button is pressed the confusion matrix for the test set should be displayed.
        self.confusion_matrix_button = tk.Button(self.controls_frame, text="Confusion Matrix", fg="green", command=self.generate_confusion_matrix)
        self.confusion_matrix_button.grid(row=0, column=3)

                
        # Drop down to Select Learning Method     
        self.label_for_learning_rule = tk.Label(self.controls_frame, text="Learning Method Type:",
                                                  justify="center")
        self.label_for_learning_rule.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)
        self.learning_rule_variable = tk.StringVar()
        self.learning_rule_dropdown = tk.OptionMenu(self.controls_frame, self.learning_rule_variable,
                                                  "Delta Rule", "Unsupervised Hebb", "Filtered Learning (Smoothing)",
                                                 command=lambda event: self.learning_rule_dropdown_callback())
        self.learning_rule_variable.set("Delta Rule")
        self.learning_rule_dropdown.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # Drop down to Select Transfer Functions     
        self.label_for_activation_function = tk.Label(self.controls_frame, text="Activation Function Type:",
                                                  justify="center")
        self.label_for_activation_function.grid(row=0, column=5, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
                                                  "Symmetric Hard Limit", "Hyperbolic Tangent", "Linear",
                                                 command=lambda event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetric Hard Limit")
        self.activation_function_dropdown.grid(row=0, column=5, sticky=tk.N + tk.E + tk.S + tk.W)
        
 
    def read_input_data_from_files(self):
        count = 0
        for filename in os.listdir("./mnist-data"):
            count+= 1
            # asssign target based on filename
            t_q = self.assign_targets_from_files(filename)
            #read_one_image_and_convert_to_vector(file_name):
            img = scipy.misc.imread("./mnist-data/" + filename).astype(np.float32) # read image and convert to float
            img_col = img.reshape(-1,1) # reshape to column vector and return it
            img_col = np.vstack((img_col, np.array([1])))
            if count == 1:
                self.P = img_col
                self.Targ = t_q
            else:
                self.P = np.hstack((self.P,img_col))
                self.Targ = np.hstack((self.Targ, t_q))
        # normalize each vector element to be in the range of -1 to 1. i.e. , divide the input numbers by 127.5 and subtract 1.0
        self.P = (1/127.5) * self.P
        self.P = np.subtract(self.P, 1)
        print(" shape of P ",self.P.shape)
        print(" shape of T ",self.Targ.shape)
        print("number of files read: ", count)
        
    def assign_targets_from_files(self, filename):
        if filename.startswith("0"):
            t_q = np.array([1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
        elif filename.startswith("1"):
            t_q = np.array([-1,1,-1,-1,-1,-1,-1,-1,-1,-1])
        elif filename.startswith("2"):
            t_q = np.array([-1,-1,1,-1,-1,-1,-1,-1,-1,-1])
        elif filename.startswith("3"):
            t_q = np.array([-1,-1,-1,1,-1,-1,-1,-1,-1,-1])
        elif filename.startswith("4"):
            t_q = np.array([-1,-1,-1,-1,1,-1,-1,-1,-1,-1])
        elif filename.startswith("5"):
            t_q = np.array([-1,-1,-1,-1,-1,1,-1,-1,-1,-1])
        elif filename.startswith("6"):
            t_q = np.array([-1,-1,-1,-1,-1,-1,1,-1,-1,-1])
        elif filename.startswith("7"):
            t_q = np.array([-1,-1,-1,-1,-1,-1,-1,1,-1,-1])
        elif filename.startswith("8"):
            t_q = np.array([-1,-1,-1,-1,-1,-1,-1,-1,1,-1])
        elif filename.startswith("9"):
            t_q = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,1])
        return t_q.reshape(-1,1)

    # separate the input data into two sets. The first set should include 80% of your data set (randomly selected). This is your training set. The second set (the other 20%) is the test set.
    def split_into_training_test_sets(self):
        X_train, X_test, y_train, y_test = train_test_split(self.P.T, self.Targ.T, stratify=self.Targ.T, train_size=0.80, test_size=0.20, shuffle=True)
        self.P_train = X_train
        self.P_test = X_test
        self.T_train = y_train
        self.T_test = y_test

        print(" shape of P_train : ",  self.P_train.shape)
        print(" shape of P_test : ",  self.P_test.shape)
        print(" shape of T_train : ",  self.T_train.shape)
        print(" shape of T_test : ",  self.T_test.shape)
        
    def initialize_weights_random(self):
        print("weights randomized.")
        w = np.random.uniform(-0.001,0.001,7850)
        self.W = w.reshape(10, 785)
        print("shape of W : ", self.W.shape)
        #print(self.W)
        
    def train_data(self):
        print("Started learning of the data set using ", self.learning_type, " ", self.activation_type)
        self.W_old = self.W
        
        # run the learning for 100 epochs
        for x in range(0, 100):
            self.apply_learning_rule_on_training_set()
            self.get_error_percentage()
            self.counter = self.counter+1
            self.error_points = np.vstack((self.error_points, np.array([self.counter, self.error_percent])))
            #print("error points to plot : ", self.error_points.shape, " ", self.error_points)
        self.display_points()
            
      
    # This module applies the corresponding learning Rule
    def apply_learning_rule_on_training_set(self):
        if self.learning_type == "Filtered Learning (Smoothing)":
            for p_row, t_row in zip(self.P_train, self.T_train):
                #print("shape of p_row ", p_row.shape)
                #print("shape of t_row ", t_row.shape)
                p_q = p_row.reshape(-1,1)
                t_q = t_row.reshape(-1,1)
                
                self.W = ((1-self.alpha) * self.W_old) + (self.alpha * ((t_q).dot(p_q.T)))
                self.W_old = self.W
                
        elif self.learning_type == "Delta Rule":
            for p_row, t_row in zip(self.P_train, self.T_train):
                #print("shape of p_row ", p_row.shape)
                #print("shape of t_row ", t_row.shape)
                p_q = p_row.reshape(-1,1)
                t_q = t_row.reshape(-1,1)
                a_q = calculate_activation_function(self.W_old, p_q, self.activation_type)
                
                self.W = self.W_old + (self.alpha * ((t_q - a_q).dot(p_q.T)))
                self.W_old = self.W
            
        elif self.learning_type == "Unsupervised Hebb":
            for p_row, t_row in zip(self.P_train, self.T_train):
                #print("shape of p_row ", p_row.shape)
                #print("shape of t_row ", t_row.shape)
                p_q = p_row.reshape(-1,1)
                t_q = t_row.reshape(-1,1)
                a_q = calculate_activation_function(self.W_old, p_q, self.activation_type)
                
                self.W = self.W_old + (self.alpha * ((a_q).dot(p_q.T)))
                self.W_old = self.W
            
    def get_error_percentage(self):
        self.actual = calculate_activation_function(self.W, self.P_test.T, self.activation_type)
        #print("shape of actuals ", self.actual.shape)
        self.predicted_classes = np.argmax(self.actual, axis=0)
        #print("shape of predicted_classes ", self.predicted_classes.shape)
        #print("predicted_classes ", self.predicted_classes)
        self.target_classes = np.argmax(self.T_test.T, axis=0)
        #print("shape of target_classes ", self.target_classes.shape)
        #print("target_classes ", self.target_classes)
        number_of_correct_predictions = (self.predicted_classes == self.target_classes).sum()
        #print("number of correct predictions :: ", number_of_correct_predictions)
        self.error_percent = (1 - number_of_correct_predictions/200) * 100
        
    def display_points(self):
        self.axes.cla()
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Error in %')
        self.axes.plot(self.error_points[:,0], self.error_points[:,1], 'rx', markersize=2)
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()
        
    def alpha_slider_callback(self):
        self.alpha = np.float(self.alpha_slider.get())
            
    def learning_rule_dropdown_callback(self):
        self.learning_type = self.learning_rule_variable.get()
        
    def activation_function_dropdown_callback(self):
        self.activation_type = self.activation_function_variable.get()
        
    def generate_confusion_matrix(self):
        print("generating confusion matrix")
        self.get_error_percentage()
        print("\n Confusion matrix:")
        crossTab = pd.crosstab(self.target_classes, self.predicted_classes, rownames=['True'], colnames=['Predicted'], margins=True)
        self.confusion_matrix = (crossTab.as_matrix())
        self.display_numpy_array_as_table()
        
    def display_numpy_array_as_table(self):
    # This function displays a 1d or 2d numpy array (matrix).
    # Note that the last column and the last row displays the total number of samples
        input_array = self.confusion_matrix
        if input_array.ndim==1:
            num_of_columns,=input_array.shape
            temp_matrix=input_array.reshape((1, num_of_columns))
        elif input_array.ndim>2:
            print("Input matrix dimension is greater than 2. Can not display as table")
            return
        else:
            temp_matrix=input_array
        number_of_rows,num_of_columns = temp_matrix.shape
        plt.figure()
        self.axes.cla()
        tb = self.axes.table(cellText=np.round(temp_matrix,2), loc=(0,0), cellLoc='center')
        for cell in tb.properties()['child_artists']:
            cell.set_height(1/number_of_rows)
            cell.set_width(1/num_of_columns)

        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.canvas.draw()

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

main_window = MainWindow(debug_print_flag=False)
main_window.wm_state('zoomed')
main_window.title('Venkatesh')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()