# Venkatesh, Ravi Chandra Kumar

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.backends.tkagg as tkagg

import random
import math
from skimage.util import view_as_windows # the magic function
import tensorflow as tf
import sklearn.datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np


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
        
def generate_data_samples_with_classes(dataset_name, n_samples, n_classes):
    if dataset_name == 'swiss_roll':
        data = sklearn.datasets.make_swiss_roll(n_samples, noise=1.5, random_state=99)[0]
        data = data[:, [0, 2]]
    if dataset_name == 'moons':
        data = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.15)[0]
    if dataset_name == 'blobs':
        data = sklearn.datasets.make_blobs(n_samples=n_samples, centers=n_classes*2, n_features=2, cluster_std=0.85*np.sqrt(n_classes), random_state=100)
        return data[0]/10., [i % n_classes for i in data[1]]
    if dataset_name == 's_curve':
        data = sklearn.datasets.make_s_curve(n_samples=n_samples, noise=0.15, random_state=100)[0]
        data = data[:, [0,2]]/3.0

    ward = AgglomerativeClustering(n_clusters=n_classes*2, linkage='ward').fit(data)
    return data[:]+np.random.randn(*data.shape)*0.03, [i % n_classes for i in ward.labels_]

def one_hot_encoding_of_input_labels(train_onehot,train_labels):
    count = 0
    for i in train_labels:
        train_onehot[count,i]=1
        count += 1
    return train_onehot


class LeftFrame:
    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        self.xmin = 0
        self.xmax = 1000
        self.ymin = 0
        self.ymax = 100
        
        self.input_dimension = 2
        #learning rate
        self.alpha = 0.1
        #weight regularization
        self.lambda_regularizer = 0.01
        #number of iterations
        self.epochs = 10
        #number of nodes in the hidden layer
        self.number_of_nodes = 100
        #number of data samples
        self.number_of_samples = 200
        #number of classes in data
        self.number_of_classes = 4
        #type of data generated
        self.type_of_data = "s_curve"
        #activation function to be used on hidden layer
        self.activation_type = "Relu"
        
        #create weight matrix with random numbers between -0.001 to 0.001
        #Size of weight matrix depends on the number of nodes in the hidden layer => everytime no. of nodes changes, weights of hidden layer are reinitialzed
        #And the no. of nodes also determines the size of weights in output layer => everytime no. of nodes changes, weights of output layer also needs to be reset 
        #reinitialize the weights also when a new no. of classes, as the no. of nodes in output layer changes
        self.initialize_weights_and_bias()
        
        # generate input data samples based on the : no_of_samples, no_of_samples, data_sample_category
        #variables to hold input data samples and their corresponding class
        self.input_data, self.input_labels = generate_data_samples_with_classes(self.type_of_data, self.number_of_samples, self.number_of_classes)
        
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=2)
        
        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
        
        #actual plotting of the graph
        self.figure = plt.figure(figsize=(20,10)) #setting the size of the plot 
        #self.figure = plt.figure("")
        self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.6], autoscaley_on=True)      #
        self.axes = self.figure.gca()
        self.axes.set_xlabel('X')    #setting label for x-axis
        self.axes.set_ylabel('Y')   #setting label for y-axis
        self.axes.set_title(self.type_of_data)
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
        
        # slider 1 : learning rate - alpha slider
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Alpha(learning rate)",
                                            command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # slider 2 : weight regularization
        self.lambda_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Lambda",
                                            command=lambda event: self.lambda_slider_callback())
        self.lambda_slider.set(self.lambda_regularizer)
        self.lambda_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # slider 3 : Number of nodes in hidden layer
        self.nodes_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=1, to_=500, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Num. of Nodes in Hidden Layer",
                                            command=lambda event: self.nodes_slider_callback())
        self.nodes_slider.set(self.number_of_nodes)
        self.nodes_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # slider 4 : Number of samples in input data
        self.number_of_samples_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=4, to_=1000, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Number of Samples",
                                            command=lambda event: self.number_of_samples_slider_callback())
        self.number_of_samples_slider.set(self.number_of_samples)
        self.number_of_samples_slider.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # slider 5 : Number of Classes in input data
        self.number_of_classes_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=2, to_=10, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Number of Classes",
                                            command=lambda event: self.number_of_classes_slider_callback())
        self.number_of_classes_slider.set(self.number_of_classes)
        self.number_of_classes_slider.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)
         
        # button 1 - When this button is pushed train the model on the data and get the weights and plots should be displayed
        self.adjust_weight_button = tk.Button(self.controls_frame, text="Adjust Weight(Train)", fg="red", command=self.train_data)
        self.adjust_weight_button.grid(row=0, column=5)
        
        # button 2 - When this button is pushed all the weights and biases should be initialized between -0.001 and +0.001
        self.init_weight_button = tk.Button(self.controls_frame, text="Reset Weights", fg="blue", command=self.initialize_weights_and_bias)
        self.init_weight_button.grid(row=0, column=6)

        # Drop down 1 - to Select Hidden Layer Transfer Function    
        self.label_for_activation_function = tk.Label(self.controls_frame, text="Activation Function Type:",
                                                  justify="center")
        self.label_for_activation_function.grid(row=0, column=7, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
                                                  "Relu", "Sigmoid",
                                                 command=lambda event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Relu")
        self.activation_function_dropdown.grid(row=0, column=7, sticky=tk.N + tk.E + tk.S + tk.W)
        
        
        # Drop down 2 - to Select Type of generated data    
        self.label_for_type_of_data = tk.Label(self.controls_frame, text="Type of generated data:",
                                                  justify="center")
        self.label_for_type_of_data.grid(row=0, column=8, sticky=tk.N + tk.E + tk.S + tk.W)
        self.type_of_data_variable = tk.StringVar()
        self.type_of_data_dropdown = tk.OptionMenu(self.controls_frame, self.type_of_data_variable,
                                                  "s_curve", "blobs", "swiss_roll", "moons",
                                                 command=lambda event: self.type_of_data_dropdown_callback())
        self.type_of_data_variable.set("s_curve")
        self.type_of_data_dropdown.grid(row=0, column=8, sticky=tk.N + tk.E + tk.S + tk.W)
        
    def alpha_slider_callback(self):
        self.alpha = np.float(self.alpha_slider.get())   
    
    def lambda_slider_callback(self):
        self.lambda_regularizer = np.float(self.lambda_slider.get())
        
    def nodes_slider_callback(self):
        self.number_of_nodes = np.int_(self.nodes_slider.get())
        self.initialize_weights_and_bias()
        
    def number_of_samples_slider_callback(self):
        self.number_of_samples = self.number_of_samples_slider.get()
        self.input_data, self.input_labels = generate_data_samples_with_classes(self.type_of_data, self.number_of_samples, self.number_of_classes)
        self.display_points()
        
    def number_of_classes_slider_callback(self):
        self.number_of_classes = self.number_of_classes_slider.get()
        self.initialize_weights_and_bias()
        self.input_data, self.input_labels = generate_data_samples_with_classes(self.type_of_data, self.number_of_samples, self.number_of_classes)
        self.display_points()
        
    def initialize_weights_and_bias(self):
        print("weights and bias initialized.")
        self.Weights_hidden_layer = np.random.uniform(-0.001,0.001, self.input_dimension * self.number_of_nodes)
        self.Weights_hidden_layer = self.Weights_hidden_layer.reshape(self.input_dimension, self.number_of_nodes)
        self.Bias_hidden_layer = np.random.uniform(-0.001,0.001, self.number_of_nodes)
        print("shape of hidden layer weights and biases :: ", self.Weights_hidden_layer.size, self.Bias_hidden_layer.size)
        
        self.Weights_output_layer = np.random.uniform(-0.001,0.001, self.number_of_nodes * self.number_of_classes)
        self.Weights_output_layer = self.Weights_output_layer.reshape(self.number_of_nodes, self.number_of_classes)
        self.Bias_output_layer = np.random.uniform(-0.001,0.001, self.number_of_classes)
        print("shape of Output layer weights and biases :: ", self.Weights_output_layer.size, self.Bias_output_layer.size)
        
    def train_data(self):
        print("Training the model on the data.")
        input_data = self.input_data
        input_labels = self.input_labels
        training_epochs = 10
        lambda_regularizer = self.lambda_regularizer
        num_of_nodes_hidden_layer = self.number_of_nodes
        input_dimension = self.input_dimension
        num_of_nodes_output_layer = self.number_of_classes

        labels_onehot = np.zeros((input_data.shape[0], num_of_nodes_output_layer))
        labels_onehot_encoded = one_hot_encoding_of_input_labels(labels_onehot, input_labels)

        # build the model.... tensorflow graph input
        learning_rate = tf.Variable(self.alpha, dtype=tf.float32)
        x = tf.placeholder('float', [None, input_dimension])
        y = tf.placeholder('float', [None, num_of_nodes_output_layer])

        weights_hidden_layer = tf.Variable(self.Weights_hidden_layer, dtype=tf.float32)
        bias_hidden_layer = tf.Variable(self.Bias_hidden_layer, dtype=tf.float32)
        
        net_hidden = tf.add(tf.matmul(x, weights_hidden_layer), bias_hidden_layer)
        
        if( self.activation_type == "Relu" ):
            output_hidden_layer = tf.nn.relu(net_hidden)
        elif (self.activation_type == "Sigmoid"):
            output_hidden_layer = tf.nn.sigmoid(net_hidden)

        weights_output_layer = tf.Variable(self.Weights_output_layer, dtype=tf.float32)
        bias_output_layer = tf.Variable(self.Bias_output_layer, dtype=tf.float32)
        output_output_layer = tf.matmul(output_hidden_layer, weights_output_layer) + bias_output_layer

        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_output_layer, labels=y))
        regularizers = tf.nn.l2_loss(weights_hidden_layer) + tf.nn.l2_loss(weights_output_layer)
        cost = tf.reduce_mean(cross_entropy_loss + lambda_regularizer * tf.cast(regularizers, tf.float32))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        actuals = tf.argmax(output_output_layer, 1)
        targets = tf.argmax(y, 1)
        # For each index, equal() determines if the element in the first tensor equals the one in the second. We get an array of bools (True and False)
        num_of_correct_prediction = tf.equal(tf.argmax(output_output_layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(num_of_correct_prediction, 'float'))


        # initialize variables
#         init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        # launch graph
        with tf.Session() as sess:
            sess.run(init)
            # define training cycle
            for epoch in range(10):

                batch_x, batch_y = input_data,labels_onehot_encoded
                _cost_, _cross_entropy_loss, _actuals_, _accuracy, _optimizer_, _weights_hidden_layer_, _bias_hidden_layer_, _weights_output_layer_, _bias_output_layer_ = sess.run([cost, cross_entropy_loss, actuals, accuracy, optimizer, weights_hidden_layer, bias_hidden_layer, weights_output_layer, bias_output_layer], feed_dict = {x: batch_x, y: batch_y})
                
                self.Weights_hidden_layer = _weights_hidden_layer_
                self.Bias_hidden_layer = _bias_hidden_layer_
                self.Weights_output_layer = _weights_output_layer_
                self.Bias_output_layer = _bias_output_layer_
                
                print("Cost :::::  ",_cost_, " , cross_entropy_loss :: ", _cross_entropy_loss)
                print(" Accuracy :: ", _accuracy)
            
                self.display_line()

            print('Training finished for 10 epochs')
            


    def prediction_for_points(self, data_points):
        x = tf.placeholder('float', [None, self.input_dimension])

        weights_hidden_layer = tf.Variable(self.Weights_hidden_layer, dtype=tf.float32)
        bias_hidden_layer = tf.Variable(self.Bias_hidden_layer, dtype=tf.float32)
        net_hidden = tf.add(tf.matmul(x, weights_hidden_layer), bias_hidden_layer)
        
        if( self.activation_type == "Relu" ):
            output_hidden_layer = tf.nn.relu(net_hidden)
        elif (self.activation_type == "Sigmoid"):
            output_hidden_layer = tf.nn.sigmoid(net_hidden)

        weights_output_layer = tf.Variable(self.Weights_output_layer, dtype=tf.float32)
        bias_output_layer = tf.Variable(self.Bias_output_layer, dtype=tf.float32)
        output_output_layer = tf.matmul(output_hidden_layer, weights_output_layer) + bias_output_layer
        
        actuals = tf.argmax(output_output_layer, 1)
    
        init = tf.global_variables_initializer()
        # launch graph
        with tf.Session() as sess:
            sess.run(init)

            _x_, _actuals_ = sess.run([x, actuals], feed_dict = {x: data_points})
            
        return _actuals_
            
            
    def display_line(self):
        self.axes.cla()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        
        resolution=100
        xs = np.linspace(np.amin(self.input_data[:, 0]) - 0.5, np.amax(self.input_data[:, 0]) + 0.5, resolution)
        ys = np.linspace(np.amin(self.input_data[:, 1]) - 0.5, np.amax(self.input_data[:, 1]) + 0.5, resolution)
        xx, yy = np.meshgrid(xs, ys)
        count = 0
        data_points = np.zeros((resolution ** 2, 2))
        for i in range(resolution):
            for j in range(resolution):
                data_points[count, 0] = xx[i,j]
                data_points[count, 1] = yy[i,j]
                count += 1
                
        zz= self.prediction_for_points(data_points)
        zz = np.reshape(zz, (resolution, resolution))
        
        self.axes.pcolormesh(xx, yy, zz, cmap="Accent", zorder=1)
        
        self.axes.scatter(self.input_data[:, 0], self.input_data[:, 1], c=self.input_labels, zorder=2)
        
        self.axes.xaxis.set_visible(True)
        
        self.canvas.draw()
        
    def activation_function_dropdown_callback(self):
        self.activation_type = self.activation_function_variable.get()
        print("Activation Function chosen :: ", self.activation_type )
    
    def type_of_data_dropdown_callback(self):
        self.type_of_data = self.type_of_data_variable.get()
        self.initialize_weights_and_bias()
        self.input_data, self.input_labels = generate_data_samples_with_classes(self.type_of_data, self.number_of_samples, self.number_of_classes)
        self.display_points()
        
    def display_points(self):
        self.axes.cla()
        self.axes.set_title(self.type_of_data)
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.axes.scatter(self.input_data[:, 0], self.input_data[:, 1], c=self.input_labels)
        self.axes.xaxis.set_visible(True)
        self.canvas.draw()
   
def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

main_window = MainWindow(debug_print_flag=False)
main_window.wm_state('zoomed')
main_window.title('Assignment_05 --  Venkatesh')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()