# Deep-Learning-with-TensorFlow
Learn Deep Learning from this repository which help you to understand clearly about Deep Neural Network, SPL, MPL Algorithm, CNN, RNN, and RBM etc.

# Table of Content
1. Deep Learning Introdution
2. Neural Network
3. Activation Function
4. Perceptorn
5. Single Layer Perceptron
6. Multi Layer Perceptron
7. Tensor Flow
8. Tensor Board
9. Linear Regression
10. Deep Network
11. Gradient Decent
12. Convolutional Neural Networks (CNN)
13. Recurrent Neural Networks (RNN)
14. Restricted Boltzmann Machine(RBM)
15. Autoencoders
16. KERAS
17. TFLearn

# 1. Deep Learning Introdution
Deep Learning is a branch of Machine Learning based on a set of algorithms that attempt to model high-level abstraction in the data by using a deep graph with multiple processing layers. It is composed of multiple linear and non-linear transformations.

Deep learning mimics the way our brain functions i.e. it learns from experience.

* A collection of statistical machine learning techniques used to learn feature hierarchies often based on artificial neural networks.

* Deep learning is a specific approach used for building and training neural networks, which can perform task like recognizing objects from an image, understanding speech and languages, and playing board games.

* An algorithm is considered to be deep if the input data passed through a series of nonlinearities or nonlinear transformation before become output.



# 2. Neural Network

### What is Neuorn?
The neuron is the basic working unit of the brain, a specialized cell designed to transmit information to other nerve cells, muscle, or gland cells. Neurons are cells within the nervous system that transmit information to other nerve cells, muscle, or gland cells. Most neurons have a cell body, an axon, and dendrites.

* **Soma (cell body)** — this portion of the neuron receives information. It contains the cell's nucleus.
* **Dendrites** — these thin filaments carry information from other neurons to the soma. They are the "input" part of the cell.
* **Axon** — this long projection carries information from the soma and sends it off to other cells. This is the "output" part of the cell. It normally ends with a number of synapses connecting to the dendrites of other neurons.

Both dendrites and axons are sometimes referred to as nerve fibers.

Axons vary in length a great deal. Some can be tiny, whereas others can be over 1 meter long. The longest axon is called the dorsal root ganglion (DRG), a cluster of nerve cell bodies that carries information from the skin to the brain. Some of the axons in the DRG travel from the toes to the brain stem — up to 2 meters in a tall person.

#### Types of neurons
Neurons can be split into types in different ways, for instance, by connection or function.
##### Connection
* ** Efferent neurons** — these take messages from the central nervous system (brain and spinal cord) and deliver them to cells in other parts of the body.
* **Afferent neurons** — take messages from the rest of the body and deliver them to the central nervous system (CNS).
* **Interneurons** — these relay messages between neurons in the CNS.

###Function
* **Sensory** — carry signals from the senses to the CNS.
* **Relay** — carry signals from one place to another within the CNS.
* **Motor** — carry signals from the CNS to muscles.

#### How do neurons carry a message?
If a neuron receives a large number of inputs from other neurons, these signals add up until they exceed a particular threshold.

Once this threshold is exceeded, the neuron is triggered to send an impulse along its axon — this is called an action potential.

An action potential is created by the movement of electrically charged atoms (ions) across the axon's membrane.

Neurons at rest are more negatively charged than the fluid that surrounds them; this is referred to as the membrane potential. It is usually -70 millivolts (mV).

### Neural Network
A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria.

A Neural Network is a network of one or more neurons

Neural Networks are a computation model that shares some properties with the animal brain where many simple units (neurons) are working in parallel with no centralized control unit.

A neural network’s main function it to receive a set of inputs, perform progressively complex computations, and then use the output to solve a problem.


# 3. Activation Function
Activation or Transformer Function translates the input into outputs. It uses threshold to produce output.

Activation functions are an extremely important feature of the artificial neural networks. They basically decide whether a neuron should be activated or not. Whether the information that the neuron is receiving is relevant for the given information or should it be ignored. 

#### Important Activation Functions are:
1. Linear or Identity Function
2. Unit or Binary Step Function
3. Sigmoid or Logistic Function
4. TanH Function
5. ReLU Function
6. Softmax Function


# 4. Perceptorn
Perceptron is a fundamental unit of the neural network which takes weighted inputs, process it and capable of performing binary classifications.

The Perceptron is a linear model used for binary classification. It models the neurons. It receives n inputs (corresponding to each feature). It then sums those inputs, applies a transformation and produce an output. It has two functions, Summation and Transformation (activation)

In the modern sense, the Perceptron is an algorithm for learning a binary classifier called a threshold function: a function that maps its input (a real-valued vector) to an output value (a single binary value).


# 5. Single Layer Perceptron
A single layer perceptron (SLP) is a feed-forward network based on a threshold transfer function. SLP is the simplest type of artificial neural networks and can only classify linearly separable cases with a binary target (1 , 0).		

**Algorithm**	The single layer perceptron does not have a priori knowledge, so the initial weights are assigned randomly. SLP sums all the weighted inputs and if the sum is above the threshold (some predetermined value), SLP is said to be activated (output=1). 
		
The input values are presented to the perceptron, and if the predicted output is the same as the desired output, then the performance is considered satisfactory and no changes to the weights are made. However, if the output does not match the desired output, then the weights need to be changed to reduce the error. 

Because SLP is a linear classifier and if the cases are not linearly separable the learning process will never reach a point where all the cases are classified properly. The most famous example of the inability of perceptron to solve problems with linearly non-separable cases is the XOR problem.

**Weights** 	The weight determine the slope of the classifier line, bias allows us to shift the line towards left or right.

**Bias**	The threshold thing. Normally bias treated as another weighted input with input values x0 = 1

**Gradient Descent**	The name for one commonly used optimization function that adjusts weights according to the error they caused is called “gradient descent.”


# 6. Multi Layer Perceptron
### Multi-layer Perceptron - Backpropagation algorithm		

A multi-layer perceptron (MLP) has the same structure of a single layer perceptron with one or more hidden layers. The backpropagation algorithm consists of two phases: the forward phase where the activations are propagated from the input to the output layer, and the backward phase, where the error between the observed actual and the requested nominal value in the output layer is propagated backwards in order to modify the weights and bias values. 		
 		
#### Forward propagation:		
Propagate inputs by adding all the weighted inputs and then computing outputs using sigmoid threshold.


# 7. Tensor Flow


# 8. Tensor Board


# 9. Linear Regression


# 10. Deep Network


# 11. Gradient Decent


# 12. Convolutional Neural Networks (CNN)


# 13. Recurrent Neural Networks (RNN)


# 14. Restricted Boltzmann Machine(RBM)


# 15. Autoencoders


# 16. KERAS


# 17. TFLearn
