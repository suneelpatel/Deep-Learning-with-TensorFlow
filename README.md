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

**Weights** The weight determine the slope of the classifier line, bias allows us to shift the line towards left or right.

**Bias** The threshold thing. Normally bias treated as another weighted input with input values x0 = 1

**Gradient Descent** The name for one commonly used optimization function that adjusts weights according to the error they caused is called “gradient descent.”


# 6. Multi Layer Perceptron
### Multi-layer Perceptron - Backpropagation algorithm		

A multi-layer perceptron (MLP) has the same structure of a single layer perceptron with one or more hidden layers. The backpropagation algorithm consists of two phases: the forward phase where the activations are propagated from the input to the output layer, and the backward phase, where the error between the observed actual and the requested nominal value in the output layer is propagated backwards in order to modify the weights and bias values. 		
 		
#### Forward propagation:		
Propagate inputs by adding all the weighted inputs and then computing outputs using sigmoid threshold.

# 7. Tensor Flow
TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.

#### Easy model building
TensorFlow offers multiple levels of abstraction so you can choose the right one for your needs. Build and train models by using the high-level Keras API, which makes getting started with TensorFlow and machine learning easy.

If you need more flexibility, eager execution allows for immediate iteration and intuitive debugging. For large ML training tasks, use the Distribution Strategy API for distributed training on different hardware configurations without changing the model definition.

#### Robust ML production anywhere
TensorFlow has always provided a direct path to production. Whether it’s on servers, edge devices, or the web, TensorFlow lets you train and deploy your model easily, no matter what language or platform you use.

Use TensorFlow Extended (TFX) if you need a full production ML pipeline. For running inference on mobile and edge devices, use TensorFlow Lite. Train and deploy models in JavaScript environments using TensorFlow.js.

#### Powerful experimentation for research
Build and train state-of-the-art models without sacrificing speed or performance. TensorFlow gives you the flexibility and control with features like the Keras Functional API and Model Subclassing API for creation of complex topologies. For easy prototyping and fast debugging, use eager execution.

TensorFlow also supports an ecosystem of powerful add-on libraries and models to experiment with, including Ragged Tensors, TensorFlow Probability, Tensor2Tensor and BERT.

# 8. Tensor Board
### TensorBoard: TensorFlow's visualization toolkit
TensorBoard provides the visualization and tooling needed for machine learning experimentation:
* Tracking and visualizing metrics such as loss and accuracy
* Visualizing the model graph (ops and layers)
* Viewing histograms of weights, biases, or other tensors as they change over time
* Projecting embeddings to a lower dimensional space
* Displaying images, text, and audio data
* Profiling TensorFlow programs

# 9. Linear Regression
Linear Regression Model Using TensorFlow
Linear Regression Model is used for predicting the unknown value of a variable (Dependent Variable) from the known value of another variables (Independent Variable) using linear regression equation as shown below:
	
		Y = b + wX
	
	Therefore, for creating a linear model, you need:
		Dependent or Output Variable (Y)
		Slope Variable (w)
		Y – Intercept or Bias (b)
		Independent or Input Variable (X)


# 10. Deep Network
Deep Network have more number of neurons, more number of neurons also facilitate better weight setting using backpropogation, more neurons add more weight to adjust. Using this Deep Networks are able to learn or say adjust their weight better, giving greater precision output. 

#### How is it different?	
* More Neurons than previous Networks
* More complex way of connecting layers
* Advancement of computing power
* Automatic Feature Extraction


# 11. Gradient Decent
Gradient Decent one of the most popular algorithm to perform optimization and by far the most common way to optimize neural networks.

* In GD optimization, we compute the cost gradient based on the complete training set, hence, we sometimes also call it a batch Gradient Decent.
* In case of very large dataset, using GD can be quite costly since we need to calculate the gradients for the whole dataset to perform just one update.
* Batch gradient decent can be very slow and is hard to control for datasets that don’t fit in memory.
* Batch gradient decent also doesn’t allow us to update our model online, i.e. with new example on the run.

#### Batch Gradient Decent
* In GD optimization, we compute the cost gradient based on the complete training set, hence, we sometimes also call it a batch Gradient Decent.
* In case of very large dataset, using GD can be quite costly since we need to calculate the gradients for the whole dataset to perform just one update.
* Batch gradient decent can be very slow and is hard to control for datasets that don’t fit in memory.
* Batch gradient decent also doesn’t allow us to update our model online, i.e. with new example on the run.

#### Stochastic Gradient Decent
* SDG eliminates the redundancy by performing one update at a time and therefore is usually much faster. It can also be used for online training.
* The term “Stochastic” comes from the fact that the gradient based on single training sample is a “Stochastic approximation” of the “true” cost gradient.
* Due to it stochastic nature, the path towards the global cost minimum is not “direct” as in GD, but may go “zig-zag” if we are visualizing the cost surface in a 2D space.

#### Mini-batch Gradient Decent
* Mini-batch gradient decent (MB-GD) a compromise between Batch Gradient Decent (BGD) and Stochastic Gradient Decent (SGD).
* In MB-GD, we update the model based on smaller groups of training samples (mini-batch).
* Therefore, instead of computing the gradient from  1 sample (Stochastic Gradient Decent) or all n training sample (Batch Gradient Decent), we compute the gradient from 1<k<n training samples.
* MB-GD converges in fewer iterations than BDG because the weight are updated more frequently.


# 12. Convolutional Neural Networks (Convnets or CNN)
Convolutional Neural networks allow computers to see, in other words, Convnets are used to recognize images by transforming the original image through layers to a class scores. 

CNNs, like neural networks, are made up of neurons with learnable weights and biases. 

CNN was inspired by the visual cortex.

Each neuron receives several inputs, takes a weighted sum over them, pass it through an activation function and responds with an output. 

#### How Do Convolutional Neural Networks Work?
There are four layered concepts we should understand in Convolutional Neural Networks:
1. Convolution,
2. ReLu,
3. Pooling and
4. Full Connectedness (Fully Connected Layer)

**Convolutional layer:** When we use Feedforward Neural Networks (Multi Layer Perceptron) for image classification, there are many challenges with it. The most frustrating challenge is that, it introduces a lot of parameters, consider the video tutorial on CNN.
To overcome this challenge Convolution Layer was introduced. it is assumed that, pixels that are spatially closer together will “cooperate” on forming a particular feature of interest much more than ones on opposite corners of the image. Also, if a particular (smaller) feature is found to be of great importance when defining an image’s label, it will be equally important, if this feature was found anywhere within the image, regardless of location.

**ReLU Layer:** Rectified Linear Unit (ReLU) transform function only activates a node if the input is above a certain quantity, while the input is below zero, the output is zero, but when the input rises above a certain threshold, it has a linear relationship with the dependent variable. 
* In this layer we remove every negative values from the filtered images and replaces it with zero’s
* This is done to avoid the values from summing up to zero

**Pooling Layer:** Pooling Layer performs a function to reduce the spatial dimensions of the input, and the computational complexity of our model. And it also controls Overfitting. It operates independently on every depth slice of the input. There are different functions such as Max pooling, average pooling, or L2-norm pooling. However, Max pooling is the most used type of pooling which only takes the most important part (the value of the brightest pixel) of the input volume.

**Fully Connected Layer:** Fully connected layers connect every neuron in one layer to every neuron in another layer. The last fully-connected layer uses a softmax activation function for classifying the generated features of the input image into various classes based on the training dataset.


# 13. Recurrent Neural Networks (RNN)
Recurrent Neural Networks (RNN) are a type of artificial neural network designed to recognize patterns in sequences of data, such as text, genomes, handwriting, spoken word, numerical time’s series data emanating from sensors, stock markets and government agencies.

RNNs are deep nets that can capture the short and long term temporal dependencies in the data, for sequential data such as speech or text.

Each neuron or unit of RNN uses its internal memory to maintain information about the previous input.
Maintaining sequential information is very important for language processing because for predicting the next word in a sentence one needs to know which words came before it.

### Training Recurrent Neural Networks

Recurrent Neural Networks use backpropagation algorithm for training, but it is applied for every timestamp. It is commonly known as Back-propagation Through Time (BTT).

There are some issues with Back-propagation such as:

* Vanishing Gradient
* Exploding Gradient

#### Vanishing Gradient
When making use of back-propagation the goal is to calculate the error which is actually found out by finding out the difference between the actual output and the model output and raising that to a power of 2.

#### Exploding Gradient
The working of the exploding gradient is similar but the weights here change drastically instead of negligible change. 

### Long Short-Term Memory Networks

Long Short-Term Memory networks are usually just called “LSTMs”.

They are a special kind of Recurrent Neural Networks which are capable of learning long-term dependencies.

#### What are long-term dependencies?

Many times only recent data is needed in a model to perform operations. But there might be a requirement from a data which was obtained in the past.

Let’s look at the following example:

Consider a language model trying to predict the next word based on the previous ones. If we are trying to predict the last word in the sentence say “The clouds are in the sky”.

The context here was pretty simple and the last word ends up being sky all the time. In such cases, the gap between the past information and the current requirement can be bridged really easily by using Recurrent Neural Networks.

So, problems like Vanishing and Exploding Gradients do not exist and this makes LSTM networks handle long-term dependencies easily.

# 14. Restricted Boltzmann Machine(RBM)
### What are Restricted Boltzmann Machines?
Restricted Boltzmann Machine is an undirected graphical model that plays a major role in Deep Learning Framework in recent times. It was initially introduced as Harmonium by Paul Smolensky in 1986 and it gained big popularity in recent years in the context of the Netflix Prize where Restricted Boltzmann Machines achieved state of the art performance in collaborative filtering and have beaten most of the competition.

It is an algorithm which is useful for dimensionality reduction, classification, regression, collaborative filtering, feature learning, and topic modeling.

#### Layers in Restricted Boltzmann Machine
Restricted Boltzmann Machines are shallow, two-layer neural nets that constitute the building blocks of deep-belief networks. The first layer of the RBM is called the visible, or input layer, and the second is the hidden layer. Each circle represents a neuron-like unit called a node. The nodes are connected to each other across layers, but no two nodes of the same layer are linked.

The restriction in a Restricted Boltzmann Machine is that there is no intra-layer communication. Each node is a locus of computation that processes input and begins by making stochastic decisions about whether to transmit that input or not.

#### Working of Restricted Boltzmann Machine
Each visible node takes a low-level feature from an item in the dataset to be learned. At node 1 of the hidden layer, x is multiplied by a weight and added to a bias. The result of those two operations is fed into an activation function, which produces the node’s output, or the strength of the signal passing through it, given input x.

At each hidden node, each input x is multiplied by its respective weight w. That is, a single input x would have three weights here, making 12 weights altogether (4 input nodes x 3 hidden nodes). The weights between the two layers will always form a matrix where the rows are equal to the input nodes, and the columns are equal to the output nodes.

Each hidden node receives the four inputs multiplied by their respective weights. The sum of those products is again added to a bias (which forces at least some activations to happen), and the result is passed through the activation algorithm producing one output for each hidden node.

Now that you have an idea about how Restricted Boltzmann Machine works, let’s continue our Restricted Boltzmann Machine Tutorial and have a look at the steps involved in the training of RBM.

#### Training of Restricted Boltzmann Machine
The training of the Restricted Boltzmann Machine differs from the training of regular neural networks via stochastic gradient descent.

The Two main Training steps are:

* Gibbs Sampling
* Contrastive Divergence step

#### The process from training to the prediction phase goes as follows:

* Train the network on the data of all users
* During inference-time, take the training data of a specific user
* Use this data to obtain the activations of hidden neurons
* Use the hidden neuron values to get the activations of input neurons
* The new values of input neurons show the rating the user would give yet unseen movies

# 15. Autoencoders
### What are Autoencoders?
An autoencoder neural network is an Unsupervised Machine learning algorithm that applies backpropagation, setting the target values to be equal to the inputs. Autoencoders are used to reduce the size of our inputs into a smaller representation. If anyone needs the original data, they can reconstruct it from the compressed data.

#### Applications of Autoencoders
* **Image Coloring:** Autoencoders are used for converting any black and white picture into a colored image. Depending on what is in the picture, it is possible to tell what the color should be.

* **Feature variation:** It extracts only the required features of an image and generates the output by removing any noise or unnecessary interruption.

* **Dimensionality Reduction:** The reconstructed image is the same as our input but with reduced dimensions. It helps in providing the similar image with a reduced pixel value.

* **Denoising Image:** he input seen by the autoencoder is not the raw input but a stochastically corrupted version. A denoising autoencoder is thus trained to reconstruct the original input from the noisy version.

* **Watermark Removal:** It is also used for removing watermarks from images or to remove any object while filming a video or a movie.

# 16. KERAS


# 17. TFLearn
