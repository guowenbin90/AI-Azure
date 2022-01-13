# Introduction to machine learning
## What is machine learning
Machine learning is a data science technique used to extract patterns from data, 
allowing computers to identify related data, and forecast future outcomes, behaviors, and trends.
## Applications of machine learning
1. Natural Language Processing
- Text - summarization, topic detection, similarity, search
- Speech - speech to text, text to speech translation
2. Computer Vision
- Self-driving cars
- Object dectection
- Object identification
- LIDAR and Visible Spectrum
3. Decision Making
- Sequential decision making problems
- Recommenders
4. Analytics
- Regression, Classification, Forecasting
- Clustering
## Data Science Process
- Collect data: Write code
- Prepare data: Wrtie queries & code
- Train model: Write code & do some math
- Evaluate model: Write code & do some math
- Delopy model -> Re-train model
## Types of Data
- Numerical
- Time-Series
- Categorical
- Text
- Image
## Scaling data
Standardization rescales data so that it has a mean of 0 and a standard deviation of 1. (𝑥 − 𝜇)/𝜎  
Normalization rescales the data into the range [0, 1]. (𝑥 −𝑥𝑚𝑖𝑛)/(𝑥𝑚𝑎𝑥 −𝑥𝑚𝑖𝑛)
## Encoding Categorical Data
- Ordinal encoding: One of the potential drawbacks to this approach is that it implicitly assumes an order across the categories. In the above example, Blue (which is encoded with a value of 2) seems to be more than Red (which is encoded with a value of 1), even though this is in fact not a meaningful way of comparing those values.
- One hot encoding: One drawback of one-hot encoding is that it can potentially generate a very large number of columns.
## Text Data
**Text normalization** is the process of transforming a piece of text into a canonical (official) form.  
Lemmatization is an example of normalization. A lemma is the dictionary form of a word and lemmatization is the process of reducing multiple inflections to that single dictionary form. For example, is, am, are -> be  
**Stop words** are high-frequency words that are unnecessary (or unwanted) during the analysis.  
**tokenized** the text (i.e., split each string of text into a list of smaller parts or tokens)  
## Two Perspectives on ML
CS: Output = Program(Input Features)  
We are using input features to create a program that can generate the desired output.
Statistics: Output Variable = f(Input Variables), Dependent Variable = f(Independent Variables)    
We are trying to find a mathematical function that, 
given the values of the independent variables can predict the values of the dependent variables.
## Libraries
### Core Framework and Tools
- Python is a very popular high-level programming language that is great for data science. Its ease of use and wide support within popular machine learning platforms, coupled with a large catalog of ML libraries, has made it a leader in this space.
- Pandas is an open-source Python library designed for analyzing and manipulating data. It is particularly good for working with tabular data and time-series data.
- NumPy, like Pandas, is a Python library. NumPy provides support for large, multi-dimensional arrays of data, and has many high-level mathematical functions that can be used to perform operations on these arrays.

### Machine Learning and Deep Learning
- Scikit-Learn is a Python library designed specifically for machine learning. It is designed to be integrated with other scientific and data-analysis libraries, such as NumPy, SciPy, and matplotlib (described below).
- Apache Spark is an open-source analytics engine that is designed for cluster-computing and that is often used for large-scale data processing and big data.
- TensorFlow is a free, open-source software library for machine learning built by Google Brain.
- Keras is a Python deep-learning library. It provide an Application Programming Interface (API) that can be used to interface with other libraries, such as TensorFlow, in order to program neural networks. Keras is designed for rapid development and experimentation.
- PyTorch is an open source library for machine learning, developed in large part by Facebook's AI Research lab. It is known for being comparatively easy to use, especially for developers already familiar with Python and a Pythonic code style.

### Data Visualization
- Plotly is not itself a library, but rather a company that provides a number of different front-end tools for machine learning and data science—including an open source graphing library for Python.
- Matplotlib is a Python library designed for plotting 2D visualizations. It can be used to produce graphs and other figures that are high quality and usable in professional publications. You'll see that the Matplotlib library is used by a number of other libraries and tools, such as SciKit Learn (above) and Seaborn (below). You can easily import Matplotlib for use in a Python script or to create visualizations within a Jupyter Notebook.
- Seaborn is a Python library designed specifically for data visualization. It is based on matplotlib, but provides a more high-level interface and has additional features for making visualizations more attractive and informative.
- Bokeh is an interactive data visualization library. In contrast to a library like matplotlib that generates a static image as its output, Bokeh generates visualizations in HTML and JavaScript. This allows for web-based visualizations that can have interactive features.

## Classical ML vs. Deep Learning

## Approaches to machine learning:
1. Supervised learning
- **Classification**: Outputs are categorical.
- **Regression**: Outputs are continuous and numerical.
- **Similarity learning**: Learns from examples using a similarity function that measures how similar two objects are.
- **Feature learning**: Learns to automatically discover the representations or features from raw data.
- **Anomaly detection**: A special form of classification, which learns from data labeled as normal/abnormal.
2. Unsupervised learning
- **Clustering**: Assigns entities to clusters or groups.
- **Feature learning**: Features are learned from unlabeled data.
- **Anomaly detection**: Learns from unlabeled data, using the assumption that the majority of entities are normal.
3. Reinforcement learning
- **Markov decision process**: A mathematical process to model decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. Does not assume knowledge of an exact mathematical model.

## The Trade-Offs
**Bias** measures how inaccurate the model prediction is in comparison with the true output. It is due to erroneous assumptions made in the machine learning process to simplify the model and make the target function easier to learn. High model complexity tends to have a low bias.  
**Variance** measures how much the target function will change if different training data is used. Variance can be caused by modeling the random noise in the training data. High model complexity tends to have a high variance.  
**Overfitting** refers to the situation in which models fit the training data very well, but fail to generalize to new data.  
**Underfitting** refers to the situation in which models neither fit the training data nor generalize to new data.

prediction error = Bias error + variance + error + irreducible error  

**Overfitting vs. Underfitting**
- **k-fold cross-validation**: it split the initial training data into k subsets and train the model k times. In each training, it uses one subset as the testing data and the rest as training data.
- hold back a **validation dataset** from the initial training data to estimatete how well the model generalizes on new data.
- **simplify** the model. For example, using fewer layers or less neurons to make the neural network smaller.
- use **more data**.
- **reduce dimensionality** in training data such as PCA: it projects training data into a smaller dimension to decrease the model complexity.
- **Stop the training early** when the performance on the testing dataset has not improved after a number of training iterations.

# Model training
