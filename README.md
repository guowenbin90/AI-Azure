
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
