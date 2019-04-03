# data-science-portfolio
##### This repository contains data analysis, visualization and machine learning projects I worked on during my course, as part of competitions, my hobby, curiosity and endeavours to expand and practice my skillset.
-------------------------------------------------------------------------------------------------------------------------------------

## Python

### Programming related:
1. Pandas
2. Advanced Python
3. Matplotlib
4. Scikit learn
5. h20
6. xgboost
7. Tensorflow
8. keras
9. OpenCV
10. Data scraping



### Data Cleaning
1. Real Estate data wrangling
2. [Text data preperation from xml files](https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Wrangling/Xml-Text-preprocessing.ipynb "Python Jupyter notebook")
Given a xml file containing patent abstract data, I attempt to extract and preprocess for downstream analytical tasks.
Libraries used - `BeautifulSoup`, `pandas`, `numpy`, `nltk`
3. [Data wrangling using json files](https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Wrangling/Json-data-wrangling.ipynb "Python Jupyter notebook")
Given a json file containing real estate data, I attempt to clean it up for downstream analytical tasks.
Libraries used - `json`, `pandas`, `seaborn`, `matplotlib`, `numpy`, `sklearn`
4. [Text data preprocessing of news articles](https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Machine%20Learning/text-classification/News-classification/pre-processing/News-Preprocessing.ipynb "Python Jupyter notebook")
Given a set of training and test dataset of news articles, I attempt to preprocess them for classification task. 
Libraries used - `nltk`, `pandas`, `numpy`

### Data analysis

1. Inside Airbnb Exploratory Data analysis (to be uploaded)
2. Myki data Exploratory Data Analysis (to be uploaded)

### Machine learning projects

#### Kaggle
1. Titanic dataset - Predicting survivors
2. Mnist digit classification
3. Object detection
4. Object localization


#### Reinforcement learning
1. Coming soon

#### Text Classification
1. [News classification](https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Machine%20Learning/text-classification/News-classification/modelling/News-Classification.ipynb "Python Jupyter notebook")
Preprocessed news article are classified using different machine learning models (Naive Bayes, Multilayer Perceptron, Logistic regression, Linear SVM, CNN using wordembeddings, CNN+LSTM, Fasttext) for a kaggle style university competition, in an attempt to get highest accuracy. Libraries used - `sklearn`, `keras`, `pandas`, `numpy`, `fastext`


## R

### Data analysis & interactive visualizations
1.[  Car accidents in Victoria interactive visualization](https://vm24.shinyapps.io/Victoria-car-accidents/ "ShinyApp")
I used Victoria's vehicle accident and hospital data to create this interactive R Shiny visualization for deriving insights. Data cleaning and wrangling was performed in python, and the visualization was created in R Shiny. Libraries used: `R Shiny`, `chorddiag`, `ggplot2`, `leaflet`, `plotly` <br>
2. Datathon 2018 - Congestion and speed limit violation interactive visualization -
Given myki data and average car speed and location data taken from sensors at Datathon 2018, I combined the average car speed data with speed limit data shape file and traffic volume data shape file in QGIS, cleaned, transformed and wrangled the data in python to create this interactive visualization in R shiny. Using this visualization we can spot congested roads during different time of the day, sections of the road where speed limits are frequently violated, and also under utilized roads. This visualization can help in improving the road network in victoria. Libraries used: `R Shiny`,  `leaflet`

## Scala
1. Coming soon

## Pig & Hive
1. Coming soon
