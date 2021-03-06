# data-science-portfolio
##### This repository contains data analysis, visualization and machine learning projects I worked on during my course, as part of competitions, my hobby, curiosity and endeavours to expand and practice my skillset.
-------------------------------------------------------------------------------------------------------------------------------------

## Python


### Data Cleaning

1. [Text data preperation from xml files](https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Wrangling/Xml-Text-preprocessing.ipynb "Python Jupyter notebook"): <br>
Given a xml file containing patent abstract data, I attempt to extract and preprocess for downstream analytical tasks.
Libraries used - `BeautifulSoup`, `pandas`, `numpy`, `nltk`
2. [Data wrangling using json files](https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Wrangling/Json-data-wrangling.ipynb "Python Jupyter notebook") :<br>
Given a json file containing real estate data, I attempt to clean it up for downstream analytical tasks.
Libraries used - `json`, `pandas`, `seaborn`, `matplotlib`, `numpy`, `sklearn`
3. [Text data preprocessing of news articles](https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Machine%20Learning/text-classification/News-classification/pre-processing/News-Preprocessing.ipynb "Python Jupyter notebook"): <br>
Given a set of training and test dataset of news articles, I attempt to preprocess them for classification task. <br>
`pip install -r requirements.txt` should help get the environment setup to run this notebook. The requirements.txt file can be found at <a href="https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Machine%20Learning/text-classification/News-classification/requirements.txt">here</a>. <br>
Libraries used - `nltk`, `pandas`, `numpy`


### Machine learning projects

#### Text Classification
1. [News classification](https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Machine%20Learning/text-classification/News-classification/modelling/News-Classification.ipynb "Python Jupyter notebook"): <br>
Preprocessed news article are classified using different machine learning models (Naive Bayes, Multilayer Perceptron, Logistic regression, Linear SVM, CNN using wordembeddings, CNN+LSTM, Fasttext) for a kaggle style university competition, in an attempt to get highest accuracy. 
`pip install -r requirements.txt` should help get the environment setup to run this notebook. The requirements.txt file can be found at <a href="https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Machine%20Learning/text-classification/News-classification/requirements.txt">here</a>. Instructions on how to install keras (using tensorflow backend) can be found <a href="https://keras.io/#installation">here</a>.<br>
Main libraries used - `sklearn`, `keras`, `pandas`, `numpy`, `fastext`


#### Kaggle
1. [Bike share rental demand prediction on AWS Sagemaker](https://github.com/VarunM24/data-science-portfolio/tree/master/Python/Machine%20Learning/Kaggle/Bike-Share "Python Jupyter notebook")  :
In this Kaggle <a href="https://www.kaggle.com/c/bike-sharing-demand/overview">competition</a > we are given a set of historical bike rental data spanning 2 years. We need to use first 19 days of hourly bike rental data to predict the hourly bike rental demand count for the rest of the month. <br>
- Data was preprocessed and explored in this <a href="https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Machine%20Learning/Kaggle/Bike-Share/kaggle_bikeshare_data_preparation.ipynb">notebook</a> on AWS Sagemaker.
- Data modelling was done using xgboost in this <a href="https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Machine%20Learning/Kaggle/Bike-Share/kaggle_bikeshare_xgboost_cloud_training.ipynb">notebook</a> on AWS Sagemaker cloud inatace.
- The deployed endpoint was used to make predictions for test data using this <a href="https://github.com/VarunM24/data-science-portfolio/blob/master/Python/Machine%20Learning/Kaggle/Bike-Share/kaggle_bikeshare_xgboost_cloud_prediction.ipynb">notebook </a> on AWS Sagemaker. <br>
Main libraries/tools used - `boto3`, `sagemaker`, `pandas`, `numpy`, `matplotlib`


## R

### Data analysis & interactive visualizations
1. [  Car accidents in Victoria interactive visualization app](https://vm24.shinyapps.io/victoria-car-accidents/ "ShinyApp"): <br>
I used Victoria's vehicle accident and hospital data to create this interactive R Shiny visualization app for deriving insights. Data cleaning and wrangling was performed in python, and the visualization was created in R Shiny. Libraries used: `R Shiny`, `chorddiag`, `ggplot2`, `leaflet`, `plotly` 
2. Datathon 2018 - Congestion and speed limit violation interactive visualization (to be uploaded): <br>
Given myki data and average car speed and location data taken from sensors at Datathon 2018, I combined the average car speed data with speed limit data shape file and traffic volume data shape file in QGIS, cleaned, transformed and wrangled the data in python to create this interactive visualization in R shiny. Using this visualization we can spot congested roads during different time of the day, sections of the road where speed limits are frequently violated, and also under utilized roads. This visualization can help in improving the road network in victoria. Libraries used: `R Shiny`,  `leaflet`

## Tableau visualizations:
1. [Movie rating dashboard](https://public.tableau.com/views/Moviedashboard/Moviepopularity?:embed=y&:display_count=yes&publish=yes "Tableau dashboard"): Using rotten tomatoes data containing hollywood movies data, I created this dashboard to see compare rating by audience and rotten tomatoes critics.

2. [Journalist deaths in various countries storytelling](https://public.tableau.com/views/Journalist_Deaths_15543637300810/Story1?:embed=y&:display_count=yes&publish=yes "Tableau storytelling"): This story is about notoriousness of various countries for deaths of journalist over the years.

## Scala
1. Coming soon

