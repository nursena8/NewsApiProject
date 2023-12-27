# NewsAPI Project
NewsAPI have used a news sites api ,This project is designed to gather and process data using the API of a news website to gain insights into various news categories.
- [Usage api](https://newsapi.org)

## Overview

The primary aim of this project is to extract data from a news website using its API. The gathered data includes the name of the news source, country, language, URL, ID, and the actual content of the news. The project processes this data to gain insights into different news categories.

## Workflow

1. **Data Gathering**: The project uses the NewsAPI to fetch news data, including source details, URLs, languages, and content.
2. **Data Processing**: The fetched data is cleaned and preprocessed to remove unnecessary elements like special characters, stop words, and non-alphabetic characters.
3. **Model Building**: Text classification models are constructed using machine learning algorithms to predict news categories based on their titles or content.
4. **Analysis and Visualization**: The project utilizes various visualization techniques like word clouds and bar plots to showcase the most common words in different news categories.

## Requirements

To run this project, you'll need:
- Python 3
- Necessary Python libraries (nltk, wordcloud, matplotlib, pandas, scikit-learn,numpy)
- Access to the NewsAPI with an API key

## Usage

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Obtain a NewsAPI key and replace `YOUR_API_KEY` in the code with your actual key.
4. Install `pip install newsapi-python`,'pip install nltk' to use apiclient and nltk library.
5. You can also run app.py too.
6. Run the provided scripts to fetch, process, and analyze the news data.

## Conclusion

The project demonstrates how to use the NewsAPI to collect news data, preprocess the information, build predictive models, and visualize insights based on news categories. This can be further expanded to encompass more sophisticated analysis and visualization techniques or integrated into larger-scale application
- We can predict category with text that is awsome :)
### Let's have a quick look up  to the project
![](https://github.com/nursena8/NewsApiProject/assets/115145369/1249f52c-34c7-4321-8ab5-e536563d6d23)
- Most coomon words

![](https://github.com/nursena8/NewsApiProject/assets/115145369/be57d7fd-7ba9-4544-94b6-d84d82bb6b75)
- news languages

Feel free to contribute or use this project as a reference for working with news-related APIs and data processing tasks.
