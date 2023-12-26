class NEWS_API_Project:

    def main():
        """importing some necesarry libraries"""

        import pandas as pd #data preprocessing
        from newsapi import NewsApiClient #to use api
        from IPython.display import JSON #display json
        from  config import NEW_API # importing api keys from config file
        #plotting and cleaning text libraries
        import nltk
        from nltk.corpus import stopwords
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        from collections import Counter
        import string
        import numpy as np # linear algebra
        import re
        from sklearn.feature_extraction.text import TfidfVectorizer # for vektorizing text
        from sklearn.model_selection import train_test_split # splitting dataframe
        from sklearn.ensemble import RandomForestClassifier # random forest modeling
        from sklearn.metrics import accuracy_score, classification_report # useful metrics

        """# data collecting"""

        newsapi=NewsApiClient(api_key=NEW_API) # using Client to use api
        top_headlines=newsapi.get_top_headlines(q="bitcoin",sources="bbc-news,the-verge",
                                                language="en")

        sources=newsapi.get_sources() #get source from api

        sources["sources"]

        #checking parsing data what will show you

        data=[]
        for source in sources["sources"]:
            id = source["id"]
            name=source["name"]
            description=source["description"]
            url=source["url"]
            kategori=source["category"]
            language=source["language"]
            country=source["country"]

            data.append({"name":name,"id":id,"description":description,"url":url,"kategori":kategori,"language":language,"country":country})
        df=pd.DataFrame(data)
        df

        """# EDA"""

        df.head()

        df.info()

        #crearting workflow
        def json_to_dataframe(api_key):
            """
            convert given json to dataframe
            :param api_key :string indicate

            this method use your api_key and collect sources from api.
            """
            newsapi=NewsApiClient(api_key=api_key)
            top_headlines=newsapi.get_top_headlines(q="bitcoin",sources="bbc-news,the-verge",language="en")
            sources=newsapi.get_sources()
            # parsing data
            data=[]
            for source in sources["sources"]:
                id=source["id"]
                name=source["name"]
                category=source["category"]
                description=source["description"]
                country=source["country"]
                language=source["language"]
                url=source["url"]
                # append parsing data to list
                data.append({"id":id,"name":name,"category":category,"description":description,"country":country,"language":language,"url":url})
            df=pd.DataFrame(data)
            return df

        df=json_to_dataframe(NEW_API) #call the method

        df["category"].nunique() #checking number of unique variables for "category"

        df["category"]=df["category"].astype("category") #chance dtype to categorical

        """# useful methods

        if you want to see only url by categories you can use this method   ✨
        """

        def choose_category_to_see_url(category_type):
            """
            :param category_type: string
            pandas DataFrame.loc
            """
            print(df.loc[df["category"]==category_type,"url"])

        choose_category_to_see_url("science") #call the method

        #selecting language
        def choose_language(language_type):
            """
            :param language_type:string indicate
            """
            language=df.loc[df.language=="en",df.columns]
            return language

        choose_language("en")

        def find_your_news(category,country,language):  # filtering  dataframe by country,language and category.
            """
            :param category:string indicate
            :param country:string indicate
            :param language:string indicate
            """
            return df[(df["category"]==category)&(df["country"]==country)&(df["language"]==language)]
        find_your_news("general","us","en").head()

        df[df["description"].str.contains("breaking|news")].head() # looking any word in description feature

        def search_news_with_a_word(word):
            """
            :param word string indicate
            Series.str.contains
            pandas doc link:https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html
            """
            return df[df["description"].str.contains(word)]
        search_news_with_a_word("sport")

        df.query("category=='general'&country=='us'&language=='en'").head(6)

        # visualizatitons

        df["category"].value_counts().plot(kind="bar") # plotting  counts of each category  of the dataframe
        plt.title("plotting  counts of each category  of the dataframe")

        """general has the most news against to the others"""

        df["country"].value_counts().plot(kind="bar") # plotting each country  of the dataframe
        plt.title(" plotting each country  of the dataframe")

        df["language"].value_counts().plot(kind="bar") # plotting  each language of the dataframe
        plt.title(" plotting  each language of the dataframe")

        df[df["description"].str.contains("spor")]["country"].value_counts().plot(kind="bar")
        plt.title(" plotting each country that contains spor news count of description")

        """some insights
        *   we see that the great britain has the most sport news against to other countries
        *   most common language is english and "us" has the most new sites of dataframe


        """

        def plotting_most_common_words(language):

            """
            :param:language_type:string indicate
            cleaning desciription and plotting most common words in description feature
            """
            nltk.download('stopwords') # downloading stopword for repated words

            stop_words = set(stopwords.words(language)) #choosing language

            special_chars = set(string.punctuation) # choosing puctuation for pure text

            all_descriptions = " ".join(df['description']) # join all description of dataframe

            words = all_descriptions.lower().split() # convert into lower case for consistency

            filtered_words = [word for word in words if word not in stop_words and word not in special_chars ] # select text

            word_freq = Counter(filtered_words) # count filtered list

            res = word_freq.most_common(20) # add most common 20 words to res

            wc = WordCloud(background_color='white', width=800, height=600) # to plotting use WordCloud
            # plotting
            plt.figure(figsize=(15, 7))
            plt.imshow(wc.generate_from_frequencies({k: v for k, v in res}))
            plt.axis("off")
            plt.show()

        plotting_most_common_words("english")

        """We see that the most common word is news as we predict :)
        Some category names are most common too.
        """

        def plot_most_common_words_in_category(category, df):# this method will give you some information about which words are in there category
            """
            :param catgory:string indicate
            :param df:dataframe you have used
            """
            category_df = df[df['category'] == category]
            all_titles = " ".join(category_df['description'])

            stop_words = set(stopwords.words('english'))
            special_chars = set(string.punctuation)

            words = all_titles.lower().split()
            filtered_words = [word for word in words if word not in stop_words and word not in special_chars]

            word_freq = Counter(filtered_words)
            res = word_freq.most_common(20)

            wc = WordCloud(background_color='white', width=800, height=600)

            plt.figure(figsize=(15, 7))
            plt.imshow(wc.generate_from_frequencies({k: v for k, v in res}))
            plt.axis("off")
            plt.show()

            plot_most_common_words_in_category("business",df)

        #column ekleme _name__main readme projeleri string metoları plotlar textden hangi kategori olduğunu bulma

if __name__=="main":
    main()