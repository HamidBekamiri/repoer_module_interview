import pandas as pd
import numpy as np
import streamlit as st
from itertools import cycle



class Interview_report:
    def __init__ (self, df):
        self.organization = df['Organization']
        self.expert = df['Expert_id']
        self.answer = df['Answer']

    def value_count(self):
        organization_count = self.organization.value_counts()
        expert_count = self.expert.value_counts()
        df_records = len(df)
        return df_records, organization_count, expert_count

    def clean(self, txt):
        txt = txt.str.replace("()", "")
        txt = txt.str.replace('(<a).*(>).*()', '')
        txt = txt.str.replace('(&amp)', '')
        txt = txt.str.replace('(&gt)', '')
        txt = txt.str.replace('(&lt)', '')
        txt = txt.str.replace('(\xa0)', ' ')
        return txt

    def preprocessing(self):
        answer_clean = self.clean(self.answer)
        # Converting to lower case
        answer_clean = answer_clean.apply(lambda x: " ".join(x.lower() for x in x.split()))
        # Removing the Punctuation
        answer_clean = answer_clean.str.replace('[^\w\s]', '')
        # Removing Stopwords
        import nltk
        from nltk.corpus import stopwords
        nltk.download("stopwords")
        stop = stopwords.words('english')
        answer_clean = answer_clean.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        answer_clean.head()
        # Remove the Rare Words
        freq = pd.Series(' '.join(answer_clean).split()).value_counts()
        less_freq = list(freq[freq ==1].index)
        answer_clean = answer_clean.apply(lambda x: " ".join(x for x in x.split() if x not in less_freq))
        # Spelling Correction
        from textblob import TextBlob, Word, Blobber
        answer_clean.apply(lambda x: str(TextBlob(x).correct()))
        answer_clean.head()
        import nltk
        nltk.download('wordnet')
        answer_clean = answer_clean.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        interview_len = answer_clean.astype(str).apply(len)
        word_count = answer_clean.apply(lambda x: len(str(x).split()))
        polarity = answer_clean.map(lambda text: TextBlob(text).sentiment.polarity)
        df_answer_clean = answer_clean.to_frame('answer_clean')
        df_interview_len = interview_len.to_frame('interview_len')
        df_word_count = word_count.to_frame('word_count')
        df_polarity = polarity.to_frame('polarity')
        df_all = pd.concat([self.organization, self.expert, df_answer_clean, df_interview_len, df_word_count, df_polarity], join='outer', axis=1,)
        return df_all, freq, less_freq

    def get_top_n_words(self, corpus, n=None):
        from sklearn.feature_extraction.text import CountVectorizer
        vec=CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        common_words = words_freq[:n]
        df1 = pd.DataFrame(common_words, columns = ['Answer_clean_1', 'Count'])
        df1.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Top Words", ylabel = "Count", title = "Bar Chart of Top Words Frequency")
        # Frequency Charts
        return words_freq[:n]

    def get_top_n_bigram(self, corpus, n=None):
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(ngram_range=(2,2)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        common_words2 = words_freq[:n]
        df2 = pd.DataFrame(common_words2, columns=['Answer_clean_1', "Count"])
        df2.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Bigram Words", ylabel = "Count", title = "Bar chart of Bigrams Frequency")
        return words_freq[:n]

    def get_top_n_trigram(self, corpus, n=None):
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        common_words3 = words_freq[:n]
        df3 = pd.DataFrame(common_words3, columns = ['Answer_clean_1' , 'Count'])
        df3.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Trigrams Words", ylabel = "Count", title = "Bar chart of Trigrams Frequency")
        return words_freq[:n]

    def visulization_report(self):
        df_all, freq, less_freq = Interview_report(df).preprocessing()
        df_all[["interview_len", "word_count", "polarity"]].hist(bins=20, figsize=(15, 10))
        # Polarity vs expert
        from matplotlib import pyplot as plt
        import seaborn as sns
        plt.figure(figsize = (10, 8))
        sns.set_style('whitegrid')
        sns.set(font_scale = 1.5)
        sns.boxplot(x = 'Expert_id', y = 'polarity', data = df_all)
        plt.xlabel("Expert_id")
        plt.ylabel("Polatiry")
        plt.title("Expert_id vs Polarity")
        # save the figure
        plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_1.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure(figsize = (10, 8))
        sns.set_style('whitegrid')
        sns.set(font_scale = 1.5)
        sns.boxplot(x = 'Organization', y = 'polarity', data = df_all)
        plt.xlabel("Organization")
        plt.ylabel("Polatiry")
        plt.title("Organization vs Polarity")
        plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_2.png', dpi=300, bbox_inches='tight')
        plt.show()
        mean_pol = df_all.groupby('Expert_id')['polarity'].agg([np.mean])
        mean_pol.columns = ['mean_polarity']
        fig, ax = plt.subplots(figsize=(11, 6))
        plt.bar(mean_pol.index, mean_pol.mean_polarity, width=0.3)
        #plt.gca().set_xticklabels(mean_pol.index, fontdict={'size': 14})
        mean_pol = df_all.groupby('Expert_id')['polarity'].agg([np.mean])
        mean_pol.columns = ['mean_polarity']
        fig, ax = plt.subplots(figsize=(11, 6))
        for i in ax.patches:
            ax.text(i.get_x(), i.get_height()+0.01, str("{:.2f}".format(i.get_height())))
        plt.title("Polarity of Expert_id", fontsize=22)
        plt.ylabel("Polarity", fontsize=16)
        plt.xlabel("Expert_id", fontsize=16)
        plt.ylim(0, 0.35)
        plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_3.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure(figsize=(11, 6))
        sns.countplot(x='Expert_id', data=df_all)
        plt.xlabel("Expert_id")
        plt.title("Number of questions data of each Expert_id")
        plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_4.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(11, 6))
        sns.countplot(x='Organization', data=df_all)
        plt.xlabel("Organization")
        plt.title("Number of questions data of each Expert_id")
        plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_5.png', dpi=300, bbox_inches='tight')
        plt.show()
        # Length of the interview vs the Rating
        # plt.figure(figsize=(11, 6))
        # sns.pointplot(x = "Expert_id", y = "interview_len", data = df_all)
        # plt.xlabel("Expert_id")
        # plt.ylabel("Expert_id Length")
        # plt.title("Product Expert_id vs Interview Length")
        # plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_6.png', dpi=300, bbox_inches='tight')
        # plt.show()
        # # Length of the Interview vs the Expert_id
        # plt.figure(figsize=(11, 6))
        # sns.pointplot(x = "Organization", y = "interview_len", data = df_all)
        # plt.xlabel("Organization")
        # plt.ylabel("Organization Length")
        # plt.title("Product Organization vs Interview Length")
        # plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_7.png', dpi=300, bbox_inches='tight')
        # plt.show()
        # # Top 5 products based on the Polarity
        # product_pol = df_all.groupby('Expert_id')['polarity'].agg([np.mean])
        # product_pol.columns = ['polarity']
        # product_pol = product_pol.sort_values('polarity', ascending=False)
        # product_pol = product_pol.head()
        # product_pol
        # WordCloud
        # conda install -c conda-forge wordcloud
        # text = " ".join(interview for interview in df_all.answer_clean)
        # from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
        # stopwords = set(STOPWORDS)
        # stopwords = stopwords.union(["ha", "thi", "now", "onli", "im", "becaus", "wa", "will", "even", "go", "realli", "didnt", "abl"])
        # wordcl = WordCloud(stopwords = stopwords, background_color='white', max_font_size = 50, max_words = 5000).generate(text)
        # plt.figure(figsize=(14, 12))
        # plt.imshow(wordcl, interpolation='bilinear')
        # plt.axis('off')
        # plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_8.png', dpi=300, bbox_inches='tight')
        # plt.show()
        # Frequency Charts
        from sklearn.feature_extraction.text import CountVectorizer
        common_words = Interview_report(df).get_top_n_words(df_all['answer_clean'], 20)
        df1 = pd.DataFrame(common_words, columns = ['Answer_clean_1', 'Count'])
        st.write(df1.head())
        df1.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot.bar()
        plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_9.png', dpi=300, bbox_inches='tight')
        # Frequency Charts
        common_words2 = Interview_report(df).get_top_n_bigram(df_all['answer_clean'], 30)
        df2 = pd.DataFrame(common_words2, columns=['Answer_clean_1', "Count"])
        df2.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Bigram Words", ylabel = "Count", title = "Bar chart of Bigrams Frequency")
        plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_10.png', dpi=300, bbox_inches='tight')
        # Frequency Charts
        common_words3 = Interview_report(df).get_top_n_trigram(df_all['answer_clean'], 30)
        df3 = pd.DataFrame(common_words3, columns = ['Answer_clean_1' , 'Count'])
        df3.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Trigrams Words", ylabel = "Count", title = "Bar chart of Trigrams Frequency")
        plt.savefig('/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_11.png', dpi=300, bbox_inches='tight')
        # Part-of -Speech Tagging
        from textblob import TextBlob, Word, Blobber
        blob = TextBlob(str(df_all['answer_clean']))
        pos_df = pd.DataFrame(blob.tags, columns = ['word', 'pos'])
        pos_df = pos_df.pos.value_counts()[:30]
        pos_df.plot(kind='bar', xlabel = "Part Of Speech", ylabel = "Frequency", title = "Bar Chart of the Frequency of the Parts of Speech", figsize=(10, 6))
        from sklearn.feature_extraction.text import CountVectorizer
        vec=CountVectorizer().fit(df.Answer)
        bag_of_words = vec.transform(df.Answer)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        common_words = words_freq[:10]
        df1 = pd.DataFrame(common_words, columns = ['Answer_clean_1', 'Count'])
        df1.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Top Words", ylabel = "Count", title = "Bar Chart of Top Words Frequency")
        # Frequency Charts

st.header("Statistical Report Module - Interview analysis app")
st.subheader("Upload CSV of Interview")
data_file = st.file_uploader("Please upload CSV file with ID, Organization, Expert_id, Question, and Answer columns!",type=['csv'])
if st.button("Process"):
    if data_file is not None:
        file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        st.write(file_details)

        df = pd.read_csv(data_file)
        st.dataframe(df)
        Interview_report.visulization_report(df)
        filteredImages = ['/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_1.png', '/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_2.png', '/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_4.png', '/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_5.png', '/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_8.png', '/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_9.png', '/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_10.png', '/Users/zhyarsozhyn/Documents/AIGrowthLab/NLP_CA/images/plot_11.png'] # your images here
        caption = ['Expert_id vs Polarity', 'Organization vs Polarityd', 'Number of questions data of each Expert_id', 'Number of questions data of each Expert_id', 'WordCloud', 'Bar Chart of Top Words Frequency', 'Bar chart of Bigrams Frequency', 'Bar chart of Trigrams Frequency'] # your caption here
        cols = cycle(st.columns(2)) # st.columns here since it is out of beta at the time I'm writing this
        for idx, filteredImage in enumerate(filteredImages):
            next(cols).image(filteredImage, width=350, caption=caption[idx])
