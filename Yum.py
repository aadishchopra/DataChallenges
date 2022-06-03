import pandas as pd
import numpy as np
import nbformat
from matplotlib import pyplot as plt
from gensim.models import Word2Vec,KeyedVectors
from nltk.corpus import stopwords
import nltk as nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import re
from nltk.cluster import KMeansClusterer,euclidean_distance,cosine_distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px


df=pd.read_csv('Menu Items.csv')
df.head()
df.shape

#Data Quality Checks

dtype=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
dtype=pd.concat([dtype,pd.DataFrame(df.isnull().sum()).T.rename(index={0:'nulls'})])
dtype=pd.concat([dtype,pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0:'%nulls'})])

'''

for i in df.columns:
    if i not in ['Price', 'Description']:
        df_new = pd.DataFrame(df[i].value_counts()).sort_values(by=i, ascending=False).reset_index().rename(columns={'index':i, i:'Count'}).head(10)
        fig = px.pie(df_new, names=i, values='Count', color_discrete_sequence=px.colors.sequential.RdBu, title='Top 10 {}'.format(i))
        fig.show()
        
'''

#Pre-ProcessData
'''
1. Remove $ from the price amounts so that it could be aggregated.
2. Removal of unwanted words/digits from text
3. Removal of NaN from Items and setting it to 
4. Removal of Punctuation and stop words
'''
#1
df['Price'] = df['Price'].map(lambda x: x.lstrip('$'))
df['Price'] = df['Price'].map(lambda x: float(x))
#2
df['Section'] = df['Section'].map(lambda x: x.replace('must be 0 to purchase',''))
df['Item']=df['Item'].fillna('ItemNullFill')
df['Item'] = df['Item'].map(lambda x: x.replace('must be 0 to purchase',''))
#3

#4
stop = set(stopwords.words("english"))

def preprocess(text):
    text_input = re.sub('[^a-zA-Z1-9]+', ' ', str(text))
    output = re.sub(r'\d+', '',text_input)
    return output.lower().strip()

def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


df['Item'] = df.Item.map(preprocess)
df['Item'] = df.Item.map(remove_stopwords)

'''
1. Top 10 Sellers by Revenue
2. Word-Embeddings (For Item and Section)

'''

#1.
rev_rest=df.groupby('Restaurant').agg({'Price':sum}).sort_values(by='Price',ascending=0)[:10].plot(kind='bar',figsize=(14,8),title="Revenue by Restaurants")
rev_rest.bar_label(rev_rest.containers[0], label_type='edge')



#Restaurants having most Items
df_new = pd.DataFrame(df['Restaurant'].value_counts()).sort_values(by='Restaurant', ascending=False).reset_index().rename(columns={'index':'Restaurant', 'Restaurant':'Count'}).head(10)
plt.title(label='Restaurants having most Items')
plt.pie(df_new['Count'],labels=df_new['Restaurant'])
plt.show()



#Most Popular categories
df_new = pd.DataFrame(df['Section'].value_counts()).sort_values(by='Section', ascending=False).reset_index().rename(columns={'index':'Section', 'Section':'Count'}).head(10)
plt.title(label='Most Popular categories')
plt.pie(df_new['Count'],labels=df_new['Section'])
plt.show()


#Most Popular Items
df_new = pd.DataFrame(df['Item'].value_counts()).sort_values(by='Item', ascending=False).reset_index().rename(columns={'index':'Item', 'Item':'Count'}).head(10)
plt.title(label='Most Popular Items')
plt.pie(df_new['Count'],labels=df_new['Item'])
plt.show()








#2
#Item Word-Embeddings
Items=df['Item'].values
ItemVec=[nltk.word_tokenize(item) for item in Items]
model=Word2Vec(ItemVec)


#--------------------------------------------------------------------------------#




# fit a 2d PCA model to the vectors
words = list(model.wv.key_to_index)
vectors=model.wv.vectors
pca = PCA(n_components=2)
PCA_result = pca.fit_transform(vectors)
#PCA_result['x_values'] =PCA_result.iloc[0:, 0]
#PCA_result['y_values'] =PCA_result.iloc[0:, 1]
PCA_final = pd.merge(pd.DataFrame(words,columns=['words']), pd.DataFrame(PCA_result,columns=['x_values','y_values']), left_index=True, right_index=True)



def elbow_method(PCA_result):
    """
    This is the function used to get optimal number of clusters in order to feed to the k-means clustering algorithm.
    """

    number_clusters = range(1, 10)  # Range of possible clusters that can be generated
    kmeans = [KMeans(n_clusters=i, max_iter=600) for i in number_clusters]  # Getting no. of clusters

    score = [kmeans[i].fit(PCA_result).score(PCA_result) for i in
             range(len(kmeans))]  # Getting score corresponding to each cluster.
    score = [i * -1 for i in score]  # Getting list of positive scores.

    plt.plot(number_clusters, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Method')
    plt.show()


elbow_method(PCA_result)
# Optimal Clusters = 3



clusterer=KMeansClusterer(3,distance=cosine_distance)
clusters=clusterer.cluster(vectors,True,trace=True)



cluster_groups=pd.DataFrame(data={'words':words,'group':clusters})
#print(clusterer.classify(vector=vectors))

#Method Using sklearn
kmeans = KMeans(n_clusters= 3, max_iter=400, algorithm = 'auto')# Partition 'n' no. of observations into 'k' no. of clusters.
fitted = kmeans.fit(PCA_result)
prediction = kmeans.predict(PCA_result)


for i, word in enumerate(words):
    print(word + ":" + str(clusters[i]))


plt.figure(figsize=(20,20))
for i, word in enumerate(words):
    colors=['red','green','blue']
    plt.annotate(word, xy=(PCA_final['x_values'][i], PCA_final['y_values'][i]),color=colors[cluster_groups['group'][i]])
plt.show()





def classify_sentences(ItemVector):
    l=[]
    for i,token in enumerate(ItemVector):
        try:
            for word in token:
                try:
                    for i,word in enumerate(cluster_groups['words']):
                        print(cluster_groups['group'][i])
                except:
                    print("0 Didn't find ")
        except:
            print("Didn't find")

classify_sentences(ItemVec)









