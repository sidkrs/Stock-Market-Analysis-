from newsapi import NewsApiClient
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
from kneed import KneeLocator
from scipy.spatial import distance_matrix

KEYWORD = 'AAPL'

def get_basket():
    basket = []
    on = 0
    while on == 0:
        ticker = input("Enter a ticker (one by one) or 'done' to finish: ")
        if ticker == 'done':
            on = 1
        else:
            ticker = ticker.capitalize()
            basket.append(ticker)
    return basket


def get_news(keyword=KEYWORD):
    '''connects to the newsapi and returns a dataframe of articles with the keyword'''
    newsapi = NewsApiClient(api_key='d5bc67ff273a429f950a14d711c0df2b') 
    all_articles = newsapi.get_everything(q=keyword)  
    df = pd.json_normalize(all_articles['articles'])
    return df


def apd_stock(basket, pcas):
    apd_val = []
    for i in pcas:
        apd_val.append(average_pairwise_distance(i))
    
    apds = {}
    for i in range(len(basket)):
        apds[basket[i]] = apd_val[i]
    
    return apds


def process_content(df):
    '''takes a dataframe of articles and returns a vectorized representation of the articles' content'''
    # Replace NaN with empty strings
    df['content'] = df['content'].fillna('')
    # Use TF-IDF to vectorize the articles' content
    vectorizer = TfidfVectorizer(stop_words='english')
    # Fit and transform the vectorizer on the content
    tfidf = vectorizer.fit_transform(df['content'])
    return tfidf


def reduce_dimensionality(tfidf):
    '''takes a vectorized representation of the articles' content and returns a 2D representation of the articles' content'''
    # Use PCA to reduce to 2 components
    pca = PCA(n_components=2)
    pca_reduced = pca.fit_transform(tfidf.toarray())
    return pca_reduced


def find_optimal_clusters(data):
    '''takes a 2D representation of the articles' content and returns the optimal number of clusters'''
    inertias = []
    # Find the inertia for each value of k
    ks = range(1, 10)
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=10).fit(data)
        inertias.append(kmeans.inertia_)
    
    # Assuming the elbow method, find the elbow point
    kn = KneeLocator(ks, inertias, curve='convex', direction='decreasing')
    optimal_k = kn.elbow
    return optimal_k, inertias


def cluster_data(data, k):
    '''takes a 2D representation of the articles' content and the optimal number of clusters and returns the cluster labels'''
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(data)
    # Return the cluster labels
    return kmeans.labels_


def plot_interia(ks, inertias, optimal_k):
    '''takes a range of k values, the inertia for each k value, and the optimal k value and 
    plots the inertia for each k value and the optimal k value'''
    plt.plot(ks, inertias, '-o', color='black')
    plt.plot(optimal_k, inertias[optimal_k - 1], 'rx')
    # Plot label and title
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.legend(['Inertia', 'Optimal k'])
    plt.show()

def plot_clusters(pca_results, labels, keyword=KEYWORD):
    '''takes a 2D representation of the articles' content 
    and the cluster labels and plots the clusters'''
    # Plot the clusters and sort the labels so that they match
    for cluster in np.unique(labels):
        indices = np.where(labels == cluster)
        plt.scatter(pca_results[indices, 0], pca_results[indices, 1], label=f"Cluster {cluster}")
    plt.title(f'PCA - KMeans Clustering for {keyword}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def average_pairwise_distance(points):
    # Convert list of points to a numpy array
    points_array = np.array(points)

    # Calculate the distance matrix
    dist_matrix = distance_matrix(points_array, points_array)

    # Sum up all distances and count the number of distances
    sum_distances = np.sum(dist_matrix)
    count_distances = len(points) * (len(points) - 1) # exclude distance from point to itself

    # Calculate the average distance
    avg_distance = sum_distances / count_distances
    return avg_distance

def get_tick_sent(keyword):
    df = get_news(keyword)
    tfidf = process_content(df)
    pca = reduce_dimensionality(tfidf)

    optimal_k, inertias = find_optimal_clusters(pca)
    
    # What value of k is optimal based on inertia?
    # print("Optimal value of k based on inertia:", optimal_k)
    labels = cluster_data(pca, optimal_k)
    
    # How many articles are in each cluster?
    # print("Number of articles in each cluster:", clusters_counts)
    
    # Plot Graphs
    return pca, labels, keyword

def plot(apds):
    ticker = max(apds, key=apds.get)
    pca, labels, keyword = get_tick_sent(ticker)
    plot_clusters(pca, labels, keyword)

def main():
    df = get_news()
    tfidf = process_content(df)
    pca = reduce_dimensionality(tfidf)
    
    # What keyword(s) did you use in your query?
    print("Keyword used:", KEYWORD)

    # After PCA has been applied what are the two values of the first article in your dataset?
    print("PCA first article components:", pca[0])
    optimal_k, inertias = find_optimal_clusters(pca)
    
    # What value of k is optimal based on inertia?
    print("Optimal value of k based on inertia:", optimal_k)
    labels = cluster_data(pca, optimal_k)
    clusters_counts = dict(Counter(labels))
    
    # How many articles are in each cluster?
    print("Number of articles in each cluster:", clusters_counts)
    
    # Plot Graphs
    plot_interia(range(1, 10), inertias, optimal_k)
    plot_clusters(pca, labels)

if __name__ == '__main__':
    main()

