import os
import numpy as np
import matplotlib.pyplot as plt


# def correlation(dist1, dist2):
#     from scipy.stats import spearmanr, pearsonr
#     coef, p_val = pearsonr(dist1, dist2)
#     return coef

def TrustScore(x, x_new, nei):
    from sklearn.manifold import trustworthiness
    score = trustworthiness(x, x_new, n_neighbors=nei)
    return score


# Clustering
def kmeans(X,y):
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score
    
    n = len(np.unique(y))
    cl = KMeans(n_clusters=n).fit_predict(X)
    
    ARI = adjusted_rand_score(cl,y)
    FMI = fowlkes_mallows_score(cl,y)
    NMI = normalized_mutual_info_score(cl,y)
    
    Score = [ARI,NMI,FMI]
    return Score
    
# Clustering    
def Agglomerative(X,y):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score
    
    n = len(np.unique(y))
    cl = AgglomerativeClustering(n_clusters=n).fit_predict(X)
    
    ARI = adjusted_rand_score(cl,y)
    FMI = fowlkes_mallows_score(cl,y)
    NMI = normalized_mutual_info_score(cl,y)
    
    Score = [ARI,NMI,FMI]
    return Score



# Classification
def Knn(X,y,nei):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=nei)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    Score = [acc,pre,rec,f1]
    return Score

# Classification
def RFC(X,y,ntrees):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    rfc = RandomForestClassifier(n_estimators=ntrees)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    Score = [acc,pre,rec,f1]
    return Score
  

def plot_latent_space(model_decoder, x_embedding=None, digit_size = 28, n=30, dataname="data",
                      methodname="method", res_plots=None, figsize=15, sc=2.0):
    # if res_plots is None:
    #     res_plots = %pwd
    #     res_plots = res_plots+'/'
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    if x_embedding is None:
        scale = sc
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]
    else:
        grid_x = np.linspace(np.quantile(x_embedding, 0.2, axis=0)[0], np.quantile(x_embedding, 0.8, axis=0)[0], n)
        grid_y = np.linspace(np.quantile(x_embedding, 0.2, axis=0)[1], np.quantile(x_embedding, 0.8, axis=0)[1], n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = model_decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel(methodname+"1", fontsize=15)
    plt.ylabel(methodname+"2", fontsize=15)
    plt.imshow(figure, cmap="Greys_r")

    if res_plots is None:
        plt.savefig(dataname+'_'+methodname+'_embedding_mapping.pdf')
        plt.show()
    else:    
        newpath = res_plots+dataname+'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plt.savefig(newpath+dataname+'_'+methodname+'_embedding_mapping.pdf')
        plt.show()



def generate(model_decoder, x_embedding, sample = 10, dataname="data", methodname="method",res_plots=None, digit_size=28):
    # if res_plots is None:
    #     res_plots = %pwd
    #     res_plots = res_plots+'/'
    pred = np.random.uniform(low=np.quantile(x_embedding, 0.2, axis=0), high=np.quantile(x_embedding, 0.8, axis=0), size=(sample,2))
    predictions_high = model_decoder.predict(pred)
    fig, ax = plt.subplots(int(sample/5), 5, figsize=(12,6))
    for i in range(sample):
        plt.gray()
        ax[i//5, i%5].imshow(predictions_high[i].reshape(digit_size,digit_size))

    if res_plots is None:
        plt.savefig(dataname+'_'+methodname+'_samples.pdf')
        plt.show()
    else:    
        newpath = res_plots+dataname+'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plt.savefig(res_plots+dataname+'/'+dataname+'_'+methodname+'_samples.pdf')
        plt.show()



def conditional_generation(model_encoder, model_decoder, input, dataname="data", methodname="method",
                           sample=5, res_plots=None, digit_size=28):
    pred = np.eye(input.shape[1])
    fig, ax = plt.subplots(input.shape[1], sample, figsize=(10,20))
    for s in range(sample):
        predictions_low = model_encoder.predict(pred)
        predictions_high = model_decoder.predict(predictions_low[-1])
        for i in range(input.shape[1]):
            plt.gray()
            ax[i, s].imshow(predictions_high[i].reshape(digit_size,digit_size))

    if res_plots is None:
        plt.savefig(dataname+'_'+methodname+'_conditional_samples.pdf')    
        plt.show()
    else:
        newpath = res_plots+dataname+'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plt.savefig(res_plots+dataname+'/'+dataname+'_'+methodname+'_conditional_samples.pdf')
        plt.show()



################################ End of Code ##################################################



    
