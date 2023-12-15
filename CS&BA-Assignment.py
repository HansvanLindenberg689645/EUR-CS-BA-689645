#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:07:51 2023

@author: hansvanlindenberg
"""

import json
import pandas as pd
import re 
import numpy as np
import random
import matplotlib.pyplot as plt
import math


json_file_path = '/Users/hansvanlindenberg/Documents/a. Master Econometrie/a. Blok 2/Computer Science for Business Analytics/Assignment /TVs-all-merged.json'

# Open the JSON file in read mode
with open(json_file_path, 'r') as json_file:
    # Load the JSON data from the file
    dataDic = json.load(json_file)

keyslist = list(dataDic.keys())

# Create a dictionary with all the 'titles' of the tv's
titles = {}
for key in keyslist:
    titles[key] = dataDic[key][0]["title"]

# Create a list for different features
shop_vector = []
url_vector = []
modelID_vector = []
features_map_vector = []
title_vector = []


for tv_id, entries in dataDic.items():
    for entry in entries:
        # Split the values into separate variables
        
        url_value = entry['url']
        url_vector.append(url_value)
    
        modelID_value = entry['modelID']
        modelID_vector.append(modelID_value)
    
        features_map_value = entry['featuresMap']
        features_map_vector.append(features_map_value)
        
        shop_value = entry['shop']
        shop_vector.append(shop_value)
        
        title_value = entry['title']
        title_vector.append(title_value)
        
data_dict = {
    'URL': url_vector,
    'Shop': shop_vector,
    'ModelID': modelID_vector,
    'Features Map': features_map_vector,
    'Title': title_vector
}

# Create a DataFrame from the dictionary
data = pd.DataFrame(data_dict)
data["Brand"] = data["Features Map"].apply(lambda x: x.get("Brand") if isinstance(x, dict) else float('nan'))
data["UPC"] = data["Features Map"].apply(lambda x: x.get("UPC") if isinstance(x, dict) else float('nan'))
data["Product Height (with stand)"] = data["Features Map"].apply(lambda x: x.get("Product Height (with stand)") if isinstance(x, dict) else float('nan'))
data["Product Height (without stand)"] = data["Features Map"].apply(lambda x: x.get("Product Height (without stand)") if isinstance(x, dict) else float('nan'))
data["Screen Size (Measured Diagonally)"] = data["Features Map"].apply(lambda x: x.get("Screen Size (Measured Diagonally)") if isinstance(x, dict) else float('nan'))
data["TV Type"] = data["Features Map"].apply(lambda x: x.get("TV Type") if isinstance(x, dict) else float('nan'))
data["Maximum Resolution"] = data["Features Map"].apply(lambda x: x.get("Maximum Resolution") if isinstance(x, dict) else float('nan'))
data["Vertical Resolution"] = data["Features Map"].apply(lambda x: x.get("Vertical Resolution") if isinstance(x, dict) else float('nan'))
data["Refresh Rate"] = data["Features Map"].apply(
    lambda x: next((v for k, v in x.items() if "Refresh Rate" in k), None) 
    if isinstance(x, dict) else None)
data["HDMI"] = data["Features Map"].apply(
    lambda x: next((v for k, v in x.items() if "HDMI" in k), None) 
    if isinstance(x, dict) else None)


#edit the columns to get more general data
def generalizeData(data: pd.DataFrame):
    
    data['Title'] = data['Title'].str.lower()
    data['Title'] = data['Title'].apply(lambda x: re.sub('(-inch|"|inches| inch|"|in.)', 'inch', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('(\.0inch)', 'inch', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('(hz|-hz|hertz)', 'hz', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('(newegg.com|thenerds.net|best buy|diagonal|diag|class)', '', x))
    data['Title'] = data['Title'].apply(lambda x: re.sub('(ledlcd|LED-LCD)', 'led lcd', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('( \. )', ' ', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('( -| /|\(|\)|)', '', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('(refurbished|class|series|diag\.|led|hdtv)', '', x ))
    data['Title'] = data['Title'].apply(lambda x: re.sub('( : )', '', x ))

    
    data['URL'] = data['URL'].apply(lambda x: re.sub('(http://www.)', '', x ))
    data['URL'] = data['URL'].apply(lambda x: re.sub('(.com)', '', x ))
    
    data['Product Height (with stand)'] = data['Product Height (with stand)'].apply(lambda x: re.sub('(-inch|"|inches| inch|"|in.)', 'inch', x) if isinstance(x, str) else x)
    
    data['Product Height (without stand)'] = data['Product Height (without stand)'].apply(lambda x: re.sub('(-inch|"|inches| inch|"|in.)', 'inch', x) if isinstance(x, str) else x)

    data['Screen Size (Measured Diagonally)'] = data['Screen Size (Measured Diagonally)'].apply(lambda x: re.sub('(")', 'inch', x) if isinstance(x, str) else x)

    data['Refresh Rate'] = data['Refresh Rate'].apply(lambda x: re.sub(r'\D', '', str(x)) + 'hz' if x is not None else None)

    data['HDMI'] = data['HDMI'].apply(lambda x: re.sub(r'\D*(\d).*', r'\1', str(x)) if x is not None else None)

    return data

generalizeData(data)



different_features =['URL', 'Title', 'Brand', 'UPC', 'Product Height (with stand)', 'Product Height (without stand)', 'Screen Size (Measured Diagonally)', 'TV Type', 'Maximum Resolution', 'Vertical Resolution','Refresh Rate', 'HDMI']

def deleteSingleTitleWords(data: pd.DataFrame):
    allWords = []
    
    for feature_i in data["Title"]:
        for word in feature_i.split():
            allWords.append(word)
    
    wordCount = pd.Series(allWords).value_counts()
    wordCount.name = "Count"
    
    wordCount = wordCount.rename_axis('Word').reset_index()
    singleWord = wordCount[wordCount['Count'] == 1]
    
    for i in singleWord:
        data["Title"] = data["Title"].str.replace(r'\b{}\b'.format(i), ' ')
    
    return data

deleteSingleTitleWords(data)



#words that only show up once will not detect duplicates therefore we delete them
def deleteSingleWords(data: pd.DataFrame, different_features):
    
    for feature in different_features: 
        allWords = []
        
        for feature_i in data[feature]:
            if feature_i is not None:
                for word in feature_i.split():
                    allWords.append(word)
            
        wordCount = pd.Series(allWords).value_counts()
        wordCount.name = "Count"
        
        wordCount = wordCount.rename_axis('Word').reset_index()
        singleWord = wordCount[wordCount['Count'] == 1]
        
        for i in singleWord:
            data[feature] = data[feature].str.replace(r'\b{}\b'.format(i), ' ')        
    
    return data

deleteSingleWords(data, different_features)



#similar to different_features but now put them in a dictionary because of hashing
def importantFeatures(data: pd.DataFrame):
    importantFeatures = {}
    for index, row in data.iterrows(): 
            importantFeatures[index] = {
                "Index": index,
                "URL": row["URL"],
                "Title": row["Title"],
                "Brand": row["Brand"],
                "UPC": row["UPC"],
                "Product Height (with stand)": row["Product Height (with stand)"],
                "Product Height (without stand)": row["Product Height (without stand)"],
                "Screen Size (Measured Diagonally)": row["Screen Size (Measured Diagonally)"],
                "TV Type": row["TV Type"],
                "Maximum Resolution": row["Maximum Resolution"],
                "Vertical Resolution": row["Vertical Resolution"],
                "Refresh Rate": row["Refresh Rate"],
                "HDMI": row["HDMI"],
                }
    return importantFeatures

importantFeatures(data)


def shingles(feature_str):
    splitted = feature_str.split()
    return(set(splitted))

def bin_vec(different_features, data):
    n = len(data)
    total_feature_vect = np.empty((1,n))

    for feature in different_features:
      shingle_union = set([])
      column = data[feature]
      for i in column:
          if i is not None:
              shingle_union.update(shingles(i))

      vector_repr = [[0]*len(data) for i in range(len(shingle_union))]
      # One-hot encoding
      # Transform the sets into lists for indexing
      # Maybe first clean so that no spaces are in the way
      # and things that are very similar are counted as well (using particular distance)
      shingle_union = list(shingle_union)
      for index in range(n):
          ft_string = data[feature][index]
          if ft_string is not None:
            shingle_i = list(shingles(ft_string))
            match_index = [shingle_union.index(j) if j in shingle_union else -10 for j in shingle_i]
            for j in match_index:
                if j>=0:
                    vector_repr[j][index] = 1

      # Convert nested list to numpy array for easier use later on
      vector_repr = np.array(vector_repr)

      # Add this to the total matrix
      total_feature_vect = np.append(total_feature_vect, vector_repr, axis=0)

    # Delete first 'empty' row
    total_feature_vect = np.delete(total_feature_vect, 0, axis=0)

    return total_feature_vect

l = bin_vec(different_features, data)



def minhashing(feature_vec, h):
    # feature_vec is a matrix with each column representing the feature vector 
    # of one product from the dataset
    [m,n] = feature_vec.shape    
    # Create an empty signature matrix S of size (h, #products)
    S = np.zeros((h, n), dtype=int)
    
    for i in range(h):
        # Generate a random permutation for each iteration
        permutation = random.sample(range(m), m)
        # Generate the corresponding permutated feature_vector
        feature_vec_perm = feature_vec[permutation,:]
        # Determine the hash values for each product/column
        for j in range(n):
            row_index = np.where(feature_vec_perm[:,j]==1)[0][0]

            # Set the corresponding entry in the signature matrix
            S[i, j] = row_index
            
    return S

S = minhashing(l, h=1000)



#calculate Jaccard Similarity  
def JaccardSim(a, b):
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    if union == 0:
        return 0
    Jsim = float(intersection) / union  
      
    return Jsim 
        

#LSH, find candidate pairs hashed
def candidatePairs(S, bandsize):
    bandsize = int(bandsize)
    candidatePairs = []
    [v,w] = S.shape
    u = math.ceil(v/bandsize)
    BandedSignature = [[0]*w for l in range(u)]
    for i in range(w):
        band_cnt = 0
        #seperate the signature into bands
        splitted_bands = [bandsize*l for l in range(1, u)]
        for bandSignature in np.array_split(S[:,i], splitted_bands):
            my_list = str(bandSignature)
            hashValue = hash(my_list)
            BandedSignature[band_cnt][i] = hashValue
            band_cnt += 1
        band_cnt = 0
        #seperate the signature into bands
        split_indices = [bandsize*l for l in range(1, u)]
        for bandSignature in np.array_split(S[:,i], split_indices):
            my_list = str(bandSignature)
            hashValue = hash(my_list) 
            BandedSignature[band_cnt][i] = hashValue
            band_cnt += 1
            
    for i in range(w-1):
        for k in range(i+1, w):
            for b in range(u):
                if BandedSignature[b][i] == BandedSignature[b][k]:
                    candidatePairs.append([i, k])
                    break
                
    return (candidatePairs)


bandsize=10
cp = candidatePairs(S, bandsize)


def CosineSim(a, b):
    similarity = np.dot(a ,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return similarity

#Filter out some candidate pairs using classification 
def classification (candidatePairs, threshold, data, ft_vector):
    candidatePairsC = []
    ft_array = np.array(ft_vector)
    
    for candidatePair in candidatePairs:
        c1 = candidatePair[0]
        c2 = candidatePair[1]
        
        if (data["Shop"][c1] != data["Shop"][c2]):
            if (data["Brand"][c1] is None) or (data["Brand"][c2] is None) or (data["Brand"][c1] == data["Brand"][c2]):   
                # sim = JaccardSim(ft_array[:,c1], ft_array[:,c2])
                simC = CosineSim(ft_array[:,c1], ft_array[:,c2])
                if  simC >= threshold:
                    candidatePairsC.append(candidatePair)
    return candidatePairsC 
    
threshold=0.6 
test = classification(cp, threshold, data, S)



def realDuplicates(data):
    countedModelIDList = data["ModelID"].value_counts()
    duplicateCount = 0               
    for i in countedModelIDList:        
        if i == 2:
            duplicateCount += 1
        elif i == 3:
            duplicateCount += 3
        elif i == 4:
            duplicateCount += 6

    return countedModelIDList, duplicateCount

[totalDuplicates, count_d] = realDuplicates(data)

        
def estimatedDuplicates(cp, data):
    truePositive = 0
    falsePositive = 0
    tpPairs = []
    for candidatePair in cp:
        c1 = candidatePair[0]
        c2 = candidatePair[1]
        if  data["ModelID"][c1] == data["ModelID"][c2]:
            truePositive += 1
            tpPairs.append(candidatePair)
        else:
            falsePositive += 1
            
    return truePositive, falsePositive

TP, FP = estimatedDuplicates(cp, data)


# F1 score
def F1(cp, data):   
    totalDuplicates = realDuplicates(data)[1]
    TP, FP = estimatedDuplicates(cp, data)
    if TP == 0:
        return 0, 0, 0, 0
    FN = (totalDuplicates - TP)
    precision = TP / (TP+FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall, TP

# F1star score
def F1star(candidatepairs, data):
    n_real_dupl = realDuplicates(data)[1]
    n_compared = len(candidatepairs)
    n_est_dupl = estimatedDuplicates(candidatepairs, data)[0]
    if n_compared != 0 and n_real_dupl !=0 and n_est_dupl != 0:
        pair_quality = n_est_dupl/ n_compared 
        pair_completeness = n_est_dupl/n_real_dupl
        f1_star = 2 * ((pair_quality * pair_completeness) / (pair_quality + pair_completeness))
    else:
        pair_quality = 0 
        pair_completeness = 0 
        f1_star = 0
    
    return  f1_star, pair_quality, pair_completeness



I = 5 # number of bootstraps
p = 0.63 # training/test split

H = 500 # number of minhashes/size of signature matrix
R = np.arange(1, H, 3) # number of bands (all factors of the size of signature matrix)

comparisons_made =[0 for k in range(len(R))]
metrics = [[0]*len(R) for k in range(4)] # pair quality, pair completeness, f1*, f1

# fraction of comparisons 
q = 0
while q < I:
    ## Split data randomly into train and test data
    rnd_data = data.sample(frac=1)
    N = len(rnd_data)
    train_size = round(N*p)
    train_data = rnd_data[:train_size]
    test_data = rnd_data[train_size:]
    train_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)
    
    if realDuplicates(train_data) != 0:
        # Your code to execute when there are no real duplicates in train_data
        q += 1
        
    for index in range(len(R)):
        ## Run the algorithm for different b and r
        feature_matrix = bin_vec(different_features, train_data)
        signature_matrix = minhashing(feature_matrix, H)
        cp = candidatePairs(signature_matrix, R[index])
        comparisons_made[index] += len(cp)
        F1_score = F1(cp, train_data)
        F1_star_score = F1star(cp, train_data)
        
        # Save the scores in the 'metrics' matrix
        metrics[0][index] += F1_star_score[0]
        metrics[1][index] += F1_star_score[1]
        metrics[2][index] += F1_star_score[2]
        metrics[3][index] += F1_score[0]

metrics_final = [[a/I for a in l] for l in metrics]
comparisons_per_bootstrap = [a/I for a in comparisons_made]

#make a plot 
total_possible_comparisons = len(train_data)*((len(train_data)-1) / 2)
labels = ["F1*", "Pair quality", "Pair completeness", "F1"]
total_comparisons = len(train_data)*(len(train_data)-1)/2
fraction_of_comparisons = [m/total_comparisons for m in comparisons_per_bootstrap]

for i in range(4):
    plt.plot(fraction_of_comparisons, metrics_final[i])
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("{}".format(labels[i]))
    plt.show()
    
    


#gridsearch 
I = 5 # number of bootstraps
p = 0.63 # training/test split
H = 100 # number of minhashes/size of signature matrix
parameters = np.arange(0.1, 1, 0.1)
R = np.arange(1, H/10, 1) # number of rows per band
B = [int(H/i) for i in R] 
BR = [(1/b)**(1/bandsize) for b,bandsize in zip(B,R)]

R_select = []
for x in parameters:
    BR_t = [abs(z-x) for z in BR]
    R_select.append(R[BR_t.index(min(BR_t))])
    
param_t = np.arange(0.5, 0.8, 0.1) # possible values of threshold in classification

best_parameters = [[0]*I for k in range(len(R_select))]
metrics = [[0]*len(R_select) for k in range(4)] 
fraction_of_comparisons = [0 for k in range(len(R_select))]

max_score = 0 #F1
i = 0
while i < I:
    ## Split data randomly in train and test data
    rnd_data = data.sample(frac=1)
    N = len(rnd_data)
    train_size = round(N*p)
    train_data = rnd_data[:train_size]
    test_data = rnd_data[train_size:]
    train_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)
    
    if realDuplicates(train_data)[1]/len(train_data) < 0.15:
        i += 1
    
        for index in range(len(R_select)):
            for t in param_t:
                ## Run the algorithm for different b and r
                feature_matrix = bin_vec(different_features, train_data)
                signature_matrix = minhashing(feature_matrix, H)
                cp = candidatePairs(signature_matrix, R_select[index])
                cp2 = classification(cp, t, train_data, signature_matrix)
                F1_score = F1(cp2, train_data)[0]
                F1_star_score = F1star(cp, train_data)
                
                if F1_score > max_score:
                    best_t = t
                    max_score = F1_score
            
                print(i, index, t)
                
            feature_matrix = bin_vec(different_features, test_data)
            signature_matrix = minhashing(feature_matrix, H)
            cp = candidatePairs(signature_matrix, R_select[index])
            cp2 = classification(cp, t, train_data, signature_matrix)
            F1_score = F1(cp2, test_data)[0]
            F1_star_score = F1star(cp, test_data)
            total_comparisons = len(test_data)*(len(test_data)-1)/2
            fraction_of_comparisons[index] += (len(cp)/total_comparisons)
            
            best_parameters[index][i-1]= best_t
                
            # Save the scores in the 'metrics' matrix
            metrics[0][index] += F1_star_score[0]
            metrics[1][index] += F1_star_score[1]
            metrics[2][index] += F1_star_score[2]
            metrics[3][index] += F1_score

metrics_final = [[a/I for a in l] for l in metrics]
print(max_score)

for i in range(4):
    plt.scatter(fraction_of_comparisons, metrics_final[i])
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("{}".format(labels[i]))
    plt.show()
