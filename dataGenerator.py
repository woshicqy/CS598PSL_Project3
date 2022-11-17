import pandas as pd
import numpy as np   
import os

data = pd.read_csv("alldata.tsv",sep='\t', header=0)
testIDs = pd.read_csv("project3_splits.csv", header=0)
data_top = list(data.columns.values)

data_arr = data.to_numpy()
testID_arr = testIDs.to_numpy()

for i in range(1,6):
    
    folderName = 'split_' + str(i)
    isExist = os.path.exists(folderName)
    if not isExist:
        os.mkdir(folderName)
    ### python start from 0, R starts from 1 ###
    index = testID_arr[:,i-1]-1
    # print(index.shape)
    # exit()

    train_split = data.iloc[~index,:]
    test_split = data.iloc[index,:]
    test_y = data.iloc[index,:]

    train_split = train_split[["id", "sentiment", "review"]]
    test_split = test_split[["id", "review"]]
    test_y = test_y[["id", "sentiment", "score"]]

    train_file = folderName + '/' + 'train.tsv'
    test_file = folderName + '/' + 'test.tsv'
    test_y_file = folderName + '/' 'test_y.tsv'

    np.savetxt(train_file, train_split, delimiter="\t",fmt='%s',header='\t'.join(["id", "sentiment", "review"]),comments='')
    np.savetxt(test_file, test_split, delimiter="\t",fmt='%s',header='\t'.join(["id", "review"]),comments='')
    np.savetxt(test_y_file, test_y, delimiter="\t",header='\t'.join(["id", "sentiment", "score"]),comments='')

    print(f'>>>>> Split {i} saving is done <<<<<')

print('>>>>> All data has already been splitted <<<<<')

