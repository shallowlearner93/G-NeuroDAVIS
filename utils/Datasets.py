import numpy as np
import pandas as pd

path = '/home/tih_isi_7/G-NeuroDAVIS/Data/PreprocessedData/'
# save = ''

def LoadData(dataname, DType='both', LType=True):

    if dataname in ['Mnist', 'FMnist']:
        from keras import datasets as ds
        if dataname == 'Mnist':
            (x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = ds.fashion_mnist.load_data()

        if DType == 'train':
            x_train = np.expand_dims(x_train, -1).astype("float32") / 255
            data = np.array([x_train[i].flatten() for i in range(x_train.shape[0])])
            if LType == True:
                return data, y_train
            else:
                return data

        elif DType == 'test':
            x_test = np.expand_dims(x_test, -1).astype("float32") / 255
            data = np.array([x_test[i].flatten() for i in range(x_test.shape[0])])
            if LType == True:
                return data, y_test
            else:
                return data

        elif DType == 'None':
            return y_train, y_test
            
        else:
            x_train = np.expand_dims(x_train, -1).astype("float32") / 255
            x_test = np.expand_dims(x_test, -1).astype("float32") / 255
            data_train = np.array([x_train[i].flatten() for i in range(x_train.shape[0])])
            data_test = np.array([x_test[i].flatten() for i in range(x_test.shape[0])])
            if LType == True:
                return data_train, data_test, y_train, y_test
            else:
                return data_train, data_test
        
    elif dataname in ['3Rings', 'Olympics', 'Shape', 'EllipticRing', 'SwissRoll']:
            if LType == 'data':
                x = pd.read_csv(path+dataname+'/'+dataname+'.csv', index_col=0)
                return x
            elif LType == 'label':
                y = pd.read_csv(path+dataname+'/'+dataname+'_groundTruth.csv', index_col=0)
                y = y.astype(int)
                y=y.iloc[:,0]
                return y
            else:
                x = pd.read_csv(path+dataname+'/'+dataname+'.csv', index_col=0)
                y = pd.read_csv(path+dataname+'/'+dataname+'_groundTruth.csv', index_col=0)
                y = y.astype(int)
                y=y.iloc[:,0]
                return x, y
    
    elif dataname == 'Coil20':

        
        y_train = pd.read_csv(path+dataname+'/'+dataname+'_groundTruth.csv',header=0, index_col=0)
        
        if DType == 'None':
            return y_train

        x_train = pd.read_csv(path+dataname+'/'+dataname+'.csv',header=0, index_col=0)
        x_train = np.array(x_train.drop(['Label'], axis=1))

        if LType == True:
            return x_train, y_train
    
        else:
            return x_train

    elif dataname == 'OlivettiFaces':
        from sklearn import datasets as ds
        faces = ds.fetch_olivetti_faces()
        x_train = faces.images
        y_train = faces.target
        data = np.array([x_train[i].flatten() for i in range(x_train.shape[0])])
        
        if DType == 'None':
            return y_train
        
        if LType == True:
            return data, y_train
    
        else:
            return data

    else:
        datas = ['Mnist', 'FMnist', 'Coil20', 'OlivettiFaces']
        print("Data not found")
        print('Available datasets:',datas)
        
        
        























########################################## End of Code ############################################



