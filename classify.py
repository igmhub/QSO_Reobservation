import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
from sklearn.preprocessing import LabelEncoder
import speclite
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

# Routine to read brick files (to be modified with the new standard)
def readBricks(path_in,brick_name):
    hdus = []
    for channel in 'brz':
        filename = 'brick-{}-{}.fits'.format(channel,brick_name)
        hdulist = fits.open(os.path.join(path_in,filename))
        hdus.append(hdulist)
    return hdus
# Routine to downsample the input spectra (we use speclite)
# si is an offset to make the number of points an integer divisible
# by ndownsample
def downsample(hdus,camera,nspec,ndownsample, si=0):
    data = np.ones((nspec,len(hdus[camera][2].data[si:])), dtype=[('flux', float), ('ivar',float)])
    data['flux']=hdus[camera][0].data[:,si:]
    data['ivar']=hdus[camera][1].data[:,si:]
    return speclite.downsample(data,ndownsample,axis=1,weight='ivar')
# Routine to encode the labels on the training sample
def encode_truth(table_train):
    hzqso = table_train['TRUE_OBJTYPE']=='HZQSO'
    bin1 = np.logical_and(table_train['Z']>=2.1,table_train['Z']<2.6)
    bin2 = np.logical_and(table_train['Z']>=2.6,table_train['Z']<3.1)
    bin3 = table_train['Z']>=3.1
    hzqso_1 = np.logical_and(hzqso,bin1)
    hzqso_2 = np.logical_and(hzqso,bin2)
    hzqso_3 = np.logical_and(hzqso,bin3)
    table_train['TRUE_OBJTYPE'][hzqso_1]='HZ_QSO_1'
    table_train['TRUE_OBJTYPE'][hzqso_2]='HZ_QSO_2'
    table_train['TRUE_OBJTYPE'][hzqso_3]='HZ_QSO_3'
    elgs = table_train['TRUE_OBJTYPE']=='ELG'
    table_train['TRUE_OBJTYPE'][elgs]='GALAXY'
    encoder = LabelEncoder()
    encoder.fit(table_train['TRUE_OBJTYPE'])
    encoded_Y = encoder.transform(table_train['TRUE_OBJTYPE'])
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y
# Routine to encode the outputs of REDMONSTER
def encode_rm(tab_rm):
    category = np.zeros(len(tab_rm),dtype=int)
    category[tab_rm['SPECTYPE']=='spEigenStar']=5
    category[tab_rm['SPECTYPE']=='ssp_em_galaxy']=0
    category[np.logical_and(tab_rm['SPECTYPE']=='QSO',tab_rm['Z']<2.1)]=4
    zbin1 = np.logical_and(tab_rm['Z']>=2.1,tab_rm['Z']<2.6)
    zbin2 = np.logical_and(tab_rm['Z']>=2.6,tab_rm['Z']<3.1)
    zbin3 = tab_rm['Z']>=3.1
    category[np.logical_and(tab_rm['SPECTYPE']=='QSO',zbin1)]=1
    category[np.logical_and(tab_rm['SPECTYPE']=='QSO',zbin2)]=2
    category[np.logical_and(tab_rm['SPECTYPE']=='QSO',zbin3)]=3
    return category
# We define the classifier architecture here
def quassify_dense(xtrain,ytrain):
    model = Sequential()
    model.add(Dense(128, input_dim=xtrain.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(24, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrain, ytrain, batch_size=len(ytrain), epochs=1000, verbose=0, validation_split=0.2)
    return model
def quassifier_run(xtrain,ytrain,xvec):
    """Run the classifier
    Args:
    -----
        xtrain : Array containing the training spectra with shape (ntrain,npoints_spectra)
        ytrain : Array containing the training classes with shape (nsamples,)
        xvec: Array containing the spectra to classify with shape (nsamples,npoints_spectra)

    Returns:
    --------
        predicted_output: Array containing the output probabilities with shape (nsamples,6)
    """
    model = quassify_dense(xtrain,ytrain)
    predicted_output = model.predict_on_batch(xvec)
    return predicted_output
