from data import *
from model import *
from utils import *


import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras import  optimizers
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


max_elms = 3  #4
min_elms = 3
max_sites = 20  #40

dataframe = pd.read_csv('YOUR_DATA.csv')

RICR_representation, Nsites, rotation_matrices = RICR_represent(dataframe, max_elms, max_sites, return_Nsites=True)

RICR_representation = pad(RICR_representation, 2)


X, scaler_X = minmax(RICR_representation)

prop = ['formation_energy_per_atom', 'band_gap',]
Y = dataframe[prop].values
scaler_y = MinMaxScaler()
Y = scaler_y.fit_transform(Y) 


ind_train, ind_test = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=21)
X_train, X_test = X[ind_train], X[ind_test]
y_train, y_test = Y[ind_train], Y[ind_test]

# Get model
VAE, encoder, decoder, regression, vae_loss = RICR(X_train, y_train, coeffs=(2, 10,))
# Train model
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=4, min_lr=1e-6)

def scheduler(epoch, lr):
    if epoch == 50:
        lr = 4e-4
    elif epoch == 100:
        lr = 3e-4
    elif epoch == 150:
        lr = 2e-4
    elif epoch == 250:
        lr = 1e-4
    elif epoch == 350:
        lr = 5e-5
    return lr
schedule_lr = LearningRateScheduler(scheduler)

VAE.compile(optimizer=optimizers.RMSprop(learning_rate=5e-4), loss=vae_loss)
VAE.fit([X_train, y_train], 
        X_train,
        shuffle=True, 
        batch_size=256,
        epochs=10,   #350
        callbacks=[reduce_lr, schedule_lr],
        )

train_latent = encoder.predict(X_train, verbose=1)
y_train_, y_test_ = scaler_y.inverse_transform(y_train), scaler_y.inverse_transform(y_test)

font_size = 26
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size-2
plt.rcParams['ytick.labelsize'] = font_size-2

fig, ax = plt.subplots(1, 2, figsize=(18, 7.3))
s0 = ax[0].scatter(train_latent[:,0], train_latent[:,1], s=7, c=np.squeeze(y_train_[:,0]))
cbar = plt.colorbar(s0, ax=ax[0], ticks=list(range(-1, -8, -2)))
s1 = ax[1].scatter(train_latent[:,0], train_latent[:,1], s=7, c=np.squeeze(y_train_[:,1]))
plt.colorbar(s1, ax=ax[1], ticks=list(range(0, 10, 2)))
fig.text(0.016, 0.92, '(A) $E_\mathrm{f}$', fontsize=font_size)
fig.text(0.533, 0.92, '(B) $E_\mathrm{g}$', fontsize=font_size)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, top=0.85)
plt.show()


X_test_recon = VAE.predict([X_test, y_test], verbose=1)
X_test_recon_ = inv_minmax(X_test_recon, scaler_X)
X_test_recon_[X_test_recon_ < 0.1] = 0
X_test_ = inv_minmax(X_test, scaler_X)


def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true+1e-12), np.array(y_pred+1e-12)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=0)

def MAE_site_coor(SITE_COOR, SITE_COOR_recon, Nsites):
    site = []
    site_recon = []
    for i in range(len(SITE_COOR)):
        site.append(SITE_COOR[i, :Nsites[i], :])
        site_recon.append(SITE_COOR_recon[i, :Nsites[i], :])
    site = np.vstack(site)
    site_recon = np.vstack(site_recon)
    return np.mean(np.ravel(np.abs(site - site_recon)))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


elm_str = joblib.load('data/element.pkl')
abc = X_test_[:, len(elm_str), :3]
abc_recon = X_test_recon_[:, len(elm_str), :3]
print('abc (MAPE): ', MAPE(abc,abc_recon))
print('abc (r2_score): ', r2_score(abc,abc_recon))


ang = X_test_[:, len(elm_str)+1, :3]
ang_recon = X_test_recon_[:, len(elm_str)+1, :3]
print('angles (MAPE): ', MAPE(ang, ang_recon))
print('angles (r2_score): ', r2_score(ang, ang_recon))

coor = X_test_[:, len(elm_str)+2:len(elm_str)+2+max_sites, :3]
coor_recon = X_test_recon_[:, len(elm_str)+2:len(elm_str)+2+max_sites, :3]
print('coordinates (MAE): ', MAE_site_coor(coor, coor_recon, Nsites[ind_test]))

elm_accu = []
for i in range(max_elms):
    elm = np.argmax(X_test_[:, :len(elm_str), i], axis=1)
    elm_recon = np.argmax(X_test_recon_[:, :len(elm_str), i], axis=1)
    elm_accu.append(metrics.accuracy_score(elm, elm_recon))
print(f'Accuracy for {len(elm_str)} elements are respectively: {elm_accu}')

y_test_hat = regression.predict(X_test, verbose=1)
y_test_hat_ = scaler_y.inverse_transform(y_test_hat)
print(f'The regression MAE for {prop} are respectively', MAE(y_test_, y_test_hat_))



target_Ef, target_Eg = -1.5, 1.5
Nsamples = 10
ind_constraint = np.squeeze(np.argwhere(y_train_[:, 0] < target_Ef))
ind_temp = np.argsort(np.abs(y_train_[ind_constraint, 1] - target_Eg))
ind_sample = ind_constraint[ind_temp][:Nsamples]
Nperturb = 3
Lp_scale = 0.6

samples = train_latent[ind_sample, :]
samples = np.tile(samples, (Nperturb, 1))
gaussian_noise = np.random.normal(0, 1, samples.shape)
samples = samples + gaussian_noise * Lp_scale
RICR_designs = decoder.predict(samples, verbose=1)
RICR_designs = inv_minmax(RICR_designs, scaler_X)

np.save('RICR_designs.npy', RICR_designs)

