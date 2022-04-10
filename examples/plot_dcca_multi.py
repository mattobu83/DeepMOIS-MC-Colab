import sys
sys.path.insert(1,'/home/pdutta/DGCCA/cca_zoo')

"""
Deep CCA for more than 2 views
=================================

This example demonstrates how to easily train Deep CCA models and variants
"""

import numpy as np
import pandas as pd
import cca_zoo.packages.pytorch_lightning.pytorch_lightning as pl
from torch.utils.data import Subset
from torch import optim

# %%
from cca_zoo import data
from cca_zoo.deepmodels import (
    DCCA,
    CCALightning,
    get_dataloaders,
    process_data,
    architectures,
    objectives,
    DTCCA,
)



df = pd.read_csv("/home/pdutta/DGCCA/data/TCGA_BRCA/methyl_rnaseq_mirna_minmax.tsv", sep = "\t", header =None)

print (df.shape)
print (df.iloc[1,0:10])





X_feature_list =  []
Y_feature_list =  []
Z_feature_list =  []

for i in range(0,1233):
    #print (len(df.iloc[i, 4]), len(df.iloc[i, 5]), len(df.iloc[i, 6]))
    #print (len(df.iloc[i, 4].split()), len(df.iloc[i, 5].split()), len(df.iloc[i, 6].split()))
    X_feature_list.append(df.iloc[i, 4].split()), Y_feature_list.append(df.iloc[i, 5].split()), Z_feature_list.append(df.iloc[i, 6].split())


X_features= np.array([[float(i) for i in row] for row in X_feature_list])
Y_features= np.array([[float(i) for i in row] for row in Y_feature_list])
Z_features= np.array([[float(i) for i in row] for row in Z_feature_list])
print(X_features.shape, Y_features.shape, Z_features.shape)
print (Z_features)
print (X_features.shape[1])
input()

dataset = data.CCA_Dataset([X_features, Y_features, Z_features])
train_dataset, val_dataset = process_data(dataset, val_split=0.2)
loader = get_dataloaders(dataset)
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)
# for item in next(iter(train_loader)):
#     print ("#",len(item))
#     print ( item[0].shape)
#     input()
#     print (item)
#     input()



#print ("$$$",len(train_loader))
# for item in next(iter(train_loader)):
#     print ("#",item,len(item))
#     print (item[0], item[0].shape)

# The number of latent dimensions across models
latent_dims = min(X_features.shape[1], Y_features.shape[1], Z_features.shape[1])
print("latent dims", latent_dims)
input()


# number of epochs for deep models
epochs = 100
no_of_views  = 3






encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=X_features.shape[1], layer_sizes=[500, 100])
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=Y_features.shape[1], layer_sizes=[500, 100])
encoder_3 = architectures.Encoder(latent_dims=latent_dims, feature_size=Z_features.shape[1], layer_sizes=[500, 100])


# Deep GCCA
dcca = DCCA(
    latent_dims=latent_dims, encoders=[encoder_1, encoder_2, encoder_3], objective=objectives.GCCA
)
print ("Return DCCA")


optimizer = optim.Adam(dcca.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
dcca = CCALightning(dcca, optimizer=optimizer, lr_scheduler=scheduler)

print ("CCALightning dcca", dcca)
        

trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
print ("Check")
trainer.fit(dcca, train_loader, val_loader)
print ("ded")


 