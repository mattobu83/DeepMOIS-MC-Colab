import sys
sys.path.insert(1,'/home/pdutta/Github/Multiview_clustering_DGCCCA')

"""
Deep CCA for more than 2 views
=================================

This example demonstrates how to easily train Deep CCA models and variants
"""
import argparse, json
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch import optim

from cca import data
from cca.deepmodels import (
    DCCA,
    CCALightning,
    get_dataloaders,
    process_data,
    architectures,
    objectives,
    DTCCA,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, required=True, help='Tab-separated view data')
    parser.add_argument('--n', default=None, type=int, required=True, help='Number of views')
    parser.add_argument('--arch', default=None, help='Architecture for each view network, given as list of lists of layer widths')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to run for')
    parser.add_argument('--latDim', default=100, type=int, help='Size of the latent dimensions for each view')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--log_path', default=None, required=True, help='path to log files')
    parser.add_argument('--embedPath', default=None, required=True, help='path to save embedding vector')
    parser.add_argument('--num_workers', default=8, type=int, required=True, help='number of process for dataloading')
    parser.add_argument('--train_batch_size', default=16, type=int, required=True, help='training batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, required=True, help='validation batch size')
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep = "\t", header =None)
    no_of_views = args.n
    architecture_list= json.loads(args.arch)
    num_of_epochs= args.epochs
    latent_dims= args.latDim
    lr= args.lr
    

    ## Preprocess the dataframe into list of lists 
    feature_lists = []
    for i in range(no_of_views):
        feature_lists.append([])
    for i in range(0,df.shape[0]):
        #print (len(df.iloc[i, 4]), len(df.iloc[i, 5]), len(df.iloc[i, 6]))
        #print (len(df.iloc[i, 4].split()), len(df.iloc[i, 5].split()), len(df.iloc[i, 6].split()))
        feature_lists[0].append(df.iloc[i, 4].split()), feature_lists[1].append(df.iloc[i, 5].split()), feature_lists[2].append(df.iloc[i, 6].split())
    for k in range(no_of_views):
        globals()['view_%s_features' % k] = np.array([[float(i) for i in row] for row in feature_lists[k]])
    print(view_0_features.shape, view_1_features.shape, view_2_features.shape)

    ## Creating training dataset
    dataset = data.CCA_Dataset([view_0_features, view_1_features, view_2_features])
    train_dataset, val_dataset = process_data(dataset, val_split=0.2)
    loader = get_dataloaders(dataset)
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, num_workers= (args.num_workers-10))


    # Developing the architecture
    for i in range(no_of_views):
        globals()['Encoder_%s' % i]=architectures.Encoder(latent_dims=latent_dims, feature_size=globals()["view_%s_features" %i].shape[1], layer_sizes=architecture_list[i])



    # Deep Generalized Canonical Correlation Analysis
    dcca = DCCA(
        latent_dims=latent_dims, encoders=[Encoder_0, Encoder_1, Encoder_2], objective=objectives.GCCA
    )
    optimizer = optim.Adam(dcca.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
    dcca = CCALightning(dcca, optimizer=optimizer, lr_scheduler=scheduler)
    trainer = pl.Trainer(default_root_dir=args.log_path, max_epochs=num_of_epochs, enable_checkpointing=False, accelerator="gpu", devices=4, strategy="ddp")
    #, accelerator="gpu", devices=4, strategy="ddp"#### gpus=2
    trainer.fit(dcca, train_loader, val_loader)



