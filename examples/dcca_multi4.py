import sys, json, re, os, itertools
from os import listdir
from os.path import isfile
sys.path.insert(1,'/content/DeepMOIS-MC/')

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
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, BatchSampler


from cca import data
from cca.deepmodels import (
    DCCA,
    CCALightning,
    get_dataloaders,
    process_data,
    architectures,
    objectives,
)

def save_embedding(all_embedding_files_path, base_name, final_embd_csv_path, indices, dict_parameters):
    all_filenames = [f for f in listdir(all_embedding_files_path) if isfile(join(all_embedding_files_path, f))] 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]  
    sorted_file_list = sorted(all_filenames, key=alphanum_key)
    temp = []
    for file in sorted_file_list[-2:]:
        print(file)
        if (file.endswith(".npy")):
            file_path = join(all_embedding_files_path, file)
            X= np.load(file_path)
            print (X)
            print (len(X))
            temp.append(X)
            
    emb = np.vstack(temp)
    #!mkdir -p $final_embd_csv_path
    if not os.path.exists(final_embd_csv_path):
        os.makedirs(final_embd_csv_path)
    final_emb_array = "final_embedding_"+base_name+".npz"
    final_emb_csv = "final_embedding_"+base_name+".csv" 
    np.savez(join(final_embd_csv_path, final_emb_array), emb)
    df_embedding = pd.DataFrame(emb)
    df_embedding['shuffled_index'] = shuffled_indices
    df_embedding = df_embedding.sort_values(by=['shuffled_index'], ascending=True)
    df_embedding = df_embedding.drop(columns= 'shuffled_index').reset_index(drop=True)
    df_embedding = df_embedding.rename(columns={k:'DGCCA_'+str(k+1) for k in df_embedding.columns})
    df_embedding.to_csv(join(final_embd_csv_path, final_emb_csv), index=False)
    dict_parameters['csv_path']=join(final_embd_csv_path, final_emb_csv)
    np.save(join(final_embd_csv_path, 'parameters_details.npy'), dict_parameters)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, required=True, help='Tab-separated view data')
    parser.add_argument('--n', default=None, type=int, required=True, help='Number of views')
    parser.add_argument('--arch', default=None, help='Architecture for each view network, given as list of lists of layer widths')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to run for')
    parser.add_argument('--latDim', default=100, type=int, help='Size of the latent dimensions for each view')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--log_path', default=None, required=True, help='path to log files')
    parser.add_argument('--model_path', default=None, required=True, help='path to saved models')
    parser.add_argument('--embedPath', default=None, required=True, help='path to save all generated embedding vectors')
    parser.add_argument('--final_embed_path', default=None, required=True, help='path to save final embedding vector in a .csv file')
    parser.add_argument('--base_name', default=None, required=True, help='base name of the final embedding vector')
    parser.add_argument('--cancer_type', default=None, required=True, help='type of the cancer')
    parser.add_argument('--num_workers', default=8, type=int, required=True, help='number of process for dataloading')
    parser.add_argument('--train_batch_size', default=None, type=int, help='training batch size')
    parser.add_argument('--val_batch_size', default=None, type=int, help='validation batch size')
    args = parser.parse_args()
    dict_parameters = vars(args)

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
        feature_lists[0].append(df.iloc[i, 4].split()), feature_lists[1].append(df.iloc[i, 5].split()), feature_lists[2].append(df.iloc[i, 6].split()),feature_lists[3].append(df.iloc[i, 7].split())
    for k in range(no_of_views):
        globals()['view_%s_features' % k] = np.array([[float(i) for i in row] for row in feature_lists[k]])
    print(view_0_features.shape, view_1_features.shape, view_2_features.shape, view_3_features.shape)
    
    # Developing the architecture
    Encoders = []
    for i in range(no_of_views):
        globals()['Encoder_%s' % i]=architectures.Encoder(latent_dims=latent_dims, feature_size=globals()
                                                          ["view_%s_features" %i].shape[1], layer_sizes=architecture_list[i])

    
    
    ### Creating all combination of the views
    views = [view_0_features, view_1_features, view_2_features, view_3_features]
    view_names = ['methyl','rnaseq','mirna','imaging']
    encoders = [Encoder_0, Encoder_1, Encoder_2, Encoder_3]
    
    for L in range(2, len(views)+1):
        print(L)
        view_list = list(itertools.combinations(views, L))
        view_names_list = list(itertools.combinations(view_names, L))
        encoders_list = list(itertools.combinations(encoders, L))
        for i , j, z in zip(view_list, view_names_list, encoders_list):
            try:
                view_name = '_'.join(j)
                print (view_name)
                dataset = data.CCA_Dataset(i)
                dataset_size = len(dataset)
                
                
                ## Creating training and validation dataset
                indices = list(range(dataset_size))
                split = int(np.floor(0.9 * dataset_size))
                np.random.seed(0)
                np.random.shuffle(indices)
                train_indices, val_indices = indices[:split], indices[split:]
                shuffled_indices = train_indices + val_indices
                train_sampler = SubsetRandomSampler(train_indices)
                val_sampler = SubsetRandomSampler(val_indices)
                embed_sampler = SequentialSampler(list(range(dataset_size)))
                if (args.train_batch_size == None):
                    train_batch_size = len(dataset)
                else:
                    train_batch_size = args.train_batch_size
                if (args.val_batch_size == None):
                    val_batch_size = len(dataset)
                else:
                    val_batch_size = args.val_batch_size
                train_loader = DataLoader(dataset, sampler=train_sampler, num_workers= int((args.num_workers)), batch_size = train_batch_size)
                val_loader = DataLoader(dataset, sampler=val_sampler, num_workers= int((args.num_workers)), batch_size = val_batch_size)
                embed_loader = DataLoader(dataset, sampler=embed_sampler, num_workers= int((args.num_workers)), batch_size = dataset_size)
                #BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False)
    
    
    
                # Deep Generalized Canonical Correlation Analysis
                embed_path = os.path.join(args.embedPath , view_name)
                model_path = os.path.join(args.model_path, view_name)
                log_path = os.path.join(args.log_path, view_name)
                final_embd_csv_path = os.path.join(args.final_embed_path, view_name)
                paths = [embed_path, model_path, log_path, final_embd_csv_path]
                for path in paths:
                    if not os.path.exists(path):
                        os.makedirs(path)
                dcca = DCCA(embedding_path= embed_path, latent_dims=latent_dims, encoders=list(z), objective=objectives.GCCA, epochs= num_of_epochs)
                optimizer = optim.Adam(dcca.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
                dcca = CCALightning(dcca, optimizer=optimizer, lr_scheduler=scheduler, embedding_path = embed_path)
                trainer = pl.Trainer(default_root_dir=log_path, max_epochs=num_of_epochs, enable_checkpointing=True, 
                                     accelerator="gpu", devices=1, profiler="simple", 
                                     callbacks= [ModelCheckpoint(monitor='val loss', mode='min', dirpath=model_path, 
                                                                 filename='{epoch}-{step}-{val loss:.2f}')])
                #, accelerator="gpu", devices=4, strategy="ddp"#### gpus=2
                trainer.fit(dcca, train_loader, val_loader)
                trainer.validate(dataloaders=embed_loader, ckpt_path='best')
                embeding_csv = pd.DataFrame(np.load(os.path.join(embed_path,'embedding_view.npy')))
                embeding_csv = embeding_csv.rename(columns={k:'DGCCA_'+str(k+1) for k in embeding_csv.columns})
                embeding_csv.to_csv(os.path.join(final_embd_csv_path, "final_embedding.csv"), index=False)
                #dict_parameters['csv_path']=os.path.join(final_embd_csv_path, "final_embedding.csv")
                np.save(os.path.join(final_embd_csv_path, 'parameters_details.npy'), dict_parameters)
                #save_embedding(args.embedPath, args.base_name, args.final_embed_path, shuffled_indices, dict_parameters)
            except:
                print("-x-"*25)
                print(" ")
                print("did not converge")
                print(" ")
                print("-x-"*25)
                continue



    
    

