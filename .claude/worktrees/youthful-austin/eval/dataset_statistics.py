from torch.utils.data import Dataset
import s3fs
import numpy as np
from tqdm import tqdm
import os
from torch.nn import functional as F
from gdr.evaluation.fidelity_diversity.features import get_fixed_length_motion_features, get_variable_length_motion_features
import torch
import boto3
from rich.pretty import pprint


   
    

class S3StreamingDataset(Dataset):
    
    def __init__(self, s3_uri_or_path):
        self.s3_uri = s3_uri_or_path
        
        print(f'Processing dataset from {s3_uri_or_path}')

        if s3_uri_or_path.startswith('s3://'):
            self.s3 = True
            s3 = boto3.resource('s3')
            bucket = self.s3_uri.split('/')[2]
            self.bucket_name = bucket
            bucket = s3.Bucket(bucket)
            prefix = '/'.join(self.s3_uri.split('/')[3:])
            
            
            objects = bucket.objects.filter(Prefix=prefix)
            
            self.files = [obj.key for obj in objects if obj.key.endswith('.npy')]
            
            self.fs = s3fs.S3FileSystem()
        else:
            self.s3 = False
            #crawl the directory
            
            print(f'Crawling the directory {os.listdir(s3_uri_or_path)}')
            
            self.files = []
            for root, dirs, files in os.walk(s3_uri_or_path):
                self.files.extend([os.path.join(root, file) for file in files if file.endswith('.npy')])
            self.fs = None
        
        self.stats = None
        
    def __len__(self):
        return len(self.files)
    
    def get_single_file(self, idx):
        if self.s3:
            file = self.files[idx]
            with self.fs.open(f's3://{self.bucket_name}/{file}', 'rb') as f:
                data = np.load(f)
            return data, file
        else:
            file = self.files[idx]
            data = np.load(file)
            return data, file
        
    
    def __getitem__(self, idx):
        return self.get_single_file(idx)
    
    
    def get_stats(self,n=-1, fixed_length=64):
        
        
        if self.stats is not None and self.stats['n'] == n and self.stats['computed_from'] == self.s3_uri:
            return self.stats
        
        
        self.sequence_stats = OnlineStats('sequence')
        
        self.get_fixed_length_motion_features_stats = OnlineStats('get_fixed_length_motion_features')
        
        self.get_variable_length_motion_features_stats = OnlineStats('get_variable_length_motion_features')
        
        
        pprint('-------- Computing Sequence Stats --------')
        
        self.incremental_mean(self.sequence_stats, lambda x: x.mean(0), n)
        self.incremental_covariance_matrix(self.sequence_stats, lambda x: x.mean(0), n)
        
        pprint('-------- Computing fixed length Stats --------')
        
        self.incremental_mean(self.get_fixed_length_motion_features_stats, get_fixed_length_motion_features, n, fixed_length=fixed_length)
        self.incremental_covariance_matrix(self.get_fixed_length_motion_features_stats, get_fixed_length_motion_features, n, fixed_length=fixed_length)
        
        pprint('-------- Computing variable length Stats --------')
        
        self.incremental_mean(self.get_variable_length_motion_features_stats, get_variable_length_motion_features, n, fixed_length=fixed_length)
        self.incremental_covariance_matrix(self.get_variable_length_motion_features_stats, get_variable_length_motion_features, n, fixed_length=fixed_length)
        
    
        
        sequence_stats = self.sequence_stats.dump()
        sequence_stats.update({'computed_from': self.s3_uri})
        get_fixed_length_motion_features_stats = self.get_fixed_length_motion_features_stats.dump()
        get_fixed_length_motion_features_stats.update({'computed_from': self.s3_uri})
        get_variable_length_motion_features_stats = self.get_variable_length_motion_features_stats.dump()
        get_variable_length_motion_features_stats.update({'computed_from': self.s3_uri})
        
        self.stats = {
            'sequence_stats.pkl': sequence_stats,
            'fixed_length_motion_features_stats.pkl': get_fixed_length_motion_features_stats,
            'variable_length_motion_features_stats.pkl': get_variable_length_motion_features_stats
        }
        
        print('Computed stats')
        print(self.stats)
        
        
        return self.stats
    
    def save_stats(self, path):
        import pickle as pkl
        if self.stats is None: self.get_stats()
        
        for k,v in self.stats.items():
            
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, k), 'wb') as f:
                pkl.dump(v, f)
                
            print(f'Saved stats to {path}')
            
            
    def sequence(self, data, **kwargs):
        return data.mean(0)
    
    def incremental_mean(self, onlinestats, preprocessing_step, n=-1, **kwargs):
        #compute the mean of the dataset incrementally

        n = len(self) if n < 0 or n > len(self) else n
        
        prog_bar = tqdm(range(n))
        
        #loop over the progress bar
        for i in prog_bar:
            data,file = self.get_single_file(i)
            
            
            # if mean is None:
            #     mean = preprocessing_step(data, **kwargs)
            # else:
            #     mean = mean + preprocessing_step(data, **kwargs)
                
            # it += 1
            
            onlinestats.update(data, preprocessing_step, kwargs)
            
            prog_bar.set_description(f'Processed {i+1} files, {file}')
            
        
            
        print(f'final mean shape: {onlinestats.mean.shape}')
    
    def incremental_covariance_matrix(self, online_stats, preprocessing_step,  n=-1, **kwargs):
        # if mean is None:
        #     mean = self.incremental_mean(n)
        
        
        if online_stats.mean is None:
            self.incremental_mean(online_stats, preprocessing_step, n, **kwargs)
            
        # mean = online_stats.get_mean()
        

        total_files = len(self)
        n = total_files if n < 0 or n > total_files else n

        prog_bar = tqdm(range(n))

        for i in prog_bar:
            data, file = self.get_single_file(i)
            data = np.asarray(data, dtype=np.float32) if isinstance(data, np.ndarray) else data.float()
            

            # deviation = preprocessing_step(data, **kwargs) - mean
            # sample_cov = np.outer(deviation.T, deviation)

            # if cov is None:
            #     cov = sample_cov
            # else:
            #     cov = cov + sample_cov
                
            # it += 1
            
            online_stats.update_covar_matrix(data, preprocessing_step, kwargs)
            
            
            prog_bar.set_description(f'Processed {i+1} files, {file}')
            
        
        print(f'Final covariance matrix shape: {online_stats.covar_matrix.shape}')
        
    
            

import argparse
def process_dataset(s3_uri, n=-1, output='statistics', save = False, fixed_length=64):
    dataset = S3StreamingDataset(s3_uri)
    stats = dataset.get_stats(n, fixed_length= fixed_length)
    dataset.save_stats(output) if save else None
    return stats


class OnlineStats:
    def __init__(self, name):
        self.name = name
        self.sum = None
        self.unscaled_covar = None
        self.count = 0
    
    
    @property
    def mean(self):
        return self.sum / self.count if self.sum is not None else None
    
    @property
    def covar_matrix(self):
        return self.unscaled_covar / (self.count - 1) if self.unscaled_covar is not None else None
    
    def reset(self):
        self.sum = None
        self.unscaled_covar = None
        self.count = 0
        
    def update(self, data, preprocess_fn=None, preprocess_kwargs={}):
        
        data = preprocess_fn(data, **preprocess_kwargs) if preprocess_fn is not None else data
        
        self.count += 1
        self.sum = data if self.sum is None else self.sum + data

        
    def update_covar_matrix(self, data, preprocess_fn=None, preprocess_kwargs={}):
        
        assert self.mean is not None, 'Mean should be computed first'
        
        if self.mean is None:
            self.update(data, preprocess_fn, preprocess_kwargs)
            
        data = preprocess_fn(data, **preprocess_kwargs) if preprocess_fn is not None else data
        
        deviation = data - self.mean
        deviation = deviation.numpy() if isinstance(deviation, torch.Tensor) else deviation
        
        sample_cov = np.outer(deviation.T, deviation)
        
        if self.covar_matrix is None:
            self.unscaled_covar = sample_cov
        else:
            self.unscaled_covar = self.unscaled_covar + sample_cov
            
    
    
    def dump(self):
        return {
            'name': self.name,
            'mean': self.mean,
            'covar_matrix': self.covar_matrix,
            'count': self.count
        }
        
    def save_to_pkl(self, path):
        import pickle as pkl
        with open(path, 'wb') as f:
            pkl.dump(self.dump(), f)
        print(f'Saved {self.name} to {path}')
        
    def load_from_pkl(self, path):
        import pickle as pkl
        
        if 's3://' in path:
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(path, 'rb') as f:
                data = pkl.load(f)
        else:
            with open(path, 'rb') as f:
                data = pkl.load(f)
            
        self.name = data['name']
        self.count = data['count']
        self.sum = data['mean'] * self.count
        self.unscaled_covar = data['covar_matrix'] * (self.count - 1)
        
        print(f'Loaded {self.name} from {path}')
        print({k:v.shape for k,v in self.dump().items() if isinstance(v, np.ndarray)})
        
        
    def load(self, name, mean, covar_matrix, count):
        self.name = name
        self.count = count
        self.sum = self.mean * self.count
        self.unscaled_covar = self.covar_matrix * (self.count - 1)
        
        
    def fit(self, iterable, preprocess_fn=None, preprocess_kwargs={}, verbose=False):
        
        if verbose:
            from tqdm.rich import tqdm
            iterable = tqdm(iterable)
            iterable2 = tqdm(iterable)
        else:
            iterable2 = iterable
        
        for i in iterable:
            self.update(i, preprocess_fn, preprocess_kwargs)
            
        for i in iterable2:
            self.update_covar_matrix(i, preprocess_fn, preprocess_kwargs)
        
        
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset from S3 URI.')
    parser.add_argument('--path', type=str, help='The S3 URI or path of the dataset')
    parser.add_argument('--output', type=str, help='The output path for the statistics', default='statistics')
    parser.add_argument('--n', type=int, help='The number of samples to process', default=-1)
    parser.add_argument('--save', action='store_true', help='Save the statistics to the output path')
    parser.add_argument('--fixed_length', type=int, help='The fixed length of the motion features', default=64)
    
    args = parser.parse_args()
    
    output = args.output
    
    stats = process_dataset(args.path, args.n, output, args.save, args.fixed_length)
    
    print({k:v.shape for k,v in stats.items() if isinstance(v, np.ndarray)})