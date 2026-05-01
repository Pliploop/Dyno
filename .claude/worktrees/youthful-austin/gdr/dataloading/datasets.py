from torch.utils.data import Dataset
from .loading_utils import *
import torch
import os
import random
from tqdm import tqdm
import pandas as pd
from hydra.utils import instantiate

import logging

KEEP_KEYS = ['file_path', 'file_index', 'split', 'caption', 'audio', 'prompt']

class BasicProcessor:
    def __init__(self, probability = 1.0, split = None):
        self.probability = probability
        self.split = split
    
    def process(self, annot):
        """
        Override in subclasses. Applied only when probability/split gate passes.
        """
        return annot
        
    def __call__(self, annot):
        """
        Gate by probability and split; then run process().
        """
        if random.random() >= self.probability:
            return annot
        if self.split is not None and annot.get('split') not in self.split:
            return annot
        return self.process(annot)
    
class MultiCaptionProcessor(BasicProcessor):

    def __init__(self, probability=1, split=None, separator = '§§', mode_ = 'random'):
        super().__init__(probability, split)
        self.separator = separator
        self.mode_ = mode_

    def process(self, annot):
        possible_captions = annot['prompt']
        if self.separator not in possible_captions:
            return annot
        else:
            possible_captions = [cap.strip() for cap in possible_captions.split(self.separator) if cap.strip()]
            
        if self.mode_ == 'random':
            annot['prompt'] = random.choice(possible_captions)
        elif self.mode_ == 'all':
            annot['prompt'] = '. '.join(possible_captions)
        elif self.mode_ == 'first':
            annot['prompt'] = possible_captions[0]
        else:
            raise ValueError(f"Invalid mode_: {self.mode_}")
        return annot

    
class RandomNSentencesProcessor(BasicProcessor):

    def process(self, annot):
        prompt = annot['prompt']
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        n_sentences = len(sentences)
        if n_sentences == 0:
            return annot
        keep_n_sentences = random.randint(1, n_sentences)
        random_sentences = random.sample(sentences, keep_n_sentences)
        prompt = '. '.join(random_sentences)
        annot['prompt'] = prompt
        return annot
    
class ShuffleSentencesProcessor(BasicProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def process(self, annot):
        prompt = annot['prompt']
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        if not sentences:
            return annot
        random.shuffle(sentences)
        prompt = '. '.join(sentences)
        annot['prompt'] = prompt
        return annot

class ShuffleTagsProcessor(BasicProcessor):
    def __init__(self, replace_caption_p = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.replace_caption_p = replace_caption_p
    
    def process(self, annot):
        ## if tags is in the annotation, it will be a comma-separated string
        ## drop anywhere from all but 1 to 1 tag, and shuffle
        # check that tags is not nan
        if 'tags' in annot and isinstance(annot['tags'], str):
            tags = [tag.strip() for tag in annot['tags'].split(',') if tag.strip()]
        else:
            return annot
        len_tags = len(tags)
        if len_tags <= 1:
            # Can't drop any tags if there's only 1 or 0 tags
            return annot
        drop_tags = random.randint(1, len_tags-1)
        tags_to_drop = random.sample(tags, drop_tags)
        tags = [tag for tag in tags if tag not in tags_to_drop]
        random.shuffle(tags)
        annot['tags'] = '. '.join(tags)

        # if replace caption, replace else add it to the caption
        if random.random() < self.replace_caption_p:
            annot['prompt'] = annot['tags']
        else:
            annot['prompt'] = annot['prompt'] + '.' + annot['tags']
        return annot

class TextAudioDataset(Dataset):
    def __init__(self,
                 annotations = None,
                 get_annotations_function = None,
                 task_kwargs = None,
                 target_n_samples = 96000,
                 target_sr = 48000,
                 return_audio = True,
                 return_text = True,
                 concept = None,
                 return_full_audio = False,
                 preextracted_features = False,
                 truncate_preextracted = 50,
                 split = None,
                 filter_split = None,
                 root_dir = None,
                 new_dir = None,
                 limit_n = None,
                 processors = [],
                 **kwargs
                 ):
        
        
        # Get annotations either directly or from function
        if annotations is not None:
            self.annotations = annotations
        elif get_annotations_function is not None:
            # Support both string and callable
            if isinstance(get_annotations_function, str):
                task_kwargs = task_kwargs or {}
                import importlib

                # parse the fully qualified function name
                module_name, func_name = get_annotations_function.rsplit('.', 1)
                module = importlib.import_module(module_name)
                get_annotations_func = getattr(module, func_name)
                self.annotations = get_annotations_func(**task_kwargs)
            else:
                task_kwargs = task_kwargs or {}
                self.annotations = get_annotations_function(**task_kwargs)
        else:
            raise ValueError("Must provide either annotations or get_annotations_function")
        



        self.target_n_samples = target_n_samples
        self.target_sr = target_sr
        self.return_audio = return_audio
        self.return_text = return_text
        self.concept = concept
        self.return_full_audio = return_full_audio
        self.preextracted_features = preextracted_features
        self.truncate_preextracted = truncate_preextracted
        self.split = split
        self.root_dir = root_dir
        self.new_dir = new_dir
        self.limit_n = limit_n
        # Update split if needed
        if split is not None and split != 'keep':
            for annot in self.annotations:
                annot['split'] = split
        elif split == 'keep':
            # Keep original splits from annotations, or set to 'train' if not present
            for annot in self.annotations:
                if 'split' not in annot or annot['split'] not in ['train', 'val', 'test']:
                    annot['split'] = 'train'
        elif split is None and len(self.annotations) > 0 and 'split' not in self.annotations[0].keys():
            for annot in self.annotations:
                annot['split'] = 'train'

        if filter_split is not None:
            self.annotations = [annot for annot in self.annotations if annot['split'] in filter_split]
                
        annot_df = pd.DataFrame(self.annotations)
        
        try:
            annot_df['file_index'] = pd.factorize(annot_df['file_path'])[0]
        except Exception as e:
            print(e)
        
        annot_df['file_path'] = annot_df['file_path'].apply(lambda x: x.replace(root_dir, new_dir) if root_dir is not None and new_dir is not None else x)
        
        self.annotations = annot_df.to_dict('records')

        if self.limit_n is not None and self.limit_n < len(self.annotations):
            self.annotations = self.annotations[:self.limit_n]
            print(f"Limiting dataset to {self.limit_n} samples")
        else:
            print(f"Dataset has {len(self.annotations)} samples")

        
        assert return_audio or return_text, "At least one of return_audio or return_text must be True (duh)"

        self.base_processors = [
            MultiCaptionProcessor(probability=1, mode_='random', split=['train']),
            MultiCaptionProcessor(probability=1, mode_='first', split=['val', 'test']),
        ]

        self.processors = processors

    

    def purge(self):
        if self.return_audio and not self.preextracted_features:
            raise NotImplementedError("Purging your audio dataset is probably a bad idea")
        else:
            file_paths = [annot['file_path'] for annot in self.annotations]
            for file_path in file_paths:
                os.remove(file_path)
            print(f"Removed {len(file_paths)} files")
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx, return_full_audio = False, hop = None, verbose = False):
        
        
        return_full_audio = self.return_full_audio if return_full_audio is None else return_full_audio
        
        annot = self.annotations[idx]
        
        
        if self.return_audio:
            if not self.preextracted_features:
                audio = load_full_and_split(
                    annot['file_path'],
                    self.target_sr,
                    self.target_n_samples,
                    hop=hop,
                    verbose=verbose
                    ) if return_full_audio else load_audio_chunk(
                    annot['file_path'],
                    target_sr=self.target_sr,
                    target_n_samples=self.target_n_samples,
                    verbose=verbose
                    )
                audio = audio.mean(0,keepdim=True) if not return_full_audio else audio.mean(1,keepdim=True)
            else:
                file_path = annot['file_path'].replace('.mp3','.npy').replace('.wav','.npy')
                try:
                    
                    audio = np.load(file_path,mmap_mode='r')
                    if audio.shape[0] > self.truncate_preextracted:
                        rand_start = random.randint(0,audio.shape[0]-self.truncate_preextracted)
                        audio = audio[rand_start:rand_start+self.truncate_preextracted]
                        audio = torch.tensor(audio)
                    else:
                        #repeat the audio to match the target_n_samples
                        n_repeat = self.truncate_preextracted // audio.shape[0] +1
                        audio = np.repeat(audio, n_repeat, axis=0)
                        rand_start = random.randint(0,audio.shape[0]-self.truncate_preextracted)
                        audio = audio[rand_start:rand_start+self.truncate_preextracted, :]
                        audio = torch.tensor(audio)
                        
                except Exception as e:
                    return self.__getitem__(idx+1)
        
        if self.return_text:
            possible_captions = annot['caption']
            # ramdomly choose a caption hash
            random_hash = random.choice(list(possible_captions.keys()))
            caption = possible_captions[random_hash]
                    
        return_dict = {}
        
        if self.return_audio:
            return_dict['audio'] = audio
            return_dict['file_path'] = annot['file_path']
            
        if self.return_text:
            return_dict['prompt'] = caption
                
        return_dict['file_idx'] = annot['file_index']
        return_dict['split'] = annot.get('split', 'train')
        
        # Copy tags to return_dict if they exist in annotation
        if 'tags' in annot:
            return_dict['tags'] = annot['tags']

        if '§§' in return_dict.get('prompt', '') and idx%1000 == 0:
            # print the split and the prompt
            print(f"Split: {return_dict['split']}, Prompt: {return_dict['prompt']}...") if verbose else None

            

        for processor in self.base_processors:
            return_dict = processor(return_dict)
        for processor in self.processors:
            return_dict = processor(return_dict)


        return_dict = {k: v for k, v in return_dict.items() if k in KEEP_KEYS}

        return return_dict
    
    
    @staticmethod
    def _extract_output_tensor(output, out_key):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            if out_key in output:
                return output[out_key]
            if 'embedding_proj' in output:
                return output['embedding_proj']
            if 'last_hidden_state' in output:
                return output['last_hidden_state']
        raise KeyError(f"Could not find output key '{out_key}' in model output of type {type(output).__name__}")

    def extract_features(self, model, extract_method = 'extract_features', extract_kwargs = {}, out_key = 'embedding',hop = None, return_full_audio = True, verbose = False, batch_size = 1, num_workers = 0, max_batch_chunks = 200):
        from torch.utils.data import DataLoader, Dataset

        device = next(model.parameters()).device
        print(f"Extracting features with {extract_method} method on {device} device") if verbose else None

        for param in model.parameters():
            param.requires_grad = False
        try:
            model.eval()
        except:
            pass

        parent = self

        class _ExtractionDataset(Dataset):
            def __len__(self):
                return len(parent)

            def __getitem__(self, idx):
                item = parent.__getitem__(idx, return_full_audio=return_full_audio, hop=hop, verbose=verbose)
                file_path = parent.annotations[idx]['file_path'].replace('.mp3', '.npy').replace('.wav', '.npy')
                return {
                    'audio': item['audio'],
                    'file_path': file_path,
                }

        def _collate_fn(batch):
            audios = []
            file_paths = []
            chunk_counts = []

            for item in batch:
                audio = item['audio']
                if audio.ndim == 3:
                    audio = audio.squeeze(1)
                elif audio.ndim == 1:
                    audio = audio.unsqueeze(0)

                audios.append(audio)
                file_paths.append(item['file_path'])
                chunk_counts.append(audio.shape[0])

            return {
                'audio': torch.cat(audios, dim=0),
                'file_paths': file_paths,
                'chunk_counts': chunk_counts,
            }

        loader = DataLoader(
            _ExtractionDataset(),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
            collate_fn=_collate_fn,
        )

        for batch in loader:
            try:
                audio = batch['audio'].to(device, non_blocking=device.type == 'cuda')
                file_paths = batch['file_paths']
                chunk_counts = batch['chunk_counts']

                if audio.shape[0] > max_batch_chunks:
                    feature_parts = []
                    for chunk in torch.split(audio, max_batch_chunks, dim=0):
                        output = getattr(model, extract_method)(chunk, **extract_kwargs)
                        feature_parts.append(self._extract_output_tensor(output, out_key))
                    flat_features = torch.cat(feature_parts, dim=0)
                else:
                    output = getattr(model, extract_method)(audio, **extract_kwargs)
                    flat_features = self._extract_output_tensor(output, out_key)

                start = 0
                for file_path, n_chunks in zip(file_paths, chunk_counts):
                    audio_features = flat_features[start:start + n_chunks]
                    start += n_chunks
                    print(f"Extracted features for {file_path}, shape: {audio_features.shape}") if verbose else None
                    yield audio_features, file_path
            except Exception as e:
                print(f"Error extracting batched features: {e}") if verbose else None
                continue
            
    def extract_and_save_features(self, model, save_dir = None, extract_method = 'extract_features', extract_kwargs = {}, out_key = 'embedding', hop = None, return_full_audio = True, limit_n = None, save = False, verbose = True, root_path = None, done_ids = None, batch_size = 1, num_workers = 0, max_batch_chunks = 200):
        
        
        print(self.__len__())
        
        audio_features_all = []
        counter = 0
        skipped_count = 0
        
        save_dir = '' if save_dir is None else save_dir
        done_ids = done_ids or set()
        
        if 's3://' in save_dir:
            import boto3
            import io
            client = boto3.client('s3')
        else:
            client = None
            import io

        # filter self.annotations to only include files that are not in done_ids
        new_annotations = []
        for annot in self.annotations:
            fp = annot['file_path']
            fp = fp.replace(root_path+'/','')
            # remove extension
            fp = fp.replace('.mp3','').replace('.wav','').replace('.npy','')

            if fp not in done_ids:
                new_annotations.append(annot)
        
        self.annotations = new_annotations
        
        for audio_features, file_path in (pbar:= tqdm(self.extract_features(model, extract_method = extract_method, extract_kwargs = extract_kwargs, out_key = out_key, hop = hop, return_full_audio = return_full_audio, verbose = verbose, batch_size = batch_size, num_workers = num_workers, max_batch_chunks = max_batch_chunks))):
            
            # print(file_path, root_path, save_dir)
            
            if root_path is not None:
                file_path = file_path.replace(root_path+'/','')

            save_path = os.path.join(save_dir, file_path)
            
            if save and audio_features is not None:
                

                #remove the root path from the file path
                
                
                if 's3://' in save_dir:
                    bucket, key = save_dir.replace("s3://", "").split("/", 1)
                    key = f"{key}/{file_path}"
                    
                    # local_path = os.path.join(local_temp_dir, file_path)
                    # os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    # np.save(local_path, audio_features.detach().cpu().numpy())
                    
                    pbar.set_description(f"Uploading features to s3://{bucket}/{key}") if verbose else None
                    try:
                        # client.upload_file(save_path, bucket, key)
                        
                        buffer = io.BytesIO()
                        np.save(buffer, audio_features.detach().cpu().numpy())
                        buffer.seek(0)
                        client.put_object(Bucket=bucket, Key=key, Body=buffer)
                    except Exception as e:
                        print(f"Error uploading to s3: {e}") if verbose else None
                        
                    # os.remove(local_path)
                else:
                    pbar.set_description(f"Saving features in {save_path}, shape: {audio_features.shape}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    np.save(save_path, audio_features.detach().cpu().numpy())
            
            if not save and audio_features is not None:
                pbar.set_description(f"{file_path}, shape: {audio_features.shape}")
                pass
                
            audio_features_all.append(audio_features.detach().cpu()) if audio_features is not None else None
            
            counter += 1
            if limit_n and counter >= limit_n:
                break
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} already processed items") if verbose else None
        
        try:   
            print(f"Returning {len(audio_features_all)} features") if verbose else None
            all_= torch.stack(audio_features_all)
            print(f"Stacked features, shape: {all_.shape}") if verbose else None
            return all_
        
        
        except:
            return None
