from torch.utils.data import Dataset
from .loading_utils import *
import torch
import os
import random
from tqdm import tqdm
import pandas as pd
from hydra.utils import instantiate

import logging

KEEP_KEYS = ['file_path', 'file_index', 'split', 'audio', 'attention_mask']


class AudioDataset(Dataset):
    def __init__(self,
                 annotations=None,
                 get_annotations_function=None,
                 task_kwargs=None,
                 target_n_samples=96000,
                 target_sr=48000,
                 return_audio=True,
                 return_full_audio=False,
                 preextracted_features=False,
                 n_frames=50,
                 random_crop=True,
                 truncate_preextracted=None,
                 split=None,
                 filter_split=None,
                 root_dir=None,
                 new_dir=None,
                 limit_n=None,
                 processors=[],
                 **kwargs
                 ):

        if annotations is not None:
            self.annotations = annotations
        elif get_annotations_function is not None:
            if isinstance(get_annotations_function, str):
                task_kwargs = task_kwargs or {}
                import importlib
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
        self.return_full_audio = return_full_audio
        self.preextracted_features = preextracted_features
        # truncate_preextracted kept as alias; n_frames is the canonical name
        self.n_frames = truncate_preextracted if truncate_preextracted is not None else n_frames
        self.random_crop = random_crop
        self.split = split
        self.root_dir = root_dir
        self.new_dir = new_dir
        self.limit_n = limit_n

        if split is not None and split != 'keep':
            for annot in self.annotations:
                annot['split'] = split
        elif split == 'keep':
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

    def __getitem__(self, idx, return_full_audio=False, hop=None, verbose=False):
        return_full_audio = self.return_full_audio if return_full_audio is None else return_full_audio

        annot = self.annotations[idx]

        if self.return_audio:
            if not self.preextracted_features:
                try:
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
                    audio = audio.mean(0, keepdim=True) if not return_full_audio else audio.mean(1, keepdim=True)
                except Exception as e:
                    logging.warning(f"Skipping corrupted file {annot['file_path']}: {e}")
                    return self.__getitem__(idx + 1, return_full_audio=return_full_audio, hop=hop, verbose=verbose)
            else:
                file_path = annot['file_path'].replace('.mp3', '.npy').replace('.wav', '.npy')
                try:
                    raw = np.load(file_path, mmap_mode='r')
                    T_actual = raw.shape[0]
                    if T_actual >= self.n_frames:
                        start = random.randint(0, T_actual - self.n_frames) if self.random_crop else 0
                        audio = torch.tensor(np.array(raw[start:start + self.n_frames]), dtype=torch.float32)
                        attention_mask = torch.ones(self.n_frames, dtype=torch.bool)
                    else:
                        audio = torch.zeros(self.n_frames, raw.shape[-1], dtype=torch.float32)
                        audio[:T_actual] = torch.tensor(np.array(raw), dtype=torch.float32)
                        attention_mask = torch.zeros(self.n_frames, dtype=torch.bool)
                        attention_mask[:T_actual] = True
                except Exception:
                    return self.__getitem__(idx + 1)

        return_dict = {}

        if self.return_audio:
            return_dict['audio'] = audio
            return_dict['file_path'] = annot['file_path']
            if self.preextracted_features:
                return_dict['attention_mask'] = attention_mask

        return_dict['file_idx'] = annot['file_index']
        return_dict['split'] = annot.get('split', 'train')

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

    def extract_features(self, model, extract_method='extract_features', extract_kwargs={}, out_key='embedding', hop=None, return_full_audio=True, verbose=False, batch_size=1, num_workers=0, max_batch_chunks=200):
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
                return {'audio': item['audio'], 'file_path': file_path}

        def _collate_fn(batch):
            audios, file_paths, chunk_counts = [], [], []
            for item in batch:
                audio = item['audio']
                if audio.ndim == 3:
                    audio = audio.squeeze(1)
                elif audio.ndim == 1:
                    audio = audio.unsqueeze(0)
                audios.append(audio)
                file_paths.append(item['file_path'])
                chunk_counts.append(audio.shape[0])
            return {'audio': torch.cat(audios, dim=0), 'file_paths': file_paths, 'chunk_counts': chunk_counts}

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
                logging.exception(
                    "Feature extraction failed for a batch with %s files and %s chunks",
                    len(batch.get('file_paths', [])),
                    batch.get('audio').shape[0] if batch.get('audio') is not None else "unknown",
                )
                raise

    def extract_and_save_features(self, model, save_dir=None, extract_method='extract_features', extract_kwargs={}, out_key='embedding', hop=None, return_full_audio=True, limit_n=None, save=False, verbose=True, root_path=None, done_ids=None, batch_size=1, num_workers=0, max_batch_chunks=200):
        print(self.__len__())

        audio_features_all = [] if not save else None
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

        new_annotations = []
        for annot in self.annotations:
            fp = annot['file_path']
            fp = fp.replace(root_path + '/', '')
            fp = fp.replace('.mp3', '').replace('.wav', '').replace('.npy', '')
            if fp not in done_ids:
                new_annotations.append(annot)

        self.annotations = new_annotations
        remaining_count = len(self.annotations)
        skipped_count = len(done_ids)

        def _shape_str(features):
            return "x".join(str(dim) for dim in features.shape)

        pbar = tqdm(
            self.extract_features(
                model,
                extract_method=extract_method,
                extract_kwargs=extract_kwargs,
                out_key=out_key,
                hop=hop,
                return_full_audio=return_full_audio,
                verbose=verbose,
                batch_size=batch_size,
                num_workers=num_workers,
                max_batch_chunks=max_batch_chunks,
            ),
            desc="Extracting features",
            unit="file",
            total=remaining_count,
            dynamic_ncols=True,
            leave=True,
        )

        for audio_features, file_path in pbar:
            if root_path is not None:
                file_path = file_path.replace(root_path + '/', '')

            save_path = os.path.join(save_dir, file_path)
            display_path = file_path.replace(os.sep, "/")

            if save and audio_features is not None:
                # Sequence encoders return (N_chunks, T, D); flatten to (N_chunks*T, D) for storage
                feat_to_save = audio_features.flatten(0, 1) if audio_features.ndim == 3 else audio_features
                shape = _shape_str(feat_to_save)
                if 's3://' in save_dir:
                    bucket, key = save_dir.replace("s3://", "").split("/", 1)
                    key = f"{key}/{file_path}"
                    try:
                        buffer = io.BytesIO()
                        np.save(buffer, feat_to_save.detach().cpu().numpy())
                        buffer.seek(0)
                        client.put_object(Bucket=bucket, Key=key, Body=buffer)
                        pbar.set_postfix_str(f"{display_path} shape={shape}", refresh=True)
                        if verbose:
                            tqdm.write(f"Saved s3://{bucket}/{key} shape={shape}")
                    except Exception as e:
                        print(f"Error uploading to s3: {e}") if verbose else None
                else:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    np.save(save_path, feat_to_save.detach().cpu().numpy())
                    pbar.set_postfix_str(f"{display_path} shape={shape}", refresh=True)
                    if verbose:
                        tqdm.write(f"Saved {save_path} shape={shape}")

            if not save and audio_features is not None:
                flat = audio_features.flatten(0, 1) if audio_features.ndim == 3 else audio_features
                pbar.set_postfix_str(f"{display_path} shape={_shape_str(flat)}", refresh=True)

            if not save and audio_features is not None:
                audio_features_all.append(audio_features.detach().cpu())

            counter += 1
            if limit_n and counter >= limit_n:
                break

        if skipped_count > 0:
            print(f"Skipped {skipped_count} already processed items") if verbose else None

        try:
            if save:
                return None
            print(f"Returning {len(audio_features_all)} features") if verbose else None
            all_ = torch.stack(audio_features_all)
            print(f"Stacked features, shape: {all_.shape}") if verbose else None
            return all_
        except:
            return None
