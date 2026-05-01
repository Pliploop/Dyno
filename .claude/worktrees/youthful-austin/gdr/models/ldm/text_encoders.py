from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel
from torch import nn
import torch
import os
import logging

from dyno.models.utils.base import BaseModule
from dyno.models.clap.src.laion_clap.hook import CLAP_Module

class T5TextEncoder(BaseModule):
    
    def __init__(self, model_name = 'google-t5/t5-base', ckpt_path = None, freeze = False, max_length = 77):
        super().__init__(ckpt_path = ckpt_path, freeze = freeze)
        # Use HF_TOKEN from environment if available
        hf_token = os.getenv('HF_TOKEN', None)
        tokenizer_kwargs = {'token': hf_token} if hf_token else {}
        model_kwargs = {'token': hf_token} if hf_token else {}
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.encoder = T5EncoderModel.from_pretrained(model_name, **model_kwargs)
        self.max_length = max_length
    
    def get_text_embedding(self, text, return_dict = True, return_tokenizer_only = False, **kwargs):
        text_input = self.tokenizer(text, return_tensors = 'pt', padding = True, truncation = True, max_length = self.max_length)
            
        if return_tokenizer_only:
            return text_input
        
        device = next(self.encoder.parameters()).device
        
        if self.freeze:
            with torch.no_grad():
                text_embed = self.encoder(input_ids = text_input['input_ids'].to(device), attention_mask = text_input['attention_mask'].to(device))
        else:
            text_embed = self.encoder(input_ids = text_input['input_ids'].to(device), attention_mask = text_input['attention_mask'].to(device))
            
            
        if return_dict:
            text_input.update(text_embed)
        else:
            text_input = text_embed
        
        return text_input

class HuggingFaceTextEncoder(BaseModule):
    """
    General HuggingFace text encoder that supports any AutoModel from transformers.
    Always outputs last_hidden_state without pooling.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', ckpt_path: str = None, freeze: bool = False, max_length: int = 512, trust_remote_code: bool = True):
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        self.model_name = model_name
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        
        # Use HF_TOKEN from environment if available
        hf_token = os.getenv('HF_TOKEN', None)
        tokenizer_kwargs = {'trust_remote_code': trust_remote_code}
        model_kwargs = {'trust_remote_code': trust_remote_code}
        if hf_token:
            tokenizer_kwargs['token'] = hf_token
            model_kwargs['token'] = hf_token
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model_name = model_name
        
        # For encoder-decoder models, use only the encoder part
        # Check if model has an encoder attribute (encoder-decoder models)
        
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def get_text_embedding(self, text, return_dict=True, return_tokenizer_only=False, use_tensor=True, **kwargs):
        """
        Get text embeddings from HuggingFace model.
        
        Args:
            text: Text input (str or list of str)
            return_dict: If True, return dict with last_hidden_state and attention_mask
            return_tokenizer_only: If True, return only tokenizer output
            use_tensor: For compatibility (ignored, always returns tensors)
            **kwargs: Additional arguments passed to tokenizer
            
        Returns:
            Dict with 'last_hidden_state' and 'attention_mask' if return_dict=True,
            or just tokenizer output if return_tokenizer_only=True
        """
        # Tokenize input
        tokenizer_kwargs = {
            'return_tensors': 'pt',
            'padding': 'max_length',
            'truncation': True,
            'max_length': self.max_length,
            **kwargs
        }
        text_input = self.tokenizer(text, **tokenizer_kwargs)
        
        if return_tokenizer_only:
            return text_input
        
        # Move to device and encode
        device = next(self.model.parameters()).device
        text_input = text_input.to(device)
        
        # Forward through model
        if self.freeze:
            with torch.no_grad():
                if 't5gemma' in self.model_name.lower(): #hack for seq2seq models
                    outputs = self.model.encoder(**text_input)
                else:
                    outputs = self.model(**text_input)
        else:
            if 't5gemma' in self.model_name.lower(): #hack for seq2seq models
                outputs = self.model.encoder(**text_input)
            else:
                outputs = self.model(**text_input)
        
        # Always use last_hidden_state (no pooling)
        last_hidden_state = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs['last_hidden_state']
        if return_dict:
            # Return dict with last_hidden_state and attention_mask
            return {
                'last_hidden_state': last_hidden_state.float(),
                'attention_mask': text_input.get('attention_mask', None)
            }
        else:
            return last_hidden_state.float()

class CLAPTextEncoder(BaseModule):
    def __init__(self, clap_kws ={}, clap_ckpt = None, ckpt_path = None, freeze = False):
        super(CLAPTextEncoder, self).__init__(ckpt_path = ckpt_path, freeze = freeze)
        self.encoder_pair = CLAP_Module(**clap_kws)
        self.encoder_pair.load_ckpt(clap_ckpt, verbose=False)
    
    def get_text_embedding(self, text, return_dict = True, return_tokenizer_only = False, **kwargs):
        if self.freeze:
            with torch.no_grad():
                return self.encoder_pair.get_text_embedding(text, return_dict = return_dict, return_tokenizer_only = return_tokenizer_only, **kwargs)
        else:
            return self.encoder_pair.get_text_embedding(text, return_dict = return_dict, return_tokenizer_only = return_tokenizer_only, **kwargs)

class CLAPAudioEncoder(BaseModule):
    def __init__(self, clap_kws ={}, clap_ckpt = None, ckpt_path = None, freeze = False):
        super(CLAPAudioEncoder, self).__init__(ckpt_path = ckpt_path, freeze = freeze)
        self.encoder_pair = CLAP_Module(**clap_kws)
        self.encoder_pair.load_ckpt(clap_ckpt, verbose=False)
    
    @torch.no_grad()
    def extract_features(self, audio, return_dict = True, return_tokenizer_only = False, **kwargs):
        return self.encoder_pair.get_audio_embedding_from_data(audio, return_dict = return_dict, return_tokenizer_only = return_tokenizer_only, **kwargs)

    def get_audio_embedding_from_data(self, audio, **kwargs):
        return self.extract_features(audio, **kwargs)


class MuQAudioEncoder(BaseModule):
    """
    Audio encoder wrapper for OpenMuQ (MuQ/MuQMuLan) models.
    
    This encoder wraps MuQ models from the 'muq' package and provides
    a consistent interface compatible with the GDR codebase.
    
    The model is called directly with audio of shape (B, N_samples) and returns
    embeddings of shape (B, embedding_dim).
    
    Args:
        model_name: HuggingFace model identifier (e.g., "OpenMuQ/MuQ-MuLan-large" or "OpenMuQ/MuQ-large-msd-iter")
        ckpt_path: Optional path to a checkpoint file to load (not used for MuQ.from_pretrained)
        freeze: If True, freeze model parameters and use torch.no_grad() during inference
        model_type: Type of model to load - "MuQMuLan" or "MuQ". If None, auto-detects from model_name
    """
    
    def __init__(
        self, 
        model_name: str = "OpenMuQ/MuQ-MuLan-large",
        ckpt_path: str = None,
        freeze: bool = False,
        model_type: str = None,
        device: str = None,
        sampling_rate: int = 24000,
        **kwargs
    ):
        super(MuQAudioEncoder, self).__init__(ckpt_path=ckpt_path, freeze=freeze)
        
        try:
            from muq import MuQ, MuQMuLan
        except ImportError:
            raise ImportError(
                "The 'muq' package is required for MuQAudioEncoder. "
                "Please install it with: pip install muq"
            )
        
        resolved_device = device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = resolved_device
        self.sampling_rate = sampling_rate
    
        
        # Load the appropriate model from HuggingFace
        if "MuLan" in model_name:
            self.muq = MuQMuLan.from_pretrained(model_name)
        else:
            self.muq = MuQ.from_pretrained(model_name)
        
        # Set model to eval mode
        self.muq = self.muq.to(self.device).eval()
        
        # Freeze parameters if requested
        if self.freeze:
            for param in self.muq.parameters():
                param.requires_grad = False
    
    def extract_features(self, audio, return_dict=True, return_tokenizer_only=False, **kwargs):
        """
        Extract audio features using MuQ model.
        
        Args:
            audio: Audio tensor of shape (B, N_samples) where B is batch size and N_samples is audio length.
                  The dataloading already handles the correct shape, so minimal preprocessing is needed.
            return_dict: If True, return a dictionary with 'last_hidden_state' and 'embedding_proj'.
                        If False, return just the tensor.
            return_tokenizer_only: Not used for audio encoders, kept for interface compatibility.
            **kwargs: Additional arguments (ignored for MuQ models)
            
        Returns:
            If return_dict=True: Dictionary with 'last_hidden_state' and 'embedding_proj' (both same tensor)
            If return_dict=False: Tensor of shape (B, embedding_dim)
        """
        if return_tokenizer_only:
            # Not applicable for audio encoders, but kept for interface compatibility
            return audio
        
        # Ensure audio is a tensor and on the correct device
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        
        device = next(self.muq.parameters()).device
        audio = audio.to(device)
        
        # Ensure audio is 2D: (batch_size, audio_length)
        # The dataloading already provides this shape, but handle edge cases
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim > 2:
            # If audio is (batch, channels, length), take mean across channels or first channel
            if audio.shape[1] > 1:
                audio = audio.mean(dim=1)
            else:
                audio = audio.squeeze(1)
        
        # Align with the other repo's MuQ calling convention while keeping
        # compatibility with installs that only support positional inputs.
        with torch.set_grad_enabled(not self.freeze):
            try:
                out = self.muq(wavs=audio)
            except TypeError:
                out = self.muq(audio)

        if isinstance(out, torch.Tensor):
            embeddings = out
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            embeddings = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            hidden_states = out.last_hidden_state
            embeddings = hidden_states[:, 0] if hidden_states.ndim == 3 else hidden_states
        else:
            raise TypeError(
                f"MuQ model returned {type(out).__name__}; expected tensor or "
                "pooler_output/last_hidden_state."
            )
        
        if return_dict:
            # Return dict for compatibility with code that expects 'embedding_proj' or 'last_hidden_state'
            return {
                'last_hidden_state': embeddings,
                'embedding_proj': embeddings,
            }
        else:
            return embeddings
    
    def get_audio_embedding_from_data(self, audio, **kwargs):
        """
        Get audio embeddings from audio data.
        
        This method is the main interface used by the GDR codebase.
        It calls extract_features with appropriate defaults.
        
        Args:
            audio: Audio tensor of shape (B, N_samples)
            **kwargs: Additional arguments passed to extract_features
            
        Returns:
            Audio embeddings dictionary with 'last_hidden_state' and 'embedding_proj'
        """
        return self.extract_features(audio, return_dict=True, **kwargs)


# Alias for backward compatibility
OpenMuQAudioEncoder = MuQAudioEncoder
