import torch
import torch.nn as nn
from torch.utils.data import Dataset


# BillingualDataset di sini inherit dari class Dataset yang nantinya berhubungan dengan pyTorch DataLoader
# __len__ digunakan untuk return seberapa banyak dataset yang dipassing (yang nantinya digunakan untuk iterator di DataLoader)
# __getitem__ digunakan untuk mereturn value apa saja yang akan direturn dari Dataset (yang nantinya digunakan untuk mendapatkan batch pada setiap index di iterator)
class BillingualDataset(Dataset):
  # BillingualDataset karena ini adalah translation model yang menerima input dan menghasilkan output dari 2 bahasa yang berbeda
  def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
    super().__init__()
    
    self.ds = ds
    self.tokenizer_src = tokenizer_src
    self.tokenizer_tgt = tokenizer_tgt
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    self.seq_len = seq_len
    
    # token_to_id digunakan untuk mencari id dari [SOS]
    # lalu diconvert ke torch.tensor dengan dtype int64 karena biasanya dictionary ukurannya sangat besar
    self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
    self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
    
  def __len__(self):
    return len(self.ds)
  
  def causal_mask(self, size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
  
  def __getitem__(self, index):
    # original pair from huggingface opus_books dataset
    src_target_pair = self.ds[index]
    
    # src_text = source language text
    # tgt_text = target translation language text
    src_text = src_target_pair['translation'][self.src_lang]
    tgt_text = src_target_pair['translation'][self.tgt_lang] 
    
    # Mengambil input tokens dari sequence encoder dan decoder, lalu direturn dalam bentuk array of ids dari tiap id token yang ada di vocabulary
    enc_input_tokens = self.tokenizer_src.encode(src_text).ids
    dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
    
    # Dikarenakan setiap sequence tidak pasti sama panjanganya, maka diberikan [PAD] token untuk menyamaratakan setiap sequence
    # Pada encoder ditambah dua special token yaitu [SOS] dan [EOS]
    # Sedangkan pada decoder hanya membutuhkan satu special token yaitu [SOS]
    # self.seq_len di sini adalah didapatkan dari config yang memperkirakan maksimal kalimat paling panjang yang ada di dataset
    # default 350 karena di dataset opus_books untuk en-it, kalimat paling panjang adalah 350 kata
    enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
    dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
    
    if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
      raise ValueError('Sentence is too long')
    
    # Will be sent to the input of encoder
    # Add [SOS] and [EOS] to encoder
    encoder_input = torch.cat([
      self.sos_token,
      torch.tensor(enc_input_tokens, dtype=torch.int64),
      self.eos_token,
      torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
    ])
    
    # Will be sent to the input of decoder
    # Add [SOS] to decoder
    decoder_input = torch.cat([
      self.sos_token,
      torch.tensor(dec_input_tokens, dtype=torch.int64),
      torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
    ])
    
    # Expect of the output of the decoder (label/target)
    # Maka dari itu, decoder tidak memiliki [EOS] special token. karena nantinya akan diexpect/ditambah [EOS] token dari label ini
    # Label digunakan untuk teacher forcing agar decoder bisa menebak next token dari label (expected output)
    label = torch.cat([
      torch.tensor(dec_input_tokens, dtype=torch.int64),
      self.eos_token,
      torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
    ])
    
    # For debugging purposes
    assert encoder_input.size(0) == self.seq_len
    assert decoder_input.size(0) == self.seq_len 
    assert label.size(0) == self.seq_len
    
    # Encoder mask digunakan untuk masking padding token seperti [PAD] dan menghasilkan output matriks 3D dengan susunan (1, 1, Seq_Len) atau (batch_dimenson, attention_heads, seq_len)
    # Decoder selain digunakan untuk masking [PAD] token juga digunakan untuk causal-masking agar token hanya bisa melihat token sebelumnya dengan
    # membuat upper triangle matriks
    # Output dari function causal_mask (misal size=3)
    # tensor([[[ True, False, False],
    #         [ True,  True, False],
    #         [ True,  True,  True]]]) 
    # Lalu, susunan Decoder adalah (1, Seq_Len, Seq_Len) atau (batch_dimension, seq_len, seq_len)

    return {
      "encoder_input": encoder_input, # (Seq_Len)
      "decoder_input": decoder_input, # (Seq_Len)
      "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_Len)
      "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & self.causal_mask(decoder_input.size(0)), #(1, Seq_Len, Seq_Len)
      "label": label, # (Seq_Len)
      "src_text": src_text,
      "tgt_text": tgt_text
    }
    
    
    