import torch
import torch.nn as nn

from dataset import BillingualDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from model import build_transformer
from config import get_config, get_weights_file_path

import warnings


def get_all_sentences(ds, lang):
  for item in ds:
    yield item['translation'][lang]
    
def get_or_build_tokenizer(config, ds, lang):
  tokenizer_path = Path(config['tokenizer_file'].format(lang))
  if not Path.exists(tokenizer_path):
    # Unknown token digunakan jika tidak ada kata yang relevan di dictionary dan direplace oleh [UNK]
    # WordLevel model merupakan salah satu cara untuk mengubah kalimat jadi token yang dipisah per kata
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    # min_frequency digunakan untuk memasukkan token ke dictionary jika muncul minimal 2x dalam wordlist yang ditrain
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
  return tokenizer

def get_ds(config):
  # Load dataset dari opus_books. dan karena orpus books hanya memiliki dataset untuk split train, maka spesifikasi data yang diambil 
  # adalah hanya train saja dan harus menulis kode sendiri untuk split train-validation data
  ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
  
  # Build tokenizers
  tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
  
  # Keep 90% for training and 10% for validation
  train_ds_size = int(0.9 * len(ds_raw))
  val_ds_size = len(ds_raw) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) 
  
  train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
  val_ds = BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
  
  # inisiasi variabel untuk maksimal seq_len pada src dan tgt
  max_len_src = 0
  max_len_tgt = 0
  
  # Mencari longest sequence length pada src dan tgt yang sudah ditokenisasi
  for item in ds_raw:
    src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
    tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
  print(f'Max length of source sentence: {max_len_src}')
  print(f'Max length of target sentence: {max_len_tgt}')
  
  # Pada val_dataloader tidak menggunakan batch, karena pada contoh ini, pada validation data, sequence akan diproses satu persatu (tidak menggunakan batch/parallel processing)
  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
  
  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
  model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
  return model

def train_model(config):
  # Define the computation device
  
  # CUDA
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # MPS
  device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
  
  print(f'Using device {device}')
  
  # Membuat folder di path ['model_folder]. parents=True artinya jika ada parent directory yang missing, maka akan dibuat juga parent directorynya
  # exist_ok=True artinya jika folder sudah ada maka ignore Error. jika exist_ok=False maka akan error FileExistError
  Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
  
  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
  
  # Tensorboard
  writer = SummaryWriter(config['experiment_name'])
  
  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
  
  # Membuat function jika terjadi error di tengah train. maka akan melanjutkan sesuai state terakhir
  initial_epoch = 0
  global_step = 0
  if config['preload']:
    model_filename = get_weights_file_path(config, config['preload'])
    print(f"Preloading model {model_filename}")
    state = torch.load(model_filename)
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
    
  
  # menghitung loss function dengan entropy loss. ignore index berarti menghilangkan [PAD] dalam perhitungan entropy loss
  # label smoothing digunakan agar model less overfit dan less confident dengan mendistribusikan probabilitas tertinggi ke kandidat lainnya
  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
  
  # Train the model. Karena get_model mereturn class yang menginherit nn.Module, maka mendapatkan function .train()
  for epoch in range(initial_epoch, config['num_epochs']):
    # tells your model that you are training the model. This helps inform layers such as Dropout and BatchNorm, which are designed to behave differently during training and evaluation. 
    # For instance, in training mode, BatchNorm updates a moving average on each new batch; whereas, for evaluation mode, these updates are frozen.
    # Intinya, membuat flag self.training menjadi True, agar parameter dapat 'belajar'
    # kebalikannya adalah model.eval() yang membuat parameter self.train menjadi False
    model.train()
    
    # train_dataloader berasal dari class DataLoader dan DataLoader meretrun iterable object
    batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
    for batch in batch_iterator:
      
      # DataLoader memanggil DataClass yang dipassing pada DataLoader, pada contoh ini adalah memanggil class BillingualDataset
      # Memanggil __getitem__ pada BillingualDataset dengan index yang dipassing secara otomatis oleh DataLoader
      
      # Pada encoder mask menghasilkan matriks dengan dimensi (batch, 1, 1, seq_len) karena keperluan untuk menyamakan dengan higher dimension pada attention matriks
      # misalnya attention matriks memiliki dimensi (batch, num_heads, seq_len_query, seq_len_key). Jadi, fungsi dari matrix broadcasting adalah untuk mengubah smaller tensor
      # menjadi higher dimension agar match pada large tensor. 
      
      # The second 1 allows broadcasting across all attention heads. This is because the attention scores are computed for multiple heads simultaneously, and you want to apply the same mask to each head.
      # The third 1 allows broadcasting across the query length (which may differ during multi-head attention). Since the mask operates on the key sequence (which determines what is attended to), you want to apply the same mask to every query position in the sequence.

      # Intinya, pada encoder mask pada nilai '1' ketiga berfungsi untuk matching pada cross-head attention di decoder. karena bisa saja sequence length-nya berbeda
      
      # Untuk decoder mask, kurang lebih sama dengan encoder mask, bedanya adalah tidak ada nilai '1' ketiga di decoder mask, melainkan diganti dengan seq_len
      # karena pada decoder mask digunakan untuk self-attention, khususnya pada causal-mask
      
      # Encoder Self-Attention: Query and key lengths are the same because they both come from the same source sentence.
      # Decoder Self-Attention: Query and key lengths are also the same because both come from the previously generated tokens.
      # Cross-Attention (in the Decoder): Query and key lengths differ because queries come from the decoder’s current sequence, while keys come from the encoder’s output (which represents the source sequence). 
      # This is where the input (source) and output (target) sentences can have different lengths.

      encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
      decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
      encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
      decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)
      
      # Run the tensors through the transformers
      # Pada encoder_output memanggil function forward() pada class Encoder (karena class encoder inherit class nn.Module)
      # Pada decoder_output memanggil function forward() pada class Decoder (karena class encoder inherit class nn.Module)
      # Pada proj_output memanggil function forward() pada class ProjectionLayer (karena class encoder inherit class nn.Module)
      encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
      proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)
      
      # Menghitung nilai asli dari label
      label = batch['label'].to(device) # (batch, seq_len)
      
      # Menghitung loss dengan membandingkan nilai asli dari label dengan projection output
      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
      
      # Log the loss to tensorboard
      writer.add_scalar('train loss', loss.item(), global_step)
      writer.flush()
      
      # Backpropagate the loss
      loss.backward()
      
      # Update the weights
      optimizer.step()
      optimizer.zero_grad()
      
      # Global step biasanya digunakan untuk tensorboard
      global_step += 1
      
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    # Best practice-nya adalah tetap menyimpan state dari optimizer. Pada contoh ini menggunakan Adam optimizer
    # Dikarenakan jika state dari optimizer tidak disimpan, maka perhitungan juga akan dimulai dari awal lagi
    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'global_step': global_step
    })
    
  

if __name__ == '__main__':
  # warnings.filterwarnings('ignore')
  config = get_config()
  train_model(config)