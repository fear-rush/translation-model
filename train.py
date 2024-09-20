from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
  sos_idx = tokenizer_tgt.token_to_id('[SOS]')
  eos_idx = tokenizer_tgt.token_to_id('[EOS]')

  # Precompute the encoder output and reuse it for every step
  encoder_output = model.encode(source, source_mask)
  # Initialize the decoder input with the sos token
  decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
  while True:
    if decoder_input.size(1) == max_len:
        break

    # build mask for target
    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

    # calculate output
    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

    # get next token
    prob = model.project(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    decoder_input = torch.cat(
        [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
    )

    if next_word == eos_idx:
        break

  return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
  model.eval()
  count = 0

  source_texts = []
  expected = []
  predicted = []
  
  console_width = 80

  with torch.no_grad():
    for batch in validation_ds:
      count += 1
      encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
      encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

      # check that the batch size is 1
      assert encoder_input.size(
          0) == 1, "Batch size must be 1 for validation"

      model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

      source_text = batch["src_text"][0]
      target_text = batch["tgt_text"][0]
      model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

      source_texts.append(source_text)
      expected.append(target_text)
      predicted.append(model_out_text)
      
      # Print the source, target and model output
      print_msg('-'*console_width)
      print_msg(f"{f'SOURCE: ':>12}{source_text}")
      print_msg(f"{f'TARGET: ':>12}{target_text}")
      print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

      if count == num_examples:
        print_msg('-'*console_width)
        break
  
  if writer:
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    writer.add_scalar('validation cer', cer, global_step)
    writer.flush()

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    writer.add_scalar('validation wer', wer, global_step)
    writer.flush()

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    writer.add_scalar('validation BLEU', bleu, global_step)
    writer.flush()

def get_all_sentences(ds, lang):
  for item in ds:
      yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
  tokenizer_path = Path(config['tokenizer_file'].format(lang))
  if not Path.exists(tokenizer_path):
      # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        # Unknown token digunakan jika tidak ada kata yang relevan di dictionary dan direplace oleh [UNK]
  # WordLevel model merupakan salah satu cara untuk mengubah kalimat jadi token yang dipisah per kata
      tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
      tokenizer.pre_tokenizer = Whitespace()
      # min_frequency digunakan untuk memasukkan token ke dictionary jika muncul minimal 2x dalam wordlist yang ditrain
      trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
      tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
      tokenizer.save(str(tokenizer_path))
  else:
      tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer

def get_ds(config):
  # It only has the train split, so we divide it overselves
  # Load dataset dari opus_books. dan karena orpus books hanya memiliki dataset untuk split train, maka spesifikasi data yang diambil 
  # adalah hanya train saja dan harus menulis kode sendiri untuk split train-validation data
  ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

  # Build tokenizers
  tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

  # Keep 90% for training, 10% for validation
  train_ds_size = int(0.9 * len(ds_raw))
  val_ds_size = len(ds_raw) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

  train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
  val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

  # Find the maximum length of each sentence in the source and target sentence
  # inisiasi variabel untuk maksimal seq_len pada src dan tgt
  max_len_src = 0
  max_len_tgt = 0
  
  # Mencari longest sequence length pada src dan tgt yang sudah ditokenisasi
  for item in ds_raw:
      src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
      tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
      max_len_src = max(max_len_src, len(src_ids))
      max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f'Max length of source sentence: {max_len_src}')
  print(f'Max length of target sentence: {max_len_tgt}')
  
  # Pada val_dataloader tidak menggunakan batch, karena pada contoh ini, pada validation data, sequence akan diproses satu persatu (tidak menggunakan batch/parallel processing)
  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
  # Define the device
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
  print("Using device:", device)
  if (device == 'cuda'):
      print(f"Device name: {torch.cuda.get_device_name(device.index)}")
      print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
  elif (device == 'mps'):
      print(f"Device name: <mps>")
  else:
      print("NOTE: If you have a GPU, consider using it for training.")
      print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
      print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
  device = torch.device(device)

  # Make sure the weights folder exists
  # Membuat folder di path ['model_folder]. parents=True artinya jika ada parent directory yang missing, maka akan dibuat juga parent directorynya
# exist_ok=True artinya jika folder sudah ada maka ignore Error. jika exist_ok=False maka akan error FileExistError
  Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
  # Tensorboard
  writer = SummaryWriter(config['experiment_name'])

  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

  # If the user specified a model to preload before training, load it
  # Membuat function jika terjadi error di tengah train. maka akan melanjutkan sesuai state terakhir
  # Default: None
  initial_epoch = 0
  global_step = 0
  preload = config['preload']
  model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
  if model_filename:
    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
  else:
      print('No model to preload, starting from scratch')
      
  # menghitung loss function dengan entropy loss. ignore index berarti menghilangkan [PAD] dalam perhitungan entropy loss
  # label smoothing digunakan agar model less overfit dan less confident dengan mendistribusikan probabilitas tertinggi ke kandidat lainnya
  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

  # Train the model. Karena get_model mereturn class yang menginherit nn.Module, maka mendapatkan function .train()
  for epoch in range(initial_epoch, config['num_epochs']):
    torch.cuda.empty_cache()
    # tells your model that you are training the model. This helps inform layers such as Dropout and BatchNorm, which are designed to behave differently during training and evaluation. 
    # For instance, in training mode, BatchNorm updates a moving average on each new batch; whereas, for evaluation mode, these updates are frozen.
    # Intinya, membuat flag self.training menjadi True, agar parameter dapat 'belajar'
    # kebalikannya adalah model.eval() yang membuat parameter self.train menjadi False
    model.train()
    # train_dataloader berasal dari class DataLoader dan DataLoader meretrun iterable object
    batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
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
      
      encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
      decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
      encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
      decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

      # Run the tensors through the encoder, decoder and the projection layer
      
      # Run the tensors through the transformers
      # Pada encoder_output memanggil function forward() pada class Encoder (karena class encoder inherit class nn.Module)
      # Pada decoder_output memanggil function forward() pada class Decoder (karena class encoder inherit class nn.Module)
      # Pada proj_output memanggil function forward() pada class ProjectionLayer (karena class encoder inherit class nn.Module)
      encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
      proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

      # Compare the output with the label
      # Menghitung nilai asli dari label
      label = batch['label'].to(device) # (B, seq_len)

      # Compute the loss using a simple cross entropy
      # Menghitung loss dengan membandingkan nilai asli dari label dengan projection output
      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

      # Log the loss
      writer.add_scalar('train loss', loss.item(), global_step)
      writer.flush()

      # Backpropagate the loss
      loss.backward()

      # Update the weights
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)

      # Global step biasanya digunakan untuk tensorboard
      global_step += 1

    # Run validation at the end of every epoch
    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

    # Save the model at the end of every epoch
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    # Best practice-nya adalah tetap menyimpan state dari optimizer. Pada contoh ini menggunakan Adam optimizer
    # Dikarenakan jika state dari optimizer tidak disimpan, maka perhitungan juga akan dimulai dari awal lagi
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)


if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
