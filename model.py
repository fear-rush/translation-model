import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
  
  def __init__(self, d_model: int, vocab_size: int):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model)
    
  def forward(self, x):
    # Rumus dari paper menyatakan bahwa setiap input embedding dikali dengan sqrt(d_model)
    return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
  
  # d_model = size of embedding (e.g. 512)
  # seq_len = size of positional encoding because on every sequence length will be converted to n vectors based on d_model. eg: 512 x 24 (total sequence length)
  # dropout = make model not overfit
  def __init__(self, d_model: int, seq_len: int, dropout: float):
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout) 
    
    # Create a matrix of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)
    # Create a vector of shape(seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float()) * (-math.log(100000.0) / d_model)
    # Apply the sin to even positions, and cos to odd positions
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Membuat seq_len menjadi batch processing. karena nanti pas diproses tidak satu kalimat utuh, namun di-batch
    pe = pe.unsqueeze(0) # Jadi, matriksnya menjadi (1, seq_len, d_model)
    
    # Menyimpan hasil perkalian batch matriks seperti BatchNorm menjadi persistent state
    # Perlu disimpan karena BatchNorm merupakan bukan parameter dari model, namun penting untuk proses state selanjutnya
    # Buffers in PyTorch are typically tensors that are part of a model but are not updated during training (i.e., they are not trainable parameters).
    # By registering a tensor as a buffer using register_buffer(), PyTorch ensures that the tensor is:
    # Saved along with the model during model.save().
    # Moved to the appropriate device (e.g., GPU) when you call model.to(device).
    # Available in the model during inference and forward passes.
    # kenapa tidak diset di constructor 'self.pe' saja? karena jika diset di constructor, maka parameter tersebut akan dilearn (berubah tiap training)
    # maka dari itu, jika ada parameter yang tidak ditrain, lebih baik diregister di buffer saja
    self.register_buffer('pe', pe)
    
  def forward(self, x):
    # di sini bisa memanggil self.pe karena sebelumnya 'pe' sudah diassign oleh register buffer
    # di sini ada function requires_grad_ karena pada Positional Embedding, nilai dari hasil kalkulasi matriks
    # tidak dilearn lagi, dan akan digunakan berulang-kali
    # self.pe is accessible because register_buffer() stores the tensor in the model.
    # self.pe is not updated during training because it's not a parameter, it's a buffer (fixed).
    # Buffers like self.pe are used for values that should be part of the model but should not be trained.
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    return self.dropout(x) 
  
class LayerNormalization(nn.Module):
  
  # eps = epsilon. digunakan untuk number stability (agar pembagi tidak terlalu kecil) dan menghindari division by 0
  # digunakan untuk menormalisasi layer agar hasilnya seragam dan tidak membebani komputasi, serta pada layer ini
  # menggunakan alpha (weight) dan beta (bias) agar hasil perhitungan bisa fit sesuai data (self-learning)
  def __init__(self, eps: float = 10**-6):
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
    self.bias = nn.Parameter(torch.zeros(1)) # Added
    
  def forward(self, x):
    mean = x.mean(dim = -1, keepdim = True)
    std = x.std(dim = -1, keepdim = True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias
  
class FeedForwardBlock(nn.Module):
  
  def __init__(self, d_model: int, d_ff: int, dropout: float):
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1 -> from d_model to d_ff
    self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2 -> from d_ff back to d_model
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    # x input -> (Batch, Seq_Len, d_model)
    # step 1 -> convert from x input to d_ff -> (Batch, Seq_Len, d_ff)
    # step 2 -> convert d_ff back to d_model (Batch, Seq_Len, d_model)
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
  
class MultiHeadAttentionBlock(nn.Module):
  
  # d_model = total embedding. ex: 512
  # h = number of sequence division. actually not sequence bot per column on each token
  # d_k = kind of batch attention processing (see the transformers slide)
  def __init__(self, d_model: int, h: int, dropout: float):
    super().__init__()
    self.d_model = d_model
    self.h = h
    # d_model must be divisible by h
    assert d_model % h == 0, "d_model is not divisible by h"
    
    self.d_k = d_model // h
    self.w_q = nn.Linear(d_model, d_model) # Wq
    self.w_k = nn.Linear(d_model, d_model) # Wk
    self.w_v = nn.Linear(d_model, d_model) # Wv
    
    self.w_o = nn.Linear(d_model, d_model) # Wo
    self.dropout = nn.Dropout(dropout)
    
  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k = query.shape[-1]
    
    # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
      attention_scores.masked_fill_(mask == 0, -1e9)
    attention_scores = attention_scores.softmax(dim = -1) # (Batch, Seq_Len, Seq_Len)
    if dropout is not None:
      attention_scores = dropout(attention_scores)
      
    # attention_scores @ value for computation, attention_scores for visualization  
    return (attention_scores @ value), attention_scores
  
  def forward(self, q, k, v, mask):
    query = self.w_q(q) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
    key = self.w_k(k) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
    value = self.w_v(v) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
     
    # (Batch, Seql_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k,).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
    
    x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
    
    # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
    
    # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
    return self.w_o(x)
  
# ResidualConnection layer merupakan layer yang menerima 'sisa' dari layer sebelumnya
# atau layer yang menerima value 'skip' yang tidak diteruskan di Multi-Head Attention atau Feedforward layer
class ResidualConnection(nn.Module):
  
  def __init__(self, dropout: float):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()
    
  # sublayer merupakan nilai dari layer sebelumnya
  # contohnya adalah dari layer Multi-Head Attention atau dari layer Feedforward
  # untuk rumus sublayer(self.norm(x)), pada paper Attention is All You Need, implementasinya self.norm(sublayer(x))
  # Digunakan rumus ini, karena banyak paper lain menggunakan rumus sublayer(self.norm(x))
  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))
  
  
class EncoderBlock(nn.Module):
  
  def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
  def forward(self, x, src_mask):
    # Dua residual connections layer
    # 1. dari multihead connection yang diteruskan ke Add & Norm layer dan nilai x ori yang di-skip
    # 2. dari feedforward layer yang diteruskan ke Add & Norm layer dan nilai x ori yang di-skip
    # pada Encoder, self_attention_block memiliki 3 nilai x yang sama untuk query, key, dan value
    # src_mask merupakan source mask yang digunakan untuk 'menutupi' padding yang akan 'ditebak' pada proses training
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
    x = self.residual_connections[1](x, self.feed_forward_block)
    
    return x
  
class Encoder(nn.Module):
  
  def __init__(self, layers: nn.ModuleList):
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization()
    
  # Pada awalnya, x akan diinisiasi oleh input tensor x, lalu akan diproses sejumlah N layers Encoder yang akan menghasilkan x juga
  # Lalu, x hasil encoder tersebut akan di-passing lagi ke layer ke N encoder lainnya
  # Selain x yang diinisiasi, mask juga diinisiasi untuk mencegah model melakukan learning (weight) pada 
  # keyword yang tidak relevan. Biasanya padding seperti <PAD>
  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
      
    return self.norm(x)
  
class DecoderBlock(nn.Module):
  
  def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
  # pada residual connection pertama, nilai x yang didapat dari Masked-Multi Head attention berasal dari query, key, value yang sama dari positional encoding dan nilai x dari 'skip' value
  # pada residual connection kedua, nilai x yang didapat berasal dari query dan key dari encoder, dan value berasal dari masked-multi head attention dan nilai x dari 'skip' value
  # pada residual connection ketiga, nilai x yang didapat dari feed forward block dan dan nilai x dari 'skip' value
  def forward(self, x, encoder_output, src_mask, tgt_mask):
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
    x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
    x = self.residual_connections[2](x, self.feed_forward_block)
    
    return x
  
class Decoder(nn.Module):
  
  def __init__(self, layers: nn.ModuleList):
    super().__init__()
    self.layers = layers 
    self.norm = LayerNormalization()
    
  # src_mask berasal dari encoder mask yang digunakan untuk 'meniadakan' nilai pada kata (token) yang tidak relevan seperti <PAD>
  # tgt_mask berasal dari inisiasi di Decoder yang sama gunanya seperti src_mask, namun bedanya hanya ada tambahan agar pada Decoder hanya dapat memprediksi kata dari kata sebelumnya (look-ahead mask)
  # encoder_output merupakan nilai query dan key yang telah diolah di Encoder dan berasal dari layer terakhir di Encoder
  # x merupakan inisiasi tensor pertama
  # yang diinisiasi di awal berarti x dan tgt_mask  
  def forward(self, x, encoder_output, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, tgt_mask)
    
    return self.norm(x)
  
# Projection layer di sini adalah linear layer yang berfungsi untuk mengubah dari nilai yang dhihasilkan dari layer decoder terakhir menjadi kembali ke bentuk
# Vocabulary (Batch, Seq_Len, vocab_size)
class ProjectionLayer(nn.Module):
  
  def __init__(self, d_model: int, vocab_size: int):
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)
    
  def forward(self, x):
    # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, vocab_size)
    # Using log softmax for numerical stability
    return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
  
  def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.src_pos = src_pos
    self.tgt_pos = tgt_pos
    self.projection_layer = projection_layer
    
  def encode(self, src, src_mask):
    src = self.src_embed(src)
    src = self.src_pos(src)

    return self.encoder(src, src_mask)
  
  # encoder output merupakan value query, key, dan value yang dihasilkan dari layer terakhir Encoder yang dipassing ke Decoder
  # tgt_mask di sini berfungsi untuk fungsi translation. (karena dalam contoh ini, transformernya untuk translation model)
  def decode(self, encoder_output, src_mask, tgt, tgt_mask):
    tgt = self.tgt_embed(tgt)
    tgt = self.tgt_pos(tgt)
    
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
  
  def project(self, x):
    return self.projection_layer(x)
  
# TRANSFORMERS INITIALIZATION METHOD
# ---------------------------------------------------------------------------------------------------------------------------------------------#

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
  # Create the embedding layers
  src_embed = InputEmbeddings(d_model, src_vocab_size)
  tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
  
  # Create the positional encoding layers
  # Sebenarnya tidak perlu untuk menginstaniasi dua PositionalEncoding yang berbeda untuk input (src) dan output (tgt)
  # Karena PositionalEncoding melakukan hal yang sama. Namun, pada code kali ini hanya dijelaskan secara rinci bagaimana Transformer sebenarnya bekerja
  # Maka dari itu dibuatah dua PositionalEncoding
  src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
  tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
  
  # Create the encoder blocks
  encoder_blocks = []
  for _ in range(N):
    encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
    encoder_blocks.append(encoder_block)
    
  # Create the decoder blocks
  decoder_blocks = []  
  for _ in range(N):
    decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
    decoder_blocks.append(decoder_block)
    
  # Create the Encoder and Decoder
  encoder = Encoder(nn.ModuleList(encoder_blocks))
  decoder = Decoder(nn.ModuleList(decoder_blocks))
  
  # Create the projection layer
  # Untuk projection layer digunakan tgt_vocab_size (hasil akhir dari output yang diberikan). Karena pada contoh ini adalah model translation, maka digunakanlah
  # Vocab size dari vocab size yang diharapkan dari output. Karena pada akhirnya nantinya akan dikembalikan ke dictionary index dari vocabulary dari output
  # Konteks: model ini melakukan translasi dari English -> Italy. Maka yang digunakan adalah vocabulary dari Bahasa Itali
  projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
  
  # Create the Transformer
  transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
  
  # Initialize the parameters
  # parameters() pada pyTorch akan mereturn trainable parameters seperti weight dan bias. Untuk inisiasi weight dapat dilakukan dengan uniform xavier initialization
  # Dilakukan filet dim() > 1 karena weight biasanya berdimensi 2 atau lebih, sedangkan bias berdimensi 1 dan secara default akan diinisiasi 0 oleh pytorch.
  # Paper xavier initialization: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  # Paper why xavier initialization is bad on ReLU: https://arxiv.org/pdf/1704.08863
  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
      
  return transformer
  
    
  
  
  
  
    












    
    
    
    
    
    