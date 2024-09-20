from pathlib import Path

# batch_size = total batch size yang digunakan untuk memproses sequence secara parallel
# num_epochs = berapa kali percobaan iterasi untuk satu kali feed forward dan backpropagation
# lr = learning rate. umumnya learning rate berubah sesuai epoch, namun pada kode kali ini dibuat statik (karena hanya mencontohkan bagaimana transformer bekerja). tapi harusnya learning rate bisa berubah untuk menyesuaikan beban komputasi
# seq_len = sequence terpanjang dalam dataset. pada contoh kali ini, sequence terpanjang pada dataset opus_books en-it adalah 350 kata
# d_model = banyaknya embedding. default adalah 512
# lang_src = source language. default english
# lang_tgt = target language. default italy
# model_folder = tempat di mana model transformer akan disimpan
# model_basename = nama untuk model yang disimpan. default tmodel_
# preload = preload/restart model ketika crash saat training. Default None
# tokenizer_file = nama file tokenizer
# experiment_name = nama experiment hasil train model. seperti log report. pada experiment ini berisi loss value terhadap model yang ditrain
def get_config():
  
  return {
    "batch_size": 8,
    "num_epochs": 20,
    "lr": 10**-4,
    "seq_len": 350,
    "d_model": 512,
    "datasource": 'opus_books',
    "lang_src": "en",
    "lang_tgt": "it",
    "model_folder": "weights",
    "model_basename": "tmodel_",
    "preload": "latest",
    "tokenizer_file": "tokenizer_{0}.json",
    "experiment_name": "runs/tmodel"
  }
  
def get_weights_file_path(config, epoch: str):
  model_folder = config['model_folder']
  model_basename = config['model_basename']
  model_filename = f"{model_basename}{epoch}.pt"
  
  return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
  
