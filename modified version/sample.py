from read_utils import TextConverter, get_batches
from model import CharRnn, Config
import tensorflow as tf

converter = TextConverter(filename="vocab.pkl")

ckpt = tf.train.latest_checkpoint("checkpoints")

config = Config(sampling=True, use_embedding=True, embedding_size=128, num_classes=converter.vocab_size)

model = CharRnn(config)

model.load(ckpt)

start = converter.text_to_arr("今天我")

arr = model.sample(n_samples=1000, prime=start, vocab_size=converter.vocab_size)

print(converter.arr_to_text(arr))