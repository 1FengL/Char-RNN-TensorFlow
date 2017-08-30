from read_utils import TextConverter, get_batches
from model import CharRnn, Config

with open("ha.txt", "r") as f:
	text = f.read()

converter = TextConverter(text)
converter.save_to_file("vocab.pkl")

config = Config(use_embedding=True, embedding_size=128, num_classes=converter.vocab_size)

arr = converter.text_to_arr(text)
g = get_batches(arr, config.batch_size, config.num_steps)

data_len = config.batch_size * config.num_steps
num_batches = int(len(arr) / data_len)

model = CharRnn(config)

model.train(g, num_batches, keep_prob=config.keep_prob, save_path="checkpoints/", epochs=1000)


