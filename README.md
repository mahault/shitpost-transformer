# shitpost-transformer

Transformer \\
Twitter Scraper \\
Twitter bot \\

Worth looking into Tweepy: https://realpython.com/twitter-bot-python-tweepy/

https://cheapbotsdonequick.com/?fbclid=IwAR1DD534IDpwGaI9D5IpCArzUkg0ezIF6N4b5twffaBLS8Ko1CryWIamXbA

https://twitter.com/abhi1thakur/status/1470406419786698761?t=P4m7ZuswAW4FG99mVFQa4Q&s=19&fbclid=IwAR0n6i2nBXr5B8IdONvFtYZy7jVVyCPsEPBuZ6Ki7igEdlzS_yDzpZ5OsQU

https://pypi.org/project/twitter-scraper/



Shit posts must be:
non-toxic
wholesome
video games



To run: 
python main.py --cuda --epochs 6 --model Transformer --lr 5 
python generate.py --cuda --model Transformer
                                           # Generate samples from the trained Transformer model.

                                           
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU,
                        Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the
                        transformer model