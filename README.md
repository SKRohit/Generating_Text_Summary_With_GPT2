# Generating Text Summary With GPT2
Accompanying code for blog [Generating Text Summaries Using GPT-2 on PyTorch with Minimal Training](https://blog.paperspace.com/generating-text-summaries-gpt-2/).

## Dataset Preparation

##### Run max_article_sizes.py for both CNN and Daily Mail Tokenized articles separately. It will create pickle files of sizes of each CNN/DAILY MAIL articles.
    $ python max_article_sizes.py path/to/cnn_or_dailymail/tokenized/articles


##### Run below command to prepare json files which contains tokenized articles and summaries
    $ python prepare_data.py path/to/pickle_file/of/articles/sizes/created/using/above/command
    
    
## Training
Use pretrained weights to finetune the GPT2 model using tricks mentioned in [Generating Text Summaries Using GPT-2 on PyTorch with Minimal Training](https://blog.paperspace.com/improving-yolo/) on your data.
```
$ python train_gpt2_summarizer.py --batch_size 1 --root_dir path/to/json/files/created/using/prepare_data.py
```

## Credit

### [Sample Efficient Text Summarization Using a Single Pre-Trained Transformer](https://arxiv.org/abs/1905.08836)
_Urvashi Khandelwal, Kevin Clark, Dan Jurafsky, Lukasz Kaiser_ <br>

Training code in this repo has been adapted from huggingface [run_lm_finetuning.py](https://github.com/SKRohit/pytorch-transformers/blob/master/examples/run_lm_finetuning.py).
