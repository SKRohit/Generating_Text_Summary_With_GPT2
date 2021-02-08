import argparse
from datetime import datetime
import json
import os
import pickle
import random
import time

import numpy as np
from transformers import ConstantLRSchedule, GPT2Config, GPT2LMHeadModel,AdamW, GPT2Tokenizer, WarmupLinearSchedule
from tensorboardX import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tnrange, tqdm

from dataset import GPT21024Dataset 
from utils import add_special_tokens, beam_search, generate_beam_sample, generate_sample, sample_seq, set_seed, top_k_top_p_filtering



def train(args, model, tokenizer, train_dataset, valid_dataset, ignore_index):
	""" Trains GPT2 model and logs necessary details.
		Args:
			args: dict that contains all the necessary information passed by user while training
 			model: finetuned gpt/gpt2 model
			tokenizer: GPT/GPT2 tokenizer
			train_dataset: GPT21024Dataset object for training data
			ignore_index: token not considered in loss calculation
	"""
    writer = SummaryWriter('./logs')
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset,sampler=train_sampler,batch_size=args.batch_size,num_workers=args.num_workers)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation
    optimizer = AdamW(model.parameters(),lr=args.lr)
    scheduler = WarmupLinearSchedule(optimizer,100,80000)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = tnrange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = torch.tensor(batch['article']), torch.tensor(batch['article'])
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            logits = model(inputs)[0]
            idx = batch['sum_idx'].item() # index of separator token
            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx+1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                writer.add_scalar('loss', (tr_loss - logging_loss)/args.gradient_accumulation_steps, global_step)
                logging_loss = tr_loss
                print("loss:", loss.item(), end='\n\n')
                if (step + 1)/args.gradient_accumulation_steps == 1.0:
                	print('After 1st update: ', end='\n\n')
                	generate_sample(valid_dataset, tokenizer, num=2, eval_step=False,device=args.device)
                
                
            if (step + 1) % (10*args.gradient_accumulation_steps) == 0:
                results = evaluate(args, model, valid_dataset, ignore_index, global_step)
                for key, value in results.items():
                    writer.add_scalar('eval_{}'.format(key), value, global_step)
                print('After', global_step+1,'updates: ', end='\n\n')
                generate_sample(valid_dataset, tokenizer, num=2, eval_step=True,device=args.device)
                    
     

 def evaluate(args, model, eval_dataset, ignore_index, global_step=None):
 	""" Returns perplexity score on validation dataset.
 		Args:
 			args: dict that contains all the necessary information passed by user while training
 			model: finetuned gpt/gpt2 model
 			eval_dataset: GPT21024Dataset object for validation data
 			global_step: no. of times gradients have backpropagated
 			ignore_index: token not considered in loss calculation
 	"""
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    eval_output_dir = args.output_dir

    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = torch.tensor(batch['article']).to(args.device), torch.tensor(batch['article']).to(args.device)
        
        with torch.no_grad():
        	logits = model(inputs)[0]
            idx = batch['sum_idx'].item() # index of separator token
            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx+1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }
    print("perplexity:", perplexity.item())

    if global_step:
    	output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
	    with open(output_eval_file, "a") as f:
	        for key in sorted(result.keys()):
	            f.write('\n\n')
	            f.write("time = %s, %s = %s, step = %s\n" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), key, str(result[key]), str(global_step)))
    return result           

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr",default=5e-5, type=float, required=True, help="learning rate")
	parser.add_argument("--seed",default=42, type=int, required=False, help="seed to replicate results")
	parser.add_argument("--n_gpu",default=1, type=int, required=False, help="no of gpu available")
	parser.add_argument("--gradient_accumulation_steps",default=32, type=int, required=True, help="gradient_accumulation_steps")
	parser.add_argument("--batch_size",default=1, type=int, required=True, help="batch_size")
	parser.add_argument("--num_workers",default=4, type=int, required=False, help="num of cpus available")
	parser.add_argument("--device",default=torch.device('cpu'), required=False, help="torch.device object")
	parser.add_argument("--num_train_epochs",default=1, type=int, required=True, help="no of epochs of training")
	parser.add_argument("--output_dir",default=./output, type=str, required=True, help="path to save evaluation results")
	parser.add_argument("--model_dir",default=./weights, type=str, required=True, help="path to save trained model")
	parser.add_argument("--fp16",default=True, type=bool, required=False, help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
	parser.add_argument("--fp16_opt_level",default='O0', type=str, required=False, help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
	parser.add_argument("--max_grad_norm",default=1.0, type=float, help="max gradient norm.")
	parser.add_argument("--root_dir",default='./CNN/gpt2_1024_data', type=str, help="location of json dataset.")
	parser.add_argument("--ids_file",default='./CNN/ids.json', type=str, help="location of train, valid and test file indexes")
	args = parser.parse_args()

	train_data = GPT21024Dataset(args.root_dir,args.ids_file,mode='train',length=3000) #training on only 3000 datasets
	valid_data = GPT21024Dataset(args.root_dir,args.ids_file,mode='valid',length=500)  #validation on only 500 datasets
	tokenizer = add_special_tokens()
	ignore_idx = tokenizer.pad_token_id
   	model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
   	model.to(args.device)

	start = time.time()
	train(args, model, tokenizer, train_data, valid_data, ignore_index)
	print('total time: ', (time.time()-start)/60, " minutes", end='\n\n')

	print('Saving trained model...')
	model_file = os.path.join(args['model_dir'], 'model_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.bin'.format(args['fp16_opt_level'],3000,args['num_train_epochs']))
	config_file = os.path.join(args['model_dir'], 'config_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.json'.format(args['fp16_opt_level'],3000,args['num_train_epochs']))
	torch.save(model.state_dict(), model_file)
	model.config.to_json_file(config_file)


if __name__ == '__main__':
	main()
