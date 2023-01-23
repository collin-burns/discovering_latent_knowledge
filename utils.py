import os
import functools
import argparse
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F

# make sure to install promptsource, transformers, and datasets!
from promptsource.templates import DatasetTemplates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_dataset


############# Model loading and result saving #############

# Map each model name to its full Huggingface name; this is just for convenience for common models. You can run whatever model you'd like.
model_mapping = {
    "gpt-j": "EleutherAI/gpt-j-6B",
    "T0pp": "bigscience/T0pp",
    "unifiedqa": "allenai/unifiedqa-t5-11b",
    "T5": "t5-11b",
    "deberta-mnli": "microsoft/deberta-xxlarge-v2-mnli",
    "deberta": "microsoft/deberta-xxlarge-v2",
    "roberta-mnli": "roberta-large-mnli",
}


def get_parser():
    """
    Returns the parser we will use for generate.py and evaluate.py
    (We include it here so that we can use the same parser for both scripts)
    """
    parser = argparse.ArgumentParser()
    # setting up model
    parser.add_argument("--model_name", type=str, default="T5", help="Name of the model to use")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for the model and tokenizer")
    parser.add_argument("--parallelize", action="store_true", help="Whether to parallelize the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model")
    # setting up data
    parser.add_argument("--dataset_name", type=str, default="imdb", help="Name of the dataset to use")
    parser.add_argument("--split", type=str, default="test", help="Which split of the dataset to use")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Which prompt to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    # which hidden states we extract
    parser.add_argument("--use_decoder", action="store_true", help="Whether to use the decoder; only relevant if model_type is encoder-decoder. Uses encoder by default (which usually -- but not always -- works better)")
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to use (if not all layers)")
    parser.add_argument("--all_layers", action="store_true", help="Whether to use all layers or not")
    parser.add_argument("--token_idx", type=int, default=-1, help="Which token to use (by default the last token)")
    # saving the hidden states
    parser.add_argument("--save_dir", type=str, default="generated_hidden_states", help="Directory to save the hidden states")

    return parser


def load_model(model_name, cache_dir=None, parallelize=False, device="cuda"):
    """
    Loads a model and its corresponding tokenizer, either parallelized across GPUs (if the model permits that; usually just use this for T5-based models) or on a single GPU
    """
    if model_name in model_mapping:
        # use a nickname for our models
        full_model_name = model_mapping[model_name]
    else:
        # if you're trying a new model, make sure it's the full name
        full_model_name = model_name

    # use the right automodel, and get the corresponding model type
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name, cache_dir=cache_dir)
        model_type = "encoder_decoder"
    except:
        try:
            model = AutoModelForMaskedLM.from_pretrained(full_model_name, cache_dir=cache_dir)
            model_type = "encoder"
        except:
            model = AutoModelForCausalLM.from_pretrained(full_model_name, cache_dir=cache_dir)
            model_type = "decoder"
    
        
    # specify model_max_length (the max token length) to be 512 to ensure that padding works 
    # (it's not set by default for e.g. DeBERTa, but it's necessary for padding to work properly)
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, cache_dir=cache_dir, model_max_length=512)
    model.eval()

    # put on the correct device
    if parallelize:
        model.parallelize()
    else:
        model = model.to(device)

    return model, tokenizer, model_type


def save_generations(generation, args, generation_type):
    """
    Input: 
        generation: numpy array (e.g. hidden_states or labels) to save
        args: arguments used to generate the hidden states. This is used for the filename to save to.
        generation_type: one of "negative_hidden_states" or "positive_hidden_states" or "labels"

    Saves the generations to an appropriate directory.
    """
    # construct the filename based on the args
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)

    # create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # save
    np.save(os.path.join(args.save_dir, filename), generation)


def load_single_generation(args, generation_type="hidden_states"):
    # use the same filename as in save_generations
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)
    return np.load(os.path.join(args.save_dir, filename))


def load_all_generations(args):
    # load all the saved generations: neg_hs, pos_hs, and labels
    neg_hs = load_single_generation(args, generation_type="negative_hidden_states")
    pos_hs = load_single_generation(args, generation_type="positive_hidden_states")
    labels = load_single_generation(args, generation_type="labels")

    return neg_hs, pos_hs, labels


############# Data #############
class ContrastDataset(Dataset):
    """
    Given a dataset and tokenizer (from huggingface), along with a collection of prompts for that dataset from promptsource and a corresponding prompt index, 
    returns a dataset that creates contrast pairs using that prompt
    
    Truncates examples larger than max_len, which can mess up contrast pairs, so make sure to only give it examples that won't be truncated.
    """
    def __init__(self, raw_dataset, tokenizer, all_prompts, prompt_idx, 
                 model_type="encoder_decoder", use_decoder=False, device="cuda"):

        # data and tokenizer
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        
        # for formatting the answers
        self.model_type = model_type
        self.use_decoder = use_decoder
        if self.use_decoder:
            assert self.model_type != "encoder"

        # prompt
        prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
        self.prompt = all_prompts[prompt_name_list[prompt_idx]]

    def __len__(self):
        return len(self.raw_dataset)

    def encode(self, nl_prompt):
        """
        Tokenize a given natural language prompt (from after applying self.prompt to an example)
        
        For encoder-decoder models, we can either:
        (1) feed both the question and answer to the encoder, creating contrast pairs using the encoder hidden states
            (which uses the standard tokenization, but also passes the empty string to the decoder), or
        (2) feed the question the encoder and the answer to the decoder, creating contrast pairs using the decoder hidden states
        
        If self.decoder is True we do (2), otherwise we do (1).
        """
        # get question and answer from prompt
        question, answer = nl_prompt
        
        # tokenize the question and answer (depending upon the model type and whether self.use_decoder is True)
        if self.model_type == "encoder_decoder":
            input_ids = self.get_encoder_decoder_input_ids(question, answer)
        elif self.model_type == "encoder":
            input_ids = self.get_encoder_input_ids(question, answer)
        else:
            input_ids = self.get_decoder_input_ids(question, answer)
        
        # get rid of the batch dimension since this will be added by the Dataloader
        if input_ids["input_ids"].shape[0] == 1:
            for k in input_ids:
                input_ids[k] = input_ids[k].squeeze(0)

        return input_ids


    def get_encoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-only models; standard formatting.
        """
        combined_input = question + " " + answer 
        input_ids = self.tokenizer(combined_input, truncation=True, padding="max_length", return_tensors="pt")

        return input_ids


    def get_decoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-only models.
        This is the same as get_encoder_input_ids except that we add the EOS token at the end of the input (which apparently can matter)
        """
        combined_input = question + " " + answer + self.tokenizer.eos_token
        input_ids = self.tokenizer(combined_input, truncation=True, padding="max_length", return_tensors="pt")

        return input_ids


    def get_encoder_decoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-decoder models.
        There are two cases for this, depending upon whether we want to use the encoder hidden states or the decoder hidden states.
        """
        if self.use_decoder:
            # feed the same question to the encoder but different answers to the decoder to construct contrast pairs
            input_ids = self.tokenizer(question, truncation=True, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer(answer, truncation=True, padding="max_length", return_tensors="pt")
        else:
            # include both the question and the answer in the input for the encoder
            # feed the empty string to the decoder (i.e. just ignore it -- but it needs an input or it'll throw an error)
            input_ids = self.tokenizer(question, answer, truncation=True, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer("", return_tensors="pt")
        
        # move everything into input_ids so that it's easier to pass to the model
        input_ids["decoder_input_ids"] = decoder_input_ids["input_ids"]
        input_ids["decoder_attention_mask"] = decoder_input_ids["attention_mask"]

        return input_ids


    def __getitem__(self, index):
        # get the original example
        data = self.raw_dataset[int(index)]
        text, true_answer = data["text"], data["label"]

        # get the possible labels
        # (for simplicity assume the binary case for contrast pairs)
        label_list = self.prompt.get_answer_choices_list(data)
        assert len(label_list) == 2, print("Make sure there are only two possible answers! Actual number of answers:", label_list)

        # reconvert to dataset format but with fake/candidate labels to create the contrast pair
        neg_example = {"text": text, "label": 0}
        pos_example = {"text": text, "label": 1}

        # construct contrast pairs by answering the prompt with the two different possible labels
        # (for example, label 0 might be mapped to "no" and label 1 might be mapped to "yes")
        neg_prompt, pos_prompt = self.prompt.apply(neg_example), self.prompt.apply(pos_example)

        # tokenize
        neg_ids, pos_ids = self.encode(neg_prompt), self.encode(pos_prompt)

        # verify these are different (e.g. tokenization didn't cut off the difference between them)
        if self.use_decoder and self.model_type == "encoder_decoder":
            assert (neg_ids["decoder_input_ids"] - pos_ids["decoder_input_ids"]).sum() != 0, print("The decoder_input_ids for the contrast pairs are the same!", neg_ids, pos_ids)
        else:
            assert (neg_ids["input_ids"] - pos_ids["input_ids"]).sum() != 0, print("The input_ids for the contrast pairs are the same!", neg_ids, pos_ids)

        # return the tokenized inputs, the text prompts, and the true label
        return neg_ids, pos_ids, neg_prompt, pos_prompt, true_answer

    
def get_dataloader(dataset_name, split, tokenizer, prompt_idx, batch_size=16, num_examples=1000,
                   model_type="encoder_decoder", use_decoder=False, device="cuda", pin_memory=True, num_workers=1):
    """
    Creates a dataloader for a given dataset (and its split), tokenizer, and prompt index

    Takes a random subset of (at most) num_examples samples from the dataset that are not truncated by the tokenizer.
    """
    # load the raw dataset
    raw_dataset = load_dataset(dataset_name)[split]

    # load all the prompts for that dataset
    all_prompts = DatasetTemplates(dataset_name)

    # create the ConstrastDataset
    contrast_dataset = ContrastDataset(raw_dataset, tokenizer, all_prompts, prompt_idx, 
                                       model_type=model_type, use_decoder=use_decoder, 
                                       device=device)

    # get a random permutation of the indices; we'll take the first num_examples of these that do not get truncated
    random_idxs = np.random.permutation(len(contrast_dataset))

    # remove examples that would be truncated (since this messes up contrast pairs)
    prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
    prompt = all_prompts[prompt_name_list[prompt_idx]]
    keep_idxs = []
    for idx in random_idxs:
        question, answer = prompt.apply(raw_dataset[int(idx)])
        input_text = question + " " + answer
        if len(tokenizer.encode(input_text, truncation=False)) < tokenizer.model_max_length - 2:  # include small margin to be conservative
            keep_idxs.append(idx)
            if len(keep_idxs) >= num_examples:
                break

    # create and return the corresponding dataloader
    subset_dataset = torch.utils.data.Subset(contrast_dataset, keep_idxs)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    return dataloader


############# Hidden States #############
def get_first_mask_loc(mask, shift=False):
    """
    return the location of the first pad token for the given ids, which corresponds to a mask value of 0
    if there are no pad tokens, then return the last location
    """
    # add a 0 to the end of the mask in case there are no pad tokens
    mask = torch.cat([mask, torch.zeros_like(mask[..., :1])], dim=-1)

    if shift:
        mask = mask[..., 1:]

    # get the location of the first pad token; use the fact that torch.argmax() returns the first index in the case of ties
    first_mask_loc = torch.argmax((mask == 0).int(), dim=-1)

    return first_mask_loc


def get_individual_hidden_states(model, batch_ids, layer=None, all_layers=True, token_idx=-1, model_type="encoder_decoder", use_decoder=False):
    """
    Given a model and a batch of tokenized examples, returns the hidden states for either 
    a specified layer (if layer is a number) or for all layers (if all_layers is True).
    
    If specify_encoder is True, uses "encoder_hidden_states" instead of "hidden_states"
    This is necessary for getting the encoder hidden states for encoder-decoder models,
    but it is not necessary for encoder-only or decoder-only models.
    """
    if use_decoder:
        assert "decoder" in model_type
        
    # forward pass
    with torch.no_grad():
        batch_ids = batch_ids.to(model.device)
        output = model(**batch_ids, output_hidden_states=True)

    # get all the corresponding hidden states (which is a tuple of length num_layers)
    if use_decoder and "decoder_hidden_states" in output.keys():
        hs_tuple = output["decoder_hidden_states"]
    elif "encoder_hidden_states" in output.keys():
        hs_tuple = output["encoder_hidden_states"]
    else:
        hs_tuple = output["hidden_states"]

    # just get the corresponding layer hidden states
    if all_layers:
        # stack along the last axis so that it's easier to consistently index the first two axes
        hs = torch.stack([h.squeeze().detach().cpu() for h in hs_tuple], axis=-1)  # (bs, seq_len, dim, num_layers)
    else:
        assert layer is not None
        hs = hs_tuple[layer].unsqueeze(-1).detach().cpu()  # (bs, seq_len, dim, 1)

    # we want to get the token corresponding to token_idx while ignoring the masked tokens
    if token_idx == 0:
        final_hs = hs[:, 0]  # (bs, dim, num_layers)
    else:
        # if token_idx == -1, then takes the hidden states corresponding to the last non-mask tokens
        # first we need to get the first mask location for each example in the batch
        assert token_idx < 0, print("token_idx must be either 0 or negative, but got", token_idx)
        mask = batch_ids["decoder_attention_mask"] if (model_type == "encoder_decoder" and use_decoder) else batch_ids["attention_mask"]
        first_mask_loc = get_first_mask_loc(mask).squeeze()
        final_hs = hs[torch.arange(hs.size(0)), first_mask_loc+token_idx]  # (bs, dim, num_layers)
    
    return final_hs


def get_all_hidden_states(model, dataloader, layer=None, all_layers=True, token_idx=-1, model_type="encoder_decoder", use_decoder=False):
    """
    Given a model, a tokenizer, and a dataloader, returns the hidden states (corresponding to a given position index) in all layers for all examples in the dataloader,
    along with the average log probs corresponding to the answer tokens

    The dataloader should correspond to examples *with a candidate label already added* to each example.
    E.g. this function should be used for "Q: Is 2+2=5? A: True" or "Q: Is 2+2=5? A: False", but NOT for "Q: Is 2+2=5? A: ".
    """
    all_pos_hs, all_neg_hs = [], []
    all_gt_labels = []

    model.eval()
    for batch in tqdm(dataloader):
        neg_ids, pos_ids, _, _, gt_label = batch

        neg_hs = get_individual_hidden_states(model, neg_ids, layer=layer, all_layers=all_layers, token_idx=token_idx, 
                                              model_type=model_type, use_decoder=use_decoder)
        pos_hs = get_individual_hidden_states(model, pos_ids, layer=layer, all_layers=all_layers, token_idx=token_idx, 
                                              model_type=model_type, use_decoder=use_decoder)

        if dataloader.batch_size == 1:
            neg_hs, pos_hs = neg_hs.unsqueeze(0), pos_hs.unsqueeze(0)

        all_neg_hs.append(neg_hs)
        all_pos_hs.append(pos_hs)
        all_gt_labels.append(gt_label)
    
    all_neg_hs = np.concatenate(all_neg_hs, axis=0)
    all_pos_hs = np.concatenate(all_pos_hs, axis=0)
    all_gt_labels = np.concatenate(all_gt_labels, axis=0)

    return all_neg_hs, all_pos_hs, all_gt_labels

############# CCS #############
class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCS(object):
    def __init__(self, x0, x1, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.probe = self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

        
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)    


    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

        
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1
    

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss


    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test).mean()
        acc = max(acc, 1 - acc)

        return acc
    
        
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()
    
    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss
