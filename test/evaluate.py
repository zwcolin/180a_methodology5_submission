from __future__ import print_function
from collections import Counter
from tqdm import tqdm
from datetime import datetime
import string
import re
import argparse
import json
import sys
import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

dateTimeObj = datetime.now()
timestamp = dateTimeObj.strftime("%d-%b-%Y_(%H:%M:%S.%f)")
out_file = f'test/evaluation_log_{timestamp}.txt'

def get_answer(model, tokenizer, question, context):
    input_text = "question: %s  context: %s" % (question, context)
    features = tokenizer([input_text], return_tensors='pt')
    
    output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])

    return tokenizer.decode(output[0], skip_special_tokens=True)

def print_validation_results(model, tokenizer, val_dataset):
    gold_answers = []
    predictions = []
    print()
    print(f'Evaluation Begins - You will be able to see evaluation results in the evaluation_logs_{timestamp}.txt after the process is done. To demonstrate the pipeline, I will only make evaluations on the first 100 samples in the validation dataset.')
    for data in tqdm(val_dataset):
        answers = data['answers']['text']
        if len(answers) != 0:
            output = get_answer(model, tokenizer, data['question'], data['context'])

            print(f'Model Output: {output}', file=open(out_file, "a"))
            print('-------------------------', file=open(out_file, "a"))
            for answer in answers:
                print(f'Sample Answer: {answer}', file=open(out_file, "a"))
            print('', file=open(out_file, "a"))
            predictions.append(output)
            gold_answers.append(answers)
    return predictions, gold_answers

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

def test_model():
    path = 'test/testdata'
    if not os.path.exists(path):
        os.makedirs(path)

    dataset = load_dataset('squad_v2', split='validation[:100]', cache_dir=path)
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-squadv2")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-squadv2")
    
    predictions, gold_answers = print_validation_results(model, tokenizer, dataset)
    results = evaluate(gold_answers, predictions)
    print('-------------------------', file=open(out_file, "a"))
    print('Evaluation Metrics', file=open(out_file, "a"))
    print('-------------------------', file=open(out_file, "a"))
    print(f"Exact Match Score: {results['exact_match']}", file=open(out_file, "a"))
    print(f"F1 Score: {results['f1']}", file=open(out_file, "a"))