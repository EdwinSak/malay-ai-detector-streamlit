import torch
from transformers import AutoModelForSequenceClassification
from pathlib import Path
max_seq_length = 512
mistral_threshold = 0.7059877514839172
electra_threshold = 0.9441768527030945
mallam_threshold = 0.13432104885578156
svm_threshold = 0.6771792032000351
ensemble_threshold = 0.6190963642043329

MODULE_DIR = Path(__file__).resolve().parent.parent

def calculate_tokens(tokenizer, input_text):
    tokens = tokenizer.encode(input_text)
    count = len(tokens)
    print(f"has {count} tokens")
    return count


def initialize_mistral():
    mistral_path = MODULE_DIR /'detector-model/models/malaysian-mistral-64M-4096/checkpoint-873'
    mistral_path = 'SmartestBoy/malaysian-mistral-ai-detector'


    id2label = {0: "Human", 1: "AI"}
    label2id = {"Human": 0, "AI": 1}

    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_path)
    mistral_model = AutoModelForSequenceClassification.from_pretrained(
        str(mistral_path),
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        device_map="cpu" 
    )
    mistral_model.eval()
    return mistral_model, mistral_tokenizer

def inference(
        model, 
        tokenizer, 
        threshold, 
        text=None
    ):
    if text==None:
        text = input()
    with torch.inference_mode():
        print(f'running inference for {model.config._name_or_path}')
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length).to("cpu")
        print(f"dtype: {inputs['input_ids'].dtype}")        
        outputs = model(**inputs)
        token_count = inputs["input_ids"].shape[1]
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        ai_probs = probs[0][1].item()
        if ai_probs >= threshold:
            pred =  'AI'
        else:
            pred = 'Human'
    return pred, ai_probs, token_count

from peft import PeftModel

def initialize_mallam():
    mallam_path = 'SmartestBoy/mallam-ai-detector'
    
    tokenizer = AutoTokenizer.from_pretrained(mallam_path)
    
    id2label = {0: "Human", 1: "AI"}
    label2id = {"Human": 0, "AI": 1}

    base_model = AutoModelForSequenceClassification.from_pretrained(
        mallam_path, 
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        device_map="cpu",        
        dtype=torch.float32
    )

    model = PeftModel.from_pretrained(base_model, mallam_path)
    
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def initialize_electra():
    electra_path = MODULE_DIR /'detector-model/models/electra-base-discriminator-bahasa-cased-512-seq/checkpoint-873'
    electra_path = 'SmartestBoy/electra-malay-ai-detector'
    id2label = {0: "Human", 1: "AI"}
    label2id = {"Human": 0, "AI": 1}

    electra_model = AutoModelForSequenceClassification.from_pretrained(
        str(electra_path),
        num_labels = 2,
        id2label = id2label,
        label2id = label2id,
        device_map='cpu'
    )

    electra_tokenizer = AutoTokenizer.from_pretrained(electra_path)
    electra_model.eval()
    return electra_model, electra_tokenizer


import joblib
def initialize_svm():
    svm = joblib.load('models/svm/tuned-svm.pkl')
    return svm 


def svm_inference(svm, text=None):
    if text == None:
        text = input()
    vectorizer = svm.named_steps['tfidf']
    feature_vector = vectorizer.transform([text])
    
    count = feature_vector.nnz

    pred = svm.predict_proba([text])
    ai_prob = pred[0][1].item()
    if ai_prob >= svm_threshold:
        label =  'AI'
    else:
        label = 'Human'
    return label, ai_prob, count


import numpy as np

def ensemble_inference(
        mistral,
        mistral_tok,
        electra,
        electra_tok,
        svm,
        text=None):
    print('Ensembling...')
    if text == None:
        text = input()
    print('text is ' + text)
    mistral_pred, mistral_probs, _  = inference(mistral, mistral_tok, mistral_threshold, text)
    svm_pred, svm_probs, svm_tokens = svm_inference(svm, text)
    electra_pred, electra_probs, _  = inference(electra, electra_tok, electra_threshold, text)
    trust_scores = {
        'mistral': 0.5,
        'electra': 1,
        'svm': 0.33,
    }
    model_names = [
        'electra', 
        'mistral', 
        'svm'
    ]
    prob_data_map = {
        'electra': electra_probs,
        'mistral': mistral_probs,
        'svm': svm_probs
    }

    mistral_tokens = calculate_tokens(mistral_tok, input_text=text)
    electra_tokens = calculate_tokens(electra_tok, input_text=text)
    token_counts = {
        'mistral': mistral_tokens,
        'svm': svm_tokens,
        'electra': electra_tokens
    }
    all_probs = [prob_data_map[m] for m in model_names]
    active_scores = [trust_scores[m] for m in model_names]

    final_probs = np.average(all_probs, axis=0, weights=active_scores).item()
    
    print(final_probs)
    print('Ensembling done!')

    if final_probs > ensemble_threshold:
        pred = 'AI'
    else:
        pred = 'Human'
            

    ai_sum = sum([
        mistral_pred == 'AI',
        svm_pred == 'AI',
        electra_pred == 'AI',
        pred =='AI'
    ])


    uncalibrated_probs = {
        'mistral': mistral_probs,
        'electra': electra_probs,
        'svm': svm_probs,
        'ensemble': final_probs
    }
    print('uncalibrated probs: ' + str(uncalibrated_probs))

    calibrated_probs = calibrate_probs(uncalibrated_probs)
    final_object = {
        'pred': pred,
        'prob': final_probs,
        'calibrated_probs': calibrated_probs,
        'mistral_probs': mistral_probs,
        'electra_probs': electra_probs,
        'svm_probs': svm_probs,
        'ai_sum': ai_sum,
        'token_counts': token_counts,
    }
    return final_object



def map_score(raw_score, threshold=0.5):
    if raw_score <= threshold:
        return (raw_score / threshold) * 0.5
    else:
        return 0.5 + (raw_score - threshold) / (1.0 - threshold) * 0.5


def calibrate_probs(raw):
    mistral_probs =  map_score(raw['mistral'], mistral_threshold)
    electra_probs =  map_score(raw['electra'], electra_threshold)
    svm_probs =  map_score(raw['svm'], svm_threshold)
    ensemble_probs =  map_score(raw['ensemble'], ensemble_threshold)

    results = {
        "Mistral": mistral_probs,
        "Electra": electra_probs,
        "SVM": svm_probs,
        "Ensemble": ensemble_probs,
    }

    print('calibrated probs: ' + str(results))
    return results
