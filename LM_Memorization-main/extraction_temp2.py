import logging
import argparse
import numpy as np
import torch
import zlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level='ERROR')

def calculatePerplexity(sentence, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, _ = outputs[:2]
    return torch.exp(loss).item()

def parse_commoncrawl(wet_file):
    lines = open(wet_file, 'r', encoding='utf-8').readlines()
    eng_lines = [line for line in lines if 'WARC-Identified-Content-Language: eng' in line]
    return ' '.join(eng_lines)

def generate_text(model, tokenizer, prompt, device, max_length=256, top_k=40, top_p=0.95):
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    if len(encoded_prompt[0]) == 0:
        encoded_prompt = torch.tensor([[tokenizer.eos_token_id]]).to(device)
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length + len(encoded_prompt[0]),
        temperature=1.0,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1
    )
    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    return text

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    models = {
        'XL': GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device),
        'Large': GPT2LMHeadModel.from_pretrained('gpt2-large').to(device),
        'Medium': GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device),
        'Small': GPT2LMHeadModel.from_pretrained('gpt2').to(device),
    }

    cc_text = ""
    if args.internet_sampling and args.wet_file:
        cc_text = parse_commoncrawl(args.wet_file)[:1000]  # Use the first 1000 characters to condition
        print("Common crawl text loaded for conditioning.")

    results = []
    for name, model in models.items():
        model.eval()
        print(f"\nGenerating for model: {name}")
        for _ in tqdm(range(args.N), desc=f"Generating with {name}"):
            text = generate_text(model, tokenizer, cc_text, device)
            perplexity = calculatePerplexity(text, model, tokenizer, device)
            zlib_size = len(zlib.compress(text.encode('utf-8')))
            results.append({
                'Model': name,
                'Perplexity': perplexity,
                'Zlib Size': zlib_size,
                'Sample Text': text[:50] + '...'
            })

    df_results = pd.DataFrame(results)
    print("\n" * 2)  # Prints two empty lines for spacing before the DataFrame
    print(df_results)

    # Optionally, save the results to a CSV file
    df_results.to_csv("model_analysis_results_with_commoncrawl.csv", index=False)
    print("Results saved to model_analysis_results_with_commoncrawl.csv.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate GPT-2 samples and analyze memorization using common crawl.")
    parser.add_argument('--N', type=int, default=5, help="Number of samples to generate for each model.")
    parser.add_argument('--batch-size', type=int, default=1, help="Currently unused, set for future implementation.")
    parser.add_argument('--internet-sampling', action='store_true', help="Use common crawl for conditioning.")
    parser.add_argument('--wet-file', type=str, help="Path to a commoncrawl WET file for conditioning.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
