import torch
import json
from datasets import load_dataset
from tokenizers import Tokenizer
import sacrebleu

from transformer import build_transformer

device="mps" if torch.backends.mps.is_available() else "cpu"

# load config
with open("final_model/config.json") as f:
    config=json.load(f)

# load tokenizers
tokenizer_src=Tokenizer.from_file("final_model/tokenizer_en.json")
tokenizer_tgt=Tokenizer.from_file("final_model/tokenizer_it.json")

# build model
model=build_transformer(
    tokenizer_src.get_vocab_size(),
    tokenizer_tgt.get_vocab_size(),
    config["seq_len"],
    config["seq_len"],
    d_model=config["d_model"]
).to(device)

ckpt=torch.load("final_model/model.pt",map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
torch.set_grad_enabled(False)


def greedy_decode(model,src,max_len,start_id,end_id):

    src_mask=torch.ones(1,1,1,src.size(1),device=device)
    encoder_output=model.encode(src,src_mask)

    ys=torch.ones(1,1,dtype=torch.long,device=device).fill_(start_id)

    for i in range(max_len):

        tgt_mask=torch.tril(torch.ones(1,1,ys.size(1),ys.size(1),device=device))

        out=model.decode(encoder_output,src_mask,ys,tgt_mask)

        prob=model.project(out[:,-1])
        next_word=torch.argmax(prob,dim=1).item()

        ys=torch.cat([
            ys,
            torch.ones(1,1,dtype=torch.long,device=device).fill_(next_word)
        ],dim=1)

        if next_word==end_id:
            break

    return ys


# load full dataset
dataset=load_dataset("opus_books","en-it",split="train")

# create proper test split (10%)
dataset=dataset.train_test_split(test_size=0.1)
dataset=dataset["test"].select(range(200))


predictions=[]
references=[]

start_id=tokenizer_tgt.token_to_id("[SOS]")
end_id=tokenizer_tgt.token_to_id("[EOS]")

with torch.no_grad():

    for sample in dataset:

        src_text=sample["translation"]["en"]
        tgt_text=sample["translation"]["it"]

        src_tokens=tokenizer_src.encode(src_text).ids
        src=torch.tensor(src_tokens).unsqueeze(0).to(device)

        out=greedy_decode(model,src,config["seq_len"],start_id,end_id)

        pred_text=tokenizer_tgt.decode(out[0].tolist())

        predictions.append(pred_text)
        references.append(tgt_text)


bleu=sacrebleu.corpus_bleu(predictions,[references])

print("Validation BLEU Score:",bleu.score)