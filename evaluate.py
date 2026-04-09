import torch
import json
from datasets import load_dataset
from tokenizers import Tokenizer
import sacrebleu
from transformer import build_transformer

device="mps" if torch.backends.mps.is_available() else "cpu"

with open("final_model/config.json") as f:
    config=json.load(f)

tokenizer_src=Tokenizer.from_file("final_model/tokenizer_en.json")
tokenizer_tgt=Tokenizer.from_file("final_model/tokenizer_it.json")

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

# ✅ get correct PAD id
pad_id=tokenizer_tgt.token_to_id("[PAD]")

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


# dataset (fixed seed)
dataset=load_dataset("opus_books","en-it",split="train")
dataset=dataset.train_test_split(test_size=0.1,seed=42)
dataset=dataset["test"].select(range(200))

predictions=[]
references=[]

start_id=tokenizer_tgt.token_to_id("[SOS]")
end_id=tokenizer_tgt.token_to_id("[EOS]")

# ✅ use correct pad id
criterion=torch.nn.CrossEntropyLoss(ignore_index=pad_id)

total_loss=0
count=0

with torch.no_grad():

    for i,sample in enumerate(dataset):

        src_text=sample["translation"]["en"]
        tgt_text=sample["translation"]["it"]

        # ---- SOURCE ----
        src_tokens=tokenizer_src.encode(src_text).ids
        if len(src_tokens)==0:
            continue

        src=torch.tensor(src_tokens).unsqueeze(0).to(device)

        # ---- BLEU ----
        out=greedy_decode(model,src,config["seq_len"],start_id,end_id)
        pred_text=tokenizer_tgt.decode(out[0].tolist(),skip_special_tokens=True)

        predictions.append(pred_text)
        references.append(tgt_text)

        # loss
        tgt_tokens=tokenizer_tgt.encode(tgt_text).ids

        # skip bad samples
        if len(tgt_tokens)<2:
            continue

        tgt=torch.tensor(tgt_tokens).unsqueeze(0).to(device)

        input_tgt=tgt[:,:-1]
        label=tgt[:,1:]

        if input_tgt.size(1)==0:
            continue

        src_mask=torch.ones(1,1,1,src.size(1),device=device)
        tgt_mask=torch.tril(torch.ones(1,1,input_tgt.size(1),input_tgt.size(1),device=device))

        enc=model.encode(src,src_mask)
        out=model.decode(enc,src_mask,input_tgt,tgt_mask)
        logits=model.project(out)

        #reshape safely
        logits=logits.view(-1,logits.size(-1))
        label=label.contiguous().view(-1)

        #skip if all pad
        if (label!=pad_id).sum()==0:
            continue

        loss=criterion(logits,label)

        if torch.isnan(loss):
            continue

        total_loss+=loss.item()
        count+=1


        if i<2:
            print("EN:",src_text)
            print("Pred:",pred_text)
            print("True:",tgt_text)
            print("------")


bleu=sacrebleu.corpus_bleu(predictions,[references])

val_loss = total_loss/count if count>0 else 0

print(f"\nValidation BLEU Score: {bleu.score:.2f}")
print(f"Validation Loss: {val_loss:.3f}")