from pit.dataset.mathematical_shapes import MathematicalShapesDataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config

num_shapes=102
train_dataset = MathematicalShapesDataset( # sum to not 100
                                      train=True,
                                      rule_indices=[0,4], 
                                      num_shapes=num_shapes,
                                      num_samples=1000000, 
                                      return_rule_label=True)

test_dataset = MathematicalShapesDataset(  # sum to 100
                                      train=False,
                                      rule_indices=[0,4], 
                                      num_shapes=num_shapes,
                                      num_samples=1000000, 
                                      return_rule_label=True)

print(train_dataset[0])
print(test_dataset[0])

device='cuda:0'
config=GPT2Config(n_layer=1, vocab_size=train_dataset.vocab_len, eos_token_id=train_dataset.eos_token)
model = GPT2LMHeadModel(config=config).to(device)



loader = DataLoader(train_dataset, batch_size=4)
for i in loader:
    i['input_ids'] = i['input_ids'].to(device)
    i['labels'] = i['input_ids'].to(device)
    output = model(**i)
    # print(output)
    print(output.loss)
    print(output.logits.size())
    break

# -------------
model.eval()
model.pad_token_id = train_dataset.eos_token
loader = DataLoader(test_dataset, batch_size=4)
for i in loader:
    i['input_ids'] = i['input_ids'].to(device)
    i['labels'] = i['input_ids'].to(device)
    output = model(**i)
    # print(output)
    print(output.loss)
    print(output.logits.size())
    break

# -------------
for i in loader:
    i['input_ids'] = i['input_ids'].to(device)
    print(i) 
    output = model.generate(i['input_ids'][:,:5], max_new_tokens=2, min_new_tokens=2)
    print(output) 
    break


