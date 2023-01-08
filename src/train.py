
import torch
import random
import numpy as np
from transformers import CamembertTokenizer, CamembertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset 
import pandas as pd

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device: ", device)


TRAIN_DATAFILE = "../data/train.xml"
train_dataframe : pd.DataFrame = pd.read_xml(TRAIN_DATAFILE)

train_dataframe.replace('\n',' ', regex=True, inplace=True)
train_dataframe.replace('\r',' ', regex=True, inplace=True)
train_dataframe.replace('\t',' ', regex=True, inplace=True)


train_dataframe = pd.read_csv("../data/train.tsv", sep="\t")
train_dataframe = train_dataframe.replace(np.nan, '', regex=True)

dev_dataframe = pd.read_csv("../data/dev.tsv", sep="\t")
dev_dataframe = train_dataframe.replace(np.nan, '', regex=True)


# You can replace "camembert-base" with any other model from the table, e.g. "camembert/camembert-large".
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

 

 
train_reviews = train_dataframe["commentaire"].values.tolist()
train_sentiments = train_dataframe["note"].values.tolist()
train_user_id = train_dataframe["user_id"].values.tolist()


MAX_LENGTH = tokenizer.max_model_input_sizes['camembert-base']
batch_size = 16

# foreach note in train_sentiments multiply by 2 and subtract 1
train_sentiments = torch.tensor([int(2 * float(x.replace(",", ".")) - 1) for x in train_sentiments])

# La fonction batch_encode_plus encode un batch de donnees
train_encoded_batch = tokenizer.batch_encode_plus(train_reviews,
                                            add_special_tokens=True,
                                            max_length=MAX_LENGTH,
                                            padding=True,
                                            truncation=True,
                                            return_attention_mask = True,
                                            return_tensors = 'pt')

print("train data loaded")

train_dataset = TensorDataset(train_encoded_batch['input_ids'], train_encoded_batch['attention_mask'], train_sentiments)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# On la version pre-entrainee de camemBERT 'base'
model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels = 10)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
epochs = 2



# Pour enregistrer les stats a chaque epoque
training_stats = []

# Boucle d'entrainement
for epoch in range(0, epochs):

    print("")
    print(f'########## Epoch {epoch+1} / {epochs} ##########')
    print('Training...')
    # On initialise la loss pour cette epoque
    total_train_loss = 0

    # On met le modele en mode 'training'
    # Dans ce mode certaines couches du modele agissent differement 
    model.train()

    # Pour chaque batch
    for step, batch in enumerate(train_loader):

        # On fait un print chaque 40 batchs
        if step % 500 == 0 and not step == 0:
            print(f'  Batch {step}  of {len(train_loader)}.')

        # On recupere les donnees du batch
        input_id = batch[0].to(device)
        attention_mask = batch[1].to(device)
        sentiment = batch[2].to(device)

        # On met le gradient a 0
        model.zero_grad()        

        # On passe la donnee au model et on recupere la loss et le logits (sortie avant fonction d'activation)
        loss, logits = model(input_id, 
                            token_type_ids=None, 
                            attention_mask=attention_mask, 
                            labels=sentiment,
                            return_dict=False)

        # On incremente la loss totale
        # .item() donne la valeur numerique de la loss
        total_train_loss += loss.item()

        # Backpropagtion
        loss.backward()
        
        # On actualise les parametrer grace a l'optimizer
        optimizer.step()

    # On calcule la  loss moyenne sur toute l'epoque
    avg_train_loss = total_train_loss / len(train_loader)


    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))  

    # Enregistrement des stats de l'epoque
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
        }
    )

print("Model saved!")
torch.save(model.state_dict(), "./sentiments.pt")