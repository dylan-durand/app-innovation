
import torch
import random
import numpy as np
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset 
from sklearn import metrics
import pandas as pd

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device: ", device)

def load_test_data():
    TEST_DATAFILE = "../data/dev.xml"
    print("Loading test data from: ", TEST_DATAFILE)
    test_dataframe : pd.DataFrame = pd.read_xml(TEST_DATAFILE)

    print("Cleaning test data")
    test_dataframe.replace('\n',' ', regex=True, inplace=True)
    test_dataframe.replace('\r',' ', regex=True, inplace=True)
    test_dataframe.replace('\t',' ', regex=True, inplace=True)
    test_dataframe.replace(np.nan, '', regex=True, inplace=True)

    # You can replace "camembert-base" with any other model from the table, e.g. "camembert/camembert-large".
    print("Loading tokenizer")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    test_reviews = test_dataframe["commentaire"].values.tolist()
    test_sentiments = test_dataframe["note"].values.tolist()
    test_review_ids = test_dataframe["review_id"].values.tolist()

    # foreach note in train_sentiments multiply by 2 and subtract 1
    test_sentiments = torch.tensor([int(2 * float(x.replace(",", ".")) - 1) for x in test_sentiments])

    batch_size = 8

    # foreach note in train_sentiments multiply by 2 and subtract 1
    # test_sentiments = torch.tensor([int(2 * float(x.replace(",", ".")) - 1) for x in test_sentiments])
    print("Encoding test data")
    encoded_batch = tokenizer.batch_encode_plus(test_reviews,
                                            truncation=True,
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_tensors = 'pt')
    print("Creating test dataloader")
    test_dataset = TensorDataset(encoded_batch['input_ids'], encoded_batch['attention_mask'])

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader, test_review_ids, test_sentiments
    
def revert_sentiments(sentiments):
    return [str((float(x) + 1) / 2).replace(".", ",") for x in sentiments]

def predict(dataloader):
    # instantiate CamembertForSequenceClassification and load the saved state_dict
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=10)
    model.load_state_dict(torch.load("../models/sentiments.pt", map_location=device))
    model = model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx % 100 == 0 and not idx == 0:
                print(f'  Batch {idx}  of {len(dataloader)}.')

            input_id, attention_mask = batch
            input_id = input_id.to(device)
            attention_mask = attention_mask.to(device)

            res = model(input_id,
                            token_type_ids=None,
                            attention_mask=attention_mask,
                            return_dict=False)
            logits = res[0]
            logits = logits.detach().cpu().numpy()
            predictions.extend([np.argmax(x) for x in logits])
    return predictions

def accuracy_on_three_classes(predictions, sentiments):
    
    # accurate if prediction is 0, 1, 2, 3 and sentiment is 0, 1, 2, 3
    # or if prediction is 4 and sentiment is 4
    # or if prediction is 5, 6, 7, 8, 9 and sentiment is 5, 6, 7, 8, 9
    accurate = 0
    for prediction, sentiment in zip(predictions, sentiments):
        if (prediction < 4 and sentiment < 4) or (prediction == 4 and sentiment == 4) or (prediction > 4 and sentiment > 4):
            accurate += 1
    return accurate / len(predictions)

def accuracy(predictions, sentiments):
    return metrics.accuracy_score(sentiments, predictions)


test_dataloader, test_review_ids, test_sentiments = load_test_data()
predictions = predict(test_dataloader)

str_predictions = revert_sentiments(predictions)

# save results to csv
results = pd.DataFrame({"review_id": test_review_ids, "note": str_predictions})

print ("Accuracy on 3 classes: ", accuracy_on_three_classes(predictions, test_sentiments))
print ("Accuracy: ", accuracy(predictions, test_sentiments))