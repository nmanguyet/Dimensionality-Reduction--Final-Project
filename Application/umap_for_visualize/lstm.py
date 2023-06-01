##
import torch
import gc
import torch.nn as nn # Using and create module, class
import torch.optim as optim # Optimization
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence # Padding 
from torch.utils.data import DataLoader, TensorDataset # Divide into batch, create dataset for training
from tqdm.notebook import tqdm # Information when train, test
from sklearn.model_selection import train_test_split # Split train - test for lstm model

def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = [torch.tensor(input, dtype=torch.long) for input in inputs]
    seq_len = [len(input) for input in inputs]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    seq_len = torch.tensor(seq_len)
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    return inputs, seq_len, labels


def create_attention_mask(sequence_lengths, max_length):
    batch_size = len(sequence_lengths)
    attention_masks = torch.zeros(batch_size, max_length, dtype=torch.float32)

    for i, seq_len in enumerate(sequence_lengths):
        attention_masks[i, :seq_len] = 1.0

    return attention_masks


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                              min=1e-9)


class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix, n_class):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float),
                                                      freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_matrix.shape[1], hidden_size=128, bidirectional=True)
        self.fc = nn.Linear(256, n_class)

    def forward(self, x, attn_mask):
        x = self.embedding(x)
        output, (hidden, _) = self.lstm(x)
        output = mean_pooling(output, attn_mask)
        x = self.fc(output).squeeze()
        return x, output

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)


def lstm_embedding(X_train, y_train, X_test, y_test, embedding_matrix, n_class):
    # Split train - test for lstm task, 
    X_train_lstm, X_valid_lstm, y_train_lstm, y_valid_lstm = split_train_test(X_train, y_train)
    # Set up train-test data
    train_dataset = NewsDataset(X_train_lstm, y_train_lstm)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    valid_dataset = NewsDataset(X_valid_lstm, y_valid_lstm)
    valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

    # Initialize the model, loss, and optimizer
    device = 'cuda:0'
    model = LSTMModel(embedding_matrix, n_class)
    # .to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-3, weight_decay=1e-4)

    # Training loop
    num_epochs = 1
    torch.cuda.empty_cache()
    gc.collect()
    ans = []
    for epoch in range(num_epochs):
        model.train()
        train_tqdm = tqdm(train_loader, leave=True, desc='Training: ')
        for inputs, seq_len, targets in train_tqdm:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            seq_len = seq_len.to(device)
            attn_mask = create_attention_mask(seq_len, inputs.size(1)).to(device)
            targets = targets.to(device)
            outputs, embed = model(inputs, attn_mask)
            ans.append(embed)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                _, pred = torch.max(outputs, dim=1) 
                acc = (pred == targets).sum()/targets.size(0)
            train_tqdm.set_postfix(loss=loss.item(), accuracy=acc.item())

        # Validation
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            valid_tqdm = tqdm(valid_loader, leave=True, desc='Validation: ')
            for inputs, seq_len, targets in valid_tqdm:
                inputs = inputs.to(device)
                seq_len = seq_len.to(device)
                attn_mask = create_attention_mask(seq_len, inputs.size(1)).to(device)
                targets = targets.to(device)
                outputs, embed = model(inputs, attn_mask)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                with torch.no_grad():
                    _, pred = torch.max(outputs, dim=1)
                    acc = (pred == targets).sum()/targets.size(0)
                valid_tqdm.set_postfix(loss=loss.item(), accuracy=acc.item())

    dataset = NewsDataset(X_train, y_train)
    embed_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    model.train()
    embed_tqdm = tqdm(embed_loader, leave=True, desc='Training: ')
    embedding_matrix_lstm_train = []
    target_matrix_train = []
    for inputs, seq_len, targets in embed_tqdm:
        inputs = inputs.to(device)
        seq_len = seq_len.to(device)
        attn_mask = create_attention_mask(seq_len, inputs.size(1)).to(device)
        targets = targets.to(device)
        outputs, embed = model(inputs, attn_mask)
        for i in range(len(embed)):
            embedding_matrix_lstm_train.append(embed[i].tolist())
            target_matrix_train.append(targets[i].tolist())

    dataset = NewsDataset(X_test, y_test)
    embed_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    model.train()
    embed_tqdm = tqdm(embed_loader, leave=True, desc='Training: ')
    embedding_matrix_lstm = []
    target_matrix = []
    for inputs, seq_len, targets in embed_tqdm:
        inputs = inputs.to(device)
        seq_len = seq_len.to(device)
        attn_mask = create_attention_mask(seq_len, inputs.size(1)).to(device)
        targets = targets.to(device)
        outputs, embed = model(inputs, attn_mask)
        for i in range(len(embed)):
            embedding_matrix_lstm.append(embed[i].tolist())
            target_matrix.append(targets[i].tolist())
    return embedding_matrix_lstm_train, target_matrix_train, embedding_matrix_lstm, target_matrix

