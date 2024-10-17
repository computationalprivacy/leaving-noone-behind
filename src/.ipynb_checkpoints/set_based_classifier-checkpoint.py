import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch.optim as optim
import torch
import torch.nn as nn

from .data_prep import normalize_cont_cols
from .feature_extractors import apply_ohe
from .distance import compute_distances

torch.set_num_threads(1)
    
# create a dataset loader
class SyntheticShadowDataLoader(torch.utils.data.Dataset):
    
    def __init__(self, X_train, y_train):
        '''
        X_train: we expect a list of pandas df, each of which corresponds to a synthetic dataset
        y_train: the corresponding labels for membership
        '''
        self.X_train = X_train
        self.y_train = y_train
        
    def preprocess(self, method: str, target_record_ohe_values: np.array, top_X: int,
                          categorical_cols: list, continuous_cols: list,
                          ohe_cat_indices: list, continous_indices: list):
        '''
        Now we wish to transform each pandas df into the a list of unique records with frequency
        '''
        results = []
        
        if method == 'all':
            for df in self.X_train:
                value_counts = df.value_counts()
                one_results = [list(value_counts.index.values[i]) + [value_counts.values[i]] for i in range(len(value_counts))]

                # do the padding
                for i in range(len(df) - len(one_results)):
                    one_results.append([0] * len(one_results[0]))
                results.append(one_results)

        elif method == 'top_closest':
            
            for df in self.X_train:
                value_counts = df.value_counts()

                # determine the distance
                all_values = np.array([np.array(value_counts.index.values[i]) for i in range(len(value_counts))])
                distances = compute_distances(record=target_record_ohe_values, values=all_values,
                                                   ohe_cat_indices=ohe_cat_indices, continous_indices=continous_indices,
                                                   n_cat_cols=len(categorical_cols), n_cont_cols=len(continuous_cols))

                # we want the closest records, so the records for which the distance is the smallest
                sorted_idx = np.argsort(distances)[:top_X]

                one_results = [list(value_counts.index.values[i]) + [value_counts.values[i], distances[i]] for i in sorted_idx]
                
                # add padding if needed
                if len(one_results) <= top_X:
                    for i in range(top_X - len(one_results)):
                        one_results.append([0] * len(one_results[0]))
                
                results.append(one_results)
            
        self.value_freq = np.array(results)
        return None

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        prepped_data = self.value_freq[index]
        
        label = self.y_train[index]
        
        return (
            torch.tensor(prepped_data, dtype=torch.float), torch.tensor(label, dtype=torch.float).reshape(1,),
        )   
    
# some modular building blocks
class BaseMLP(nn.Module):

    def __init__(self, num_in_features, hidden_size, num_out_features):
        super(BaseMLP, self).__init__()
        self.mlp = nn.Sequential(\
                    nn.Linear(num_in_features, hidden_size),\
                    nn.ReLU(), \
                    nn.Linear(hidden_size, num_out_features))

    def forward(self, features):                                
        return self.mlp(features)    

class SetBasedClassifier(nn.Module):
    
    def __init__(self, num_in_features, embedding_hidden_size, embedding_size, prediction_hidden_size, 
                dropout_rate = 0.1):
        super(SetBasedClassifier, self).__init__()
        self.num_in_features = num_in_features
        self.embedding_hidden_size = embedding_hidden_size
        self.embedding_size = embedding_size
        self.prediction_hidden_size = prediction_hidden_size
        self.use_cuda = False

        self.embed = BaseMLP(self.num_in_features, self.embedding_hidden_size, self.embedding_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.emb_to_predict = BaseMLP(self.embedding_size, self.prediction_hidden_size, 1)

    def forward(self, x):
        
        # x has size N_unique_values x (N_cols + 1)
        # where the number of rows can be different across datasets, but not the number of cols
        embedding = self.embed(x)
        embedding = self.dropout(embedding)
        
        # take the mean embedding
        mean_embedding = torch.mean(embedding, dim = 1)
        
        # last layer
        last_layer = self.emb_to_predict(mean_embedding)
        
        return last_layer
    
class SetBasedClassifier_w_Attention(nn.Module):
    
    def __init__(self, num_in_features, embedding_hidden_size, embedding_size, attention_size, n_records,
                 prediction_hidden_size, dropout_rate = 0.1):
        
        super(SetBasedClassifier_w_Attention, self).__init__()
        self.num_in_features = num_in_features
        self.embedding_hidden_size = embedding_hidden_size
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.n_records = n_records
        self.prediction_hidden_size = prediction_hidden_size
        self.use_cuda = False

        self.embed = BaseMLP(self.num_in_features, self.embedding_hidden_size, self.embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # attention 
        self.query = nn.Linear(self.embedding_size, self.attention_size, bias = False)
        self.key = nn.Linear(self.embedding_size, self.attention_size, bias = False)
        self.value = nn.Linear(self.embedding_size, self.attention_size, bias = False)
        
        # aggregate record embeddings to dataset embedding with linear combination
        self.agg = nn.Linear(self.n_records, 1, bias = False)
        
        self.emb_to_predict = BaseMLP(self.attention_size, self.prediction_hidden_size, 1)

    def forward(self, x):
        
        # x has size N_unique_values x (N_cols + 1)
        # where the number of rows can be different across datasets, but not the number of cols
        embedding = self.embed(x)
        embedding = self.dropout(embedding)
        
        # attention 
        # compute keys, query, value, 
        # all of shape batch_size x n_records x attention_size
        queries = self.query(embedding)
        keys = self.key(embedding)
        values = self.value(embedding)
                
        # compute attention scores
        # should be of shape batch_size x n_records x n_records
        attn_scores = queries @ torch.transpose(keys, 1, 2)
        attn_scores_softmax = torch.softmax(attn_scores, dim=-1)
        
        # multiply attention scores with values
        # should be of shape batch_size x n_records x attention_size
        weighted_values = attn_scores_softmax @ values
        
        # get the dataset embedding
        # first transpose the weighted values so we have batch_size x attention_size x n_records
        embedding_reshaped = torch.transpose(weighted_values, 1, 2)
        # then pass through a linear layer with weights n_records x 1
        # to get dataset-level embedding of shape batch_size x attention_size (x 1)
        agg_embedding = self.agg(embedding_reshaped).squeeze()
        
        # take this dataset level embedding and pass it through the prediction layers
        last_layer = self.emb_to_predict(agg_embedding)
        
        return last_layer
    
class SetBasedClassifier_w_TargetAttention(nn.Module):
    
    def __init__(self, target_record, num_in_features, embedding_hidden_size, embedding_size, attention_size, n_records,
                 prediction_hidden_size, dropout_rate = 0.1):
        
        super(SetBasedClassifier_w_TargetAttention, self).__init__()
        self.target_record = target_record
        self.num_in_features = num_in_features
        self.embedding_hidden_size = embedding_hidden_size
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.n_records = n_records
        self.prediction_hidden_size = prediction_hidden_size
        self.use_cuda = False

        self.embed = BaseMLP(self.num_in_features, self.embedding_hidden_size, self.embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # attention 
        self.query_target = nn.Linear(self.embedding_size, self.attention_size, bias = False)
        self.key = nn.Linear(self.embedding_size, self.attention_size, bias = False)
        self.value = nn.Linear(self.embedding_size, self.attention_size, bias = False)
        
        # prediction
        self.emb_to_predict = BaseMLP(self.attention_size, self.prediction_hidden_size, 1)

    def forward(self, x):
        
        # x has size N_unique_values x (N_cols + 1)
        # where the number of rows can be different across datasets, but not the number of cols
        embedding = self.embed(x)
        embedding = self.dropout(embedding)
        
        embedding_target = self.embed(self.target_record)
        
        # attention 
        # compute query of target record of shape 1 x attention_size
        query_target = self.query_target(embedding_target)

        # compute keys and values of all other records
        # all of shape batch_size x n_records x attention_size
        keys = self.key(embedding)
        values = self.value(embedding)
                
        # compute attention scores
        # should be of shape batch_size x n_records x 1
        attn_scores = query_target @ torch.transpose(keys, 1, 2)
        attn_scores_softmax = torch.softmax(attn_scores, dim=-1)
        
        # multiply attention scores with values
        # should be of shape batch_size x 1 x attention_size
        weighted_values = attn_scores_softmax @ values
        
        # take this dataset level embedding and pass it through the prediction layers
        last_layer = self.emb_to_predict(weighted_values.squeeze())
        
        return last_layer
    
def save_model(epoch, model, optimizer, loss, path):
    
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path)
    
    return None  
  
def train_model(train_loader, val_loader, model_type, target_record_tensor, model_params, n_epochs: int = 500, verbose = False,
               path_best_model = './best_model.pt'):
    all_train_losses, all_train_acc = [], []
    all_val_losses, all_val_acc = [], []
    lowest_val_loss = 100
    
    if model_type == 'Attention':
        model = SetBasedClassifier_w_Attention(**model_params)
    elif model_type == 'TargetAttention':
        model = SetBasedClassifier_w_TargetAttention(target_record_tensor, **model_params)
    else:
        model = SetBasedClassifier(**model_params)
    
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: ', n_trainable_params)
    
    criterion = nn.BCELoss()
    
    optimizer = optim.Adamax(model.parameters(), lr=0.001)
        
    #for epoch in tqdm(range(n_epochs)):
    for epoch in range(n_epochs):
        model.train()
        
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            sigmoid_outputs = torch.sigmoid(outputs)
            loss = criterion(sigmoid_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if epoch % 5 == 0:
            model.eval()
            training_losses, training_acc = [], []
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                outputs = model(inputs)
                sigmoid_outputs = torch.sigmoid(outputs)
                loss = criterion(sigmoid_outputs, labels)
                training_losses.append(loss.item())
                accuracy = sum((sigmoid_outputs>0.5).float() == labels) / len(labels)
                training_acc.append(accuracy.item())
            
            all_train_losses.append(np.mean(training_losses))
            all_train_acc.append(np.mean(training_acc))
            
            val_losses, val_acc = [], []
            for i, val_data in enumerate(val_loader, 0):
                val_inputs, val_labels = val_data
                val_outputs = model(val_inputs)
                val_sigmoid_outputs = torch.sigmoid(val_outputs)
                loss = criterion(val_sigmoid_outputs, val_labels)
                val_losses.append(loss.item())
                accuracy = sum((val_sigmoid_outputs>0.5).float() == val_labels) / len(val_labels)
                val_acc.append(accuracy.item())
            
            all_val_losses.append(np.mean(val_losses))
            all_val_acc.append(np.mean(val_acc))
            
            if np.mean(val_losses) <= lowest_val_loss:
                lowest_val_loss = np.mean(val_losses)
                save_model(epoch, model, optimizer, lowest_val_loss, path = path_best_model)
            
            if verbose:
                print('Training loss: ', np.mean(training_losses))
                print('Training accuracy: ', np.mean(training_acc))
                print('Validation loss: ', np.mean(val_losses))
                print('Validation accuracy: ', np.mean(val_acc))
                    
    return model, all_train_losses, all_train_acc, all_val_losses, all_val_acc

def validate_model(model_type, path_best_model, target_record_tensor, model_params, train_loader, test_loader, criterion = nn.BCELoss()):
        
    if model_type == 'Attention':
        model = SetBasedClassifier_w_Attention(**model_params)
    elif model_type == 'TargetAttention':
        model = SetBasedClassifier_w_TargetAttention(target_record_tensor, **model_params)
    else:
        model = SetBasedClassifier(**model_params)
        
    checkpoint = torch.load(path_best_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    training_losses, train_pred_proba, train_y_true = [], [], []
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        outputs = model(inputs)
        sigmoid_outputs = torch.sigmoid(outputs)
        loss = criterion(sigmoid_outputs, labels)
        training_losses.append(loss.item())
        train_pred_proba += [k.item() for k in sigmoid_outputs.detach().numpy()]
        train_y_true += [k.item() for k in labels.detach().numpy()]
    
    train_accuracy = sum((np.array(train_pred_proba)>0.5) == train_y_true) / len(train_y_true)
    print('Training loss: ', np.mean(training_losses))
    print('Training accuracy: ', train_accuracy)
    try:
        train_auc = roc_auc_score(train_y_true, train_pred_proba)
        print('Training AUC: ', train_auc)
    except:
        train_auc=0
        print('Only one class')
            
    test_losses, test_pred_proba, test_y_true = [], [], []
    for i, test_data in enumerate(test_loader, 0):
        test_inputs, test_labels = test_data
        test_outputs = model(test_inputs)
        if test_outputs.shape != test_labels.shape:
            test_outputs = test_outputs.reshape(test_labels.shape[0], test_labels.shape[1])
        test_sigmoid_outputs = torch.sigmoid(test_outputs)
        loss = criterion(test_sigmoid_outputs, test_labels)
        test_losses.append(loss.item())
        test_pred_proba += [k.item() for k in test_sigmoid_outputs.detach().numpy()]
        test_y_true += [k.item() for k in test_labels.detach().numpy()]
        
    #This only test if we are in full_synthetic mode
    if int(sum(test_y_true)) == len(test_y_true) or int(sum(test_y_true)) == 0 :
        test_pred_proba_avg = sum(test_pred_proba)/len(test_pred_proba) 
        prediction = (test_pred_proba_avg > 0.5)
        test_accuracy = int((prediction == test_y_true[0]))
    else :
        #all other cases
        test_accuracy = sum((np.array(test_pred_proba)>0.5) == test_y_true) / len(test_y_true)
    print('Test loss: ', np.mean(test_losses))
    print('Test accuracy: ', test_accuracy)
    try:
        test_auc = roc_auc_score(test_y_true, test_pred_proba)
        print('Test AUC: ', test_auc)
    except:
        test_auc=0
        print('Only one class')
    
    return np.mean(training_losses), train_accuracy, train_auc, np.mean(test_losses), test_accuracy, test_auc

def fit_set_based_classifier(all_datasets_train: list, datasets_test: list,
                             all_labels_train: list, labels_test: list, 
                             model_params: dict, 
                             target_record: pd.DataFrame, 
                             ohe: OneHotEncoder, categorical_cols: list, ohe_column_names: list,
                             continuous_cols: list, meta_data: list, df_aux: pd.DataFrame,
                             path_best_model : str,
                             validation_size: float = 0.1, batch_size: int = 20,
                             model_type: str = 'TargetAttention', top_X: int = 1000) -> tuple:

    # preprocess the datasets
    for i, df in enumerate(all_datasets_train):
        # Note that we previously only normalized the float columns,
        # here we also want to normalize the integer columns
        df_normalized_ints = normalize_cont_cols(df.copy(), meta_data, df_aux=df_aux, types = ('Integer',))
        all_datasets_train[i] = apply_ohe(df_normalized_ints, ohe, categorical_cols, ohe_column_names, continuous_cols)

    for i, df in enumerate(datasets_test):
        df_normalized_ints = normalize_cont_cols(df.copy(), meta_data, df_aux=df_aux, types = ('Integer',))
        datasets_test[i] = apply_ohe(df_normalized_ints, ohe, categorical_cols, ohe_column_names, continuous_cols)

    # do the same for the target record
    target_record_normalized_ints = normalize_cont_cols(target_record.copy(), meta_data, df_aux=df_aux, types= ('Integer',))
    target_record_ohe = apply_ohe(target_record_normalized_ints, ohe, categorical_cols, ohe_column_names, continuous_cols)

    target_record_ohe_values = target_record_ohe.values[0]
    all_columns = list(target_record_ohe.columns)
    ohe_cat_indices = [all_columns.index(col) for col in ohe_column_names]
    continous_indices = [all_columns.index(col) for col in continuous_cols]

    # prepare all train, val and test data
    datasets_train, datasets_val, labels_train, labels_val = train_test_split(all_datasets_train, all_labels_train,
                                                                  test_size=validation_size)
    train_data = SyntheticShadowDataLoader(datasets_train, labels_train)
    train_data.preprocess(method = 'top_closest', target_record_ohe_values = target_record_ohe_values, top_X = top_X,
                          categorical_cols=categorical_cols, continuous_cols=continuous_cols,
                          ohe_cat_indices=ohe_cat_indices, continous_indices=continous_indices)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = SyntheticShadowDataLoader(datasets_val, labels_val)
    val_data.preprocess(method = 'top_closest', target_record_ohe_values = target_record_ohe_values, top_X = top_X,
                          categorical_cols=categorical_cols, continuous_cols=continuous_cols,
                          ohe_cat_indices=ohe_cat_indices, continous_indices=continous_indices)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    test_data = SyntheticShadowDataLoader(datasets_test, labels_test)
    test_data.preprocess(method = 'top_closest', target_record_ohe_values = target_record_ohe_values, top_X = top_X,
                          categorical_cols=categorical_cols, continuous_cols=continuous_cols,
                          ohe_cat_indices=ohe_cat_indices, continous_indices=continous_indices)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # define the target record tensor - only needed for the target attention though
    target_record_tensor_values = list(target_record_ohe_values) + [1, 1] # add multiplicity 1 and similarity 1
    target_record_tensor = torch.tensor(target_record_tensor_values, dtype=torch.float).reshape(1, len(target_record_tensor_values))
    
    # train the model
    model, all_train_losses, all_train_acc, all_val_losses, all_val_acc = train_model(train_loader, val_loader, 
                                                                       model_type, target_record_tensor, model_params,
                                                                       path_best_model = path_best_model, 
                                                                       verbose = False)

    training_loss, train_accuracy, train_auc, test_loss, test_accuracy, test_auc = validate_model(model_type, path_best_model, 
                                                                      target_record_tensor, model_params, train_loader, test_loader)
    
    return [model], [(train_accuracy, train_auc, test_accuracy, test_auc)]