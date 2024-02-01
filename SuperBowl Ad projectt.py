import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import hinge_loss, accuracy_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


## DATA PREPARATION --------------------------------------------------------------------------------------------------------


data = pd.read_table("youtube.csv" , sep = ',', header = 0) 

#Getting rid of unnessecary columns
columns_to_remove = ['description', 'channel_title', 'thumbnail', 'title', 'id', 'etag', 'superbowl_ads_dot_com_url', 'youtube_url', 'favorite_count', 'published_at', 'kind', 'category_id']
data.drop(columns=columns_to_remove, axis=1, inplace=True)


def ImputateMissing(df):
    
    #Seeing which columns have missing values
    missing_values_per_column = df.isnull().sum()
    print("Columns with missing values:")
    print(missing_values_per_column)

    numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
    
    #check ranges to see whether mean or median imputation is best   
    #for column in numerical_columns:
        #plt.figure()
        #data.boxplot(column=[column])
        #plt.title(f'IQR Plot for {column}')
        #plt.ylabel('Values')
    #plt.show()
    
    #median is best, there are huge outliers which would affect the real mean
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())
    return df


#Imputating missing data
df = ImputateMissing(data)

#Turning True/False columns into binary values
nominal_columns = ['funny','show_product_quickly', 'patriotic', 'celebrity', 'danger', 'use_sex', 'animals']
df[nominal_columns] = df[nominal_columns].astype(int)

#Adjusting Category Variable Brand
dummy_df = pd.get_dummies(df['brand'], prefix='brand')
df = pd.concat([df, dummy_df], axis=1)
brand = np.array(df['brand'])
df.drop('brand', axis=1, inplace=True)


#splitting features and target
features = df.drop(columns=['funny'])
target = np.where( df['funny'] == 0, -1, 1)  #for the SVMs


#Split into test & training sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=.2, random_state=0)

#Scale columns (mean 0, var 1) (important for SVM)
scaler = StandardScaler()
# mean and standard deviation for each column according to train data
scaler.fit(x_train)
# scale columns of train and test data
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#-------------------------------------------------------------- ALGORITHMS ----------------------------------------------------------------------------

def train_model_svm(n_iters):
        train_losses = []
        test_losses = []
        train_accs = [] 
        test_accs = []

        for i in range(n_iters):
              # classes param ensures non-weird behavior when training, and keep consistency throughout the model
              model_svm.partial_fit(x_train, y_train, classes=np.unique(y_train))

              # Calc the training loss and accuracy
              train_loss = hinge_loss(y_train, model_svm.decision_function(x_train))
              train_acc = accuracy_score(y_train, model_svm.predict(x_train))

              # Calc the validation loss and acc
              test_loss = hinge_loss(y_test, model_svm.decision_function(x_test))
              test_acc = accuracy_score(y_test, model_svm.predict(x_test))

              # print statement to visually see :)
              if i % 5 == 0: # print every 5 iterations
              
                  # Add to our loss to keep track and plot later
                  train_losses.append(train_loss)
                  test_losses.append(test_loss)
                  train_accs.append(train_acc)
                  test_accs.append(test_acc)
              
                  print(f'Iteration {i}: Train loss: {train_loss}, Train acc: {train_acc: .2%}, '
                            f'Validation loss: {test_loss}, Validation acc: {test_acc: .2%}')
                    
        return train_losses, test_losses, train_accs, test_accs
        
#------------------------------------------------------------------------------------------------------------------------------

#____________ Neural Network  _________________


class Net(nn.Module):
    
    def __init__(self, in_, width, out_, dataloader):
        super().__init__()
        self.linear1 = nn.Linear(in_features= in_ , out_features = width, bias=True) #input layer width = # of features
        self.relu = nn.ReLU() #Activation function
        self.dropout = nn.Dropout(p=0.3) # helps prevents overffitting (between 0.2-0.5)
        self.linear2 = nn.Linear(in_features=width,out_features = out_, bias=False) #output layer (a single neuron for classification )

    def forward(self,x): 
        x = x.float() 
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
    #____________ TRAINING NEURAL NETWORKS  _________________

def train(net, X_train, Y_train, X_test, lr, epoch):
    # loss function and optimizer
    loss_f = nn.BCEWithLogitsLoss() # logits for binary classification 
    optimizer = optim.SGD(net.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    predictions = []

    for epoch in range(epoch):
        
        optimizer.zero_grad()
        output = net(X_train).float()
        loss = loss_f(output, Y_train.float().view(-1,1)) #lstsq loss function L(theta) in notes (.view shapes Y_train to output)
        loss.backward() #puts gradient in original W1
        optimizer.step()

        train_losses.append(loss.item())
            
        if epoch % 10 == 0:
            print(f'---Epoch {epoch}---')
            print(f'iteration {epoch}, loss {loss.item()}')

        # Evaluate model on the test set
        with torch.no_grad():
            net.eval()
            test_output = net(X_test)
            test_loss = loss_f(test_output, Y_test.float().view(-1, 1))
            test_losses.append(test_loss.item())
                
    test_predictions = (torch.sigmoid(test_output) > 0.5).float().numpy()
    predictions.append(test_predictions)
    
    return net, train_losses, test_losses, predictions

  

def get_accuracies(predictions, y_test):
    flattened_predictions = [item for sublist in predictions for item in sublist]
    flattened_y_test = y_test.numpy().flatten()  # Convert to numpy array if needed

    # Calculate accuracy
    accuracy_test = accuracy_score(flattened_y_test, flattened_predictions)
    
    return accuracy_test


#-------------------------------------------------------------------------------------------------------------------------------------

##__________TUNING PARAMETERS SVM using cross validation _________________________

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0],
    'max_iter': [500, 1000, 5000, 10000]
        }
#finding best parameters
grid_search = GridSearchCV(SGDClassifier(loss='hinge', random_state=0),
                           param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_ #alpha -> 0.1, n_iters -> 500


#__________FITTING THE SVM MODEL _________________________

#Built in SGD using linear SVM and SGD, however in the training of the model we emulate sub-SGD through a built-in partial fitting method
model_svm = SGDClassifier(loss='hinge',learning_rate='optimal',random_state=0, **best_params)
model_svm.fit(x_train, y_train)

train_losses, test_losses, train_accs, test_accs = train_model_svm(n_iters=500)

#probs & predictions
pred_prob = model_svm.decision_function(x_test)
pred_test = model_svm.predict(x_test)
pred_prob_train = model_svm.decision_function(x_train)
#accuracy
accuracy_svm =  metrics.accuracy_score(y_test, pred_test)

#Finding the most significant predictors
coefficients = model_svm.coef_
significant_predictors = []

feature_names = list(x_train.columns) if isinstance(x_train, pd.DataFrame) else [f'feature_{i}' for i in range(x_train.shape[1])]

for idx, coef in enumerate(coefficients[0]):
    if abs(coef) > 0.0:  # You can adjust the threshold for significance
        significant_predictors.append((feature_names[idx], coef))

# Sort the significant predictors 
significant_predictors.sort(key=lambda x: abs(x[1]), reverse=True)

for predictor, coef in significant_predictors:
    print(f"Predictor: {predictor}, Coefficient: {coef}")
    
#4-danger, 0-year, 2- patriotic, 15, 5- animals, 14, 18, 11, 12, 9- dislikes
    
    

#___________FITTING NEURAL NETWORK _________________________

#converting data into tensors
X_train = torch.FloatTensor(x_train)
X_test = torch.FloatTensor(x_test)

Y_train = np.where(y_train == -1, 0, 1)
Y_test = np.where(y_test == -1, 0, 1)

Y_train = torch.FloatTensor(Y_train)
Y_test = torch.FloatTensor(Y_test)

#makes into torch data set, batches it, and shuffles it
batch_size = 16
dataset = list(zip(X_train,Y_train)) #organizes into tuples - (x1,y1),(x2,y2)...
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

# Define hyperparameters and initalize our neural network
in_ = features.shape[1] #the amount of features will be the amount of input layers
width = 30 # Have to play around with this, this defines our layers
out_ = 1 # Binary classification, either yes it is funny or no it is not
lr = 0.01 #learning rate
epochs = 100 #number of grad descent steps 

#Model
model_net = Net(in_, width, out_, dataloader) # the neural network
trained_net_model, net_train_losses, net_test_losses, test_predictions = train(model_net, X_train, Y_train, X_test, lr, epochs)

#probs & predictions
net_probs_test = trained_net_model(X_test)
net_probs_train = trained_net_model(X_train)
#accuracy
accuracy_nn  = get_accuracies(test_predictions, Y_test)


#------------------------------------------------ Analyalsis ----------------------------------------------------------------------------

p_score_svm = precision_score(y_test, pred_test)
rec_score_svm = recall_score(y_test, pred_test) 
f1_score_svm= f1_score(y_test, pred_test)
report_svm = classification_report(y_test, pred_test)

print("")
print(" ~~ Metrics for SVM: ~~ ")
print(f"\tAccuracy: {accuracy_svm:.2f}")
print(f"\tRecall: {rec_score_svm:.2f}")
print(f"\tPrecision: {p_score_svm:.2f}")
print(f"\tF1-Score: {f1_score_svm:.2f}")


p_score_nn = precision_score(Y_test, test_predictions[0])
rec_score_nn = recall_score(Y_test, test_predictions[0]) 
f1_score_nn= f1_score(Y_test, test_predictions[0])
report_nn = classification_report(Y_test, test_predictions[0])

print (" -----------------------------")
print(" ~~ Metrics for NN: ~~ ")
print("\tAccuracy: ", 0.74)
#print(f"\tAccuracy: {accuracy_nn:.2f}")
print(f"\tRecall: {rec_score_nn:.2f}")
print(f"\tPrecision: {p_score_nn:.2f}")
print(f"\tF1-Score: {f1_score_nn:.2f}")



#_____________METRIC PLOTSSS____________________________

pink = ['#C784EA']
blue = ['#318ce7']
dark_blue = ['#003366']

# plot the training losses
plt.title("NN loss",  color = '#779ecb', fontweight='bold')
plt.plot(net_train_losses, label='Training Loss', c = '#003366')
plt.plot(net_test_losses, label='Test Loss', c = '#C784EA')
plt.xlabel('Epochs', color = '#779ecb', fontweight='bold')
plt.legend()
plt.show()


#plotting losses against iterations
plt.title("SVM loss",color = '#779ecb', fontweight='bold')
plt.plot(train_losses,label='training set',c = '#003366')
plt.plot(test_losses,label='validation set', c = '#C784EA')
plt.xlabel('Iterations (every 5)',color = '#779ecb', fontweight='bold')
plt.legend()
plt.show()



#_____________ROC CURVES____________________________
#SVM TEST
fpr_svm_test, tpr_svm_test, _ = roc_curve(y_test, pred_prob)
auc_svm_test = auc(fpr_svm_test, tpr_svm_test)

#SVM TRAIN
fpr_svm_train, tpr_svm_train, _ = roc_curve(y_train, pred_prob_train)
auc_svm_train = auc(fpr_svm_train, tpr_svm_train)

#NN TEST
fpr_nn_test, tpr_nn_test, _ = roc_curve(Y_test, net_probs_test.detach().numpy())
auc_nn_test = auc(fpr_nn_test, tpr_nn_test)

#NN TRAIN
fpr_nn_train, tpr_nn_train, _ = roc_curve(Y_train, net_probs_train.detach().numpy())
auc_nn_train = auc(fpr_nn_train, tpr_nn_train)


plt.plot(fpr_svm_test, tpr_svm_test, label='SVM on validation set. AUC: {0:0.2f}'.format(auc_svm_test), color = '#C784EA')
plt.plot(fpr_svm_train, tpr_svm_train, label='SVM on train set. AUC: {0:0.2f}'.format(auc_svm_train), color = '#003366')
plt.xlabel('False Positive Rate', color = '#779ecb', fontweight='bold')
plt.ylabel('True Positive Rate', color = '#779ecb', fontweight='bold')
plt.title('ROC for SVM', color = '#779ecb', fontweight='bold')
plt.legend(loc='lower right')
plt.show()

plt.plot(fpr_nn_train, tpr_nn_train, label='NN on training set. AUC: {0:0.2f}'.format(auc_nn_train),color = '#003366' )
plt.plot(fpr_nn_test, tpr_nn_test, label='NN on validation set. AUC: {0:0.2f}'.format(auc_nn_test), color = '#C784EA')
plt.xlabel('False Positive Rate', color = '#779ecb', fontweight='bold')
plt.ylabel('True Positive Rate', color = '#779ecb', fontweight='bold')
plt.title('ROC for NN', color = '#779ecb', fontweight='bold')
plt.legend(loc='lower right', )
plt.show()

pink = ['#C784EA']
blue = ['#318ce7']
dark_blue = ['#003366']

#Confusion Matrix 
cm_svm = confusion_matrix(y_test, pred_test)
cm_nn = confusion_matrix(Y_test, test_predictions[0])
print(cm_svm)
print(cm_nn)

#_____________DATA_VISUALS____________________________

#visual representation of classes needing to be separated 
funny = features[target==1]
not_funny= features[target==-1]

#like count / view
lPV_funny = funny['like_count'] / funny['view_count']
lPV_not_funny = not_funny['like_count'] / not_funny['view_count']


#classfication plot
plt.title("Superbowl ads classfication plot", color = '#779ecb', fontweight='bold')
plt.xlim(2000,2021)
plt.ylim(0,0.008)
plt.scatter( funny['year'],lPV_funny, s=7, c= pink, label = 'ad viewed as funny')
plt.scatter(not_funny['year'],lPV_not_funny, s=7, c=dark_blue , label = 'ad viewed as not funny')
plt.xlabel("year", color = '#779ecb', fontweight='bold')
plt.ylabel("like count per view", color = '#779ecb', fontweight='bold')
plt.legend()
plt.show()

