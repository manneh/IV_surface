import torch
from nets import MultiOptionVolatility, SingleOptionVolatility
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
import os
from params import MODEL_PARAMS, TRAINING_PARAMS
from sklearn.metrics import r2_score as r2
from torch.utils.tensorboard import SummaryWriter


data = pd.read_csv(PATH_TO_DATA)

data = data.drop_duplicates()

data = data[data['v']>0]
for col in ['m', 't']:
    data[col].fillna(data[col].mean(), inplace=True)
    mn = data[col].min()
    mx = data[col].max()
    data[col] = (data[col] - mn)/(mx-mn)

train, test = train_test_split(data, random_state=23)

train_x = torch.Tensor(train[['m', 't']].values)
train_y = torch.Tensor(train[['v']].values)

train_x = train_x.reshape(train_x.shape[0], 1, 2)
train_y = train_y.reshape(train_y.shape[0], 1)

test_x = torch.Tensor(test[['m', 't']].values)
test_x = test_x.reshape(test_x.shape[0], 1, 2)
test_y = torch.Tensor(test[['v']].values)




learning_rate = TRAINING_PARAMS['learning_rate']
n_epoch = TRAINING_PARAMS['epochs']
test_display_step = TRAINING_PARAMS['display_step']
writing_step = TRAINING_PARAMS['writing_step']
weight_decay=TRAINING_PARAMS['weight_decay']

singlenet = SingleOptionVolatility(MODEL_PARAMS['hidden_sizes_list'], 2)
single_optimizer = torch.optim.Adam(singlenet.parameters(), lr=learning_rate, weight_decay=weight_decay)
best_model_wts = copy.deepcopy(singlenet.state_dict())
best_loss = 1000.0
model_saving_path = TRAINING_PARAMS['model_saving_path']

msec = TRAINING_PARAMS['mse_coef']
mslec = TRAINING_PARAMS['mlse_coef']
mspec = TRAINING_PARAMS['mspe_coef']
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
criterion3 = nn.MSELoss()


os.makedirs(model_saving_path, exist_ok=True)


gpu = torch.cuda.is_available()
if gpu:
    singlenet = singlenet.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()


writer = SummaryWriter(TRAINING_PARAMS['writing_dir'])
for epoch in range(n_epoch):
    output = singlenet(train_x)
    singlenet.train()
    single_optimizer.zero_grad()

    mse = msec * criterion1(train_y, output)
    mspe = mspec * criterion3(train_y/train_y, output/train_y)
    msle = mslec * criterion2(torch.log(train_y), torch.log(output))
    loss =  mse + mspe + mslec


    loss.backward()
    single_optimizer.step()
    if epoch % writing_step == 0:
        singlenet.eval()
        test_output = singlenet(test_x)
        test_mse = msec * criterion1(test_y, test_output)
        test_mspe = mspec * criterion3(train_y/train_y, output/train_y)
        test_msle = mslec * criterion2(torch.log(test_y),
                                       torch.log(test_output))
        test_loss = test_mse + test_mspe + test_msle
        writer.add_scalar('Loss/val', test_loss, epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(singlenet.state_dict())
            torch.save(best_model_wts, f"{model_saving_path}/singlemodel{epoch}.pt")
    writer.add_scalar('Loss/train', loss, epoch)

    if epoch % test_display_step == 0:
        # print("Train loss in epoch", epoch, "is", loss.item())
        print("EPOCH", epoch, " "*(7 - len(str(epoch))),
              "|", f"{test_loss.item()}"[:10], "|" f"{test_mse}"[:10],
              "|" f"{test_mspe}"[:10], "|", 
              f"{test_msle}"[:10], "|", str(r2(test_y.detach().numpy(), test_output.detach().numpy()))[:10], "|", loss.item())


torch.save(best_model_wts, os.path.join(model_saving_path, "final.pt"))
