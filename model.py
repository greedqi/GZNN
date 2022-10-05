from turtle import forward
import numpy as np
from sympy import N
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
from sklearn.metrics import mean_squared_error, r2_score

from parameters import *


def extrac_subseq(data,sebseq_num):
    sebseq_len=data.shape[-2]//(2**(sebseq_num-1))
    if sebseq_len < 1 :
        print("erro:sebseq_len < 1")
    sebseq=[[[] for i in range(data.shape[1])] for i in range(sebseq_num)]
    for i in range(sebseq_num) :
        if i==0:
            sebseq[i]=data[:,2**(sebseq_num-i-1)-1 :2**(sebseq_num-i-1)-1 + sebseq_len*2**(sebseq_num-i-1) :2**(sebseq_num-i-1),:]
        else:
            sebseq[i]=data[:,2**(sebseq_num-i-1)-1 :2**(sebseq_num-i-1)-1 + sebseq_len*2**(sebseq_num-i) :2**(sebseq_num-i),:]
    return sebseq



class MyDataset(Dataset):

    def __init__(self, data, label):
        self.data=data
        self.label=label
    
    def __getitem__(self, index):
        return self.data[index,:,:], self.label[index,:,:]

    #数据集大小
    def __len__(self):
        return self.data.shape[0]

class MyGRU(nn.Module):
    def __init__(self, input_dim,input_dim2, hidden_dim):
        super(MyGRU,self).__init__()

        self.hidden_dim=hidden_dim

        self.Wxr = nn.Parameter(torch.Tensor(input_dim, hidden_dim), requires_grad=True)
        self.Wxm = nn.Parameter(torch.Tensor(input_dim, hidden_dim),requires_grad=True)
        self.Whr = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim), requires_grad=True)
        self.Whm = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim),requires_grad=True)
        self.br = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        self.bm = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        self.Whx = nn.Parameter(torch.Tensor(input_dim, hidden_dim), requires_grad=True)
        self.Whh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim),requires_grad=True)
        self.bh = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)

    def forward(self,x,m,h):
        rt=torch.sigmoid(x@self.Wxr+h@self.Whr+self.br)
        # mt=torch.sigmoid(x@self.Wxm+m@self.Whm+self.bm)

        
        h_r=torch.tanh(x@self.Whx+h@self.Whh+self.bh)
        # h=(1-rt)*h+rt*((1-mt)*h_r+mt*m)
        h=(1-rt)*h_r+rt*m
 
        return h


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRU,self).__init__()

        self.hidden_dim=hidden_dim

        self.Wxr = nn.Parameter(torch.Tensor(input_dim, hidden_dim), requires_grad=True)
        self.Wxz = nn.Parameter(torch.Tensor(input_dim, hidden_dim),requires_grad=True)
        self.Whr = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim), requires_grad=True)
        self.Whz = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim),requires_grad=True)
        self.br = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        self.bz = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        self.Whx = nn.Parameter(torch.Tensor(input_dim, hidden_dim), requires_grad=True)
        self.Whh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim),requires_grad=True)
        self.bh = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)

    def forward(self,x,h):
        rt=torch.sigmoid(x@self.Wxr+h@self.Whr+self.br)
        zt=torch.sigmoid(x@self.Wxz+h@self.Whz+self.bz)
        
        h_r=torch.tanh(x@self.Whx+rt*h@self.Whh+self.bh)
        h=(1-zt)*h+zt*h_r
 
        return h



class GraphZoomNeuralNetwork(nn.Module):
    def get_graph(self,graph_parameter):
        graph_layers_num=len(graph_parameter)
        layer_indexes_len=graph_parameter[-1]*(2**(graph_layers_num-1))
        layer_indexes=np.zeros((layer_indexes_len), dtype=int)
        for i in range(graph_layers_num):
            a=graph_parameter[i]
            temp_list=[]

            if i!=graph_layers_num-1:
                for j in range(a):
                    temp_list.append( 2**i+j*(2**(i+1)) )
            else:
                for j in range(a):
                    temp_list.append( 2**i+j*(2**(i)) )
            for n in temp_list:
                layer_indexes[n-1]=i+1
            


        return np.flipud(layer_indexes)
    
    def __init__(self, input_dim, hidden_dim, output_dim,seb_num):
        super(GraphZoomNeuralNetwork,self).__init__()

        self.seb_num=seb_num
        self.hidden_dim=hidden_dim

        self.GRUh=GRU(hidden_dim,hidden_dim)
        self.GRUx=GRU(input_dim,hidden_dim)
        self.GRU=MyGRU(input_dim,hidden_dim,hidden_dim)

        self.Wx = nn.Parameter(torch.Tensor(input_dim, hidden_dim), requires_grad=True)
        self.Wh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim),requires_grad=True)
        self.Whl = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim),requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)

        self.Wzx =nn.Parameter(torch.Tensor(input_dim, 1), requires_grad=True)
        self.Wzh =nn.Parameter(torch.Tensor(hidden_dim, 1), requires_grad=True)
        self.bz = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.Softmax=nn.Softmax(dim=0)
        self.fc = nn.Linear(hidden_dim*seb_num, output_dim, bias=True)

    def forward(self, X):
        batch_size, seq_len, _ = X.size()  
        
        graph_parameter=[7,7,7,7,7]
        layer_num=len(graph_parameter)
        layer_indexes=self.get_graph(graph_parameter)#长度和输入相同
        seq_len=len(layer_indexes)
        h_layers=[]
        for i in range(len(graph_parameter)):
            h_layers.append(torch.zeros( batch_size, self.hidden_dim).to(X.device))
       
        
        for t in range(seq_len):
            id=layer_indexes[t]
            if id==0:
                continue
            x_t=X[:,t,:]
            if id==layer_num:
                # h_layers[id-1] = self.GRU(x_t,torch.zeros( batch_size, self.hidden_dim).to(X.device),h_layers[id-1])
               h_layers[id-1] = torch.tanh(x_t @ self.Wx + h_layers[id-1] @ self.Wh + torch.zeros( batch_size, self.hidden_dim).to(X.device) @ self.Whl+self.bias)    
            else:
                for j in range(t,-1,-1):
                    if layer_indexes[j] > id:
                        id_u=layer_indexes[j]
                        break
                # h_layers[id-1] = self.GRU(x_t,h_layers[id_u-1],h_layers[id-1])
                z=torch.diag(layer_zoom[id-1,:,0])
                h_z=z@h_layers[id_u-1]
                h_layers[id-1] = torch.tanh(x_t @ self.Wx + h_layers[id-1] @ self.Wh + h_z @ self.Whl+self.bias)    
            

            layer_zoom=torch.ones(layer_num,batch_size,1).to(X.device)/layer_num#保持权重一致，不用zoom的链接的权重要除softmax的个数
            layer_zoom_input_n= np.zeros((layer_num), dtype=int)
            for i in range(t+1,seq_len):
                t_id=layer_indexes[i]
                if t_id !=0:
                    min_id=t_id
                    break
            if min_id>id:
                continue

            
            for i in range(min_id,id+1):
                for j in range(t+1,seq_len):
                    t_id=layer_indexes[j]
                    if t_id ==i:
                        layer_zoom_input_n[i-1]=j
                        break
            # zoom_n=id+1-min_id#当前zoom的数量，即当前h与其他输入链接的数量
            # layer_zoom=torch.ones(layer_num,batch_size,1).to(X.device)*0.4
            # for i in range(min_id,id+1):   #遍历layer_num-1
            #     tt=layer_zoom_input_n[i-1]
            #     x_tt=X[:,tt,:]
            #     layer_zoom[i-1]=torch.sigmoid( x_tt@ self.Wzx +h_layers[id-1] @ self.Wzh+ self.bz)
            for i in range(layer_num):   #遍历layer_num-1
                tt=layer_zoom_input_n[i]
                if tt==0:
                    layer_zoom[i]=torch.tanh(h_layers[i] @ self.Wzh+ self.bz)
                else:
                    x_tt=X[:,tt,:]
                    layer_zoom[i]=torch.tanh( x_tt@ self.Wzx +h_layers[id-1] @ self.Wzh+ self.bz)

            
            layer_zoom = self.Softmax(layer_zoom)


            # layer_zoom[min_id-1:id,:]=layer_zoom_none_zero
            # layer_zoom[min_id-1:id] = self.Softmax(layer_zoom[min_id-1:id,:])
            # for i in range(zoom_n):   #遍历layer_num-1
            #     tt=layer_zoom_input_n[i]
            #     x_tt=X[:,tt,:]
            #     layer_zoom_none_zero[i]=torch.tanh( x_tt@ self.Wzx +h_layers[id-1]  @ self.Wzh+ self.bz)#@ self.Wzh+h_layers[x_id-1]
            
            # layer_zoom_none_zero = self.Softmax(layer_zoom_none_zero)
            # layer_zoom_none_zero = layer_zoom_none_zero*zoom_n/layer_num

            # layer_zoom[min_id-1:id,:]=layer_zoom_none_zero
            # softmax归一化
            # layer_zoom = self.Softmax(layer_zoom)
                    

            



        
        h_out=torch.zeros(batch_size, self.hidden_dim*self.seb_num).to(X.device)
        HD = self.hidden_dim
        for i in range(len(graph_parameter)):
            h_out[:,HD*i:HD*(i+1)]=h_layers[i]
        y_pred=self.fc(h_out)
        return y_pred,h_out
        



class Moduel2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Moduel2,self).__init__()
        self.hidden_dim=hidden_dim

        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias2 = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)
        self.fc2 = nn.Linear(hidden_dim,1 , bias=True)

    def forward(self, X, init_states):
        batch_size, seq_len, _ = X.size()
        h_t = init_states
        y_pred=torch.zeros(batch_size,seq_len).to(X.device)

        HSD = self.hidden_dim
        c_t = torch.zeros(batch_size, self.hidden_dim).to(X.device)

        for t in range(seq_len):
             # 取出当前的值
            x_t = X[:, t, :]

            # 计算门值
            gates = x_t @ self.W + h_t @ self.U + self.bias2

            i_t = torch.sigmoid(gates[:, :HSD])
            f_t = torch.sigmoid(gates[:, HSD:HSD*2])
            g_t = torch.tanh(gates[:, HSD*2:HSD*3])
            o_t = torch.sigmoid(gates[:, HSD*3:])

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            y_pred[:,t:t+1]=self.fc2(h_t)
    
        return y_pred



class Encoder1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder1,self).__init__()

        self.hidden_dim=hidden_dim
        self.input_dim=input_dim
        self.module1=GraphZoomNeuralNetwork(input_dim,hidden_dim,SEQ_LEN,SEB_NUM)
        # self.module1=ModuelGRU(input_dim,hidden_dim,SEQ_LEN,SEB_NUM)
        # self.module1=Moduel1(input_dim,hidden_dim,SEQ_LEN,SEB_NUM)
        self.module2=Moduel2(input_dim,hidden_dim*SEB_NUM)
        stdv = 1.0 / math.sqrt(hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)

    def forward(self, X, init_states=None):
        batch_size, _, _ = X.size()
        seq_len=SEQ_LEN

        y_pred=torch.zeros(batch_size,SEQ_LEN+SEQ2_LEN).to(X.device)
        h_out=torch.zeros(batch_size, self.hidden_dim*SEB_NUM).to(X.device)
        # y_pred=torch.zeros(batch_size,SEQ2_LEN).to(X.device)
        

        X1=X[:,:SEQ_LEN,:]
        X2=X[:,SEQ_LEN:,:]

        y_pred[:,:SEQ_LEN],h_out=self.module1(X1)
        y_pred[:,SEQ_LEN:]=self.module2(X2,h_out)
        
        return y_pred


class Model():
    def __init__(self, X, Y, n_epoch, batch_size, device, seed=1024):
        torch.manual_seed(seed)
        
        self.X,self.Y=X,Y
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

        input_dim=X.shape[-1]
        output_dim=Y.shape[-1]

        self.model=Encoder1(input_dim,HID_DIM,output_dim).to(device)




        self.optimizer = torch.optim.Adam([
            {'params':self.model.module1.parameters(),'lr':LR1},
            {'params':self.model.module2.parameters(),'lr':LR2}],
                                        )

        self.criterion = nn.MSELoss(reduction='mean')

    def predict(self,X):

        self.model.eval()
        
        with torch.no_grad():#不构建计算图

            y=self.model(X)
             # 放上cpu转为numpy
            y = y.cpu().numpy()

        return y

    def fit(self,X,Y,X_test=None,Y_test=None):
        
        dataset= MyDataset(X,Y)

        self.model.train()

        for i in range(self.n_epoch):
            self.model.train()
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            loss_hist=0
            for batch_X, batch_Y in data_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                if batch_Y.ndim != output.ndim :
                    batch_Y=np.squeeze(batch_Y,axis=-1)
                loss = self.criterion(output, batch_Y)
                loss_hist+=loss.item()
                loss.backward()
                self.optimizer.step()
            # print('Epoch:{}, Loss:{}'.format(i + 1, loss_hist))
            if (X_test is not None) & (i%5==0):
                y_pred = self.predict(X=X_test)
                if y_pred.ndim > 2  :
                    y_pred=np.squeeze(y_pred,axis=-1)

                print('Epoch:{}, Loss:{}'.format(i + 1, loss_hist),'测试集的MSE：', mean_squared_error(Y_test, y_pred))
                # print('\n测试集的决定系数：', r2_score(Y_test, y_pred))



        return self
    

        