#Fecha de creación: 17 de junio del 2021
#Autor: Carlos Emilio Camacho Lorenzana



#########################Librerías de Pytorch###############################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.autograd import Variable

#############################################################################################

##########################Liberías complementarias###########################################
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import namedtuple
import json
import itertools
from itertools import product
from IPython import display
import pytorch_model_summary as pms
import time
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import networkx as nx
from torchviz import make_dot
from itertools import chain
##############################################################################################
"""
La siguientes funciones se codificaron con la finalidad de que el usuario pueda tener una experiencia similar a la que se tiene con trabajar
redes neuronales en Keras. 
"""




############################################################################################################
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()

"""
Una desventaja de Pytorch es que no tiene incluida su propia función para calcular el Accuracy. Por eso debemos agregar esta función para 
más adelante calcular
"""
############################################################################################################
class RunBuilder():
    @staticmethod 
    def get_runs(params):
        
        Run=namedtuple('Run', params.keys())
        
        runs=[]
        for v in product(*params.values()):
            runs.append(Run(*v))
        
        return runs
"""
Esta clase nos puede servir si deseamos realizar un barrido de parámetros en nuestro entrenamiento. 
"""
############################################################################################################
class RunManager():
    def __init__(self):
        #el método __init__ se encarga de inicializar los atributos correspondientes a esta clase
        self.epoch_count=0
        self.epoch_loss=0
        self.epoch_num_correct=0
        self.epoch_val_loss=0 #lo añadí
        self.epoch_val_num_correct=0 #lo añadí
        self.epoch_start_time=None
        
        self.run_params=None
        self.run_count=0
        self.run_data=[]
        self.run_start_time=None
        
        self.network=None
        self.loader=None
        #self.tb=None #Comando de Tensorboard
    
    def begin_run(self, run, network, loader, val_loader):
        #el argumento self lo que hace es pasarle a este método como parámetros todas la variables
        #que inicializamos previamente en __init__ con el prefijo self.
        
        #Este método inicializa una corrida con los hiperparámetros seleccionados en RunBuilder()
        self.run_start_time = time.time()
        
        self.run_params=run # ya sabemos que RunBuilder nos dará esta variable en algún paso del entrenamiento
        self.run_count += 1
        
        self.network=network #es la red que vamos a construir más adelante en nuestro notebook
        self.loader = loader #es el DataLoader que haremos con el conjunto de entrenamiento
        self.val_loader = val_loader  #es el DataLoader que haremos con el conjunto de validación
        ############ Implmenetación de Tensorboard #####################################################
        #self.tb =SummaryWriter(comment =f'-{run}')
        
        #images, labels =next(iter(self.loader))
        #grid = torchvision.utils.make_grid(images)
        
        #self.tb.add_image('images', grid)
        #self.tb.add_graph(
        #    self.network, 
        #    images.to(getattr(run, 'device', 'cpu'))
        #)
        ################################################################################################
    def end_run(self): 
        #Este método se encarga de reiniciar los atributos de una corrida, para probar otros hiperparámetros
        #self.tb.close() # es uno de los comandos de Tensorboard
        self.epoch_count=0
        run_duration = time.time()-self.run_start_time
        print('Tiempo de ejecución: ', run_duration, 's')
        
    def begin_epoch(self):
        #comenzamos con el entrenamiento en una época específica
        self.epoch_start_time=time.time()
        
        self.epoch_count+=1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_val_loss = 0
        self.epoch_val_num_correct = 0
        
    def end_epoch(self):
        #el método para terminar la época
        epoch_duration = time.time()- self.epoch_start_time
        run_duration = time.time()-self.run_start_time
        
        loss=self.epoch_loss / (len(self.loader.dataset)*self.loader.batch_size)#1000)
        accuracy= self.epoch_num_correct / len(self.loader.dataset)
        val_loss=self.epoch_val_loss / (len(self.val_loader.dataset)*self.loader.batch_size)#1000)
        val_acc=self.epoch_val_num_correct / len(self.val_loader.dataset)
        
        ########### Implementación de Tensorboard ################################
        #self.tb.add_scalar('loss', loss, self.epoch_count)
        #self.tb.add_scalar('accuracy', accuracy, self.epoch_count)
        #self.tb.add_scalar('val_loss', val_loss, self.epoch_count)
        #self.tb.add_scalar('val_accuracy', val_acc, self.epoch_count)
        
        #for name, param in self.network.named_parameters():
        #    self.tb.add_histogram(name, param, self.epoch_count)
        #    self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
            
        ##########################################################################
        
        results =OrderedDict()
        results['run']=self.run_count
        results['epoch']=self.epoch_count
        results['loss']=loss
        results['accuracy']=accuracy
        results['val_loss']=val_loss
        results['val_accuracy']=val_acc
        results['epoch duration'] =epoch_duration
        results['run_duration']=run_duration
        
        # En vez de imprimir los resultados con print, vamos a darle estilo a las salidas, usando pandas
        for k, v in self.run_params._asdict().items(): results[k]=v
        self.run_data.append(results)
        df=pd.DataFrame.from_dict(self.run_data, orient='columns')
        
        # comandos para mostrar la tabla de pandas en el notebook o la terminal
        display.clear_output(wait=True)
        display.display(df)
        
        
    def track_loss(self, loss):
        self.epoch_loss += loss.item()*self.loader.batch_size
        
    def track_val_loss(self, loss):
        self.epoch_val_loss += loss.item()*self.val_loader.batch_size
        
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self.get_num_correct(preds, labels)
        
    def track_val_num_correct(self, preds, labels):
        self.epoch_val_num_correct += self.get_num_correct(preds, labels)
        
    @torch.no_grad()
    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()
    
    ######Para ser más profesionales, podremos exportar nuestros resultados a una tabla de excel
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{fileName}.csv')

"""
Dado que Pytorch nos permite mayor control en nuestros entrenamientos, se realizó esta clase con la finalidad de controlar el inicio de cada época, el 
registro de los costos y accuracys, los tiempos de ejecución y los términos de las épocas y las corridas. Esta clase la vamos a llamar en la función fit 
que se describe más adelante
"""
#############################################################################################################
def plot_model(model, to_file, show_shapes=True, show_layer_names=True):
    x=Variable(torch.rand(1,10)).to('cpu')
    y=model(x)
    make_dot(y.to('cpu'), params=dict(model.named_parameters()))
    dot=make_dot(y, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render(to_file[:-4])
    return dot

"""
Esta función es análoga a la función con el mismo nombre del framework Keras
"""
#############################################################################################################
def summary(Network,n_x,n_h,n_y, batch):    
    batch_size=batch
    print(pms.summary(Network(n_x,n_h,n_y), torch.zeros((batch_size,10)), show_input=True))
    print(pms.summary(Network(n_x,n_h,n_y), torch.zeros((batch_size,10)), show_input=False))
    print(pms.summary(Network(n_x,n_h,n_y), torch.zeros((batch_size,10)), show_input=False, show_hierarchical=True))

"""
Esta función es análoga a la función con el mismo nombre del framework Keras
"""
#############################################################################################################
losses={
    'bynary_cross_entropy':F.binary_cross_entropy,
    'bynary_cross_entropy_logits': F.binary_cross_entropy_with_logits,
    'poisson_negative': F.poisson_nll_loss,
    'cosine_loss': F.cosine_embedding_loss,
    'categorical_crossentropy':F.cross_entropy,
    'ctc_loss': F.ctc_loss,
    'hinge_loss': F.hinge_embedding_loss,
    'kldiv': F.kl_div,
    'l1': F.l1_loss,
    'mse': F.mse_loss,
    'margin_rank': F.margin_ranking_loss,
    'multilabel_margin': F.multilabel_margin_loss,
    'multi_soft': F.multilabel_soft_margin_loss,
    'multi_margin': F.multi_margin_loss,
    'nll_loss':F.nll_loss,
    'smooth_l1': F.smooth_l1_loss,
    'soft_margin': F.soft_margin_loss,
    'triple_margin':F.triplet_margin_loss,
    'triple_argin_distance':F.triplet_margin_with_distance_loss
}

"""
Estas son las funciones de costo que se encuentran en Keras
"""
#############################################################################################################
class model_compile():
    def __init__(self, optimizer, loss, metrics):
        self.loss_fn=losses[loss]
        self.optim=optimizer
        
    def get_loss(self,x,y):
        loss=self.loss_fn(x.float(), y.float(), reduction='mean')
        return loss
    def get_optimizer(self):
        return self.optim
    def update_grads(self, loss, optimizer):
        optimizer.zero_grad()              
        loss.backward()
        optimizer.step()  

"""
Esta paso emula al paso COMPILE que existe en Keras. De nuevo, la idea es que el usuario se sienta familiar a la manera en que se presentan
las redes neuronales en Keras que estudiaremos a lo largo del curso
"""
#############################################################################################################
def fit(params, epochs, model, compile_step, trainsets, valsets):
    
    #graficas=OrderedDict([])
    m=RunManager()
    
    for run in RunBuilder.get_runs(params):
        
        
        device=torch.device(run.device)
        network=model
        loader=DataLoader(trainsets[run.trainset], batch_size=run.batch_size, shuffle=True, num_workers=run.num_workers)
        val_loader=DataLoader(valsets[run.trainset], batch_size=run.batch_size, num_workers=run.num_workers)
        optimizer=compile_step.get_optimizer()


        m.begin_run(run, network, loader, val_loader)
        for epoch in range(epochs): #(10)
            m.begin_epoch()
            for batch in loader:
                network.train()
                images=batch[0].to(device)
                labels=batch[1].to(device)

                #with amp.autocast():
                preds=network(images)
                loss=compile_step.get_loss(preds, labels)

                compile_step.update_grads(loss, optimizer)

                m.track_loss(loss)
                m.track_num_correct(preds, labels)

            with torch.no_grad():
                for batch in val_loader:

                    network.eval()
                    images=batch[0].to(device)
                    labels=batch[1].to(device)
                    preds_val=network(images)
                    loss_val=compile_step.get_loss(preds_val, labels)
                    m.track_val_loss(loss_val)
                    m.track_val_num_correct(preds_val, labels)

            m.end_epoch()
        m.end_run()
        #graficas=m.save()
    return m, network

"""
Se emula en esta función el método fit que se encuentra en Keras. En esta función definimos el training loop, con ayuda del runManager que deifnimos
previamente. Vamos a obtener de regreso lo que se llama HISTORY en Keras, y la misma red neuronal, pero ya entrenada
"""
#############################################################################################################
def evaluate(network, testset_onehot, params, valsets, compile_step):
    
    #graficas=OrderedDict([])
    m=RunManager()
    for run in RunBuilder.get_runs(params):
        
        
        device=torch.device(run.device)
        network=network
        loader=DataLoader(testset_onehot, batch_size=run.batch_size, shuffle=True, num_workers=run.num_workers)
        val_loader=DataLoader(valsets['normal'], batch_size=run.batch_size, num_workers=run.num_workers)
        optimizer=None


        m.begin_run(run, network, loader, val_loader)
        for epoch in range(1): #(10)
            m.begin_epoch()
            for batch in loader:
                network.eval()
                images=batch[0].to(device)
                labels=batch[1].to(device)

                with amp.autocast():
                    preds=network(images)
                    loss=compile_step.get_loss(preds, labels)

                

                m.track_loss(loss)
                m.track_num_correct(preds, labels)


            m.end_epoch()
        m.end_run()
    return m

"""
Este es el paso de evaluación, en donde igualmente volvemos a hacer el simil con Keras. Se toma de base lo que realizamos en fit
"""
#############################################################################################################