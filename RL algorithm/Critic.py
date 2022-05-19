import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#Neuronales Netz für den Critic
class CriticNetwork(nn.Module):
    

    def __init__(self, input_size, hidden_size, use_cuda, LR):

        #Falls GPU-Beschleunigung CUDA verwendet werden soll
        if use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        super(CriticNetwork, self).__init__()

        #Lernrate = 1/10 der LR des Actors
        self.LR = LR / 10

        ### INPUT ####
        self.input = nn.Linear(input_size, hidden_size)
        
        ### Hidden ###
        self.value_hidden_0 = nn.Linear(hidden_size, hidden_size)
        self.value_hidden_1 = nn.Linear(hidden_size, hidden_size)

        ### Output ###
        self.value = nn.Linear(hidden_size, 1)
    
    #Vorwärtspass
    def forward(self, x):

        #Eingangswert = Zustand
        x = Variable(x, requires_grad=True)
        x = F.relu(self.input(x))

        #Zwei versteckte Ebenen, aktiviert mit Relu
        v = F.relu(self.value_hidden_0(x))
        v = F.relu(self.value_hidden_1(v))

        #Ein Augsangsneuron, linear aktiviert
        v = self.value(x)

        return v

    #Backpropagation
    def backwards(
            self, mini_batch, critic_loss_list, critic_step_list, 
            critic_step_nb,training_epochs):

        #Optimierer Adam initialisieren
        optimizer = optim.Adam(self.parameters(), lr=self.LR, amsgrad=True)


        #Daten aus Minibatch extrahieren
        states = mini_batch[0]
        v_targ = mini_batch[6]
        v_states = mini_batch[3]
        v_states_1 = mini_batch[4]

        returns = mini_batch[5]

        value = self.forward(states.float())

        #Critic-Loss 
        delta = v_targ.detach() - value
        #Torch.mean = Erwartungswert über gesamten Mini-Batch
        loss = torch.mean(0.5*delta**2)

        #Typischer Optimierungsschritt in Pytorch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Für Visualsierung des Trainingsverlaufs
        critic_step_nb += 1
        critic_step_list.append(critic_step_nb)
        critic_loss_list.append(loss.item())

        return critic_loss_list, critic_step_list, critic_step_nb
