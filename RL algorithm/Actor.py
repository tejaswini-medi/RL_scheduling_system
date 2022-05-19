import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#Neuronales Netz für den Actor
class ActorNetwork(nn.Module):
    
    #Initialisiere
    def __init__(self, input_size, hidden_size, use_cuda, LR, GAMMA, LAMBDA, epsilon, C1, clip):

        #Falls GPU-Beschleunigung CUDA verwendet werden soll
        if use_cuda == True:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        super(ActorNetwork, self).__init__()
        ### parameter ##
        self.GAMMA = GAMMA
        self.LR = LR #Lernrate
        self.LAMBDA = LAMBDA
        self.epsilon = epsilon
        self.C1 = C1

        ### input ####
        self.input = nn.Linear(input_size, hidden_size)

        ### hidden #####        
        self.action_hidden_0 = nn.Linear(hidden_size, hidden_size)
        self.action_hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.action_hidden_2 = nn.Linear(hidden_size, hidden_size)

        ### output ###
        self.action_a_out = nn.Linear(hidden_size, 6)
        self.action_b_out = nn.Linear(hidden_size, 6)

    #Vorwärts durch das Netz
    def forward(self, x):
        #x = Zustand
        x = Variable(x, requires_grad=True)
        x = F.leaky_relu(self.input(x))
        #2 versteckte Ebenen
        action = F.leaky_relu(self.action_hidden_0(x))
        action = F.leaky_relu(self.action_hidden_1(action))

        #Ausgangsebenen für alpha und beta (6x2) mit SoftPlus + 1
        a = F.softplus(self.action_a_out(action)) + 1        
        b = F.softplus(self.action_b_out(action)) + 1
        return a, b
    
        #Berechne GAE für gesamten Batch (memory)
    def compute_advantage(self, memory, HORIZON, makespan, opt, critic):
        ### Beobachtungen aus Minibatch ###
        states = torch.tensor(memory["State"])
        actions = torch.tensor(memory["Optimal_Action"])
        states_1 = torch.tensor(memory["State_1"])
        rewards = torch.tensor(memory["Reward"])
        v_states = torch.stack(memory["v_states"]).flatten()
        v_states_1 = torch.stack(memory["v_states_1"]).flatten()
        v_states_1 = torch.stack(memory["v_states_1"]).flatten()
        ### initialisiere Tensoren ###
        returns = torch.tensor([])
        v_targ = torch.tensor([])

        ### Notwendig für laufende Variable lastgae ###
        lastgae = 0 
        first = True

        for t in reversed(range(len(states))):
            #Belohnung = theoretisches Optimum / erreichte DLZ
            if first:
                #Belohnung
                rewards[t] = opt/makespan
                #Gewinn
                g_t = rewards[t]
                #Nutzen
                v = rewards[t]
                #TD-Fehler
                delta = g_t-critic.forward(states[t])
                #letzte GAE
                lastgae = delta
                first = False
                
            else:
                #Für alle Beobachtungen außer der letzten Belohnung = 0
                rewards[t] = 0
                #Gewinn = r+V_pred(s')
                g_t = rewards[t] + critic.forward(states_1[t])
                #Nutzen = r + v(s')
                v = rewards[t] + v
                #TD-Fehler (g_t oder v)
                delta = g_t - critic.forward(states[t])
                #Letzte GAE
                lastgae = delta + self.GAMMA*self.LAMBDA*lastgae
            returns = torch.cat((returns, lastgae.reshape(1)))
            v_targ = torch.cat((v_targ, v.reshape(1)))
            
        returns = torch.flip(returns, dims=[0])
        v_targ = torch.flip(v_targ, dims=[0])
        return [
        states, actions, rewards, 
        v_states, v_states_1, returns, v_targ
        ]

    # Macht aus Aktionsliste einen Tensor, notwendig wegen 6 Parametern    
    def reshape_action(self, action):

        obs = list(zip(*action))
        a = [list(obs[x]) for x in range(len(obs))]
        actions = [torch.tensor(x) for x in a]
        return actions


    #Berechnet die log-Wahrscheinlichkeit für die Aktionen
    #unter der aktuellen Parametrierung
    def calc_logprob(self, new_pred, actions):

        actions = self.reshape_action(actions)

        a_new = new_pred[0]
        b_new = new_pred[1]

        m_new = [torch.distributions.beta.Beta(
            a,b) for a,b in zip(a_new,b_new)]

        #Entropy der Beta-Verteilung
        entropy = torch.stack([
            torch.mean(m_new[i].entropy()) for i in range(len(m_new))
            ])
        #Aktionswahrscheinlichkeit unter alpha und beta
        new_log_prob_value = [m_new[i].log_prob(
            actions[i]) for i in range(len(actions))]

        #Erzeuge Tensor aus allen Wahrschienlichkeiten
        log_probability = [new_log_prob_value[i] for i in range(len(actions))]
        log_probability = torch.stack(log_probability)

        return log_probability, entropy


    
    def backwards(
        self, mini_batch, old_policy, loss_list, 
        step_list, step_nb, training_epochs, critic):

        #Initialisiere Optimierer (Adam)
        optimizer = optim.Adam(self.parameters(), lr=self.LR, amsgrad=True)

        loss = torch.tensor([])

        states = mini_batch[0]
        actions = mini_batch[1]

        actions = torch.tensor(
            [[actions[j][i] for j in range(
                len(actions))]for i in range(len(actions[0]))])

        rewards = mini_batch[2]
        v_0 = mini_batch[3]
        v_1 = mini_batch[4]
        advantage_ests = mini_batch[5]
        advantage_ests.detach()
        v_targ = mini_batch[6]
        
        #Neue Vorhersage für Zustandsnutzen  V(s)
        new_pred = self.forward(states)
        log_probability, entropy = self.calc_logprob(new_pred, actions)
        advantage_ests = torch.tensor([[i] for i in advantage_ests])

        #Policy-Loss mit Entropy = 0.0001
        loss =- torch.mean(
            log_probability*advantage_ests) - 0.0001*torch.mean(entropy)
        
        #Adam Optimierer, Aktualisierungsschritt
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Für Visualisierung
        step_nb += 1
        step_list.append(step_nb)
        loss_list.append(loss.item())

        return loss_list, step_list, step_nb
