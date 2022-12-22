import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, num_classes, hidden_size = 768, num_layers = 3, dropout = 0.1):
        super(Discriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = hidden_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob


class AdvFriendlyQA(nn.Module):
    def __init__(self, distillBertQA, dom_weights, quest_weights, dom_classes, quest_classes, dom_lambda = 0.3, quest_lambda = 0.3):
        super(AdvFriendlyQA, self).__init__()
        self.bert = distillBertQA
        self.domDiscriminator = Discriminator(dom_classes)
        self.questDiscriminator = Discriminator(quest_classes)
        self.dom_classes = dom_classes
        self.dom_weights = dom_weights
        self.quest_classes = quest_classes
        self.quest_weights = quest_weights
        self.dom_lambda = dom_lambda
        self.quest_lambda = quest_lambda

    def bert_predict_QA(self, input_ids, attention_mask, start_positions, end_positions, labels):
        outputs = self.bert(input_ids, attention_mask, start_positions, end_positions, output_hidden_states = False)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        return start_logits, end_logits

    def disc_predict_data_label(self, disc_name, outputsQA):
        cls = outputsQA.hidden_states[-1][:,0,:]
        if disc_name == 'dom': log_probs = self.domDiscriminator(cls)
        elif disc_name == 'quest': log_probs = self.questDiscriminator(cls)
        return log_probs

    def get_qa_loss(self, input_ids, attention_mask, start_positions, end_positions):
        outputsQA = self.bert(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, output_hidden_states=True)
        log_probs_dom = self.disc_predict_data_label("dom", outputsQA)
        log_probs_quest = self.disc_predict_data_label("quest", outputsQA)
        uniform_dom = torch.ones_like(log_probs_dom) * torch.tensor(self.dom_weights).to('cuda')
        kl_div_loss = nn.KLDivLoss(reduction = "batchmean")
        kld_dom = self.dom_lambda * kl_div_loss(log_probs_dom, uniform_dom)
        uniform_quest = torch.ones_like(log_probs_quest) * torch.tensor(self.quest_weights).to('cuda')
        kl_div_loss = nn.KLDivLoss(reduction = "batchmean")
        kld_quest = self.quest_lambda * kl_div_loss(log_probs_quest, uniform_quest)
        # Alternative quest error
#        dis_loss = self.get_discriminator_loss(input_ids, attention_mask, labels, dom = False)
#        kld_quest = self.quest_lambda * dis_loss
        bert_loss = outputsQA.loss
        total_qa_loss = bert_loss + kld_dom - kld_quest
        return total_qa_loss

    def get_discriminator_loss(self, disc_name, input_ids, attention_mask, labels):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask, output_hidden_states = True)
            cls_embedding = outputs.hidden_states[-1][:,0,:]
            hidd_rep = cls_embedding.detach()
        if disc_name == 'dom': log_probs = self.domDiscriminator(hidd_rep)
        elif disc_name == 'quest': log_probs = self.questDiscriminator(hidd_rep)
        NLL = nn.NLLLoss()
        return NLL(log_probs, labels)


    def forward(self, network, input_ids, attention_mask, start_positions = None, end_positions = None, labels = None):
        if network == "qa":
            return self.get_qa_loss(input_ids, attention_mask,start_positions, end_positions)
        else:
            if network ==  'dom':
                return self.get_discriminator_loss('dom', input_ids, attention_mask, labels)
            elif network ==  'quest':
                return self.get_discriminator_loss('quest', input_ids, attention_mask, labels)


