import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import random
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils import move_to


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""
    def __init__(self, features_dim, num_head, n_layers):
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=features_dim, nhead=num_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        output = self.transformer_encoder(x)
        return output
    
class Decoder(nn.Module):
    def __init__(self, 
            features_dim,
            num_head,
            n_layers,
            mask_logits=True):
        super(Decoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(d_model=features_dim, nhead=num_head)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.project = nn.Linear(features_dim, 1)

        self.mask_logits = mask_logits
        self.decode_type = None  # Needs to be set explicitly before use

    def update_mask(self, mask, selected):
        return mask.clone().scatter_(1, selected.unsqueeze(-1), True)

    def forward(self, decoder_input, context, epoch, Training=True):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim]. 
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim] 
        """

        batch_size = decoder_input.size(1)
        outputs = []
        selections = []
        steps = range(decoder_input.size(0))
        graph_size = decoder_input.size(0)
        idxs = None
        mask = Variable(
            decoder_input.data.new().byte().new(batch_size, graph_size).zero_(),
            requires_grad=False
        )
        
        order_reference = [[i for i in range(graph_size)] for _ in range(batch_size)]
        
        for _ in steps:
            mask = self.update_mask(mask, idxs) if idxs is not None else mask
            #decoder_output = self.transformer_decoder(decoder_input, context, tgt_key_padding_mask=mask)
            #logits = self.project(decoder_output.permute(1, 0, 2)).squeeze(-1)

            decoder_input = self.transformer_decoder(decoder_input, context, tgt_key_padding_mask=mask)
            logits = self.project(decoder_input.permute(1, 0, 2)).squeeze(-1)

            if self.mask_logits:
                logits[mask] = -np.inf

            log_p = torch.log_softmax(logits, dim=1)
            probs = log_p.exp()

            if Training:
                choose = random.random()
                if choose < 1. ** (epoch // 30):
                    self.decode_type = "sampling"
                else:
                    self.decode_type = "greedy"

            idxs = self.decode(probs, mask)
            idxs = idxs.detach()  # Otherwise pytorch complains it want's a reward, todo implement this more properly?
            # Gather input embedding of selected

            #reorder = []
            #for idx in idxs:
                #reorder.append([idx.item()] + [j for j in range(graph_size) if j != idx.item()])
            """
            reorder = []
            for i in range(batch_size):
                idx = order_reference[i].index(idxs[i].item())
                reorder.append([idx] + [j for j in range(graph_size) if j != idx]) 
                order_reference[i] = [idx] + [j for j in order_reference[i] if j != idx]
                
            #order_new = torch.tensor(reorder).unsqueeze(-1).repeat(1, 1, decoder_input.size(-1)).transpose(0, 1).cuda()
            order_new = torch.tensor(reorder).unsqueeze(-1).repeat(1, 1, decoder_input.size(-1)).transpose(0, 1).cuda()
            decoder_input = decoder_input.gather(0, order_new).cuda()
            """

            # use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs)

        return torch.stack(outputs, 1), torch.stack(selections, 1)

    def decode(self, probs, mask):
        if self.decode_type == "greedy":
            _, idxs = probs.max(1)
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            idxs = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                #idxs = probs.multinomial().squeeze(1)
                idxs = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"

        return idxs

class CriticNetworkLSTM(nn.Module):
    """Useful as a baseline in REINFORCE updates"""
    def __init__(self,
            embedding_dim,
            hidden_dim,
            n_process_block_iters,
            tanh_exploration,
            use_tanh):
        super(CriticNetworkLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(embedding_dim, hidden_dim)
        
        self.process_block = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.sm = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
        inputs = inputs.transpose(0, 1).contiguous()

        encoder_hx = self.encoder.init_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = self.encoder.init_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        
        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        
        # grab the hidden state and process it via the process block 
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out


class PointerNetwork(nn.Module):

    def __init__(self,
                 embedding_dim,
                 problem,
                 n_head_encoder,
                 n_layer_encoder,
                 n_head_decoder,
                 n_layer_decoder,
                 mask_logits=True,
                 num_coordinates=11,
                 **kwargs):
        super(PointerNetwork, self).__init__()

        self.problem = problem
        assert problem.NAME == "tsp" or problem.NAME == "toposort", "Pointer Network only supported for TSP and TopoSort"
        self.input_dim = num_coordinates

        self.encoder = Encoder(
            embedding_dim,
            n_head_encoder,
            n_layer_encoder)

        self.decoder = Decoder(
            embedding_dim,
            n_head_decoder,
            n_layer_decoder,
            mask_logits=mask_logits
        )

        std = 1. / math.sqrt(embedding_dim)
        self.embedding = nn.Parameter(torch.FloatTensor(self.input_dim, embedding_dim))
        self.embedding.data.uniform_(-std, std)

        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

    def set_decode_type(self, decode_type):
        self.decoder.decode_type = decode_type

    def forward(self, inputs, labels, opts, epoch=0, Training=True, Measures=False, Plot_Data=False):

        #indices = torch.tensor([0, 9, 10]).cuda() # (level, index, memory) for input dim of dataset as 11
        #inputs = torch.index_select(inputs_11, 2, indices).cuda()

        batch_size, graph_size, input_dim = inputs.size()
        """
        embedding_inputs = torch.mm(
            inputs.transpose(0, 1).contiguous().view(-1, input_dim),
            self.embedding
        ).view(graph_size, batch_size, -1)
        """
        embedding_inputs = self.bn1(torch.mm(
            inputs.transpose(0, 1).contiguous().view(-1, input_dim),
            self.embedding
        ).view(graph_size, batch_size, -1).transpose(0, 1).transpose(1, 2)).transpose(1, 2).transpose(0, 1)

        encoder_output = self.encoder(embedding_inputs)
        #encoder_input = self.bn2(encoder_output.transpose(1, 2)).transpose(1, 2)
        
        _log_p, pi = self.decoder(embedding_inputs, encoder_output, epoch, Training)

        cost, mask, misMatch, _, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = self.problem.get_costs(inputs, pi, labels, Measures, Plot_Data, opts.graph_file)

        ll = self._calc_log_likelihood(_log_p, pi, mask)
        """
        if return_pi:
            return cost, ll, pi, misMatch, None, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min
        """

        return cost, ll, misMatch, None, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min

        """
        embedded_inputs = torch.mm(
            #inputs.transpose(0, 1).contiguous().view(-1, input_dim),
            inputs.transpose(0, 1).contiguous().view(-1, input_dim),
            self.embedding
        ).view(graph_size, batch_size, -1)
        #inputs_weight_free = inputs.detach().clone()
        #inputs_weight_free.index_fill_(2, move_to(torch.tensor([10]), opts.device), 0.)
        embedded_inputs = self.bn1(torch.mm(
            inputs.transpose(0, 1).contiguous().view(-1, input_dim),
            #move_to(inputs_weight_free, opts.device).transpose(0, 1).contiguous().view(-1, input_dim),
            self.embedding
        ).view(graph_size, batch_size, -1).transpose(0, 1).transpose(1, 2)).transpose(1, 2).transpose(0, 1)
        #).view(graph_size, batch_size, -1).transpose(1, 2)).transpose(1, 2)

        # query the actor net for the input indices 
        # making up the output, and the pointer attn 
        _log_p, pi = self._inner(embedded_inputs, eval_tours)

        #cost, mask, misMatch, _, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = self.problem.get_costs(inputs, pi, Measures, Plot_Data)
        cost, mask, misMatch, _, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = self.problem.get_costs(inputs, pi, labels, Measures, Plot_Data, opts.graph_file)
        #cost, mask, misMatch_y, misMatch_x, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = self.problem.get_costs(inputs, pi, labels, Measures, Plot_Data)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            #return cost, ll, pi, misMatch_y, misMatch_x, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min
            return cost, ll, pi, misMatch, None, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min

        #return cost, ll, misMatch_y, misMatch_x, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min
        return cost, ll, misMatch, None, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min
        """

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)
    """
    def _inner(self, inputs, eval_tours=None):

        encoder_hx = encoder_cx = Variable(
            torch.zeros(1, inputs.size(1), self.encoder.hidden_dim, out=inputs.data.new()),
            requires_grad=False
        )

        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        enc_h = self.bn2(enc_h.transpose(1, 2)).transpose(1, 2)

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)

        enc_h = self.transformer_encoder(enc_h)
        enc_h = self.bn3(enc_h.transpose(1, 2)).transpose(1, 2)

        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input,
                                                                 inputs,
                                                                 dec_init_state,
                                                                 enc_h,
                                                                 eval_tours)

        return pointer_probs, input_idxs
    """
