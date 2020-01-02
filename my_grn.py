import torch
import torch.nn as nn

def collect_neighbor_v2(representation, positions):
    # representation: [batch_size, num_nodes, feature_dim]
    # positions: [batch_size, num_nodes, num_neighbors]
    batch_size = positions.size(0)
    node_num = positions.size(1)
    neigh_num = positions.size(2)
    rids = torch.arrange(0, limit=batch_size) # [batch]
    rids = rids.reshape([-1, 1, 1]) # [batch, 1, 1]
    rids = rids.repeat(1, node_num, neigh_num) # [batch, nodes, neighbors]
    indices = torch.stack((rids, positions),3) # [batch, nodes, neighbors, 2]
    return representation[indices[:, :, :, 0], indices[:, :, :, 1], :]

class GRN(nn.Module):
    def __init__(self, options, word_dim, edgelabel_vocab, sentence_dim):
        super(GRN, self).__init__()

        self.edge_dim = edgelabel_vocab.word_dim
        self.word_dim = word_dim
        self.sentence_dim = sentence_dim

        self.edge_embedding = nn.Embedding(edgelabel_vocab.vocab_size, self.edge_dim)

        if options.with_edgelabel:
            self.u_input_dim = self.sentence_dim + self.edge_dim
        else:
            self.u_input_dim = self.sentence_dim

        self.w_in_ingate = nn.Linear(self.word_dim + self.edge_dim, sentence_dim, bias = False)
        self.u_in_ingate = nn.Linear(self.u_input_dim, self.sentence_dim, bias = False)
        self.b_ingate = nn.Parameter(torch.Tensor(self.sentence_dim))
        self.w_out_ingate = nn.Linear(self.word_dim + self.edge_dim, self.sentence_dim, bias = False)
        self.u_out_ingate = nn.Linear(self.u_input_dim, self.sentence_dim, bias = False)

        self.w_in_forgetgate = nn.Linear(self.word_dim + self.edge_dim, self.sentence_dim, bias = False)
        self.u_in_forgetgate = nn.Linear(self.u_input_dim, self.sentence_dim, bias = False)
        self.b_forgetgate = nn.Parameter(torch.Tensor(self.sentence_dim))
        self.w_out_forgetgate = nn.Linear(self.word_dim + self.edge_dim, self.sentence_dim, bias = False)
        self.u_out_forgetgate = nn.Linear(self.u_input_dim, self.sentence_dim, bias = False)

        self.w_in_outgate = nn.Linear(self.word_dim + self.edge_dim, self.sentence_dim, bias = False)
        self.u_in_outgate = nn.Linear(self.u_input_dim, self.sentence_dim, bias = False)
        self.b_outgate = nn.Parameter(torch.Tensor(self.sentence_dim))
        self.w_out_outgate = nn.Linear(self.word_dim + self.edge_dim, self.sentence_dim, bias = False)
        self.u_out_outgate = nn.Linear(self.u_input_dim, self.sentence_dim, bias = False)

        self.w_in_cell = nn.Linear(self.word_dim + self.edge_dim, self.sentence_dim, bias = False)
        self.u_in_cell = nn.Linear(self.u_input_dim, self.sentence_dim, bias = False)
        self.b_cell = nn.Parameter(torch.Tensor(self.sentence_dim))
        self.w_out_cell = nn.Linear(self.word_dim + self.edge_dim, self.sentence_dim, bias = False)
        self.u_out_cell = nn.Linear(self.u_input_dim, self.sentence_dim, bias = False)



    def forward(self, batch, options, word_repres,sentence_repres, seq_mask,):
        in_neighbor_indices = batch.in_neigh_indices
        in_neighbor_edges = batch.in_neigh_edges
        in_neighbor_mask = batch.in_neigh_mask

        out_neighbor_indices = batch.out_neigh_indices
        out_neighbor_edges = batch.out_neigh_edges
        out_neighbor_mask = batch.out_neigh_mask

        if options.forest_prob_aware and options.forest_type != '1best':
            in_neighbor_prob = batch.in_neigh_prob
            out_neighbor_prob = batch.out_neigh_prob

        batch_size = self.in_neighbor_indices.size(0)
        sentence_size_max = self.in_neighbor_indices.size(1)

        # ==== input from in neighbors
        # [batch_size, sentence_len, neighbors_size_max, edge_dim]
        in_neighbor_edge_representations = self.edge_embedding(in_neighbor_edges)
        # [batch_size, sentence_len, neighbors_size_max, word_dim]
        in_neighbor_word_representations = collect_neighbor_v2(word_repres,in_neighbor_indices)
        # [batch_size, sentence_len, neighbors_size_max, word_dim + edge_dim]
        in_neighbor_representations = torch.cat(
            [in_neighbor_word_representations, in_neighbor_edge_representations], 3)
        if options.forest_prob_aware and options.forest_type != '1best':
            in_neighbor_representations = in_neighbor_representations.mul(in_neighbor_prob.unsqueeze(-1))
        in_neighbor_representations = in_neighbor_representations.mul(in_neighbor_mask.unsqueeze(-1))
        # [batch_size, sentence_len, word_dim + edge_dim]
        in_neighbor_representations = in_neighbor_representations.sum(dim=2)
        in_neighbor_representations = in_neighbor_representations.reshape([-1, self.word_dim + self.edge_dim])

        # ==== input from out neighbors
        # [batch_size, sentence_len, neighbors_size_max, edge_dim]
        out_neighbor_edge_representations = self.edge_embedding(out_neighbor_edges)
        # [batch_size, sentence_len, neighbors_size_max, word_dim]
        out_neighbor_word_representations = collect_neighbor_v2(word_repres,out_neighbor_indices)
        # [batch_size, sentence_len, neighbors_size_max, word_dim + edge_dim]
        out_neighbor_representations = torch.concat(
            [out_neighbor_word_representations, out_neighbor_edge_representations], 3)
        if options.forest_prob_aware and options.forest_type != '1best':
            out_neighbor_representations = out_neighbor_representations.mul(out_neighbor_prob.unsqueeze(-1))
        out_neighbor_representations = out_neighbor_representations.mul(out_neighbor_mask.unsqueeze(-1))
        # [batch_size, sentence_len, word_dim + edge_dim]
        out_neighbor_representations = out_neighbor_representations.sum(2)
        out_neighbor_representations = out_neighbor_representations.reshape([-1, self.word_dim + self.edge_dim])

        node_hidden = sentence_repres
        node_cell = torch.zeros(batch_size, sentence_size_max, self.sentence_dim)

        word_repres = word_repres.reshape([-1, self.word_dim])

        graph_representations = []
        for i in range(options.num_graph_layer):
            # =============== in neighbor hidden
            # [batch_size, sentence_len, neighbors_size_max, u_input_dim]
            in_neighbor_prev_hidden = collect_neighbor_v2(node_hidden,in_neighbor_indices)
            if options.with_edgelabel:
                in_neighbor_prev_hidden = torch.cat(
                        [in_neighbor_prev_hidden, in_neighbor_edge_representations], 3)
            in_neighbor_prev_hidden = in_neighbor_prev_hidden.mul(in_neighbor_mask.unsqueeze(-1))
            # [batch_size, sentence_len, u_input_dim]
            in_neighbor_prev_hidden = in_neighbor_prev_hidden.sum(2)
            in_neighbor_prev_hidden = in_neighbor_prev_hidden.mul(seq_mask.unsqueeze(-1))
            in_neighbor_prev_hidden = in_neighbor_prev_hidden.reshape([-1, self.u_input_dim])

            # =============== out neighbor hidden
            # [batch_size, sentence_len, neighbors_size_max, u_input_dim]
            out_neighbor_prev_hidden = collect_neighbor_v2(node_hidden, out_neighbor_indices)
            if options.with_edgelabel:
                out_neighbor_prev_hidden = torch.cat(
                    [out_neighbor_prev_hidden, out_neighbor_edge_representations], 3)
            out_neighbor_prev_hidden = out_neighbor_prev_hidden.mul(out_neighbor_mask.unsqueeze(-1))
            # [batch_size, sentence_len, u_input_dim]
            out_neighbor_prev_hidden = out_neighbor_prev_hidden.sum(2)
            out_neighbor_prev_hidden = out_neighbor_prev_hidden.mul(seq_mask.unsqueeze(-1))
            out_neighbor_prev_hidden = out_neighbor_prev_hidden.reshape([-1, self.u_input_dim])

            ## ig
            edge_ingate = torch.sigmoid(self.w_in_ingate(in_neighbor_representations)
                                     + self.u_in_ingate(in_neighbor_prev_hidden)
                                     + self.w_out_ingate(out_neighbor_representations)
                                     + self.u_out_ingate(out_neighbor_prev_hidden)
                                     + self.b_ingate)
            edge_ingate = edge_ingate.reshape([batch_size, sentence_size_max, self.sentence_dim])

            ## fg
            edge_forgetgate = torch.sigmoid(self.w_in_forgetgate(in_neighbor_representations)
                                         + self.u_in_forgetgate(in_neighbor_prev_hidden)
                                         + self.w_out_forgetgate(out_neighbor_representations)
                                         + self.u_out_forgetgate(out_neighbor_prev_hidden)
                                         + self.b_forgetgate)
            edge_forgetgate = edge_forgetgate.reshape([batch_size, sentence_size_max, self.sentence_dim])

            ## og
            edge_outgate = torch.sigmoid(self.w_in_outgate(in_neighbor_representations)
                                      + self.u_in_outgate(in_neighbor_prev_hidden)
                                      + self.w_out_outgate(out_neighbor_representations)
                                      + self.u_out_outgate(out_neighbor_prev_hidden)
                                      + self.b_outgate)
            edge_outgate = edge_outgate.reshape([batch_size, sentence_size_max, self.sentence_dim])

            ## input
            edge_cell_input = torch.tanh(self.w_in_cell(in_neighbor_representations)
                                      + self.u_in_cell(in_neighbor_prev_hidden)
                                      + self.w_out_cell(out_neighbor_representations)
                                      + self.u_out_cell(out_neighbor_prev_hidden)
                                      + self.b_cell)
            edge_cell_input = edge_cell_input.reshape([batch_size, sentence_size_max, self.sentence_dim])

            temp_cell = edge_forgetgate * node_cell + edge_ingate * edge_cell_input
            temp_hidden = edge_outgate * torch.tanh(temp_cell)
            #if is_training and i < options.num_graph_layer:
            #    temp_hidden = tf.nn.dropout(temp_hidden, (1 - options.dropout_rate))
            # apply mask
            node_cell = temp_cell.mul(seq_mask.unsqueeze(-1))
            node_hidden = temp_hidden.mul(seq_mask.unsqueeze(-1))

            graph_representations.append(node_hidden)

        return graph_representations, node_hidden, node_cell

