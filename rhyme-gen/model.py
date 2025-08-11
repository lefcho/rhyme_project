import torch
import torch.nn as nn
from typing import Tuple


class NextLineModel(nn.Module):
    """
    LSTM encoder decoder for predicting the next rap line, conditioned on:
    -previous line tokens
    -rhyme id of the target line
    -normalized syllable count of the target line (0..1)
    """

    def __init__(
        self,
        vocab_size: int,
        num_rhyme_ids: int,
        embed_dim: int = 128,
        rhyme_embed_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_rhyme_ids = num_rhyme_ids
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # rhyme embedding
        self.rhyme_embedding = nn.Embedding(num_embeddings=num_rhyme_ids, embedding_dim=rhyme_embed_dim)

        # encoder and decoder LSTMs
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

        # main head
        self.output_linear = nn.Linear(hidden_dim, vocab_size)

        # Аdditional Head 
        self.rhyme_classifier = nn.Linear(hidden_dim, num_rhyme_ids)

        cond_in_dim = rhyme_embed_dim + 1
        # addition to the hidden state - short mem
        self.cond_to_hidden = nn.Linear(cond_in_dim, hidden_dim)
        # addition to the cell state - long mem
        self.cond_to_cell = nn.Linear(cond_in_dim, hidden_dim)
        # helps the h and c not to blow up
        self.cond_act = nn.Tanh()

    def encode(self, prev_seq: torch.LongTensor):
        """
        Encode previous line into final (h, c).
        prev_seq: [batch, seq_len = 30] and
        returns h, c each of shape (num_layers, batch, hidden_dim)
        """
        embedded = self.embedding(prev_seq) # [batch, seq_len, embed_dim]
        _, (h, c) = self.encoder(embedded)
        return h, c

    def _condition_hidden(self, rhyme_ids: torch.LongTensor, syllables: torch.FloatTensor):
        """
        Create conditioned hidden and cell increments from rhyme_ids and syllables.
        rhyme_ids: [batch] (LongTensor)
        syllables: [batch] (FloatTensor)
        Returns a h_cond, c_cond each shaped (num_layers, batch, hidden_dim) ready to add to encoder states.
        """
        # rhyme embedding
        rhyme_emb = self.rhyme_embedding(rhyme_ids)  # [batch, rhyme_embed_dim]

        # syllables for next line: ensure shape [batch, 1]
        syll = syllables.view(-1, 1).float()

        cond = torch.cat([rhyme_emb, syll], dim=1)  # [batch, rhyme_embed_dim + 1]
        h_vec = self.cond_act(self.cond_to_hidden(cond))  # [batch, hidden_dim]
        c_vec = self.cond_act(self.cond_to_cell(cond))  # [batch, hidden_dim]

        # expand across layers
        # shape for the right h/c dims (1, batch, hidden_dim) 
        # then repeat num_layers times -> (num_layers, batch, hidden_dim)
        h_cond = h_vec.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_cond = c_vec.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h_cond, c_cond

    def decode_step(
        self,
        input_tokens: torch.LongTensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]):
        """
        Single decoding step (one timestep).
        input_tokens: [batch] (LongTensor) token ids for current timestep
        hidden: (h, c) each (num_layers, batch, hidden_dim)
        Returns logits and new_hidden: (h, c)
        """

        # add a dim for LSMT
        token_tensor = input_tokens.unsqueeze(1)  # [batch, 1]
        embedded = self.embedding(token_tensor)   # [batch, 1, embed_dim]
        output, hidden = self.decoder(embedded, hidden)  # output: [batch, 1, hidden_dim]
        logits = self.output_linear(output.squeeze(1))   # [batch, vocab_size]
        return logits, hidden

    def forward(
        self,
        prev_seq: torch.LongTensor,
        next_seq: torch.LongTensor,
        syllables: torch.FloatTensor,
        rhyme_ids: torch.LongTensor,
        ):
        """
        Teacher-forcing forward pass.
        """
        # encode previous line
        h_enc, c_enc = self.encode(prev_seq)  # each: (num_layers, batch, hidden_dim)

        # build conditioned increments and add to encoder final states
        h_cond, c_cond = self._condition_hidden(rhyme_ids, syllables)
        h_init = h_enc + h_cond
        c_init = c_enc + c_cond
        # initial state of the decoder
        hidden = (h_init, c_init)

        # decoder loop with teacher forcing
        input_tokens = next_seq[:, 0]  # first input token for each batch sample
        outputs = []
        max_len = next_seq.size(1)

        for t in range(1, max_len):
            logits, hidden = self.decode_step(input_tokens, hidden)
            outputs.append(logits)
            # teacher forcing: use ground truth next token as next input
            input_tokens = next_seq[:, t]

        # stack outputs -> [batch, seq_len_next-1, vocab_size]
        token_logits = torch.stack(outputs, dim=1)

        final_h = hidden[0]
        final_h_top = final_h[-1]
        rhyme_logits = self.rhyme_classifier(final_h_top)

        # language modeling loss, rhyme loss
        return token_logits, rhyme_logits
