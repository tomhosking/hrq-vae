import torch
import torch.nn as nn

from math import e, floor, pow, exp



def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output * 1.0

# Return the cosine similarity between x, y
def cos_sim(x, y):
    # prod = x*y
    prod = torch.matmul(x, y.T)
    norm = torch.matmul(x.norm(dim=-1, keepdim=True), y.norm(dim=-1, keepdim=True).T)
    return prod / norm


class HierarchicalRefinementQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        num_heads=3,
        code_offset=0,
        warmup_steps=None,
        use_cosine_similarities=False,
        gumbel_temp=2.0,
        temp_min=0.5,
        temp_schedule=True,
        temp_schedule_gamma=10000,
        norm_loss_weight=None,
        init_decay_weight=0.5,
        init_embeds_xavier=True,
        init_delay_steps=None,
        init_dynamic_var=False,
        init_scale=1.0,
        head_dropout=0.3,
        head_dropout_keep_first=False,
        learnable_priors=False,
    ):

        super(HierarchicalRefinementQuantizer, self).__init__()

        self._num_embeddings = num_embeddings
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim

        self._code_offset = code_offset

        self._gumbel_temp = gumbel_temp
        self._temp_min = temp_min
        self._temp_schedule = temp_schedule
        self._temp_schedule_gamma = temp_schedule_gamma

        self._warmup_steps = warmup_steps

        self._cos_sim = use_cosine_similarities

        self._norm_loss_weight = norm_loss_weight

        if head_dropout is not None and head_dropout > 0:
            self._head_dropout = torch.distributions.Bernoulli(1 - head_dropout)
        else:
            self._head_dropout = None
        self._head_dropout_keep_first = head_dropout_keep_first

        self.dims = [self._embedding_dim] * self._num_heads

        self._embedding = nn.ModuleList(
            [nn.Embedding(self._num_embeddings, self._embedding_dim) for _ in range(num_heads)]
        )

        for hix, embedding in enumerate(self._embedding):
            torch.nn.init.xavier_uniform_(
                embedding.weight.data, gain=6.0 * init_scale * init_decay_weight**hix
            ) if init_embeds_xavier else embedding.weight.data.normal_(std=init_scale * init_decay_weight**hix)

        if learnable_priors:
            self._learnable_priors = nn.ParameterList(
                [nn.Parameter(torch.zeros(num_embeddings)) for _ in range(num_heads)]
            )
        else:
            self._learnable_priors = None

        self._init_decay_weight = init_decay_weight
        self._init_delay_steps = init_delay_steps
        self._init_dynamic_var = init_dynamic_var
        if self._init_delay_steps is not None:
            self.register_buffer("_init_cumsum", torch.zeros(embedding_dim))
            self.register_buffer("_init_cumsquared", torch.zeros(embedding_dim))
            self._init_samples = 0
            self._init_done = False
        else:
            self._init_done = True

    def encoding_to_logits(self, input, head_ix, prev_codes):
        pass

    def forward(self, inputs, global_step=None, forced_codes=None, head_mask=None):
        input_shape = inputs.shape

        quantized_list = []
        vq_codes = []
        all_probs = []

        loss = torch.zeros(input_shape[0]).to(inputs.device)

        if (
            self.training
            and not self._init_done
            and self._init_delay_steps is not None
            and global_step < self._init_delay_steps
        ):
            self._init_cumsum += inputs.squeeze(dim=1).sum(dim=0)
            self._init_cumsquared += (inputs**2).squeeze(dim=1).sum(dim=0)
            self._init_samples += input_shape[0]
        elif self.training and not self._init_done and global_step >= self._init_delay_steps:
            init_mean = self._init_cumsum / float(self._init_samples)
            init_var = (
                torch.sqrt(self._init_cumsquared / float(self._init_samples) - init_mean**2)
                if self._init_dynamic_var
                else torch.full_like(init_mean, 0.5)
            )
            for hix, embedding in enumerate(self._embedding):
                this_mean = init_mean if hix == 0 else torch.zeros_like(init_mean)
                self._embedding[hix].weight.data = torch.normal(
                    mean=this_mean.unsqueeze(0).expand(self._num_embeddings, -1),
                    std=init_var.unsqueeze(0).expand(self._num_embeddings, -1) * self._init_decay_weight**hix,
                )
            self._init_done = True

        for head_ix, embedding in enumerate(self._embedding):

            this_input = inputs[:, 0, :]

            distances = torch.zeros(input_shape[0], embedding.weight.shape[0]).to(this_input.device)
            if self._learnable_priors is not None:
                distances += self._learnable_priors[head_ix]

            # Calculate distances
            if len(all_probs) > 0:
                resid_error = this_input
                for hix in range(head_ix):
                    resid_error = resid_error - torch.matmul(
                        all_probs[hix], self._embedding[hix].weight  # .detach()
                    ).squeeze(1)

                if self._cos_sim:
                    distances += cos_sim(resid_error, self._embedding[head_ix].weight)
                else:
                    distances += -1.0 * (
                        torch.sum(resid_error**2, dim=1, keepdim=True)
                        + torch.sum(self._embedding[head_ix].weight ** 2, dim=1)
                        - 2 * torch.matmul(resid_error, self._embedding[head_ix].weight.t())
                    )
            else:

                if self._cos_sim:
                    distances += cos_sim(this_input, self._embedding[head_ix].weight)
                else:
                    distances += -1.0 * (
                        torch.sum(this_input**2, dim=1, keepdim=True)
                        + torch.sum(self._embedding[head_ix].weight ** 2, dim=1)
                        - 2 * torch.matmul(this_input, self._embedding[head_ix].weight.t())
                    )

            # Convert distances into log probs
            logits = distances

            if self._learnable_priors is not None:
                prior = torch.softmax(self._learnable_priors[head_ix], dim=-1)
                posterior = torch.softmax(logits, dim=-1)
                kl_loss = torch.nn.KLDivLoss(reduction="none")
                loss += kl_loss(posterior, prior).sum(dim=-1)

            if self.training:

                gumbel_sched_weight = 2 - 2 / (1 + exp(-float(global_step) / float(self._temp_schedule_gamma)))
                gumbel_temp = (
                    self._gumbel_temp * max(gumbel_sched_weight, self._temp_min)
                    if self._temp_schedule
                    else self._gumbel_temp
                )
                probs = torch.nn.functional.gumbel_softmax(logits, tau=gumbel_temp, hard=True, dim=-1)
            else:
                indices = torch.argmax(logits, dim=-1)
                probs = onehot(indices, N=logits.shape[-1])

            all_probs.append(probs.unsqueeze(1))

        if forced_codes is not None:
            assert (
                forced_codes.shape[1] == self._num_heads
            ), "If forced_codes is supplied, it must be the same length as the number of quantizer heads! {:} vs {:}".format(
                forced_codes.shape[1], self._num_heads
            )
            vq_codes = forced_codes.unbind(dim=1)
        elif not isinstance(self._code_offset, int) or self._code_offset > 0:
            # print("code offset", self._code_offset)
            for head_ix in range(self._num_heads):
                this_offset = (
                    self._code_offset[head_ix] if not isinstance(self._code_offset, int) else self._code_offset
                )

                min_k = torch.topk(all_probs[head_ix], this_offset + 1, dim=1, largest=False).indices
                vq_codes.append(min_k[:, this_offset])
        else:
            vq_codes = [torch.argmax(probs, dim=-1).squeeze(1) for probs in all_probs]

        # Now that we have the codes, calculate their embeddings
        for head_ix, embedding in enumerate(self._embedding):
            # If soft training, use distribution
            if self.training:
                this_quantized = torch.matmul(
                    all_probs[head_ix],
                    embedding.weight,
                    # embedding.weight,
                )

            # otherwise use one hot
            else:
                this_quantized = embedding(vq_codes[head_ix].type(torch.LongTensor).to(inputs.device)).unsqueeze(1)

            quantized_list.append(this_quantized)

        quantized = torch.cat(quantized_list, dim=1)

        if head_mask is not None:
            # print('mask found')
            # print(head_mask)
            assert (
                head_mask.shape[1] == self._num_heads
            ), "If head_mask is set, it must be the same length as the number of quantizer heads! {:} vs {:}".format(
                head_mask.shape[1], self._num_heads
            )
            # print(vq_codes[0].shape)
            # print(quantized_list[0].shape)
            # print(head_mask.shape, quantized.shape)
            quantized = quantized * head_mask.unsqueeze(-1)

        if self._head_dropout is not None and self.training:
            mask = self._head_dropout.sample(sample_shape=(*quantized.shape[:-1], 1))
            if self._head_dropout_keep_first:
                mask[:, 0, :] = 1.0
            mask = torch.cumprod(mask, dim=1).to(quantized.device)
            quantized = quantized * mask

        quantized = torch.sum(quantized, dim=1)
        quantized = quantized.view(input_shape)

        return loss, quantized, vq_codes
