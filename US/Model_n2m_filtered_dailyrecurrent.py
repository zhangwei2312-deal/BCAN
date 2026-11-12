from Layers_n2m_filtered_dailyrecurrent import *
from project_paths import validate_english_bert_files, validate_loaded_bert
from contextlib import contextmanager, nullcontext
from torch.utils.checkpoint import checkpoint


@contextmanager
def _recompute_head_state(attention, previous):
    """Recomputation must use the original R_prev and must not advance memory."""
    current, debug = attention.prev_dynamic, attention.last_debug
    attention.prev_dynamic = previous
    try:
        yield
    finally:
        attention.prev_dynamic = current
        attention.last_debug = debug

class ResidualFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 2), nn.Softmax(dim=1))
            for _ in range(4)
        ])

    def forward(self, main, feat1, feat2):
        weights = sum(layer(main) for layer in self.attn_layers) / len(self.attn_layers)
        fused = main + weights[:, 0:1] * feat1 + weights[:, 1:2] * feat2
        return fused

class DRFN(nn.Module):
    def __init__(self, pretrained_model_path, dates, tickers, num_stock, hidden_news, hidden_market, d_hidden, hidn_rnn, heads_att, hidn_att, alpha, dropout=0, static_thresh=0.5, temperature=0.1, bert_batch_size=8):
        super(DRFN, self).__init__()
        self.pretrained_model_path = str(validate_english_bert_files(pretrained_model_path))
        self.dates = dates
        self.tickers = tickers
        self.dropout = dropout
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.news_encoder = news_feature_extractor(pretrained_model_path=self.pretrained_model_path,
                                                  hidden_dim=hidden_news, max_news_per_day=30,
                                                  batch_size=bert_batch_size)
        self.market_encoder = market_feature_extractor(d_market=5, hidden_size=16, output_dim=hidden_market)
        self.tensor = Graph_Tensor(num_stock, d_hidden, hidden_market, hidden_news, 
                                  temperature=temperature,
                                  static_thresh=static_thresh)
        self.GRUs_s = GRUModel(num_stock, d_hidden, hidn_rnn)
        self.GRUs_r = GRUModel(num_stock, d_hidden, hidn_rnn)
        self.residual_fusion = ResidualFusion(dim=heads_att * hidn_att)
        self.attentions = [
            Graph_Attention(hidn_rnn, hidn_att, alpha=self.alpha, num_stock=num_stock, dropout=dropout, residual=True, concat=True) for _
            in range(heads_att)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.X2Os = nn.Linear(heads_att * hidn_att, 1, bias=True)
        self.linear_news_last = nn.Linear(hidden_news, heads_att * hidn_att)
        self.linear_market_last = nn.Linear(hidden_market, heads_att * hidn_att)
        self.last_current_dynamic_signed = None
        self.reset_parameters()
        validate_loaded_bert(self.news_encoder)
        self.debug_daily_relation = False
        self.activation_checkpointing = True
        self.reset_relation_state()

    def reset_parameters(self):
        # AutoModel.from_pretrained has already populated TinyBERT. Reinitialize
        # only the task-specific modules and leave the pretrained encoder intact.
        def non_bert_parameters():
            for name, parameter in self.named_parameters():
                if not name.startswith("news_encoder.bert."):
                    yield name, parameter

        reset_parameters(non_bert_parameters)

    def clear_dynamic_state(self):
        """Start a new chronological sequence at epoch/phase/split boundaries."""
        self.reset_relation_state()
        self.clear_relation_memory_only()

    def clear_relation_memory_only(self):
        """No Explicit Relation Memory removes D_prev, not persistent GRU_r hidden."""
        for att in self.attentions:
            att.prev_dynamic = None
        self.last_current_dynamic_signed = None

    def reset_relation_state(self):
        self._relation_hidden = None
        self._relation_window_dates = None
        self._relation_date = None
        self._relation_step = 0
        self._relation_detach_count = 0
        self.last_daily_relation_debug = None

    def get_relation_hidden(self):
        return self._relation_hidden

    def set_relation_hidden(self, hidden, window_dates):
        if hidden is None:
            raise ValueError('Use clear_dynamic_state() to reset a sequence.')
        dates = tuple(pd.Timestamp(d) for d in window_dates)
        if len(dates) != 5 or tuple(sorted(set(dates))) != dates:
            raise ValueError('Hidden requires five ordered context dates.')
        if tuple(hidden.shape) != (len(self.tickers), self.GRUs_r.hidden_dim):
            raise ValueError('Relation hidden shape mismatch.')
        self._relation_hidden = hidden
        self._relation_window_dates = dates
        self._relation_date = dates[-1]

    def detach_relation_state(self):
        if self._relation_hidden is not None:
            self._relation_hidden = self._relation_hidden.detach()
            self._relation_detach_count += 1
            if self.last_daily_relation_debug is not None:
                self.last_daily_relation_debug['detached_after_step'] = True
            if self.debug_daily_relation and self._relation_detach_count <= 3:
                print(f'DailyRelation TBPTT detach: date={self._relation_date.date()}, '
                      f'chunk={self._relation_detach_count}, hidden_has_grad_fn=False')

    def _update_daily_relation(self, fused, window_dates):
        dates = tuple(pd.Timestamp(d) for d in window_dates)
        if len(dates) != 5 or fused.size(0) != 5 or tuple(sorted(set(dates))) != dates:
            raise RuntimeError('DailyRecurrent requires the original ordered five-day prediction context.')
        previous_date = self._relation_date
        hidden = self._relation_hidden
        if hidden is None:
            if self._relation_window_dates is not None:
                raise RuntimeError('Partial relation state reset.')
            inputs = fused
        else:
            if dates[:-1] != self._relation_window_dates[1:] or dates[-1] <= previous_date:
                raise RuntimeError('Non-consecutive or repeated window. Reset state at a new sequence boundary.')
            inputs = fused[-1:]
            assert inputs.size(0) == 1, 'Overlap dates must not update the relation GRU again.'
        # Persist the pre-dropout state; prediction GRUs_s remains window-local.
        new_hidden = self.GRUs_r(inputs, hidden)
        self.set_relation_hidden(new_hidden, dates)
        self._relation_step += 1
        self.last_daily_relation_debug = {
            'current_date': str(dates[-1].date()),
            'previous_relation_date': None if previous_date is None else str(previous_date.date()),
            'current_relation_date': str(dates[-1].date()),
            'update_steps': int(inputs.size(0)),
            'processed_dates': [str(d.date()) for d in (dates if hidden is None else dates[-1:])],
            'hidden_from_previous_day': hidden is not None,
            'hidden_input_detached': hidden is None or hidden.grad_fn is None,
            'overlap_reprocessed': False,
            'detached_after_step': False,
        }
        if self.debug_daily_relation and self._relation_step <= 4:
            self.last_daily_relation_debug['hidden_norm'] = float(new_hidden.detach().norm())
            print('DailyRelation ' + str(self.last_daily_relation_debug))
        return new_hidden

    def record_pcda_debug(self, previous_date, previous_heads, effective_lambda, pair_count, enabled):
        info = self.last_daily_relation_debug
        if previous_heads is not None:
            if previous_heads.shape != (8, len(self.tickers), len(self.tickers)):
                raise RuntimeError('PCDA must receive eight head-specific previous relations.')
            if previous_heads.requires_grad:
                raise RuntimeError('PCDA selection relations must be detached.')
            if previous_date is None or str(pd.Timestamp(previous_date).date()) != info['previous_relation_date']:
                raise RuntimeError('PCDA relation date does not match the previous trading day.')
        elif previous_date is not None:
            raise RuntimeError('A PCDA date was supplied without relations.')
        info.update(pcda_previous_date=None if previous_date is None else str(pd.Timestamp(previous_date).date()),
                    pcda_target_date=info['current_date'], pcda_time_steps=1,
                    pcda_enabled=bool(enabled), pcda_active=bool(enabled and effective_lambda > 0 and pair_count > 0),
                    effective_lambda_pcda=float(effective_lambda), pcda_pairs=int(pair_count),
                    previous_head_shape=None if previous_heads is None else list(previous_heads.shape))
        if self.debug_daily_relation and self._relation_step <= 4:
            print('DailyPCDA ' + str(info))

    def encode_inputs(self, news_df, market_df):
        x_news = self.news_encoder(news_df)
        x_market = self.market_encoder(market_df)
        return x_market, x_news

    def _stateless_forward(self, function, *inputs):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *inputs, use_reentrant=False)
        return function(*inputs)

    def _attention_forward(self, attention, x_s, x_r, relation_static):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            previous = attention.prev_dynamic
            return checkpoint(
                attention, x_s, x_r, relation_static, use_reentrant=False,
                context_fn=lambda: (nullcontext(), _recompute_head_state(attention, previous)),
            )
        return attention(x_s, x_r, relation_static)

    def _collect_head_dynamics(self):
        dynamic_relations = []
        for att in self.attentions:
            if hasattr(att, 'prev_dynamic') and att.prev_dynamic is not None:
                dynamic_relations.append(att.prev_dynamic)
        if dynamic_relations:
            return torch.stack(dynamic_relations, dim=0)  # [H, N, N]
        return None

    def get_current_dynamic_relation_from_heads_signed(self):
        """
        仅用于分析/可视化：保留符号的多头均值。
        PCDA 使用 previous-day 的逐头矩阵，不使用这个可能互相抵消的均值。
        """
        stack = self._collect_head_dynamics()
        if stack is None:
            self.last_current_dynamic_signed = None
            return None
        signed = stack.mean(dim=0)
        signed = signed - torch.diag_embed(torch.diag(signed))
        self.last_current_dynamic_signed = signed.detach()
        return signed

    def forward(self, news_df, market_df, relation_static, S_pre=None, *, window_dates=None):
        if window_dates is None:
            if not isinstance(market_df, pd.DataFrame):
                raise ValueError('Precomputed inputs require explicit window_dates.')
            window_dates = sorted(market_df['Date'].unique())
        if isinstance(market_df, pd.DataFrame):
            if tuple(pd.to_datetime(sorted(market_df['Date'].unique()))) != tuple(pd.to_datetime(window_dates)):
                raise ValueError('Input dates disagree with recurrent-state dates.')
        x_market_raw, x_news_raw = self._stateless_forward(self.encode_inputs, news_df, market_df)
        x_s, sim_loss, x_news_align, x_market_align = self._stateless_forward(self.tensor,
            x_market_raw, x_news_raw, S_pre
        )
        # Shared multimodal representation for the temporal branches.
        x_r = self._update_daily_relation(x_s, window_dates)
        x_s = self._stateless_forward(self.GRUs_s, x_s)
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        x_s = F.dropout(x_s, self.dropout, training=self.training)
        # 调用get_relation会更新每个attention的prev_dynamic（当前时刻的动态关系）
        x = torch.cat([self._attention_forward(att, x_s, x_r, relation_static) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        valid_news_last = x_news_raw[-1].norm(dim=-1) > 1e-5
        x_news_last = self.linear_news_last(x_news_raw[-1])
        x_news_last = x_news_last * valid_news_last.to(x_news_last.dtype).unsqueeze(1)
        x_market_last = self.linear_market_last(x_market_raw[-1])
        x = self.residual_fusion(x, x_news_last, x_market_last)
        output = self.X2Os(x)  # Linear prediction output.
        # Return the signed head mean for the existing forward interface.
        # Current-day PCDA uses detached previous-day head-wise matrices instead.
        current_dynamic = self.get_current_dynamic_relation_from_heads_signed()
        if self.debug_daily_relation and self._relation_step <= 4:
            heads = self._collect_head_dynamics()
            self.last_daily_relation_debug['current_head_shape'] = list(heads.shape)
            print(f'DailyRelation produced: date={self._relation_date.date()}, head_shape={list(heads.shape)}')
        return output, sim_loss, x_r, x_news_align, x_market_align, current_dynamic
