import torch
import html
import re

from utils import *
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from collections import defaultdict


def clean_headlines_cell(raw_text, max_news_per_day=30):
    """Apply the shared, deliberately light headline cleaning policy."""
    if pd.isna(raw_text):
        return []
    raw_text = str(raw_text)
    if raw_text.strip() == "0":
        return []

    cleaned = []
    seen = set()
    for headline in raw_text.split('|'):
        headline = re.sub(r"\s+", " ", html.unescape(headline)).strip()
        if not headline or headline in seen:
            continue
        seen.add(headline)
        cleaned.append(headline)
        if len(cleaned) >= max_news_per_day:
            break
    return cleaned


def _headline_pivot(news_df, text_col, local_dates, local_tickers):
    """Build a stable stock-day headline table without assuming unique CSV rows."""
    grouped = (
        news_df.groupby(['Date', 'Ticker'], sort=False, dropna=False)[text_col]
        .agg(lambda values: '|'.join(str(value) for value in values if not pd.isna(value)))
        .reset_index()
    )
    return (
        grouped.pivot(index='Date', columns='Ticker', values=text_col)
        .reindex(index=local_dates, columns=local_tickers)
        .fillna("0")
    )


class news_feature_extractor(nn.Module):
    def __init__(self, pretrained_model_path: str, hidden_dim, max_news_per_day=30, batch_size=16):
        super(news_feature_extractor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.bert = AutoModel.from_pretrained(pretrained_model_path)
        # Keep the pretrained text encoder fixed throughout this pipeline.
        self.bert.requires_grad_(False)
        self.bert.eval()
        # GRU 会自动处理：输入是 TinyBERT 的 312 维，输出是你设定的 hidden_dim（默认 768）
        self.gru = nn.GRU(input_size=self.bert.config.hidden_size, 
                          hidden_size=hidden_dim, 
                          batch_first=True)
        self.max_news_per_day = max_news_per_day
        self.batch_size = batch_size
        self._precomputed_cls = None 
        self.hidden_dim = hidden_dim

    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()
        return self

    def _run_gru_on_step_inputs(self, step_inputs, T, S, device, output_dim):
        """真正的并行化计算核心 - 彻底解决 Inplace 报错并大幅提速"""
        h_prev_all = torch.zeros(1, S, output_dim, device=device)
        step_results = []
        for t in range(T):
            padded, batch_s_indices, lengths = step_inputs[t]
            if padded is not None and len(batch_s_indices) > 0:
                padded = padded.to(device)
                indices = torch.tensor(batch_s_indices, device=device)
                lengths = torch.as_tensor(lengths, dtype=torch.long, device='cpu')
                if lengths.numel() != len(batch_s_indices):
                    raise RuntimeError("Headline lengths and active-stock indices are misaligned.")
                # 批量提取并计算
                h_active = h_prev_all[:, indices, :]
                packed = pack_padded_sequence(
                    padded,
                    lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, h_new_active = self.gru(packed, h_active.contiguous())
                # 使用 scatter 避开 Inplace 报错
                expand_indices = indices.view(1, -1, 1).expand(1, -1, output_dim)
                h_prev_all = h_prev_all.scatter(1, expand_indices, h_new_active)
                
                step_res = torch.zeros(S, output_dim, device=device)
                step_res[indices, :] = h_new_active.squeeze(0)
                step_results.append(step_res)
            else:
                step_results.append(torch.zeros(S, output_dim, device=device))
        return torch.stack(step_results, dim=0)

    def preprocess(self, news_df: pd.DataFrame):
        """Apply the headline GRU to precomputed features with their true lengths."""
        df = news_df.copy()
        df.columns = df.columns.str.strip()
        local_dates = sorted(df['Date'].unique())
        local_tickers = sorted(df['Ticker'].unique())
        text_cols = [col for col in df.columns if col not in ['Date', 'Ticker']]
        text_col = text_cols[0] if text_cols else 'Headlines'

        pivot_df = _headline_pivot(df, text_col, local_dates, local_tickers)
        nested_text = pivot_df.values.tolist()
        T, S = len(nested_text), len(nested_text[0])

        if self._precomputed_cls is None or len(self._precomputed_cls) == 0:
            raise RuntimeError(
                "TinyBERT CLS cache is empty. Call precompute_bert_cls() before news inference."
            )

        # 动态获取 TinyBERT 实际维度 (312)
        sample_key = list(self._precomputed_cls.keys())[0]
        actual_bert_dim = self._precomputed_cls[sample_key].shape[1]

        step_inputs = []
        for t in range(T):
            date_str = str(local_dates[t].date())
            batch_vecs, batch_s_indices, lengths = [], [], []
            for s in range(S):
                key = (date_str, local_tickers[s])
                if key in self._precomputed_cls:
                    batch_vecs.append(self._precomputed_cls[key].unsqueeze(0))
                    batch_s_indices.append(s)
                    lengths.append(int(self._precomputed_cls[key].shape[0]))
            if batch_vecs:
                max_news = max(v.shape[1] for v in batch_vecs)
                padded = torch.zeros(len(batch_vecs), max_news, actual_bert_dim)
                for idx, v in enumerate(batch_vecs):
                    padded[idx, :v.shape[1], :] = v[0]
                step_inputs.append((padded, batch_s_indices, lengths))
            else:
                step_inputs.append((None, [], []))
        return (step_inputs, T, S)

    def forward(self, input_data):
        """Encode precomputed headline tuples or a raw news DataFrame."""
        device = next(self.parameters()).device
        if isinstance(input_data, tuple): 
            step_inputs, T, S = input_data
            return self._run_gru_on_step_inputs(step_inputs, T, S, device, self.hidden_dim)
        
        # 回测/后备逻辑 (DataFrame)
        if isinstance(input_data, pd.DataFrame):
            step_inputs, T, S = self.preprocess(input_data)
            return self._run_gru_on_step_inputs(step_inputs, T, S, device, self.hidden_dim)
        
        return input_data.to(device) # 如果是Tensor直接返回

    def precompute_bert_cls(self, news_df):
        """预计算所有新闻的BERT CLS向量并缓存（BERT冻结，只需算一次）"""
        self.bert.eval()
        device = next(self.bert.parameters()).device

        news_df = news_df.copy()
        news_df.columns = news_df.columns.str.strip()
        text_cols = [col for col in news_df.columns if col not in ['Date', 'Ticker']]
        text_col = text_cols[0] if text_cols else 'Content'

        local_dates = sorted(news_df['Date'].unique())
        local_tickers = sorted(news_df['Ticker'].unique())

        pivot_df = _headline_pivot(news_df, text_col, local_dates, local_tickers)
        nested_text = pivot_df.values.tolist()

        # 收集所有需要编码的文本
        all_texts = []
        index_map = []
        for t in range(len(local_dates)):
            for s in range(len(local_tickers)):
                raw_text = nested_text[t][s]
                news_list = clean_headlines_cell(raw_text, self.max_news_per_day)
                for news in news_list:
                    all_texts.append(news)
                    index_map.append((t, s))

        if not all_texts:
            self._precomputed_cls = {}
            raise RuntimeError("No valid headlines were available for TinyBERT CLS precomputation.")

        print(f"    [预计算] BERT编码 {len(all_texts)} 条新闻...")
        cls_vecs_list = []

        with torch.no_grad():
            for batch_start in range(0, len(all_texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(all_texts))
                batch_texts = all_texts[batch_start:batch_end]

                encoding = self.tokenizer(
                    batch_texts, padding=True, truncation=True,
                    return_tensors="pt", max_length=40
                )
                encoding = {k: v.to(device) for k, v in encoding.items()}

                outputs = self.bert(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"]
                )
                # Masked average pooling over token states, as specified in Eq. (1) of the paper:
                # padding positions are excluded so the mean runs over real tokens only.
                token_mask = encoding["attention_mask"].unsqueeze(-1).to(
                    outputs.last_hidden_state.dtype)
                pooled = (outputs.last_hidden_state * token_mask).sum(dim=1)
                pooled = pooled / token_mask.sum(dim=1).clamp(min=1e-6)
                batch_cls = pooled.cpu()
                cls_vecs_list.append(batch_cls)

        cls_vecs = torch.cat(cls_vecs_list, dim=0)
        del cls_vecs_list

        # 按索引分组缓存
        grouped = defaultdict(list)
        for idx, (t, s) in enumerate(index_map):
            grouped[(t, s)].append(cls_vecs[idx])

        self._precomputed_cls = {}
        for (t, s), vecs in grouped.items():
            date_str = str(local_dates[t].date())
            ticker = local_tickers[s]
            self._precomputed_cls[(date_str, ticker)] = torch.stack(vecs)

        total_mem = sum(v.numel() * 4 for v in self._precomputed_cls.values()) / 1024 / 1024
        print(f"    [预计算] 完成！缓存 {len(self._precomputed_cls)} 条记录，占用 {total_mem:.1f} MB")


class market_feature_extractor(nn.Module):
    def __init__(self, d_market: int, hidden_size: int , output_dim: int,
                 num_layers: int = 1):
        super().__init__()
        self.d_market = d_market
        self.input_proj = nn.Linear(d_market, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def preprocess(self, market_df: pd.DataFrame):
        """Convert a market DataFrame to an ordered feature tensor."""
        feature_cols = [col for col in market_df.columns if col not in ['Date', 'Ticker']]
        local_dates = sorted(market_df['Date'].unique())
        local_tickers = sorted(market_df['Ticker'].unique())

        pivot = market_df.pivot(index='Date', columns='Ticker', values=feature_cols)
        pivot = pivot.reindex(index=local_dates)
        # pandas returns columns as (feature, ticker). The model expects
        # [Date, Ticker, Feature], so make the column order explicit.
        pivot = pivot.swaplevel(0, 1, axis=1)
        ordered_cols = pd.MultiIndex.from_product([local_tickers, feature_cols])
        pivot = pivot.reindex(columns=ordered_cols).fillna(0)
        values = pivot.to_numpy().reshape(len(local_dates), len(local_tickers), self.d_market)
        # 预先生成 Tensor
        return torch.tensor(values, dtype=torch.float32)

    def forward(self, market_input):
        # 兼容回测（传入 DataFrame）和 提速训练（传入 Tensor）
        if isinstance(market_input, pd.DataFrame):
            x = self.preprocess(market_input).to(self.output_proj.weight.device)
        else:
            x = market_input.to(self.output_proj.weight.device)

        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        out_seq, _ = self.gru(x)
        out = self.output_proj(out_seq.permute(1, 0, 2))
        return out


class MaxSimilarityLoss(nn.Module):
    def __init__(self, margin=0.5, temperature=0.1, static_thresh=0.5):
        super().__init__()
        self.log_margin = nn.Parameter(torch.log(torch.tensor(margin)))
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))

        self.static_thresh = static_thresh

    @property
    def temp(self):
        temp = torch.exp(self.log_temp)
        return temp

    def forward(self, market_feat, news_feat, S_pre=None):

        t, num_stock, d = news_feat.shape
        loss = 0.0
        valid_items = 0  # 有效损失计数器

        for time_step in range(t):
            n_t = news_feat[time_step]  # [num_stock, d]
            m_t = market_feat[time_step]  # [num_stock, d]

            # Identify valid news anchors from nonzero news features.
            # Rows are news anchors. Zero-news rows have no textual evidence.
            valid_mask = n_t.norm(dim=-1) > 1e-5

            m_t = F.normalize(m_t, p=2, dim=-1)
            n_t = F.normalize(n_t, p=2, dim=-1)

            # 计算相似度矩阵 [num_stock, num_stock]
            sim_matrix = F.cosine_similarity(n_t.unsqueeze(1), m_t.unsqueeze(0), dim=-1) / self.temp

            if S_pre is not None:
                # 创建正样本掩码（自相关 + 互相关）
                positive_mask = torch.zeros(num_stock, num_stock, device=sim_matrix.device, dtype=torch.bool)
                
                # 对角线（自相关）：正样本
                positive_mask.fill_diagonal_(True)
                
                # 静态互相关：正样本
                static_related = (S_pre > self.static_thresh)
                positive_mask = positive_mask | static_related
                
                # Dynamic relations are not inserted into the base hard-positive mask.
                # High-confidence dynamic pairs are handled separately by PCDA.
                
                # 负样本掩码（排除正样本）
                negative_mask = ~positive_mask
                
                # 对每个样本计算 Multi-Positive InfoNCE Loss
                for i in range(num_stock):
                    # Exclude anchors without news.
                    if not valid_mask[i]:
                        continue

                    # 获取正样本索引
                    pos_indices = torch.where(positive_mask[i])[0]
                    # 获取负样本索引
                    neg_indices = torch.where(negative_mask[i])[0]
                    
                    if len(pos_indices) == 0:
                        continue
                    
                    if len(neg_indices) == 0:
                        continue
                    
                    # 分子：自相关 + 互相关
                    pos_sim = sim_matrix[i, pos_indices]

                    # 负样本（不包含正样本）
                    neg_sim = sim_matrix[i, neg_indices]

                    # Compute the probability-mass loss in log space.
                    pos_lse = torch.logsumexp(pos_sim, dim=0)        # log(sum(exp(pos)))
                    all_sim = torch.cat([pos_sim, neg_sim], dim=0)
                    all_lse = torch.logsumexp(all_sim, dim=0)        # log(sum(exp(all)))
                    loss_i = all_lse - pos_lse                       # 等价于 -log(sum(pos)/sum(all))
                    loss += loss_i
                    valid_items += 1  # Count valid anchors.
            else:
                labels = torch.arange(num_stock, device=sim_matrix.device)
                valid_indices = torch.where(valid_mask)[0]
                if valid_indices.numel() > 0:
                    loss += F.cross_entropy(
                        sim_matrix[valid_indices], labels[valid_indices]
                    )
                    valid_items += int(valid_indices.numel())

        # Normalize by the number of valid anchors.
        if valid_items > 0:
            return loss / valid_items
        else:
            return torch.tensor(0.0, device=news_feat.device, requires_grad=True)


class Graph_Tensor(nn.Module):

    def __init__(self, num_stock, d_hidden, d_market, d_news, temperature=0.1, bias=True, static_thresh=0.5):
        super(Graph_Tensor, self).__init__()
        self.num_stock = num_stock
        self.d_hidden = d_hidden
        self.d_market = d_market
        self.d_news = d_news
        self.temperature = temperature
        self.seq_transformation_news = nn.Conv1d(d_news, d_hidden, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_markets = nn.Conv1d(d_market, d_hidden, kernel_size=1, stride=1, bias=False)
        self.tensorGraph = nn.Parameter(torch.zeros(num_stock, d_hidden, d_hidden, d_hidden))
        self.W = nn.Parameter(torch.zeros(num_stock, 2 * d_hidden, d_hidden))
        self.b = nn.Parameter(torch.zeros(num_stock, d_hidden))
        self.sim_loss = MaxSimilarityLoss(temperature=self.temperature, static_thresh=static_thresh)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, market, news, S_pre=None):

        t, num_stocks = news.size()[0], news.size()[1]

        news_transformed = news.reshape(-1, self.d_news)
        news_transformed = torch.transpose(news_transformed, 0, 1).unsqueeze(0)
        news_transformed = self.seq_transformation_news(news_transformed)
        news_transformed = news_transformed.squeeze().transpose(0, 1)
        news_transformed = news_transformed.reshape(t, num_stocks, self.d_hidden)

        market_transformed = market.reshape(-1, self.d_market)
        market_transformed = torch.transpose(market_transformed, 0, 1).unsqueeze(0)
        market_transformed = self.seq_transformation_markets(market_transformed)
        market_transformed = market_transformed.squeeze().transpose(0, 1)
        market_transformed = market_transformed.reshape(t, num_stocks, self.d_hidden)

        # Rows are news anchors and columns are market candidates.
        sim_loss = self.sim_loss(market_transformed, news_transformed, S_pre)

        x_news_tensor = news_transformed.unsqueeze(2)
        x_news_tensor = x_news_tensor.unsqueeze(2)
        x_market_tensor = market_transformed.unsqueeze(-1)
        # 第一步产生的多余1在倒数第二维，用squeeze(-2)
        # Preserve signed bilinear interactions.
        temp_tensor = x_news_tensor.matmul(self.tensorGraph).squeeze(-2)
        # 第二步产生的多余1在最后一维，用squeeze(-1)
        temp_tensor = temp_tensor.matmul(x_market_tensor).squeeze(-1)
        x_linear = torch.cat((news_transformed, market_transformed), axis=-1)
        temp_linear = torch.bmm(x_linear.transpose(0, 1), self.W)
        temp_linear = temp_linear.transpose(0, 1)

        output = torch.tanh(temp_tensor + temp_linear + self.b)
        return output, sim_loss, news_transformed, market_transformed


class Graph_GRUCell(nn.Module):
    def __init__(self, num_nodes, input_size, hidden_size, bias=True):
        super(Graph_GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = (1 - inputgate) * newgate + inputgate * hidden
        return hy

class GRUModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, bias=True):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = Graph_GRUCell(num_nodes, input_dim, hidden_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.size()[1], self.hidden_dim, device=x.device, dtype=x.dtype)
        for seq in range(x.size(0)):
            hidden = self.gru_cell(x[seq], hidden)

        return hidden


class Graph_Attention(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, num_stock=105, concat=True, residual=False):
        super(Graph_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_stock = num_stock
        self.concat = concat
        self.residual = residual
        self.seq_transformation_r = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_s = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)

        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        self.W_static = nn.Parameter(torch.zeros(num_stock, num_stock).type(torch.FloatTensor), requires_grad=True)

        self.linear_pos = nn.Linear(num_stock, num_stock)
        self.linear_neg = nn.Linear(num_stock, num_stock)
        self.coef_revise = False
        self.prev_dynamic = None
        self.last_debug = None

    def manhattan_distance(self, Q, K):
        diff = Q.unsqueeze(1) - K.unsqueeze(0)
        dist = torch.abs(diff)
        dist_sum = torch.sum(dist, dim=-1)
        dist_sum = dist_sum / (dist_sum.mean() + 1e-6)
        return dist_sum

    def dynamic_to_static(self, dynamic, static):
        Q_dynamic = dynamic
        K_static = static
        V_static = static
        logits = torch.matmul(Q_dynamic, K_static.t()) / torch.sqrt(
            torch.tensor(K_static.size(1), dtype=torch.float32))
        # Mask self-attention along the diagonal.
        diag_mask = torch.eye(logits.size(0), device=logits.device) * -1e9
        logits = logits + diag_mask
        attn_sta = F.softmax(logits, dim=-1)
        attn_sta = torch.matmul(attn_sta, V_static)
        return attn_sta

    def static_to_dynamic(self, static, dynamic):
        Q_static = static
        K_dynamic = dynamic
        V_dynamic = dynamic
        logits = torch.matmul(Q_static, K_dynamic.t()) / torch.sqrt(
            torch.tensor(K_dynamic.size(1), dtype=torch.float32))
        # Mask self-attention along the diagonal.
        diag_mask = torch.eye(logits.size(0), device=logits.device) * -1e9
        logits = logits + diag_mask
        attn_dyn = F.softmax(logits, dim=-1)
        attn_dyn = torch.matmul(attn_dyn, V_dynamic)
        return attn_dyn

    def get_relation(self, input_r, relation_static, prev_dynamic=None):
        def tensor_stats(x):
            x_det = x.detach()
            flat = x_det.reshape(-1).float()
            return {
                "min": float(flat.min().item()),
                "max": float(flat.max().item()),
                "mean": float(flat.mean().item()),
                "std": float(flat.std(unbiased=False).item()),
                "abs_max": float(flat.abs().max().item()),
                "q90": float(torch.quantile(flat, 0.90).item()),
                "q95": float(torch.quantile(flat, 0.95).item()),
                "q99": float(torch.quantile(flat, 0.99).item()),
            }

        num_stock = input_r.shape[0]

        seq_r = torch.transpose(input_r, 0, 1).unsqueeze(0)  # [1, in_features, num_stocks]
        seq_fts_r = torch.transpose(self.seq_transformation_r(seq_r), 2, 1).squeeze(0)  # [N, d]

        Q = seq_fts_r
        K = seq_fts_r

        logits_pos = torch.matmul(Q, K.T) / torch.sqrt(
            torch.tensor(Q.size(1), dtype=torch.float32, device=Q.device)
        )
        D = self.manhattan_distance(Q, K)
        attn_pos = logits_pos * torch.exp(-D)

        logits_neg = torch.matmul(Q, -K.T) / torch.sqrt(
            torch.tensor(Q.size(1), dtype=torch.float32, device=Q.device)
        )
        D_1 = self.manhattan_distance(Q, -K)
        attn_neg = logits_neg * torch.exp(-D_1)

        att_pos_cross = self.dynamic_to_static(attn_pos, attn_neg)
        att_neg_cross = self.static_to_dynamic(attn_neg, attn_pos)

        # 残差式保留原始动态信号
        att_pos = attn_pos + att_pos_cross
        att_neg = attn_neg + att_neg_cross

        dynamic = self.linear_pos(att_pos) + self.linear_neg(att_neg)

        if prev_dynamic is None:
            static = torch.mul(relation_static, self.W_static)
        else:
            static = torch.sigmoid(self.alpha) * prev_dynamic + \
                     (1 - torch.sigmoid(self.alpha)) * (relation_static * self.W_static)

        att_sta = self.dynamic_to_static(dynamic, static)
        att_dyn = self.static_to_dynamic(static, dynamic)
        coefs = self.linear_pos(att_sta) + self.linear_neg(att_dyn)

        if self.coef_revise:
            coefs_eye = (torch.sigmoid(coefs) - 0.5) * 2
        else:
            coefs_eye = coefs - torch.diag_embed(torch.diag(coefs))

        self.prev_dynamic = dynamic.detach()

        # ===== 调试统计 =====
        self.last_debug = {
            "logits_pos": tensor_stats(logits_pos),
            "distance_pos": tensor_stats(D),
            "attn_pos": tensor_stats(attn_pos),

            "logits_neg": tensor_stats(logits_neg),
            "distance_neg": tensor_stats(D_1),
            "attn_neg": tensor_stats(attn_neg),

            "att_pos": tensor_stats(att_pos),
            "att_neg": tensor_stats(att_neg),

            "dynamic": tensor_stats(dynamic),
            "static": tensor_stats(static),
            "att_sta": tensor_stats(att_sta),
            "att_dyn": tensor_stats(att_dyn),
            "coefs": tensor_stats(coefs),
            "coefs_eye": tensor_stats(coefs_eye),

            "dynamic_gt_001_ratio": float((dynamic.abs() > 0.01).float().mean().item()),
            "dynamic_gt_005_ratio": float((dynamic.abs() > 0.05).float().mean().item()),
            "dynamic_gt_01_ratio": float((dynamic.abs() > 0.1).float().mean().item()),
            "dynamic_gt_03_ratio": float((dynamic.abs() > 0.3).float().mean().item()),
        }

        return coefs_eye




    def forward(self, input_s, input_r, relation_static):
        coefs_eye = self.get_relation(input_r, relation_static, prev_dynamic=self.prev_dynamic)
        seq_s = torch.transpose(input_s, 0, 1).unsqueeze(0)
        seq_fts_s = self.seq_transformation_s(seq_s)
        seq_fts_s = F.dropout(torch.transpose(seq_fts_s.squeeze(0), 0, 1), self.dropout, training=self.training)
        ret = torch.matmul(coefs_eye.unsqueeze(1), seq_fts_s).squeeze()
        if self.concat:
            return torch.tanh(ret)
        else:
            return ret

