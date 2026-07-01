from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import openood.utils.comm as comm
import math
from torch.cuda.amp import autocast

from .base_postprocessor import BasePostprocessor
import pdb


def compute_os_variance_tensor(os_tensor, th):
    """
    Calculate weighted variance for PyTorch tensors

    Args:
        os_tensor: (torch.Tensor) OOD scores tensor
        th: (float) threshold value

    Returns:
        (torch.Tensor) weighted variance
    """
    device = os_tensor.device
    
    # 二值化mask
    mask = (os_tensor >= th).float()
    
    # 计算权重
    n_pixels = os_tensor.numel()
    n_pixels1 = torch.sum(mask)
    weight1 = n_pixels1 / n_pixels
    weight0 = 1 - weight1
    
    # 处理空类别
    if weight1 == 0 or weight0 == 0:
        return torch.tensor(float('inf'), device=device)
    
    # 获取两类数值
    class1 = os_tensor[mask.bool()]
    class0 = os_tensor[~mask.bool()]
    
    # 计算方差
    var0 = torch.var(class0, unbiased=False) if class0.numel() > 0 else torch.tensor(0.0, device=device)
    var1 = torch.var(class1, unbiased=False) if class1.numel() > 0 else torch.tensor(0.0, device=device)
    
    return weight0 * var0 + weight1 * var1

def find_best_threshold(os_training_queue):
    """
    主函数：在GPU/CPU上寻找最优阈值
    
    Args:
        os_training_queue: (torch.Tensor) 输入分数张量
    
    Returns:
        best_threshold: (float) 最优阈值
    """
    # 生成阈值范围（自动匹配设备）
    threshold_range = torch.arange(0, 1, 0.01, device=os_training_queue.device)
    
    # 计算各阈值对应的指标
    criterias = torch.stack([compute_os_variance_tensor(os_training_queue, th) for th in threshold_range])
    
    # 找到所有最小方差的位置
    min_val = torch.min(criterias)
    mask = (criterias == min_val)
    candidate_indices = torch.where(mask)[0]

    # 从候选中选择中间位置的阈值
    if len(candidate_indices) == 0:
        return threshold_range[torch.argmin(criterias)].item()  # 回退机制

    # 直接取中间索引
    mid_index = len(candidate_indices) // 2
    best_threshold = threshold_range[candidate_indices[mid_index]]

    return best_threshold.item()


def kmeans_l2_normalized(x, n_clusters, max_iter=100, tol=1e-4):
    """
    L2归一化数据的高效K-Means（基于余弦相似度）
    Args:
        x: 输入数据（已L2归一化），形状 [N, D]
        n_clusters: 聚类簇数
        max_iter: 最大迭代次数
        tol: 中心点变化容忍度
        device: 计算设备（'cuda' 或 'cpu'）
    Returns:
        centroids: 聚类中心 [K, D]
        labels: 样本标签 [N]
    """
    N, D = x.shape
    # 初始化中心点：随机选择样本
    indices = torch.randperm(N)[:n_clusters].to(x.device)
    centroids = x[indices]

    for _ in range(max_iter):
        # 计算余弦相似度（等价于点积）[N, K]
        similarities = torch.mm(x, centroids.t())  # 关键优化：矩阵乘法代替距离计算

        # 分配标签：取最大相似度 [N]
        labels = torch.argmax(similarities, dim=1)

        # 更新中心点
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.any():
                # 对簇内样本求均值（保持L2归一化）
                new_centroids[k] = x[mask].mean(dim=0)
                new_centroids[k] /= torch.norm(new_centroids[k])  # 重新归一化
            else:
                # 处理空簇：随机选择一个样本
                new_centroids[k] = x[torch.randint(0, N, (1,)).to(x.device)]

        # 检查收敛条件
        if torch.norm(centroids - new_centroids) < tol:
            break
        centroids = new_centroids

    return centroids, labels

def update_queue(to_add, queue, max_len=1000, init=False):
    if init:
        queue = queue
    else:
        queue = torch.cat((queue, to_add), dim=0)
    if queue.size(0) > max_len:
        queue = queue[-max_len:, :]
    return queue

def activation_aware_score(output, id_num, ood_num, step):
    if step != 0:   ####### gamma =5 确实结果变好了，比0好，说明按照activation score 来加权确实结果更好！结果稳定了很多。
        softmax_sums = []
        step = int(step)
        for i in range(id_num, id_num + ood_num, step):  # 列索引从 1000 到 1999
            ## 下面方法每一段包含了前一段， 相当于前面的算了比较多次，权重大一些; 这样缓解了一些情况：step 1,2,10 比step 0效果好，但是还是会随着neg number 数量增加而变差; 一定要把activation score 的权重算进来。
            softmax_output = output[:,:i+step].softmax(dim=-1) 
            sum_score = softmax_output[:, :id_num].sum(dim=-1)  # 即使没做加权，结果也好了一些; 就用这个了。

            ##### 还是要做加权，试试乘以新加部分的权重!! 加权一直做不出来。
            ### 还是不行，乘上之后加过下降了！ 而且并没有想象中的对 large negative number 的稳定性， number 多的时候结果还是下降了不少； 下面的实验不再继续尝试。
            # score_weight = selected_combined_score[i-class_num:i-class_num+step].sum()
            # sum_score = sum_score * score_weight  
    
            #  如果每一段不包含前面一段呢？手动对每个进行加权？用 activation score 进行加权？
            # softmax_output = torch.cat((output[:,:class_num], output[:,i:i+step]),dim=-1).softmax(dim=-1)  # 对step 异常敏感; 即使某个step 有效也不行因为
            ############# 一定要改成，对后面的negative label apply small weights; 这样negative label 再多，其对结果的影响也会小一些。
            # sum_score = softmax_output[:, :class_num].sum(dim=-1)  # 分group 算
            # sum_score = sum_score * selected_combined_score[i-class_num:i-class_num+step].sum()  ### 结果对step 非常敏感，不能用。
            # print(selected_combined_score[class_num:i-class_num+step].sum())
            # 根绝neg score 做加权。这样 重要的negative label 可以得到更高的权重。引入更多的negative label 的影响就小了，就对neglabel number 不敏感了。否则还是敏感。
            softmax_sums.append(sum_score)
        # 将 softmax_sums 堆叠为一个张量：batch_size x 1000
        softmax_sums = torch.stack(softmax_sums, dim=-1)
        # 对列（1000 次 softmax 的结果）求均值作为最终结果
        conf_in = softmax_sums.mean(dim=-1)  # 最终为 batch_size 的向量
    else:
        full_sim = output.softmax(dim=-1)
        conf_in = full_sim[:, :id_num].sum(dim=-1)
    return conf_in


class GaussianModel(nn.Module):
    """Gaussian Distribution Model for each class with Positive and Negative distributions"""
    def __init__(self, input_shape, num_classes, clip_weights, sigma=1.0, epsilon=0.001, device='cuda'):
        super(GaussianModel, self).__init__()
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        # Positive DOTA: for target class features
        self.mu = clip_weights.T.to(device)
        self.c = torch.ones(num_classes).to(device)
        self.Sigma = sigma * torch.eye(input_shape).repeat(num_classes, 1, 1).to(device)
        self.overall_Sigma = torch.mean(self.Sigma, dim=0)
        self.Lambda = torch.pinverse(self.overall_Sigma.double()).to(device).float()
        
        # Negative DOTA: for distractor/negative features
        self.neg_mu = clip_weights.T.to(device).clone()
        self.neg_c = torch.ones(num_classes).to(device)
        self.neg_Sigma = sigma * torch.eye(input_shape).repeat(num_classes, 1, 1).to(device)
        self.neg_overall_Sigma = torch.mean(self.neg_Sigma, dim=0)
        self.neg_Lambda = torch.pinverse(self.neg_overall_Sigma.double()).to(device).float()
        
    def fit(self, x, y, is_negative=False):
        """
        Update Gaussian parameters with weighted samples
        Args:
            x: (batch_size, feature_dim) - image features
            y: (batch_size, num_classes) - soft labels (probability distribution)
            is_negative: if True, update negative distribution; else update positive
        """
        # Select which distribution to update
        if is_negative:
            mu, c, Sigma = self.neg_mu, self.neg_c, self.neg_Sigma
        else:
            mu, c, Sigma = self.mu, self.c, self.Sigma
            
        with torch.no_grad():
            with autocast():
                sum_weights = torch.sum(y, dim=0)  # (num_classes,)
                weighted_x = torch.matmul(y.T, x)  # (num_classes, feature_dim)
                
                # Update mean
                new_mu = (weighted_x + c.unsqueeze(1) * mu) / (sum_weights.unsqueeze(1) + c.unsqueeze(1))
                new_c = c + sum_weights
                
                # Update covariance matrix
                x_minus_mu = x.unsqueeze(1) - mu.unsqueeze(0)  # (batch_size, num_classes, feature_dim)
                weighted_x_minus_mu = y.unsqueeze(2) * x_minus_mu  # (batch_size, num_classes, feature_dim)
                delta = torch.einsum('bji,bjk->jik', weighted_x_minus_mu, x_minus_mu)  # (num_classes, feature_dim, feature_dim)
                Sigma = (c[:, None, None] * Sigma + delta) / (c[:, None, None] + sum_weights[:, None, None])
                
                # Update overall Sigma
                overall_Sigma = torch.mean(Sigma, dim=0)
                
                # Write back to the appropriate distribution
                if is_negative:
                    self.neg_mu = new_mu
                    self.neg_c = new_c
                    self.neg_Sigma = Sigma
                    self.neg_overall_Sigma = overall_Sigma
                else:
                    self.mu = new_mu
                    self.c = new_c
                    self.Sigma = Sigma
                    self.overall_Sigma = overall_Sigma
    
    def update_lambda(self, update_negative=False):
        """Update the precision matrix for positive and/or negative distribution"""
        if not update_negative:
            # Update positive distribution
            self.Lambda = torch.inverse(
                (1 - self.epsilon) * self.overall_Sigma + 
                self.epsilon * torch.eye(self.input_shape).to(self.device)
            ).float()
        else:
            # Update negative distribution
            self.neg_Lambda = torch.inverse(
                (1 - self.epsilon) * self.neg_overall_Sigma + 
                self.epsilon * torch.eye(self.input_shape).to(self.device)
            ).float()
    
    def predict(self, x, use_negative=False):
        """
        Predict logits using Gaussian model
        Args:
            x: (batch_size, feature_dim)
            use_negative: if True, use negative distribution; else use positive
        Returns: (batch_size, num_classes)
        """
        with torch.no_grad():
            with autocast():
                if use_negative:
                    M = self.neg_mu.T  # (feature_dim, num_classes)
                    W = torch.matmul(self.neg_Lambda, M)  # (feature_dim, num_classes)
                else:
                    M = self.mu.T  # (feature_dim, num_classes)
                    W = torch.matmul(self.Lambda, M)  # (feature_dim, num_classes)
                    
                c = 0.5 * torch.sum(M * W, dim=0)  # (num_classes,)
                scores = torch.matmul(x, W) - c  # (batch_size, num_classes)
                return scores


class DDEPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(DDEPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = self.args.beta
        self.hat_M = self.args.hat_M 
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = True
        self.proj_flag = False
        self.alpha = self.args.alpha
        self.gamma = self.args.gamma
        self.group_num = self.args.group_num
        self.group_size = self.args.group_size
        self.random_permute = self.args.random_permute
        self.reset = True
        self.thres = self.args.thres
        self.samada = self.args.samada
        self.gap = self.args.gap
        self.cluster_num = self.args.cluster_num
        self.cossim = self.args.cossim
        self.queue_len = self.args.memleng
        self.score_queue_len = 20000
        self.mute_mutual_enhancement = False
        
        # ===== 新增：TTA 相关参数（来自 dotaprompt）=====
        self.dota_sigma = 0.002  # Initial sigma for Gaussian
        self.dota_epsilon = 0.0001
        self.dota_rho = self.args.dota_rho  # Dynamic weight scaling factor
        self.dota_eta = 0.2  # Maximum weight cap
        self.neg_topk = self.args.neg_topk  # number of top-k classes used for negative distribution update
        
        self.gaussian_model = None
        self.net = None
        self.conf_buffer = np.array([], dtype=np.float32)
        self.conf_queue_length = 512
        self.entropy_cache = None
        self.eta = self.args.eta
    
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        return
        net.eval()
        self.net = net
        # Reset Gaussian model
        clip_weights = self.net.cls_text_features
        self.gaussian_model = GaussianModel(
            input_shape=clip_weights.shape[0],
            num_classes= self.net.n_cls,
            clip_weights=clip_weights,
            sigma=self.dota_sigma,
            epsilon=self.dota_epsilon,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        output_text = net.logit_scale * net.text_features.t() @ net.text_features # batch * class.
        score_output_text = torch.softmax(output_text , dim=1)
        self.score_prior = score_output_text[:1000, :1000].sum(1)   ## ID 在 ID上的得分。

    def reset_memory(self):
        self.reset = True
        # self.net = net
    

    def reset_entropy_cache(self):    
        self.entropy_cache = None

    # def reset_beta_ratio(self, beta):
    #     self.beta = beta

    def _compute_neg_scores_from_caches(self, logit_scale, text_feature_bank, class_num):
        """Estimate negative-label relevance from cached ID/OOD features."""
        score_from_id_cache = logit_scale * self.id_image_features_cache @ text_feature_bank
        score_from_ood_cache = logit_scale * self.ood_image_features_cache @ text_feature_bank

        if self.cossim:
            score_from_id_cache = score_from_id_cache[:, class_num:]
            score_from_ood_cache = score_from_ood_cache[:, class_num:]
        else:
            score_from_id_cache = torch.softmax(score_from_id_cache, dim=1)[:, class_num:]
            score_from_ood_cache = torch.softmax(score_from_ood_cache, dim=1)[:, class_num:]

        return score_from_id_cache, score_from_ood_cache

    def _select_text_features(self, id_text_features, neg_text_features, combined_score):
        """Construct the prompt bank: all ID labels + top-K salient negatives."""
        top_neg_idx = combined_score.sort(descending=True)[1][:self.hat_M]
        selected_neg = neg_text_features.t()[top_neg_idx].t()
        return torch.cat([id_text_features, selected_neg], dim=-1)

    def _initialize_runtime_state(self, net):
        """Initialize online buffers/statistics on the first inference call."""
        class_num = net.n_cls
        self.net = net
        self.class_count = torch.ones(class_num, device=net.text_features.device) * self.hat_M

        # Keep two text-feature banks:
        # - online_text_features: current inference bank
        # - online_text_features_all: full bank (ID + all negatives) used for selection.
        self.online_text_features = net.text_features.clone()
        self.online_text_features_all = net.text_features_all.clone()

        # Initialize feature caches:
        # - ID cache starts from class text prototypes
        # - OOD cache starts from pre-computed noise features.
        self.id_image_features_cache = net.text_features_all[:, :class_num].t().clone()
        self.ood_image_features_cache = net.noise_image_features.clone()

        clip_weights = net.cls_text_features
        self.gaussian_model = GaussianModel(
            input_shape=clip_weights.shape[0],
            num_classes=class_num,
            clip_weights=clip_weights,
            sigma=self.dota_sigma,
            epsilon=self.dota_epsilon,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Bootstrap salient negative labels using cache statistics.
        score_from_id_cache, score_from_ood_cache = self._compute_neg_scores_from_caches(
            net.logit_scale, net.text_features_all, class_num)
        combined_score = score_from_ood_cache.mean(0) - score_from_id_cache.mean(0)
        selected_text_features = self._select_text_features(
            net.text_features[:, :class_num],
            self.online_text_features_all[:, class_num:],
            combined_score)

        # Seed score queue with both pseudo-ID and pseudo-OOD confidence.
        output = net.logit_scale * net.text_features_all[:, :class_num].t() @ selected_text_features
        conf_in_id = torch.softmax(output, dim=1)[:, :class_num].sum(dim=-1)
        output = net.logit_scale * net.noise_image_features @ selected_text_features
        conf_in_ood = torch.softmax(output, dim=1)[:, :class_num].sum(dim=-1)
        self.scoretensor = torch.cat((conf_in_id, conf_in_ood), dim=0)
        self.reset = False

    def _compute_batch_adaptive_neg_score(self, actscore, score_from_id_cache,
                                          score_from_ood_cache, activate_indicator_id,
                                          activate_indicator_ood, class_num):
        """Blend long-term cache score with current-batch signal."""
        if torch.any(activate_indicator_id):
            score_pos_in_batch = actscore[activate_indicator_id][:, class_num:]
            ins_adaptive_pos_score = 0.95 * score_from_id_cache.mean(0) + 0.05 * score_pos_in_batch.mean(0)
        else:
            ins_adaptive_pos_score = score_from_id_cache.mean(0)

        if torch.any(activate_indicator_ood):
            score_neg_in_batch = actscore[activate_indicator_ood][:, class_num:]
            ins_adaptive_neg_score = 0.95 * score_from_ood_cache.mean(0) + 0.05 * score_neg_in_batch.mean(0)
        else:
            ins_adaptive_neg_score = score_from_ood_cache.mean(0)

        return ins_adaptive_neg_score - ins_adaptive_pos_score

    def _run_dota_fusion(self, image_features, cls_text_features, logit_scale):
        """
        Update positive/negative Gaussian branches and return fused ID prediction.
        Positive branch learns top-1 class manifold; negative branch learns top-k distractors.
        """
        output_in_vanilla = logit_scale * image_features @ cls_text_features.t()
        prob_id_normalized = torch.softmax(output_in_vanilla, dim=1)

        _, topk_indices = torch.topk(prob_id_normalized, k=1 + self.neg_topk, dim=1)
        top1_idx = topk_indices[:, 0]
        neg_idx = topk_indices[:, 1:]

        pos_prob_map = torch.zeros_like(prob_id_normalized)
        pos_prob_map.scatter_(
            1,
            top1_idx.unsqueeze(1),
            prob_id_normalized.gather(1, top1_idx.unsqueeze(1)))

        neg_prob_map = torch.zeros_like(prob_id_normalized)
        neg_prob_map.scatter_(1, neg_idx, prob_id_normalized.gather(1, neg_idx))

        self.gaussian_model.fit(image_features, pos_prob_map, is_negative=False)
        self.gaussian_model.update_lambda(update_negative=False)
        self.gaussian_model.fit(image_features, neg_prob_map, is_negative=True)
        self.gaussian_model.update_lambda(update_negative=True)

        dota_logits_pos = self.gaussian_model.predict(image_features, use_negative=False)
        dota_logits_neg = self.gaussian_model.predict(image_features, use_negative=True)
        dynamic_alpha = torch.clamp(self.dota_rho * self.gaussian_model.c.mean(), max=self.dota_eta)
        final_logits = output_in_vanilla + dynamic_alpha * dota_logits_pos - dynamic_alpha * self.beta * dota_logits_neg
        prob_all = torch.softmax(final_logits, dim=1)
        _, pred_in = torch.max(prob_all, dim=1)
        return pred_in

    def _update_confidence_buffer_and_predict_ood(self, conf_in, pred_in):
        """
        Convert ID confidence to OOD score, estimate adaptive threshold (OWTTT style),
        then mark samples above threshold as OOD (-1 label).
        """
        conf_ood = 1 - conf_in.detach().cpu().numpy()
        self.conf_buffer = np.concatenate([self.conf_buffer, conf_ood])
        self.conf_buffer = self.conf_buffer[-self.conf_queue_length:]

        threshold_range = np.arange(0, 1, 0.01)
        criterias = [self.compute_os_variance(self.conf_buffer, th) for th in threshold_range]
        threshold = threshold_range[np.argmin(criterias)]
        ood_mask = (conf_ood >= threshold)
        pred_in[ood_mask] = -1
        return pred_in


    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls

        # First call: initialize all online states and caches.
        if self.reset:
            self._initialize_runtime_state(net)

        image_features, text_features, cls_text_features, logit_scale = net(data, return_feat=True)

        output_all = logit_scale * image_features @ self.online_text_features_all
        if self.cossim:
            actscore = output_all
        else:
            actscore = torch.softmax(output_all.float(), dim=1)

        score_from_id_cache, score_from_ood_cache = self._compute_neg_scores_from_caches(
            logit_scale, self.online_text_features_all, class_num)
        combined_score = score_from_ood_cache.mean(0) - score_from_id_cache.mean(0)

        selected_text_features = self._select_text_features(
            net.text_features[:, :class_num],
            self.online_text_features_all[:, class_num:],
            combined_score)

        output = (logit_scale * image_features @ selected_text_features).float()
        logits = torch.softmax(output, dim=1)
        conf_in_domain = logits[:, :class_num].sum(dim=-1)

        self.scoretensor = torch.cat((self.scoretensor, conf_in_domain), dim=0)
        if self.scoretensor.size(0) > self.score_queue_len:
            self.scoretensor = self.scoretensor[-self.score_queue_len:]

        activate_indicator_id = conf_in_domain > (self.thres + self.gap * (1 - self.thres))
        activate_indicator_ood = conf_in_domain < 0.10

        # Re-rank negatives with a small amount of batch-specific evidence.
        combined_score = self._compute_batch_adaptive_neg_score(
            actscore,
            score_from_id_cache,
            score_from_ood_cache,
            activate_indicator_id,
            activate_indicator_ood,
            class_num)

        used_cluster_num = 0
        if used_cluster_num > 0:
            selected_text_features = torch.cat([
                net.text_features[:, :class_num], 
                reconst_clusters.t(), 
                self.online_text_features_all[:, class_num:].t()[combined_score.sort(descending=True)[1][:self.hat_M-used_cluster_num]].t()
            ], dim=-1)
        else:
            selected_text_features = self._select_text_features(
                net.text_features[:, :class_num],
                self.online_text_features_all[:, class_num:],
                combined_score)

        # Final score with re-selected salient negatives.
        output_final = logit_scale * image_features @ selected_text_features

        pred_in = self._run_dota_fusion(image_features, cls_text_features, logit_scale)
        full_sim = output_final.softmax(dim=-1)
        conf_in = full_sim[:, :class_num].sum(dim=-1)

        # Update long-term caches with high-confidence pseudo-ID/OOD samples.
        if torch.any(activate_indicator_id):
            id_features_to_add = image_features[activate_indicator_id].detach()
            self.id_image_features_cache = self._update_cache(
                id_features_to_add,
                self.id_image_features_cache,
                max_size=getattr(self, 'id_cache_size', 5000)
            )
        
        if torch.any(activate_indicator_ood):
            ood_features_to_add = image_features[activate_indicator_ood].detach()
            self.ood_image_features_cache = self._update_cache(
                ood_features_to_add,
                self.ood_image_features_cache,
                max_size=getattr(self, 'ood_cache_size', 5000)
            )

        pred_in = self._update_confidence_buffer_and_predict_ood(conf_in, pred_in)

        if self.in_score == 'sum':
            conf_in = conf_in  ## = 1-conf_out
        elif self.in_score == 'div_count':
            normalized_count = self.class_count / self.class_count.sum()  # 假设有这个属性
            for i in range(len(conf_in)):
                conf_in[i] = conf_in[i] / normalized_count[pred_in[i]] / net.n_cls
        else:
            raise NotImplementedError
        
        if torch.isnan(conf_in).any():
            pdb.set_trace()
        
        return pred_in, conf_in


    def _update_cache(self, new_features, cache, max_size):
        if cache is None:
            return new_features
        
        cache = torch.cat([cache, new_features], dim=0)
        
        if cache.size(0) > max_size:
            cache = cache[-max_size:]
        
        return cache

    def _update_entropy_cache(self, entropy, entropy_cache):
        if entropy_cache is None:
            return entropy
        
        cache = torch.cat([entropy, entropy_cache], dim=0)
        
        return cache    

    def compute_os_variance(self, os, th):
        """
        This function is borrowed from OWTTT (ICCV23): https://github.com/Yushu-Li/OWTTT
        Calculate the area of a rectangle.

        Parameters:
            os : OOD score queue.
            th : Given threshold to separate weak and strong OOD samples.

        Returns:
            float: Weighted variance at the given threshold th.
        """
        
        thresholded_os = np.zeros(os.shape)
        thresholded_os[os >= th] = 1

        # compute weights
        nb_pixels = os.size
        nb_pixels1 = np.count_nonzero(thresholded_os)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1

        # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
        # in the search for the best threshold
        if weight1 == 0 or weight0 == 0:
            return np.inf

        # find all pixels belonging to each class
        val_pixels1 = os[thresholded_os == 1]
        val_pixels0 = os[thresholded_os == 0]

        # compute variance of these classes
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

        return weight0 * var0 + weight1 * var1

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau

    def compute_entropy(self, prob):
        """计算预测分布的熵"""
        entropy = -(prob.double() * (torch.log(prob.double() + 1e-8))).sum(dim=-1)
        return entropy    