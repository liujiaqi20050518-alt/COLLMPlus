import os

_omp_threads = os.environ.get("OMP_NUM_THREADS")
if _omp_threads is None or not _omp_threads.isdigit() or int(_omp_threads) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

from minigpt4.models.rec_base_models import MatrixFactorization, LightGCN 
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from minigpt4.datasets.datasets.rec_gnndataset import GnnDataset
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random 
import time
# !!!!!!!!版本1！！！！！！！！！！！！！！！！！！！
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
# class DynamicKDLoss(nn.Module):
#     def __init__(self, gamma=0.01):
#         super().__init__()
#         # 调小 gamma 到 0.01，让更多中低频（长尾）物品能享受到蒸馏的红利
#         # 比如出现 50 次的物品，alpha 也能保留 0.6 左右的蒸馏强度
#         self.gamma = gamma

#     def forward(self, student_logits, teacher_logits, true_labels, item_freqs):
#         # 1. 基础长尾动态权重 Alpha
#         alpha = torch.exp(-self.gamma * item_freqs)
        
#         # 2. Hard Loss (学生 vs 真实世界)
#         hard_loss = F.binary_cross_entropy_with_logits(
#             student_logits, true_labels.float(), reduction='none'
#         )
        
#         # 3. 转化为概率
#         student_probs = torch.sigmoid(student_logits)
#         teacher_probs = torch.sigmoid(teacher_logits)
        
#         # 4. 🌟 新创新：平滑置信度 (Soft Confidence) 代替一刀切！
#         # 老师的预测离真实标签越近，confidence 越大 (最高为1)；越离谱，confidence 越小 (最低趋近于0)
#         teacher_error = torch.abs(true_labels.float() - teacher_probs)
#         confidence = 1.0 - teacher_error 
        
#         # 5. 🌟 推荐专属 Soft Loss：均方误差 (MSE)
#         # 强迫学生的概率分布在二维空间上向老师靠拢，不搞温度软化那套虚的
#         soft_loss = F.mse_loss(student_probs, teacher_probs, reduction='none')
        
#         # 6. 究极融合公式
#         # MSE 的值天然比 BCE 小很多，所以我们给蒸馏 Loss 乘一个放大系数 (比如 5.0)
#         # 让长尾物品 (alpha高) 且老师有把握 (confidence高) 的样本，爆发出强大的蒸馏引导力！
#         kd_weight = alpha * confidence * 8.0
        
#         total_loss = hard_loss + kd_weight * soft_loss
        
#         return total_loss.mean()


# !!!!!!!!版本2！！！！！！！！！！！！！！！！！！！
# class DynamicKDLoss(nn.Module):
#     def __init__(self, gamma=0.01):
#         super().__init__()
#         self.gamma = gamma

#     def forward(self, student_logits, teacher_logits, true_labels, item_freqs):
#         alpha = torch.exp(-self.gamma * item_freqs)
        
#         hard_loss = F.binary_cross_entropy_with_logits(
#             student_logits, true_labels.float(), reduction='none'
#         )
        
#         # 🌟 开启 Hinton 蒸馏温度魔法
#         T = 6.0  
        
#         student_probs = torch.sigmoid(student_logits)
#         teacher_probs = torch.sigmoid(teacher_logits)
        
#         teacher_error = torch.abs(true_labels.float() - teacher_probs)
#         confidence = 1.0 - teacher_error 
        
#         # 🌟 使用温度软化概率分布，挖掘大模型的“暗知识”
#         student_probs_soft = torch.sigmoid(student_logits / T)
#         teacher_probs_soft = torch.sigmoid(teacher_logits / T)
        
#         # 🌟 必须乘以 T^2 来补偿梯度缩放！
#         # soft_loss = F.mse_loss(student_probs_soft, teacher_probs_soft, reduction='none') * (T * T)
#         # 用交叉熵去对齐两个分布，梯度的指向会比距离绝对值（MSE）更加敏锐和精准！
#         # 同样别忘了乘以 (T * T) 来补偿梯度缩放
#         soft_loss = F.binary_cross_entropy(student_probs_soft, teacher_probs_soft, reduction='none') * (T * T)
#         # 保持 8.0 的大模型辅导强度！
#         kd_weight = alpha * confidence * 8.0
        
#         total_loss = hard_loss + kd_weight * soft_loss
        
#         return total_loss.mean()


class DualLevelKDLoss(nn.Module):
    def __init__(self, gamma=0.01, feat_weight=0.8): 
        super().__init__()
        self.gamma = gamma
        self.feat_weight = feat_weight

    def forward(self, student_logits, teacher_logits, student_feat, teacher_feat, true_labels, item_freqs):
        alpha = torch.exp(-self.gamma * item_freqs)
        
        # 1. 硬标签
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels.float(), reduction='none')
        
        # 2. 软标签 (保持你的进度不变)
        T = 5.0  
        teacher_error = torch.abs(true_labels.float() - torch.sigmoid(teacher_logits))
        confidence = 1.0 - teacher_error 
        
        stu_probs_soft = torch.sigmoid(student_logits / T)
        tea_probs_soft = torch.sigmoid(teacher_logits / T)
        soft_loss = F.binary_cross_entropy(stu_probs_soft, tea_probs_soft, reduction='none') * (T * T)
        kd_weight = alpha * confidence * 8.0
        
        # 🌟 3. 跨模态表征对齐 (修复版：直接计算余弦相似度！)
        # cosine_similarity 输出范围是 [-1, 1]。1代表完全同向。
        # 我们用 1.0 减去它，让 Loss 的范围变成 [0, 2]，数值极其稳定！
        cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1).unsqueeze(1)
        feat_loss = 1.0 - cos_sim 
        
        # 终极融合！
        total_loss = hard_loss + (kd_weight * soft_loss) + (self.feat_weight * feat_loss)
        return total_loss.mean()

def uAUC_me(user, predict, label):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        # print(index_ui, predict.shape)
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        
        k+=1
    print("only one interaction users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.regs = 0
        self.embed_size = 64
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = "/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/"
        self.dataset = 'ml-100k' #'yahoo-s622-01' #'yahoo-small2' #'yahooR3-iid-001'
        self.layer_size='[64,64]'
        self.verbose = 1
        self.Ks='[10]'
        self.data_type='retraining'

        # lightgcn hyper-parameters
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 100
        self.A_split = False
        self.dropout = False
        self.pretrain=0
        self.init_emb=1e-4
        
    def reset(self, config):
        for name,val in config.items():
            setattr(self,name,val)
    
    def hyper_para_info(self):
        print(self.__dict__)


class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', incerase =True,patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience= patience
        # self.metrics = None
    
    def _registry(self,metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True 
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count>=self.patience:
            return True
        else:
            return False

# set random seed   
def run_a_trail(train_config,log_file=None, save_mode=False,save_file=None,need_train=True,warm_or_cold=None):
    seed=2023
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = model_hyparameters()
    args.reset(train_config)
    args.hyper_para_info()

    # load dataset
    # data_dir = "/home/zyang/LLM/MiniGPT-4/dataset/ml-100k/"
    # data_dir = "/home/sist/zyang/LLM/datasets/ml-100k/"
    data_dir = "/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/"
    train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid','iid','label']].values
    valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid','iid','label']].values
    test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label']].values



    # === 🌟 核心数据融合开始 ===
    print("🚀 正在加载大模型脑电波数据...")
    # 请填入你真实生成的 npy 文件名！
    teacher_logits = np.load("/root/autodl-tmp/CoLLM/result/20260330_214720/distill_soft_labels_20260330_214720.npy") 
    if len(teacher_logits.shape) == 1:
        teacher_logits = teacher_logits.reshape(-1, 1)
    # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    print("🧠 正在加载大模型 4096 维文本语义特征空间...")
    llm_features_path = "/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/item_llm_features.npy"
    teacher_features_all = torch.from_numpy(np.load(llm_features_path)).float().cuda()
    train_data = train_data[:teacher_logits.shape[0]]
    # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    items = train_data[:, 1].astype(int)
    item_counts = np.bincount(items)
    item_freqs = item_counts[items].reshape(-1, 1)
    
    # 将 [uid, iid, label] 扩充为 [uid, iid, label, teacher_logit, item_freq]
    train_data = np.concatenate([train_data, teacher_logits, item_freqs], axis=1)
    print(f"✅ 数据融合完毕！新的训练集形状: {train_data.shape}")
    # === 🌟 核心数据融合结束 ===

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'not_cold']]
            test_data = test_data[test_data['not_cold'].isin([1])][['uid','iid','label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'not_cold']]
            test_data = test_data[test_data['not_cold'].isin([0])][['uid','iid','label']].values
            print("cold data size:", test_data.shape[0])
            # pass

    # train_config={
    #     "lr": 1e-2,
    #     "wd": 1e-4,
    #     "epoch": 5000,
    #     "eval_epoch":1,
    #     "patience":50,
    #     "batch_size":1024
    # }

    user_num = train_data[:,0].max() + 1
    item_num = train_data[:,1].max() + 1

    lgcn_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size']),
        "embed_size": int(train_config['embedding_size']),
        "data_path": "/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/",
        "dataset": 'ml-1m', #'yahoo-s622-01' #'yahoo-small2' #'yahooR3-iid-001'
        "layer_size": '[64,64]',

        # lightgcn hyper-parameters
        "gcn_layers": train_config['gcn_layer'],
        "keep_prob" : 0.6,
        "A_n_fold": 100,
        "A_split": False,
        "dropout": False,
        "pretrain": 0,
        "init_emb": 1e-1,
        }
    lgcn_config = omegaconf.OmegaConf.create(lgcn_config)
    gnndata = GnnDataset(lgcn_config, data_dir)
    lgcn_config['user_num'] = int(gnndata.m_users)
    lgcn_config['item_num'] = int(gnndata.n_items)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)





    # model = MatrixFactorization(mf_config).cuda()
    model = LightGCN(lgcn_config).cuda()
    model._set_graph(gnndata.Graph)
    
    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'], weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    # trainig part
    # criterion = nn.BCEWithLogitsLoss() ！！！！！！！！！！！！！修改
    criterion = DualLevelKDLoss( gamma=0.01,feat_weight=0.05)
    if not need_train:
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(valid_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
            users.extend(batch_data[:,0].cpu().numpy())
        valid_auc = roc_auc_score(label,pre)
        valid_uauc, _, _ = uAUC_me(users, pre, label)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre>=thre] =  1
        pre[pre<thre]  =0
        val_acc = (label==pre).mean()

        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(test_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
            users.extend(batch_data[:,0].cpu().numpy())
        test_auc = roc_auc_score(label,pre)
        test_uauc,_,_ = uAUC_me(users, pre, label)

        print("valid_auc:{}, valid_uauc:{}, test_auc:{}, test_uauc:{}, acc: {}".format(valid_auc, valid_uauc, test_auc, test_uauc, val_acc))
        return 
    

    # ！！！！！！！！！！！！！！！！！版本12使用的！！！！！！！！！！！！！！！
    # for epoch in range(train_config['epoch']):
    #     model.train()
    #     for bacth_id, batch_data in enumerate(train_data_loader):
    #         batch_data = batch_data.cuda()
    #         # ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
    #         # loss = criterion(ui_matching,batch_data[:,-1].float())
    #         # opt.zero_grad()
    #         # loss.backward()
    #         # opt.step()！！！！！！！！！！！！！！！！！修改过了
    #         uids = batch_data[:, 0].long()
    #         iids = batch_data[:, 1].long()
            
    #         # 因为数据被拼接了，真正的 label 现在固定在第 3 列 (索引为2)
    #         # unsqueeze(1) 是为了把形状从 [Batch_Size] 变成 [Batch_Size, 1]，对齐公式计算
    #         true_labels = batch_data[:, 2].float().unsqueeze(1)    
    #         teacher_logits = batch_data[:, 3].float().unsqueeze(1) 
    #         item_freqs = batch_data[:, 4].float().unsqueeze(1)     
            
    #         # 1. 小模型前向传播，算出自己的预测
    #         student_logits = model(uids, iids).unsqueeze(1)
            
    #         # 2. 扔进你的专属超级引擎：DynamicKDLoss
    #         loss = criterion(student_logits, teacher_logits, true_labels, item_freqs)
    #         # ==========================================
    #         # 🌟 核心修改结束
    #         # ==========================================

    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()


    for epoch in range(train_config['epoch']):
        model.train()
        if epoch < 20:
            current_feat_weight = 0.0
        # elif epoch < 20:
        #     # 20轮内从 0 线性增加到 0.1
        #     current_feat_weight = 0.05 * (epoch - 10) / 10 
        else:
            current_feat_weight = 0.05

        criterion.feat_weight = current_feat_weight
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = batch_data.cuda()
            
            uids = batch_data[:, 0].long()
            iids = batch_data[:, 1].long()
            
            true_labels = batch_data[:, 2].float().unsqueeze(1)    
            teacher_logits = batch_data[:, 3].float().unsqueeze(1) 
            item_freqs = batch_data[:, 4].float().unsqueeze(1)     
            
            # 1. 小模型前向传播，算出自己的预测
            student_logits = model(uids, iids).unsqueeze(1)
            
            # 🌟 2. 获取跨模态特征对齐的“两极”
            # a) 拿到小模型 64 维经过投影仪拉伸后的 4096 维特征
            student_feat = model.get_projected_item_emb(iids)
            # b) 查表拿到大模型真实 4096 维文本语义特征
            teacher_feat = teacher_features_all[iids]
            
            # 🌟 3. 扔进双层蒸馏超级引擎！
            loss = criterion(student_logits, teacher_logits, student_feat, teacher_feat, true_labels, item_freqs)

            opt.zero_grad()
            loss.backward()
            opt.step()


        if epoch% train_config['eval_epoch']==0:
            
            model.eval()
            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
                users.extend(batch_data[:,0].cpu().numpy())
            valid_auc = roc_auc_score(label,pre)
            valid_uauc,_,_ = uAUC_me(users, pre, label)
            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(test_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
                users.extend(batch_data[:,0].cpu().numpy())
            test_auc = roc_auc_score(label,pre)
            test_uauc,_,_ = uAUC_me(users, pre, label)
            updated = early_stop.update({'valid_auc':valid_auc, 'valid_uauc': valid_uauc,'test_auc':test_auc, 'test_uauc': test_uauc,'epoch':epoch})
            if updated and save_mode:
                torch.save(model.state_dict(),save_file)


            print("epoch:{}, valid_auc:{}, valid_uauc:{}, test_auc:{}, test_uauc:{}, early_count:{}".format(epoch, valid_auc, valid_uauc, test_auc, test_uauc, early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                # print("best results:", early_stop.best_metric)
                break
            if epoch>500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.52")
                break
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric) 
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)



# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-4]
#     dw_ = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [64, 128, 256]
#     gcn_layers = [1, 2, 3]
#     try:
#         # f = open("ml100k-rec_lgcn_search_lr"+str(lr_[0])+".log",'rw+')
#         # f = open("ood-ml100k-rec_lgcn_search_lrall-int0.1_p100_1layer"+str(lr_[0])+".log",'rw+')
#         f = open("0919-oodv2-ml1m-rec_lgcn_search_lrall-int0.1_p100_1layer"+str(lr_[0])+".log",'rw+')
#     except:
#         # f = open("ml100k-rec_lgcn_search_lr"+str(lr_[0])+".log",'w+')
#         f = open("0919-oodv2-ml1m-rec_lgcn_search_lrall-int0.1_p100_1layer"+str(lr_[0])+".log",'w+')
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 for gcn_layer in gcn_layers:
#                     train_config={
#                         'lr': lr,
#                         'wd': wd,
#                         'embedding_size': embedding_size,
#                         "epoch": 5000,
#                         "eval_epoch":1,
#                         "patience":100,
#                         "batch_size":2048,
#                         "gcn_layer": gcn_layer
#                     }
#                     print(train_config)
#                     run_a_trail(train_config=train_config, log_file=f, save_mode=False)
#     f.close()



#train_config: {'lr': 0.01, 'wd': 0.0001, 'embedding_size': 64, 'epoch': 5000, 'eval_epoch': 1, 'patience': 100, 'batch_size': 2048, 'gcn_layer': 2}

# {'lr': 0.0001, 'wd': 1e-07, 'embedding_size': 256, 'epoch': 5000, 'eval_epoch': 1, 
# 'patience': 100, 'batch_size': 2048, 'gcn_layer': 2}
# save version....
# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-4] #1e-2
#     dw_ = [1e-5]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [64]
#     save_path = "/root/autodl-tmp/CoLLM/PretrainedModels/lgcn/"
#     # save_path = "/home/sist/zyang/LLM/PretrainedModels/mf/"
#     # try:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
#     # except:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
#     f=None
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 train_config={
#                     'lr': lr,
#                     'wd': wd,
#                     'embedding_size': embedding_size,
#                     "epoch": 5000,
#                     "eval_epoch":1,
#                     "patience":100,
#                     "batch_size":2048,
#                     "gcn_layer": 2
#                 }
#                 print(train_config)
#                 save_path += "0918-OODv2_lgcn_ml1m_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
#                 run_a_trail(train_config=train_config, log_file=f, save_mode=True,save_file=save_path)
#     f.close()


# # with prtrain version:


# !!!!!!!!!!!!!!train训练！！！！！！！！！！！！！！！！！！！！！！
if __name__=='__main__':
    # lr_ = [1e-1,1e-2,1e-3]
    lr_=[1e-3] #1e-2
    dw_ = [1e-4]
    # embedding_size_ = [32, 64, 128, 156, 512]
    embedding_size_ = [64]
    save_path = "/root/autodl-tmp/CoLLM/lgcnresult/"
    f=None
    for lr in lr_:
        for wd in dw_:
            for embedding_size in embedding_size_:
                train_config={
                    'lr': lr,
                    'wd': wd,
                    'embedding_size': embedding_size,
                    "epoch": 5000,
                    "eval_epoch":1,
                    "patience":50,
                    "batch_size":2048,
                    "gcn_layer": 2
                }
                
                # print(train_config)
                # if os.path.exists(save_path + "best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"):
                #     save_path += "best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
                # else:
                #     save_path += "best_model_d" + str(embedding_size) + ".pth"


                # save_path += "0918-OODv2_lgcn_ml1m_best_model_d64lr-0.01wd0.0001.pth" 生成lgcn图参数使用
                save_file_name = save_path + "student_dynamic_kd_best.pth"
                run_a_trail(train_config=train_config, log_file=f, save_mode=True, save_file=save_file_name, need_train=True, warm_or_cold=None)
    if f is not None:
        f.close()
        
# if __name__=='__main__':
#     # 🌟 1. 严格锁定你创造历史的黄金配置（不要做任何修改）
#     train_config={
#         'lr': 1e-3, 
#         'wd': 1e-4,
#         'embedding_size': 64,
#         "epoch": 5000,
#         "eval_epoch":1,
#         "patience":50,
#         "batch_size":2048,
#         "gcn_layer": 2
#     }
    
#     # 🌟 2. 指向你那颗 0.6356 的极品神丹的绝对路径
#     # （请核对一下这个路径是不是你最新跑出最好成绩的那个 pth 文件）
#     save_file_name = "/root/autodl-tmp/CoLLM/lgcnresult/student_dynamic_kd_best.pth"
    
#     print("\n" + "🔥"*10 + " 正在评估 Warm (热门/活跃) 试卷 " + "🔥"*10)
#     # 🌟 3. need_train=False 极其关键！这意味着不训练，直接加载权重去考试
#     run_a_trail(train_config=train_config, 
#                 save_file=save_file_name, 
#                 need_train=False,  
#                 warm_or_cold='warm') 
    
#     print("\n" + "❄️"*10 + " 正在评估 Cold (冷门/长尾/OOD) 试卷 " + "❄️"*10)
#     # 🌟 4. 测试 Cold 数据
#     run_a_trail(train_config=train_config, 
#                 save_file=save_file_name, 
#                 need_train=False,  
#                 warm_or_cold='cold')





        
            







