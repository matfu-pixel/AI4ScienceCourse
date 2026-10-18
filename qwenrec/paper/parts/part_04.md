# Часть 4: с. 7–9

[с. 7]
PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

Figure 3: 8-th Day Recall@10 and training loss vs retrieval SFT training step.
Impact of Pre-trained LLM. Now we look into the benefits of training from a general-purpose LLM versus training from a randomly
initialized state. A consistent pattern emerges from the results in
Table 5: models initialized from a pre-trained LLM consistently
outperform their random initialized counterparts with or without
CPT. We hypothesize that this advantage stems primarily from the
pre-trained LLM’s pre-existing understanding of natural language,
or the general sequence processing capability learned through massive LLM pre-training is directly useful for recommendation. A
further study of this observation is out of this paper’s scope.

3.4

Scaling Study

In this section, we conduct a scaling study for generative retrieval
models with YouTube production data. We use four different sizes
from the Gemini-1.5 [26] Mixture-of-Experts (MoE) model family:
MoE-110M, MoE-370M, MoE-900M, and MoE-3B (here the size
indicates the number of activated parameters per token) with their
total number of parameters spanning from less than 1B to more
than 10B. In particular, we aim to answer: (1) How does loss scale
with different model size and compute? (2) How does retrieval
metric scale with different model size and compute? (3) What is the
best model size for different compute budgets?
3.4.1

Experiment Setup.

Dataset. Our study was conducted on a production dataset sourced
from a key YouTube surface recommending what videos to watch
next given a current watched video and other contextual signals like
watch history (see Figure 4 in [38]). Here the recommender system
employs a multi-stage architecture, encompassing candidate generation, pre-ranking, and ranking, among other phases. Our study
centers on the candidate generation stage, evaluating retrieval performance against a comprehensive video corpus containing billions
of videos.
Training and Evaluation. Our input prompt has the following
format: “watch history | user features | context video features",
where watch history is a chronological sequence of watches, each
represented by the concatenation of SID tokens, and other feature
tokens. For this study, input sequence length is fixed at 1,536 tokens,
which can cover roughly 100 most recent watches along with other
features. SFT labels are selected using the next watch on YouTube

employing a downsampling mechanism based on user engagement
and satisfaction signals. SID tokens of the next watch are used
as the prediction targets. We randomly shuffle 7 continuous days
(from July 2025) of data for training. Evaluation is performed on
the subsequent day (Day 8). Our experiments were conducted with
1,024 Google’s v6e Tensor Processing Units (TPUs) each with 32GB
High Bandwidth Memory (HBM). We run 4 trainers in parallel each
taking 256 TPUs.
Hyperparameter Selection. We conducted a pilot study using the
MoE-900M model to determine the model’s sensitivity to global
batch size and learning rates (see Appendix A.1 for details). Based
on the learnings from this pilot study, in each case we set the batch
size to approximately saturate the HBM since it indicates larger
batch sizes are preferable. For learning rate, using 5 × 10−5 for MoE900M as the anchor point, we use slightly larger learning rates for
smaller models (and vice versa) based on our best guess. Note that
unlike scaling law studies in natural language processing (e.g., [11]),
we employ constant learning rates. This approach is preferred for
two reasons: (1) continuous training in production models does not
involve a specific stopping step number, and (2) constant learning
rates allow for perfect and precise interpolation of all intermediate
data points throughout the training process. Hyperparameters of
different model sizes are summarized in Table 6. All models are
warm-started from their corresponding Continued Pre-training
checkpoints (all with Iso-FLOPS 1 × 1022 ). Continued Pre-training
training FLOPs are excluded from all FLOPs numbers in this section.
Table 6: Scaling Law Study Configurations.

3.4.2

Model Size

Learning Rate

Global Batch Size

MoE-110M
MoE-370M
MoE-900M
MoE-3B

1 × 10−4

25,600
15,360
7,680
3,584

7 × 10−5
5 × 10−5
2 × 10−5

Experiment Results.

Loss Scaling with Compute and Model Size. Figure 4 displays the
correlation between training/evaluation loss (on day 8) and training Iso-FLOPS. Figure 4a reveals a distinct power-law correlation

[с. 8]
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

between Iso-FLOPS and training loss for each model size. As IsoFLOPS increase, the optimal model size, representing the training
loss frontier, progressively shifts. This shift begins with MoE-110M
(red line), moves to MoE-370M (green line), then to MoE-900M
(purple line), and is hypothesized to eventually reach MoE-3B (blue
line). Figure 4b illustrates that there is also a power-law correlation
between Iso-FLOPS and evaluation loss for each model size before
it finally begins to saturate, which we attribute to our extensive
training. We also observe that the evaluation loss frontier shifts
toward larger models significantly sooner than the training loss,
suggesting that larger models exhibit superior generalization to
future data distributions.

He et al.

constant learning rate for each model enables us to directly assess
the model performance at each Iso-FLOPS budget using a single
run. Similar to the Chinchilla paper [11], in Figure 6b there is a
peak for most of the Iso-FLOPS budgets we consider.

(a) Training loss for Fixed (b) Eval Recall@10 for Fixed Compute
Compute Budget.
Budget.

Figure 6: Iso-FLOPS curves. We evaluate training loss and
evaluation Recall@10 at various fixed Iso-FLOPS budgets,
selecting the appropriate number of steps for each model
size.
(a) Iso-FLOPS v.s. training loss.

(b) Iso-FLOPS v.s. eval loss.

Figure 4: Training and evaluation loss variation as we scale
up training Iso-FLOPS.

Retrieval Metric Scaling with Compute and Model Size. Similarly,
in Figure 5 we report the relationship between training/evaluation
Recall@10 (on day 8) and training Iso-FLOPS, with both axes presented on a logarithmic scale. The findings are similar to those in
Figure 4, except that in Figure 5 the evaluation Recall@10 shows
fewer signs of slowing down to increase, suggesting that model
keeps improving the generation of accurate and complete sequences
that lead to correct document. Additionally, the three smaller models have processed more than one epoch of data, with MoE-110M
trained as many as 4.24 epochs but still showing no signs of overfitting (see Figure 5b).

3.4.3 Limitation Discussion. We note that the MoE-3B model did
not outperform the MoE-900M model for the compute budgets we
considered. Our scaling study is constrained by compute resource.
The suboptimal hyperparameter setup might be a significant contributing factor. For instance, our pilot study indicates a correlation
between larger batch sizes and increased training efficiency for
the same model size. Since we allocate the same training resources
for each model size, using a batch size that saturates the HBMs
(see Table 6), larger models may be at a disadvantage due to their
consequently smaller batch sizes. Indeed, at the end of training,
the MoE-3B model has only processed 0.57 epochs of training data
(about 5B examples), a more than 2x gap from the second largest
model. Our scaling study on the generative retrieval task reveals
that compute-optimal training requires a synchronized scaling of
both training examples and model size.

4

(a) Iso-FLOPS v.s. training Re- (b) Iso-FLOPS v.s. eval Recall@10.
call@10.

Figure 5: Training and evaluation Recall@10 variation as we
scale up training Iso-FLOPS.
Optimal Model Size for Fixed Iso-FLOPS Budgets. In Figure 6, we
demonstrate the gradual shifts of optimal model size to larger ones
as we increase the given Iso-FLOPS budgets. Note that our use of

Conclusion and Future Work

We introduced PLUM, a framework for adapting pre-trained LLMs
for large-scale recommendation tasks. Our method enhances item
tokenization with SIDv2, uses a large-scale continued pre-training
stage to align the LLM with user behavior and ground SIDs in text,
followed by a supervised fine-tuning stage to make the model excel
at generative retrieval.
As a launched platform in YouTube, PLUM is able to add unique
value to the current system, while being our first neural model built
without large embedding tables. Our ablations validate the value of
the CPT stage, the benefit of initializing from a pre-trained LLM, and
our proposed SID techniques. This paper is a preliminary study on
aligning LLMs with real-world recommendation systems, opening
future research directions such as applying the PLUM framework
to other tasks like ranking and personalized search, developing new
decoding strategies for candidate diversity, and enabling seamless
generation of SIDs and natural languages.

[с. 9]
PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations

5

Acknowledgements

We would like to thank the following for their efforts and feedback in the application of the techniques described in this paper
to a diverse set of surfaces and use cases (alphabetical order): Saksham Agarwal, Sriraj Badam, Sourabh Bansod, Xilun Chen, Yi Chen,
Bernardo Cunha, Onkar Dalal, Mingyan Gao, Elad Ganmor, Danfeng Guo, Pooja Gupta, Ralf Gutsche, Yuan Hao, Chuan He, Lily
He, Yan Huang, Kate Jones, Jiewen Lang, Pengfei Li, Wen Li, Li
Wei, Sean Liew, Qiguang Liu, Sunny Liu, Yang Liu, Jianan Lu, Yilong Luan, Shifan Mao, Tomer Margolin, Radwa Metwali, Aniruddh
Nath, Hardik Patel, Amy Pu, Vasin Punyakanok, Murphy Ren, Yuji
Roh, Yuan Shao, Ajay Shekhawat, Fabio Soldo, Yanwei Song, Qian
Sun, Jiaxi Tang, Aahan Tyagi, Diego Uribe, Jacky Wang, Ting Wang,
Siqi Wu, Bo Yan, Shuhao Ye, Likang Yin, Qiao Zhang, Rein Zhang,
Sai Zhang, Vivian Zhang, Yaping Zhang, Gwendolyn Zhao, Zelong
Zhao, Zhile Zou, Dav Zimak. We also thank Mahesh Sathiamoorthy
and Anima Singh for their profound contributions to the conceptualization and initial development of this project.

References
[1] Yuwei Cao, Nikhil Mehta, Xinyang Yi, Raghunandan Keshavan, Lukasz Heldt,
Lichan Hong, Ed H. Chi, and Maheswaran Sathiamoorthy. 2024. Aligning Large Language Models with Recommendation Knowledge. (2024).
arXiv:2404.00245 [cs.IR] https://arxiv.org/abs/2404.00245
[2] Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, and Ed H.
Chi. 2019. Top-K Off-Policy Correction for a REINFORCE Recommender System.
In Proceedings of the Twelfth ACM International Conference on Web Search and
Data Mining (Melbourne VIC, Australia) (WSDM ’19). Association for Computing
Machinery, New York, NY, USA, 456–464. doi:10.1145/3289600.3290999
[3] Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, and Wenwu Ou. 2019. Behavior
sequence transformer for e-commerce recommendation in Alibaba. In Proceedings
of the 1st International Workshop on Deep Learning Practice for High-Dimensional
Sparse Data (Anchorage, Alaska) (DLP-KDD ’19). Association for Computing
Machinery, New York, NY, USA, Article 12, 4 pages. doi:10.1145/3326937.3341261
[4] Xiangyi Chen, Kousik Rajesh, Matthew Lawhon, Zelun Wang, Hanyu Li, Haomiao Li, Saurabh Vishwas Joshi, Pong Eksombatchai, Jaewon Yang, Yi-Ping
Hsu, Jiajing Xu, and Charles Rosenberg. 2025. PinFM: Foundation Model for
User Activity Sequences at a Billion-scale Visual Discovery Platform. (2025).
arXiv:2507.12704 [cs.LG] https://arxiv.org/abs/2507.12704
[5] Benjamin Coleman, Wang-Cheng Kang, Matthew Fahrbach, Ruoxi Wang, Lichan
Hong, Ed H. Chi, and Derek Zhiyuan Cheng. 2023. Unified embedding: battletested feature representations for web-scale ML systems. In Proceedings of the
37th International Conference on Neural Information Processing Systems (New
Orleans, LA, USA) (NIPS ’23). Curran Associates Inc., Red Hook, NY, USA, Article
2453, 22 pages.
[6] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks
for YouTube Recommendations. In Proceedings of the 10th ACM Conference on
Recommender Systems (Boston, Massachusetts, USA) (RecSys ’16). Association
for Computing Machinery, New York, NY, USA, 191–198. doi:10.1145/2959100.
2959190
[7] Hamed Firooz, Maziar Sanjabi, Adrian Englhardt, Aman Gupta, Ben Levine,
Dre Olgiati, Gungor Polatkan, Iuliia Melnychuk, Karthik Ramgopal, Kirill Talanine, Kutta Srinivasan, Luke Simon, Natesh Sivasubramoniapillai, Necip Fazil
Ayan, Qingquan Song, Samira Sriram, Souvik Ghosh, Tao Song, Vignesh Kothapalli, Xiaoling Zhai, Ya Xu, Yu Wang, and Yun Dai. 2025. 360Brew: A Decoderonly Foundation Model for Personalized Ranking and Recommendation. (2025).
arXiv:2501.16450 [cs.IR] https://arxiv.org/abs/2501.16450
[8] Yulong Gu, Zhuoye Ding, Shuaiqiang Wang, Lixin Zou, Yiding Liu, and Dawei
Yin. 2020. Deep Multifaceted Transformers for Multi-objective Ranking in LargeScale E-commerce Recommender Systems. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (Virtual Event,
Ireland) (CIKM ’20). Association for Computing Machinery, New York, NY, USA,
2493–2500. doi:10.1145/3340531.3412697
[9] Ruidong Han, Bin Yin, Shangyu Chen, He Jiang, Fei Jiang, Xiang Li, Chi Ma,
Mincong Huang, Xiaoguang Li, Chunzhen Jing, Yueming Han, Menglei Zhou,
Lei Yu, Chuan Liu, and Wei Lin. 2025. MTGR: Industrial-Scale Generative Recommendation Framework in Meituan. (05 2025). doi:10.48550/arXiv.2505.18654
[10] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk.
2016. Session-based Recommendations with Recurrent Neural Networks.
arXiv:1511.06939 [cs.LG] https://arxiv.org/abs/1511.06939

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

[11] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya,
Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes
Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models.
arXiv preprint arXiv:2203.15556 (2022).
[12] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley,
and Wayne Xin Zhao. 2024. Large Language Models are Zero-Shot Rankers for
Recommender Systems. Springer-Verlag, Berlin, Heidelberg, 364–381. doi:10.
1007/978-3-031-56060-6_24
[13] Yanhua Huang, Yuqi Chen, Xiong Cao, Rui Yang, Mingliang Qi, Yinghao Zhu,
Qingchang Han, Yaowei Liu, Zhaoyu Liu, Xuefeng Yao, Yuting Jia, Leilei Ma, Yinqi
Zhang, Taoyu Zhu, Liujie Zhang, Lei Chen, Weihang Chen, Min Zhu, Ruiwen
Xu, and Lei Zhang. 2025. Towards Large-scale Generative Ranking. (2025).
arXiv:2505.04180 [cs.IR] https://arxiv.org/abs/2505.04180
[14] Clark Mingxuan Ju, Liam Collins, Leonardo Neves, Bhuvesh Kumar, Louis Yufeng
Wang, Tong Zhao, and Neil Shah. 2025. Generative Recommendation with
Semantic IDs: A Practitioner’s Handbook. (2025). arXiv:2507.22224 [cs.IR]
https://arxiv.org/abs/2507.22224
[15] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. arXiv:1808.09781 [cs.IR] https://arxiv.org/abs/1808.09781
[16] Kaiyuan Li, Rui Xiang, Yong Bai, Yongxiang Tang, Yanhua Cheng, Xialong Liu,
Peng Jiang, and Kun Gai. 2025. BBQRec: Behavior-Bind Quantization for MultiModal Sequential Recommendation. (2025). arXiv:2504.06636 [cs.IR] https:
//arxiv.org/abs/2504.06636
[17] Zhuoran Liu, Leqi Zou, Xuan Zou, Caihua Wang, Biao Zhang, Da Tang,
Bolin Zhu, Yijie Zhu, Peng Wu, Ke Wang, and Youlong Cheng. 2022. Monolith: Real Time Recommendation System With Collisionless Embedding Table.
arXiv:2209.07663 [cs.IR] https://arxiv.org/abs/2209.07663
[18] M. Jeffrey Mei, Florian Henkel, Samuel E. Sandberg, Oliver Bembom, and Andreas F. Ehmann. 2025. Semantic IDs for Music Recommendation. arXiv preprint
arXiv:2507.18800 (2025). arXiv:2507.18800 [cs.IR]
[19] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean
Wu, Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii, Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko, Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang
Xiong, and Misha Smelyanskiy. 2019. Deep Learning Recommendation Model for
Personalization and Recommendation Systems. (2019). arXiv:1906.00091 [cs.IR]
https://arxiv.org/abs/1906.00091
[20] Netflix. 2024.
Foundation Model for Personalized Recommendation.
https://netflixtechblog.com/foundation-model-for-personalizedrecommendation-1a0bd8e02d39.
[21] Ming Pang, Chunyuan Yuan, Xiaoyu He, Zheng Fang, Donghao Xie, Fanyi Qu,
Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo, and Jingping Shao. 2025.
Generative Retrieval and Alignment Model: A New Paradigm for E-commerce
Retrieval. In Companion Proceedings of the ACM on Web Conference 2025 (Sydney
NSW, Australia) (WWW ’25). Association for Computing Machinery, New York,
NY, USA, 413–421. doi:10.1145/3701716.3715228
[22] Gustavo Penha, Edoardo D’Amico, Marco De Nadai, Enrico Palumbo, Alexandre
Tamborrino, Ali Vardasbi, Max Lefarov, Shawn Lin, Timothy Heath, Francesco
Fabbri, and Hugues Bouchard. 2025. Semantic IDs for Joint Generative Search
and Recommendation. (2025). arXiv:2508.10478 [cs.IR] https://arxiv.org/abs/
2508.10478
[23] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Keshavan, Trung
Vu, Lukasz Heidt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej
Kula, Ed H. Chi, and Maheswaran Sathiamoorthy. 2023. Recommender systems
with generative retrieval. In Proceedings of the 37th International Conference on
Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’23). Curran
Associates Inc., Red Hook, NY, USA, Article 452, 17 pages.
[24] Anima Singh, Trung Vu, Nikhil Mehta, Raghunandan Keshavan, Maheswaran
Sathiamoorthy, Yilin Zheng, Lichan Hong, Lukasz Heldt, Li Wei, Devansh Tandon,
Ed Chi, and Xinyang Yi. 2024. Better Generalization with Semantic IDs: A
Case Study in Ranking for Recommendations. In Proceedings of the 18th ACM
Conference on Recommender Systems (Bari, Italy) (RecSys ’24). Association for
Computing Machinery, New York, NY, USA.
[25] Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta,
Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, and
Donald Metzler. 2022. Transformer memory as a differentiable search index. In
Proceedings of the 36th International Conference on Neural Information Processing
Systems (New Orleans, LA, USA) (NIPS ’22). Curran Associates Inc., Red Hook,
NY, USA, Article 1587, 13 pages.
[26] Gemini Team. 2024. Gemini 1.5: Unlocking multimodal understanding across
millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024).
[27] Dongsheng Wang, Yuxi Huang, Shen Gao, Yifan Wang, Chengrui Huang, and
Shuo Shang. 2025. Generative Next POI Recommendation with Semantic ID.
(2025). arXiv:2506.01375 [cs.IR] https://arxiv.org/abs/2506.01375
[28] Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong,
and Ed Chi. 2021. DCN V2: Improved Deep & Cross Network and Practical
Lessons for Web-scale Learning to Rank Systems. In Proceedings of the Web

