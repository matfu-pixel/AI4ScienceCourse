# Часть 3: с. 5–6

[с. 5]
PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

detailed scaling study to analyze the relationship between model
size, compute, and retrieval performance (Section 3.4).

3.1

Performance of Generative Retrieval

Traditional recommender systems use large embedding models to
recommend candidates. However, generative retrieval has a number
of advantages over traditional systems. In this section, we study
the effectiveness of generative retrieval on the YouTube production
setup.
3.1.1

Figure 2: Illustration of Generative Retrieval for next video
recommendation. The input prompt is a sequence of interleaved SID tokens, text and custom tokens for numerical
features.

This fine-tuning employs a standard autoregressive, maximumlikelihood objective. The model learns to predict the SID tokens
of ground-truth videos, which are defined as clicked videos from
user logs given user context and history. Concretely, the model is
trained to minimize the following loss:

LSFT = −

𝐿
∑︁

𝑟 (user, 𝑣𝑐𝑙𝑖𝑐𝑘 )·log 𝑃 (𝑠𝑖𝑑𝑡 |Contextuser, Historyuser, 𝑠𝑖𝑑 <𝑡 ),

𝑡 =1

(3)
where [𝑠𝑖𝑑 1, .., 𝑠𝑖𝑑𝐿 ] represents the SID of clicked video 𝑣𝑐𝑙𝑖𝑐𝑘 , and
𝑟 (𝑢𝑠𝑒𝑟, 𝑣𝑐𝑙𝑖𝑐𝑘 ) is a handcrafted reward signal of each click. In practice, given the high cost of training, we sample training examples
based on this reward and then weigh the sampled examples equally
in the loss. As illustrated in Figure 2, the input prompt contains not
only SID tokens and custom tokens for numerical features, but also
other text features that can be naturally encoded by pre-trained
LLMs.
During inference, we use beam search to decode multiple SID
sequences, which serve as the set of retrieved candidates. Each
generated SID is then mapped to a real video in our billions-scale
corpus. While this generative process can potentially produce invalid SIDs (hallucination) or SID-to-video collisions, we observe
that the hallucination rate after SFT is very low (< 5%), and the
uniqueness of SID-to-video mapping remains high (see Table 4).

3

Experiment Setup.

Model. For the following experiments, we trained a 900M activatedparam PLUM model from the Gemini-1.5 Mixture-of-Experts (MoE)
family on both Long Form Video (LFV) and Shorts. The model is
warm started from Gemini and continuously finetuned on new engagement data as described above. The training data is a mixture of
the most recent data and historical data. During serving, target sequences are decoded using beam search. In our experiments, beam
search performed better than random decoding, although at the
cost of some diversity.
Baseline. We compare the performance of the above model to
traditional LEM retrieval. This baseline is the top-performing production model, being highly-optimized since early work such as
Chen et al. [2]. This model is also based on the Transformer architecture, but most of its parameters are in the embedding layer,
with O(10M) vocab sizes for input and output item IDs. Specifically,
LEM’s neural network comprises only 0.4% of its total parameters,
whereas PLUM’s neural network accounts for 90%.
3.1.2

Experiment Results.

Recommendation Quality. We assess the quality of recommendations on a few different axes. First, we measure the effective vocab
size of a recommender as the number of unique videos needed
to cover 95% of its impressions. Higher vocab sizes are generally
desirable for personalized discovery of niche content, showing the
model can generalize better. Next, we compare the effectiveness
of recommendations using two metrics: a) The click-through-rate
(CTR) and Recommendation Acceptance. For Recommendation Acceptance, we compare both the length of video watched (WT/View)
and the fraction of video watched (WF/View). In all cases, we report
the ratio of the metric for the 900M MoE to the metric for the LEM
model.
Table 2: Comparison of recommendation quality: Each number is a ratio, dividing the metric for PLUM by that of LEM.

Experiments

In this section, we conduct a comprehensive set of experiments to
validate the PLUM framework. We first evaluate the performance
of a PLUM-based generative retrieval model against a Transformerbased large-embedding retrieval model that contributes to a majority of impressions in production (Section 3.1). Following this,
Section 3.2 demonstrates the effectiveness of our proposed enhancements to SIDs. We then conduct ablation studies for two critical
components in the PLUM framework: Section 3.3 quantifies the
precise impact of the continued pre-training (CPT) stage and the
value of initializing from a pre-trained LLM. We finally present a

Metric

LFV

Shorts

Effective Vocab Size
CTR
WT/View
WF/View

2.60x
1.42x
0.72x
1.32x

13.24x
1.33x
1.13x
1.03x

We observe that the PLUM model achieves much larger effective
vocab size, while having competitive performance against LEM
based on the metrics related to user reactions.

[с. 6]
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

He et al.

Live Experiments. We ran live experiments by adding the PLUM
model recommendations to the candidate pool. For a fair comparison, we increased the quota (by the same amount) for the best
performing retrieval model in production and use this as the baseline. We denote this modified production baseline as LEM+. In the
following table, we report the metric movements observed with
respect to LEM+ on four key metrics, for both LFV and Shorts. This
demonstrates that PLUM-based retrieval can add unique value on
top of the existing system, even without large embedding tables.

LFV

Shorts

Engaged Users
Panel CTR
Views
Satisfaction

+0.07%
+0.76%
+0.80%
+0.06%

+0.28%
+4.96%
+0.39%
+0.39%

3.3

Impact of Continued Pre-training

To measure the importance of the continued pre-training phase, we
evaluate its impact on the training efficiency and retrieval performance of the end models for generative retrieval.
3.3.1 Experiment Setup. Our training methodology contains two
stages: continued pre-training and generative retrieval fine-tuning,

SIDv1 (Baseline)
SIDv2 (Ours)

94.0%
96.7%

12.3%
14.4%

Ablate
MultiResolution
Ablate
MultiEmbedding
Ablate
Cooccurence

94.8%

13.2%

96.9%

12.8%

91.8%

12.6%

• R1: retrieval SFT (random init). Transformer model is directly trained on the generative retrieval task with randomly
initialized weights.
• R2: retrieval SFT (LLM init). This model is initialized from
a pre-trained LLM checkpoint, but bypasses the CPT stage.
• CR1: CPT (random init) + retrieval SFT. The model training follows the two PLUM training stages, but the model
used in CPT is randomly initialized.
• CR2: CPT (LLM init) + retrieval SFT. This represents the
full and intended implementation of our proposed framework.

Semantic IDs Ablation Study

In Table 4 we discuss the importance of different changes we introduced to the prior SID [23]. We refer to the prior work as SIDv1,
and refer to our proposed method as SIDv2 in the table. Since each
SID model will result in a different indexing structure, we report
the index uniqueness depicting how many videos in the corpus are
uniquely represented in the SID space along with the final Video
Recall@10 after training a GenRetrieval model on the SID. The
Video Recall@K is determined by calculating the recall of successfully retrieving a specific video given PLUM’s top K SID predictions.
When an SID maps to multiple videos, we randomly sample a video
when computing the recall metric.
We conducted ablation experiments to the SIDv2 changes in a
simple setting with a 900M MoE PLUM retrieval model without
CPT and removed watch history in the prompt. As shown in Table 4
, these experiments assessed both the uniqueness of Semantic ID
and downstream generative retrieval VID Recall@10. Results show
that all changes improve generative item retrieval recall. Notably,
incorporating a Co-occurrence alignment task greatly improved
both Semantic ID uniqueness and recall.

VID Recall@10

each tailored with a specific learning objective and a corresponding
set of hyperparameters. To evaluate the distinct contributions of
each step, we design a controlled 2x2 ablation study. This involves
four model configurations that share the same decoder-only Transformer architecture. We use the MoE-900M architecture for this
study. The four models are as follows:

Sample Efficiency. In our studies, PLUM models are extremely
sample efficient. The 900M MoE model trains on roughly 250M
examples every day. By comparison, the traditional LEM trains
on several billion examples per day. This sample efficiency also
makes these models practical for production use, while training
cost per example is much larger. Compared to the LEM, the 900M
MoE model uses < 0.55x flops to train.

3.2

SID Uniqueness

Table 4: Ablation experiments on SIDv2 changes

Table 3: Comparison of engagement: Percentage change by
adding PLUM compared to LEM+
Metric

SID Model

3.3.2

Experiment Results.

Impact of Continued Pre-training. The main results are summarized in Table 5. Following the scaling study, the performance of
each of the four models is evaluated on a large, held-out next day
dataset, and we report recall@10 as the retrieval metric. There is a
large performance gap between R1 and CR1 (or R2 and CR2), showing the benefit of CPT for retrieval fine-tuning. Furthermore, Figure
3 shows the 8-th day recall@10 and the training loss against training steps. It’s clear to see that the models with CPT can converge
much faster. Therefore, CPT can benefit the training efficiency of
task-specific fine-tuning, especially when multiple downstream
tasks share the same CPT model.
Table 5: Main generative retrieval performance of the four
model configurations.
Model

Pre-trained
LLM

CPT

Recall@10
(8th-day)

R1

No

No

0.19

R2

Yes

No

0.23

CR1

No

Yes

0.27

CR2

Yes

Yes

0.28

