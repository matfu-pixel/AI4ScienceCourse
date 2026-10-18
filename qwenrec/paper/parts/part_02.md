# Часть 2: с. 3–4

[с. 3]
PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

multiple content embeddings before encoding them with an MLP,
offering the flexibility of supporting more embeddings. For incorporating user behaviors, while most works fuse collaborative-filtering
(CF) item embeddings to content embeddings during quantization,
the CF-based item embeddings are usually dynamic depending on
item popularity change, necessitating frequent retraining of the
quantizer and downstream models. We avoid this by using CF signals in a contrastive training objective to guide the quantizer to
capture useful information from content information. In addition,
we present two other key innovations for improving the alignment
of SID training with generative retrieval.

our framework to be agnostic to the specific number and type of
input sources, enabling it to ingest and fuse multiple, heterogeneous representations. The model is designed to accept a set of
𝑀 for each item. The fusion of
distinct embedding vectors {𝑥𝑚 }𝑚=1
such multiple representations is achieved through a separate embedding encoder E𝑚 that encodes 𝑥𝑚 into a latent vector 𝑧𝑚 and a
concatenation 𝑧˜ = [𝑧 1, . . . , 𝑧𝑀 ] followed by a projection, creating a
single, unified feature encoded vector 𝑧. By integrating information
from across modalities, we form a superior input for the subsequent
RQ-VAE quantization process, ensuring that the resulting SIDs are
grounded in a comprehensive understanding of the item’s content.

1.1.3 Generative Retrieval and Alignment. Generative retrieval reframes recommendation retrieval as a sequence-to-sequence task
where the model directly generates the SIDs of relevant items in
an autoregressive manner. This idea has gained significant traction
across the industry (e.g., [14, 18, 21, 27]), since the seminal works
in document search [25] and recommendation [23]. For instance,
OneRec [40] adopts an encoder-decoder architecture to generate
video SIDs, and leverages RL training to enhance the model quality
with a reward model. Other works [31, 32] propose hybrid models
that generate both SIDs and dense embeddings to alleviate the information loss from quantization. A commonality among most of
these approaches is their focus on training generative models from
scratch. Our work has a focus on aligning SIDs with LLMs, and
investigates the impact of LLM pre-training and CPT on generative
retrieval performance. While some research [1, 7] has explored
aligning LLMs with recommendation tasks, their focus is primarily
on language interfaces. There is limited study of how to effectively
incorporate SIDs as a new modality.

2.1.2 Hierarchical Refinements in Quantization. We further refine
the RQ-VAE architecture itself to produce a more efficient and
meaningful hierarchy. This involves two key innovations:

2 PLUM Framework
2.1 Semantic IDs
A Semantic ID (SID) is conceptualized as a tuple of discrete codewords derived from an item’s underlying content features. This
process involves two primary stages: (1) encoding high-level content features into a dense semantic embedding, and (2) quantizing
this embedding into a hierarchical tuple of codewords. The efficacy of a generative retrieval model is fundamentally dependent
on the semantic richness and structural integrity of the SIDs. Building upon the foundational TIGER framework, which leverages a
Residual-Quantized Variational AutoEncoder (RQ-VAE), we introduce a series of significant enhancements to this generation process.
The initial approach was limited by its reliance on a single content
embedding source and an absence of collaborative signals in the ID
structure. Our methodology is designed to produce SIDs that are
more comprehensive, better aligned with user behavior, and more
hierarchically coherent. Figure 1 illustrates our overall design for
training the SID model.
2.1.1 Fused Multi-Modal Content Representation. A primary limitation of relying on a single content representation in [23] is its
inherent inability to capture the multi-faceted nature of complex
media items. For instance, a video’s full semantic meaning is conveyed through a combination of its textual metadata, visual content,
and audio. A unimodal representation, by definition, overlooks rich,
orthogonal sources of information. To overcome this, we architect

• Multi-Resolution Codebooks: Prior work [23] employs
a fixed uniform-resolution codebooks that can be parametrically inefficient, leading to a vast and sparsely populated
SID space where the majority of potential codeword combinations remain unassigned. We replace this with a multiresolution codebook structure, where the initial SID levels
have high resolution and are maximally discriminative, while
subsequent codewords encode low-entropy residuals with
lower resolution. In particular, the codebook cardinality is a
function of the quantization level: 2048/2𝑙𝑒𝑣𝑒𝑙 −1 resulting in
a more compact and efficient SID.
• Progressive Masking: To enforce a stricter and more interpretable hierarchy during residual quantization, we introduce progressive masking. In particular, we define a binary
mask scalar 𝑚𝑙 ∈ {0, 1} where 𝑙 is the codebook level such
that 𝑚𝑙 = 1𝑙 <𝑟 where 𝑟 ∈ [1, 𝐿] is a random integer and 𝐿 is
the total number of codebook levels. This mask is applied to
select the first 𝑟 codebook levels in SID training.
2.1.3 RQ-VAE with Co-occurrence Contrastive Regularization. SIDs
derived purely from an item’s intrinsic content may not fully capture the notion of similarity as perceived by users in a recommendation context. User behavior provides a powerful, extrinsic
signal of video relatedness; videos that are frequently watched together are often semantically linked. To bridge this gap between
content-based and behavior-based similarity, we inject a strong
collaborative signal directly into the ID generation phase by introducing a co-occurrence contrastive loss term (L𝑐𝑜𝑛 ) into the
RQ-VAE training objective. This objective function is designed to
shape the embedding space based on item co-occurrence patterns
within user interaction sequences. The loss encourages the model
to generate similar SID representations for items that frequently
appear together, while pushing apart the representations of items
that do not. In particular, the contrastive loss is defined as:
L𝑐𝑜𝑛 = −

2𝑁
∑︁𝑏

𝑒𝑥𝑝 (𝑠𝑖𝑚(𝑝𝑖 , 𝑝𝑖+ ))
,
Í2𝑁𝑏
𝑖=1
𝑗=1 𝑒𝑥𝑝 (𝑠𝑖𝑚(𝑝𝑖 , 𝑝 𝑗 ))

(1)

where 𝑝𝑖 represents a video representation in a batch of 𝑁𝑏 videos,
𝑝𝑖∗ represents the representation of the video that co-occurred with
video 𝑖 and 𝑠𝑖𝑚(𝑝𝑖 , 𝑝 𝑗 ) is the dot-product similarity between videos
𝑖 and 𝑗.

[с. 4]
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

He et al.

Embed 2

DNN
Encoder 2

A =

Codebook 1

C = B

B = A -

-

+

Codebook 3

Codebook 2

+

E = D -

D = C -

-

+

+

DNN
Decoder 2

MLP

MLP

Codebook 4

=

Reconstruction Loss

DNN

Encoder 1

Concat & Project

Embed 1

Residual Quantization
Codebook 0

Contrastive Loss

Embed 2

Embed 2

Embed 2

DNN
Decoder 1

Video 2

Embed 1

Video 1

Video 2

Embed 1

Embed 1

Co-occurring Pairs
Video 1

Figure 1: Illustration of our Semantic ID model. It takes two multi-modal video embeddings, encodes them, and compresses the
result into a quantized ID using a residual quantizer. This ID is trained to both reconstruct the original inputs and semantically
cluster co-occurring videos using a contrastive loss.
2.1.4 SID Training Loss. In addition to the training loss introduced
in [24], we introduce the co-occurence contrastive loss term. In
particular, the overall loss is defined as:
L = L𝑟𝑒𝑐𝑜𝑛 + L𝑟𝑞 + L𝑐𝑜𝑛 ,

(2)

Table 1: Example schemas used in continued pre-training.
Example user behavior training data
wh = <sid_1> <channel_name> <watch_ratio> <watch_time>
<hours_since_final_watch> <sid_2> <channel_name> ... || <sid_n>
SID + video title

where L𝑟𝑒𝑐𝑜𝑛 = 𝑚=1 ||𝑥𝑚 − 𝑥ˆ𝑚 || 2 is the reconstruction loss for
Í𝐿
each multi-modal embedding and L𝑟𝑞 = 𝑙=1
𝛽 ||𝑟𝑙 − 𝑠𝑔[𝑒 ∗𝑙 ]|| 2 +
||𝑠𝑔[𝑟𝑙 ] − 𝑒 ∗𝑙 || 2 , where 𝑒 ∗𝑙 is the closest code in the codebook at
level 𝑙, 𝑟𝑙 = 𝑟𝑙 −1 − 𝑒 ∗𝑙 is the residual computed recursively at each
level starting with 𝑟 0 = 𝑧 and sg is the stop-gradient operator. As
shown in figure 1, the final quantized vector 𝑧ˆ is computed with
progressive masking as the sum of chosen codes from each level:
Í𝐿
𝑧ˆ = 𝑙=1
𝑚𝑙 𝑒 ∗𝑙 . The quantized vector is then used as an input for
each decoder D𝑚 to reconstruct the input embeddings 𝑥ˆ𝑚 .
Í𝑀

2.2

Continued Pre-training

Following the creation of the SID vocabulary, one primary objective
of the continued pre-training (CPT) stage is to develop a base model
checkpoint where the SID tokens are semantically grounded and
aligned with existing text tokens of the base language model. To
achieve this, we train the model using next-token predictions on a
large-scale corpus constructed from two primary sources:
• User behavior data: This source is essential for personalization and capturing watch histories. During training, each
example leverages user watch histories along with additional
watch features to model user behavior sequences.
• Video metadata corpus: This large-scale corpus is designed
to create a strong association between SIDs and their corresponding textual features. Each example includes a video’s
SID, title, description, ASR captions, channel name, and synthetically generated data.
Table 1 illustrates our CPT training data schema.

Video <sid> has title (en): <video_title>
SID + video topics
The topics in video <sid> are: <topics>

Data Mixture and Training. The training set for the CPT stage is
composed of a mixture of the user behavior data and the video metadata corpus, with each source accounting for 50% of the training
examples. The CPT stage undergoes 1 million training steps with a
batch size of 16, amounting to approximately 260 billion tokens in
total. We set up various evaluations to measure the performance of
generating SIDs based on user history, the ability to jointly model
SIDs and language through held-out video metadata corpus and
tracking the degradation of general language capabilities through
standard text benchmarks.
In-context Learning. The CPT models, having been trained on a
mixed corpus of SIDs and natural language, retain the flexibility to
generate free-form text, and have the in-context few-shot learning
capability. We provide some examples in Appendix A.2.

2.3

Generative Retrieval

While the continued pre-training (CPT) stage enables the model to
generate next-video SIDs from user history, a subsequent Supervised Fine-Tuning (SFT) stage is essential to specialize it for the
retrieval task. This SFT stage allows the model to incorporate a
richer set of features, particularly real-time context, and to be optimized directly for a reward signal that aligns with user experience.

