# Часть 1: с. 1–2

[с. 1]
PLUM: Adapting Pre-trained Language Models for
Industrial-scale Generative Recommendations
Ruining He∗†

Lukasz Heldt∗†

Lichan Hong∗†

Google DeepMind

YouTube

Google DeepMind

Raghunandan
Keshavan∗†

Shifan Mao∗
YouTube

YouTube

Nikhil Mehta∗†

Zhengyang Su∗

Alicia Tsai∗†

Yueqi Wang∗

Google DeepMind

YouTube

Google DeepMind

YouTube

Shao-Chuan
Wang∗

arXiv:2510.07784v1 [cs.IR] 9 Oct 2025

Google DeepMind

Xinyang Yi∗†

Lexi Baugher

Baykal Cakici

Ed Chi

Cristos Goodrow

Google DeepMind

YouTube

YouTube

Google DeepMind

YouTube

Ningren Han

He Ma

Romer Rosales

Abby Van Soest

Devansh Tandon

YouTube

YouTube

YouTube

YouTube

YouTube

Su-Lin Wu

Weilong Yang

Yilin Zheng

YouTube

YouTube

YouTube

Abstract

CCS Concepts

Large Language Models (LLMs) pose a new paradigm of modeling
and computation for information tasks. Recommendation systems
are a critical application domain poised to benefit significantly from
the sequence modeling capabilities and world knowledge inherent in these large models. In this paper, we introduce PLUM, a
framework designed to adapt pre-trained LLMs for industry-scale
recommendation tasks. PLUM consists of item tokenization using
Semantic IDs [23, 24], continued pre-training (CPT) on domainspecific data, and task-specific fine-tuning for recommendation
objectives. For fine-tuning, we focus particularly on generative retrieval, where the model is directly trained to generate Semantic IDs
of recommended items based on user context. We conduct comprehensive experiments on large-scale internal video recommendation
datasets. Our results demonstrate that PLUM achieves substantial
improvements for retrieval compared to a heavily-optimized production model built with large embedding tables. We also present
a scaling study for the model’s retrieval performance, our learnings about CPT, a few enhancements to Semantic IDs, along with
an overview of the training and inference methods that enable
launching this framework to billions of users in YouTube.

• Do Not Use This Code → Generate the Correct Terms for
Your Paper; Generate the Correct Terms for Your Paper; Generate
the Correct Terms for Your Paper; Generate the Correct Terms for
Your Paper.

∗ Core contributors. Alphabetical order.
† Corresponding authors: Xinyang Yi (xinyang@google.com), Lukasz Heldt
(heldt@google.com), Raghunandan Keshavan (hkraghunandan@google.com), Ruining He (ruininghe@google.com), Nikhil Mehta (nikhilmehta@google.com), Alicia Tsai (aliciatsai@google.com), and Lichan Hong (lichan@google.com).

Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
© 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXX

Keywords
Recommender Systems, Generative Retrieval, Large Language Models
ACM Reference Format:
Ruining He, Lukasz Heldt, Lichan Hong, Raghunandan Keshavan, Shifan
Mao, Nikhil Mehta, Zhengyang Su, Alicia Tsai, Yueqi Wang, Shao-Chuan
Wang, Xinyang Yi, Lexi Baugher, Baykal Cakici, Ed Chi, Cristos Goodrow,
Ningren Han, He Ma, Romer Rosales, Abby Van Soest, Devansh Tandon,
Su-Lin Wu, Weilong Yang, and Yilin Zheng. 2018. PLUM: Adapting Pretrained Language Models for Industrial-scale Generative Recommendations.
In Proceedings of Make sure to enter the correct conference title from your
rights confirmation email (Conference acronym ’XX). ACM, New York, NY,
USA, 11 pages. https://doi.org/XXXXXXX.XXXXXXX

1

Introduction

Recommender systems are critical in modern digital platforms,
profoundly shaping how users discover and interact with content.
The last decades have witnessed the remarkable success of deep
learning models for recommendation in both retrieval and ranking
stages. Despite the use of neural networks, the dominant paradigm
in industrial recommenders continues to rely on massive embedding tables to represent high-cardinality categorical features such
as item IDs (see, e.g., [5, 6, 17, 19]). In these Large Embedding Models (LEMs), the vast majority of parameters reside within these
embedding tables. While highly effective at memorizing user-item
interactions, this architectural choice hinders the potential benefits of deeper, more complex networks (e.g., [3, 8]). This scaling
methodology, focused on enlarging embedding tables, is in contrast

[с. 2]
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

to that of LLMs, which emphasizes on increasing neural network
sizes to learn compositions of compact input tokens.
The success of LLMs has inspired an emerging paradigm shift
for building recommendation models [4, 7, 21, 23, 36, 40]. The inherent sequence modeling capabilities and vast world knowledge
encoded in LLMs present opportunities for building more intelligent and personalized recommender systems. However, adapting
LLMs for recommendations is non-trivial. The primary challenge
lies in bridging the domain gap: LLMs are not pre-trained with
user behavior data and item corpus in the target domain, making
it harder for them to understand user preferences from user activities and nuanced item quality. Consequently, directly applying
off-the-shelf LLMs to recommendation tasks has shown a persistent performance gap, even on small public datasets [12]. Secondly,
the traditional input representation based on large embedding tables poses a scaling challenge. These tables demand large amounts
of training data, making it expensive to train large Transformer
architectures.
This paper takes a step in the direction of teaching LLMs to solve
recommendation tasks. We introduce PLUM, a framework for effectively adapting pre-trained LLMs for industrial-scale generative
recommendation. PLUM consists of three key stages:
• Item tokenization. Each item in the corpus is represented
by a sequence of discrete tokens known as Semantic IDs
(SIDs). Building on the prior work with RQ-VAE [23, 24],
we introduce a set of new techniques (referred to as SID-v2)
for incorporating user behavioral signals and multi-modal
content embeddings, and improve the hierarchical integrity
through multi-resolution codebooks and progressive masking.
• Continued pre-training (CPT). In this stage, the vocabulary of a pre-trained LLM is expanded to include the new
SID tokens. The model is further pre-trained on a mixture of
domain-specific item data and user sequences, and generaldomain text data to align the new SID modality with model’s
existing knowledge.
• Task-specific fine-tuning. Finally, the model is fine-tuned
for specific recommendation objectives. While PLUM can
support various downstream tasks, this paper focuses on
its application to generative retrieval, where a decoder-only
model is trained to autoregressively generate the SIDs of next
items a user is likely to engage with. Generative retrieval
does not need to maintain a separate index of an item corpus,
and bypasses the dot-product limitation [29] in embeddingbased retrieval [6, 35].
The PLUM framework is designed to address the aforementioned
challenges. The continued pre-training bridges the domain gap by
enriching the pre-trained LLMs with domain corpus and user behavior patterns. We observe that the model after CPT can demonstrate
basic few-shot learning capability for generating text tokens based
on SID input. The SID-based input representation circumvents the
scaling bottleneck of LEMs. Our experiments show that by shifting
the model complexity from input embeddings to the neural networks, PLUM-based generative retrieval can achieve significantly
better sample efficiency compared to a production Transformerbased retrieval model with large embedding tables. Because of this,

He et al.

while PLUM retrieval uses a Transformer architecture that has
100x dense parameters than LEM, the overall training cost for the
retrieval task is comparable due to faster convergence. We also
observe effective scaling: the retrieval performance continues to
improve up to a Mixture-of-Experts (MoE) model with over 900M
activated parameters (approximately 4.2B parameters in total).
In addition to these contributions, we will share practical learnings from deploying PLUM-based retrieval in a large-scale production environment, particularly on the comparisons between PLUM
retrieval and LEM-based retrieval in online A/B testing. The PLUM
framework is in production for YouTube recommendations, serving
the retrieval of both long-form and short-form videos on multiple
core surfaces via online and offline inference.

1.1

Related Work

We’ve seen two primary lines of work inspired by the recent advancements of LLMs. The first line of work focuses on scaling up
the neural network component of LEMs with Transformer architecture (or its variant), which we discuss in Section 1.1.1. The second
line, as detailed in Section 1.1.2 and 1.1.3, focuses on token-based
generative recommendation, and recasts recommendation retrieval
as seq2seq transduction tasks.
1.1.1 Sequential Recommendations. Modeling user sequences and
feature interactions have been key areas for improving recommendation models, ranging from using classic sequential neural networks [10, 15] to designing custom feature interaction components
[28, 37]. More recently, the emerging trend is to use a Transformerlike architecture to unify user history and heterogeneous features
in a single sequence [9, 13, 36], showing better scaling compared to
traditional MLP architectures. Furthermore, the training paradigm
itself has evolved to mirror that of LLMs, with some works adopting
self-supervised pre-training objectives followed by task-specific
fine-tuning [4, 20]. A common thread across all these advancements
is their continued reliance on massive embedding tables for categorical features. Our work departs from this paradigm by replacing
large embedding tables with compact input tokens. Besides SIDs,
we use LLMs to process heterogeneous features by tokenizing numerical features or simply use dense embedding features as soft
tokens.
1.1.2 Semantic IDs and Quantization. The concept of representing
items as sequences of discrete tokens, or Semantic IDs (SIDs), has
emerged as a powerful alternative to traditional ID embeddings.
This approach allows items to be treated as a "language" that can
be processed by sequence models. Early work demonstrated the
feasibility of training retrieval models from scratch to predict SIDs
[23], and subsequent research showed that SIDs could be hashed to
replace random item ID embeddings, enhancing generalization of
ranking models [24, 39].
A significant body of recent research has focused on improving
the quality and expressiveness of SIDs by incorporating diverse signals into the quantization process. This includes fusing multi-modal
content features [30] and injecting collaborative filtering signals
from user behaviors [16, 22, 33, 34]. PLUM contributes to this line
of research in parallel. Different from MMQ [30], where different
tokens are generated for each modality, we simply concatenate

