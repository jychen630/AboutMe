

We utilized the SAEs trained on each layer of Llama-3.1-8B model, [technical details](https://arxiv.org/pdf/2410.20526) provided by the NLP Lab at Fudan University. These SAEs were trained on text data sampled from [SlimPajama (Soboleva et al., 2023)](https://cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) with the proportions of each subset (e.g., Commoncrawl, C4, Github, Books, Arxiv, StackExchange) preserved. There are three positions in a transformer block on which the SAE were trained, specifically:
- Post-MLP Residual Stream (R): The residual stream output of the transformer block at a layer. (Denoted as $x_{i+2}$ in the below image)
- Attention Output (A): The output of each attention layer inside the transformer block. (Denoted as $\Sigma_{h\in H_i} h(x_{i})$ in the below image)
- MLP Output (M): The output of each MLP layer inside the transformer block. (Denoted as $m(x_{i})$ in the below image)

![sae_positions_by_fudan](image-11.png)

Caption: The three positions (mlp, attention and residual stream) in one transformer block. Original image is from NLP lab at Fudan University, who trained the LLaMa SAEs we are analyzing upon. The image is clipped from [their paper](https://arxiv.org/pdf/2410.20526).

![sae_positions_by_anthropic](image-12.png)

Caption: The residual stream and its sub-components (attention and mlp) in one transformer block. Original image is from Anthropic blog, "[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)" (2021).


Recalling the basic architecture of an SAE, it consisted of an encoder mapping from a matrix of activation values to a sparse vector of dimension $d$; then a ReLU activation function applied to the sparse vector; finally a decoder mapping from the sparse vector to another matrix of activation values. Below is a simple diagram for understanding. as reference The sparse vector is an encoded representation of the input activation values. The second matrix of activation values is the "reconstructed" version of original matrix. 

![simple_sae_diagram](image-13.png)

Caption: Diagram of a sparse autoencoder. Note that the intermediate activations are sparse, with only 2 nonzero values. Note that a relu function should be applied to the sparse vector, before applying the decoder, which is not shown in the diagram.The original image is from Adam Karvonen's blog, "[An Intuitive Explanation of Sparse Autoencoders for LLM Interpretability](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html)".

The authors who trained the llama SAEs picked $d$ to be a multiple of the hidden dimension of the hidden layer of the llama model. This constant factor, also called the *expansion factor* is 8 or 32 for each SAE. Specifically, by llama architecture, the input activation dimension is 4096, and the SAE’s encoded representation is a vector of length 32k (4096 x 8) or 128k (4096 x 32). Further take the expansion factor of 8 as an example. The encoder, i.e a matrix, is of shape (4096, 4096\*8) and the decoder, i.e another matrix, is of shape (4096\*8, 4096). Note that the second input shape and output shape are the same, both are of 4096, which is again by design of SAE that aims to reconstruct the original activation values.

\~footnote{During SAE training, the loss was designed to encourage the sparsity of the vector (i.e, very few values out of $d$ values are non-zero), and encourage the reconstructed matrix to be as close as possible to the original matrix. The sparsity loss is an L1 loss, summing over all the elements in the vector. The reconstruction loss is an L2 loss, summing over the error between each individual element between the original and reconstructed matrices.}


Llama-3.1-8B consists of 32 layers, resulting in 96 (32 \* 3) possible training positions. For each position, there are two SAEs trained, each with a different expansion factor. Therefore, we totally have 96 \*2 = 192 SAE candidates available to analyze. Out of these, we picked 20 SAEs, across different layers, positions and expansion factors. In the below table, the cell with green tick marks indicate the SAE we analyzed. 

| Position             | Expansion Factor | Layer 7 | Layer 15 | Layer 16 | Layer 23 | Layer 24 | Layer 31 |
|----------------------|------------------|---------|----------|----------|----------|----------|----------|
| Attention Output (A) | 8x               | ✅       | ✅        | ✅        | ✅        | ✅        | ✅        |
|                      | 32x              | ✅       | ✅        | ✅        | ✅        | ✅        | ✅        |
| MLP Output (M)       | 8x               |         |          |          |          |          |          |
|                      | 32x              | ✅       | ✅        |          | ✅        |          | ✅        |
| Residual Stream (R)  | 8x               |         |          |          |          |          |          |
|                      | 32x              | ✅       | ✅        |          | ✅        |          | ✅        |



Each SAE's activation output were obtained when feeding the humicroedit data ([huggingface](https://huggingface.co/datasets/SemEvalWorkshop/humicroedit), [original paper with useful stats](https://arxiv.org/pdf/1906.00274)), subtask-1. Each item, quoted below, consists of an original sentence with the edited position marked by `</>`, the edited word, and the 5 grades provided by human annotators. Higher mean grade (with max of 3, min of 1) indicates the sentence is "funnier". \~footnote{I'm personally not certain how to define "funniess", nor do I know how to map the sense of funniness to a numeric score. My intuition is that these edited sentences induce laughs from easier to harder, less to more, or semantically, larger to small semantic/conceptual jumps. (A personal heuristic is that if there is no single-side strong feeling toward the thing, I may rate as 2. Otherwise I rate 1 or 3 correspondingly. Many times I was asked to rate a certain event on a scale from 1 to 5, I usually just throw a random number between 2 to 4 if I dont have strong feelings. It would be harder to rate on a scale from 1 to 10, as it's harder to articulate a more granular feeling numerically.)}

```
{
  'id': 1183,
  'original': 'Kushner to visit <Mexico/> following latest trump tirades.',
  'edit': 'therapist',
  'grades': '33332',
  'meanGrade': 2.8
}
```

Among these humicroedit data, we filtered out the items with meanGrade less than or equal to 2. The remaining data out of 9652 items is 630 (only 6.5% of them are considered quite funny!).

## Title needed
Below we describe the process ot getting the dashboards. Note that the process is done the same **for each feature** (at a position {a,m,r}, at a layer {0..31}, at an expansion factor {8x,32x}) **for each SAE**.

An activation value in the feature matrix corresponds to a token from the prompt. (Please look at the below pipeline figure to understand the details.) We masked out the activation values whose position are not in the edited text.

```
## First generate the mask (using two points: left sweep and right sweep and then sandwiched)
sentence_tokens_original (decoded): ['Royal', ' wedding', ' :', ' Meg', 'han', " '", 's', ' dress', ' in', ' detail']
sentence_tokens_original (len 10) [38702 11812  1146 18158  5418   684    85  7417   276  2429]
sentence_tokens_edited (decoded): ['Royal', ' wedding', ' :', ' Meg', 'han', " '", 's', ' elbow', ' in', ' detail']
sentence_tokens_edited (len 10) [38702 11812  1146 18158  5418   684    85 23984   276  2429]
mask (len 10): [False, False, False, False, False, False, False, True, False, False]

## Apply mask to activation values
mask: [False, False, False, False, False, False, False, True, False, False]
+
activation values: [0, 0, 0, 0, 0, 0.5, 0.6, 0.3, 0, 0.1]
=
masked activation values: [0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0]
```

![pipeline1_generating_feature_matrix](image-14.png)

Caption: The pipeline generating the feature matrix at applying SAE with humor prompts.

![pipeline2_fetching_tokens](image-15.png)
Caption: The pipeline fetching the tokens corresponding to the (large) activation values, continuing from the previous figure.


We first sorted by their activation density, which is defined to be the number of non-zero values divided by the total number of values. For those features with activation density greater than 0.01, we stream the corresponding 20 centered tokens and their neighboring tokens to dashboard. So on dashboards, you should see a dropdown list, with first item the largest activation density. The number on the dropdown list of each feature is simply an index generated programmatically, which doesn't mean anything. Note that the sentences are concatenated together, so the analysis is done on a corpus level, and you should see some `<endoftext>` in the middle of the sentences.

We manually inspected the highlighted edited words for each feature for each SAE in a previously table shown with green ticks, starting from the feature with the highest activation density. Here we displayed several strong features manually found.

### Layer 7, Residual Stream, 32x expansion factor (i.e out of 128k features), feature id 1216
The majority (17/20) is about kids or childhood.

[leprechaun, circus\*5, bedtime, mommy, picnic, (break)dance, clown, (marsh)mallow, musical, (ch)ihu(ah)ua, (kind)ergarten, (sleep)over]

![l7r_32x_1216](image-1.png)

### Layer 7, Residual Stream, 32x expansion factor (i.e out of 128k features), feature id 9920
All (20/20) about sex/sexual activities/sexual organs/sexual connotations

[bed, (sexual )toy, (cop)ulate, urinate, (test)icles, (haemorh)oid, (s)perm, bra, underwear, diarrhea, brothel, grope, diaper, anal]

![l7r_32x_9920](image-2.png)

### Layer 7, Residual Stream, 32x expansion factor (i.e out of 128k features), feature id 14080

Half (10/20) about animals.

[puppy\*2, puppies, chimpan(ze)es, (par)rots, (plat)ypus\*2, (uni)corns, (deodor)ant, monkeys]

![l7r_32x_14080](image-3.png)

### Layer 23, Attention Output, 8x expansion factor (i.e out of 32k features), feature id 13056 

More than half (12/20) about beautfy and fashion.

[wife, haircut\*2, cactus, hair, shampoo, (compl)iment, underwear, mom, fashion, barber, hairdo]

![l23a_8x_13056](image-4.png)

### Layer 31, Attention Output, 8x expansion factor (i.e out of 32k features), feature id 5696

Weakly, nearly half (9/20) about health or bodily functions.

[urinate, (nut)ritionist, (g)ynecologist\*2, sneeze\*2, moustache, testicles, tanning]
![l31a_8x_5696](image-5.png)   

### Layer 23, Residual Stream, 32x expansion factor (i.e out of 128k features), feature id 6528

All (20/20) words started with character "c" - though not indicate any semantics.

![l23r_32x_6528](image-6.png)

### Layer 15, Residual Stream, 32x expansion factor (i.e out of 128k features), feature id 13952

All (20/20) words started with character "s" - though not indicate any semantics.

![l15r_32x_13952](image-7.png)

### Layer 15, Residual Stream, 32x expansion factor (i.e out of 128k features), feature id 6272

All (20/20) words ended with character "ing" - though not indicate any semantics.

![l15r_32x_6272](image-8.png)

### Layer 15, Residual Stream, 32x expansion factor (i.e out of 128k features), feature id 8000

The majority (16/20) about bodily functions.

[beard, elbow, hair\*6, ponytail, hairstyle\*2, haircut\*2, (e)ars, toupee, hairdo ]
![l15r_32x_8000](image-16.png)

### Layer 7, MLP Output, 32x expansion factor (i.e out of 128k features), feature id 8448

All (20/20) about food, though one of them is actually *ham*ster* with *ham* comes with higher activation value.

[muffin, marshmallow\*2, spaghetti, (bur)ritos, taco, (rut)abages, tacos\*2, cheeseburgers, hamster, (w)affle, noodles, bacon, (del)i, pancakes, nut, (dough)nut]

![l7m_32x_8448](image-10.png)



## Lots of questions regarding the SAE + Humor direction

- 





frequent scripts that induce humor
![humor_scripts](image-9.png)
semeval microedit article : https://arxiv.org/pdf/1906.00274    
