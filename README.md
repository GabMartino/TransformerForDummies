
# TransformerForDummies

I found that some important details of the Transformer implementation were are not totally clear 
and I needed to search for other implementation or to explanation of these details. For this reason, 
I decided to report here the most important doubts that I had hoping that could some new people entering in this field!

The explainations assume a basic knowledge of the transformer models (e.g. Encoder-Decoder architecture, Multi-Head Attention Mechanism, ...),
avoiding to create a redundant repository over millions already present on the web, and focusing mainly on the ambiguities.

## The Architecture: Questions
The very well known image that depict the transformer architecture hides a lot of important information that are useful for the correct implementation.
<p align="center">
<img src="./assets/Transformer_architecture.png" alt="Transformer" width="50%"/>
</p>

Some of the first questions that came up in my mind when I had a look to this picture were:
1) **How the Encoder and Decoder are connected??**

The encoder and the decoder can have multiple layers (N as reported). The encoder and the decoder are connected. The output of the encoder seems to be connected to the decoder. 
But! Which layer?? The last one, the first one?? All of them??

2) **How the Encoder output is connected to the 'Multi-Head Attention of the Decoder'?**

Every attention block has three inputs that should be the Query, Key and Value. Which one is what??

3) **Why only the first attention block of the decoder is depicted as 'Masked' and not the Self-Attention of the encoder and the 'Cross-Attention block'?**

Later in this markdown more on masks.

## The architecture: Answers

**The answer to these questions resides in a couple of sentence in the paper**:

<p align="center">
<img src="./assets/paragraph_1.jpg" alt="Paragraph" width="70%"/>
</p>

2) **The Keys and the Values come from the Encoder, the Queries come from the last sublayer of the decoder.**

<p align="center">
<img src="./assets/answer_2.jpg" alt="Paragraph" width="50%"/>
</p>

1) **The Encoder Output is reported to all the Decoder Layers**

This could be extracted from the phrase: **_This allows every position in the decoder to attend over all the positions in the input sequence_**, as also reported in the image:



<p align="center">
<img src="./assets/transformer_explained.png" alt="Transformer Explained" width="50%"/>
</p>
[https://www.truefoundry.com/blog/transformer-architecture]

3) **Only the first attention block of the decoder has a mask**

This is only partially true, because here we are talking about only the Look-Ahead Mask.

## The Masks: Questions

I admit that I struggled a bit to understand well how the masking is used in this model, mainly because a looot of things are given for granted,
and appear clear and obvious only when you start to implement things and problems come up.

1) **How the mask is included in the attention computation?**

3) **The Look-Ahead Mask is the only mask used?**

2) **How the padding mask is used??**

## The Masks: Answers


1) **The employment of the Look-Ahead Mask hides a couple of interesting issues**


### The Look-Ahead/Causal Mask

First of all, I would have named the "Look Ahead Mask" as "DON'T Look Ahead Mask".
This mask is used for the decoder to allow the computation of the attention only backward in the sentence. 

Yes, it has sense, but why?? Well, because at the inference time, the decoder will act in auto-regressive manner, 
that means that it only has the encoder input as complete sentence, and the decoder should generate word by word during translation, 
hence only using the already generated words. For this reason, we need to force at the training time to learn to predict the ground-truth output sentence without looking at the next words, otherwise that's cheating!

Here we report the shape of the "Don't look ahead mask" also called "Causal Mask":
$M \in \mathbb{R}^{L x L}$

$$M = \begin{bmatrix} 
0 & -inf & -inf &  -inf & -inf &  -inf  \\\
0 & 0 & -inf & -inf & -inf & -inf \\\
0 & 0 & 0 & -inf & -inf & -inf \\\
0 & 0 & 0 & 0 & -inf & -inf \\\
0 & 0 & 0 & 0 & 0 & -inf \\\
0 & 0 & 0 & 0 & 0 & 0 
\end{bmatrix}
$$

Notice that size of the mask is $L x L$ that is the lenght of the sentence. 

The matrix is composed by zeros and -inf, we'll see in a moment why:

**The computation of the masked attention is then**:


$$
    Attention(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt{d_k}} + M)V
$$

Notice the mask inside the softmax function.

This is done because if we consider $Q \in \mathbb{R}^{Lx1}, K \in \mathbb{R}^{Lx1}, V \in \mathbb{R}^{Lx1}$,
We would have $QK^{T} \in \mathbb{R}^{LxL}$

Now, **the softmax function is applied column wise**, this is just because the later multiplication with $V$ is on the right-hand side.

Remind that:
$$Softmax(x_i) = \frac{e^{x_i}}{\sum_i e^{x_i}}$$
Where the $x_i$ is in a set $X = \{x_1, x_2, ..., x_n\}$, this function just reweight the value to be summed to 1.

Hence, when the value is $-inf$ the softmax gives a weight of $0$ that means "don't consider this value".

As an example:

$$
    \frac{QK^{T}}{\sqrt{d_k}} = \begin{bmatrix} 
1 & 2 & 3 & 4 & 5 &  6  \\\
7 & 8 & 9 & 10 & 11 & 12 \\\
13 & 14 & 15 & 16 & 17 & 18 \\\
19 & 20 & 21 & 22 & 23 &  24 \\\
25 & 26 & 27 & 28 & 29 & 30  \\\
31 & 32 & 33 & 34 & 35 & 36 
\end{bmatrix}
$$

$$
    \frac{QK^{T}}{\sqrt{d_k}} + M = \begin{bmatrix} 
1 & -inf & -inf & -inf & -inf &  -inf  \\\
7 & 8 & -inf & -inf & -inf & -inf \\\
13 & 14 & 15 & -inf & -inf & -inf \\\
19 & 20 & 21 & 22 & -inf &  -inf\\\
25 & 26 & 27 & 28 & 29 & -inf  \\\
31 & 32 & 33 & 34 & 35 & 36 
\end{bmatrix}
$$

$$
    Softmax(\frac{QK^{T}}{\sqrt{d_k}} + M) = \begin{bmatrix} 
9.3344e-14 & 0 & 0 & 0 & 0 &  0  \\\
3.7658e-11 & 3.7658e-11 & 0 & 0 & 0 & 0\\\
1.5192e-08 & 1.5192e-08 & 1.5192e-08 & 0 & 0 & 0\\\
6.1290e-06 & 6.1290e-06 & 6.1290e-06 & 6.1290e-06 & 0 &  0 \\\
2.4726e-03 & 2.4726e-03 & 2.4726e-03 & 2.4726e-03 & 0.0025 & 0  \\\
9.9752e-01 & 9.9752e-01 & 9.9752e-01 & 9.9752e-01 &  0.9975 & 1.0
\end{bmatrix}
$$

The sum column-wise is always 1.0, try to believe!

Finally, we can compute the output values of the attention mechanism:

$$
    Softmax(\frac{QK^{T}}{\sqrt{d_k}} + M)V = \begin{bmatrix} 
9.3344e-14 & 0 & 0 & 0 & 0 &  0  \\\
3.7658e-11 & 3.7658e-11 & 0 & 0 & 0 & 0\\\
1.5192e-08 & 1.5192e-08 & 1.5192e-08 & 0 & 0 & 0\\\
6.1290e-06 & 6.1290e-06 & 6.1290e-06 & 6.1290e-06 & 0 &  0 \\\
2.4726e-03 & 2.4726e-03 & 2.4726e-03 & 2.4726e-03 & 0.0025 & 0  \\\
9.9752e-01 & 9.9752e-01 & 9.9752e-01 & 9.9752e-01 &  0.9975 & 1.0
\end{bmatrix} * \begin{bmatrix} 1 \\\ 2 \\\ 3 \\\ 4 \\\ 5 \\\ 6\end{bmatrix}
$$
The results should be:
$$
    \begin{bmatrix}
    9.3344e-14 \\\
    1.1297e-10 \\\
    9.1152e-08 \\\
    6.1290e-05 \\\
    0.0372 \\\
    20.9627
    \end{bmatrix}
$$
