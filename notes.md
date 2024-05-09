## Recommendation System Notes
___
## Lecture 1: Intro
### Types of Recommendation Systems
- **Collaborative**: people who agreed in the past are likely to agree in the future
    - drawback: scalability, when pairwise metrics are infeasible to compute
    - drawback: bootstrapping collaborative filtering system even if little information provided
- **Content Based**: recommend items similar to what the user liked in the past
    - drawback: sparse ratings
- **Knowledge Based**: use domain knowledge to retrieve items that match user needs
## Lecture 2: ML and Linear Algebra Basics
### Types of Machine Learning
- **Supervised Learning**: patterns in labeled data (task driven, e.g. classification, regression)
    - $D = \{(x_i,y_i) \}_{i=1}^N$ where $x_i \in \mathbb{R}^D, y_i \in \mathbb{R}$ (regression) or $ \{-1,1\}$ (classification)
- **Unsupervised Learning**: patterns in unlabeled data (data driven, e.g. clustering)
    - $D = \{x_i\}_{i=1}^N$ where $x_i \in \mathbb{R}^D$
- **Reinforcement Learning**: interact with environment to maximize performance (learning from mistakes, e.g. playing games)
    - agent interacts with environment through actions to change states
### Classification
- #### Discriminative
    - predict from conditional probability
        - calculate $P(Y \mid X)$ from training data
    - mainly in *supervised*
    - creates boundaries
    - insensitive to outliers
    - can't generate new data
- ### Generative
    - learn how the data is distributed
    - $P(Y \mid X) = \frac{P(Y)\cdot P(X\mid Y)}{P(X)}$ (Bayes)
        - $posterior = \frac{prior \times likelihood}{evidence}$ 
    - mainly in *unsupervised*
    - sensitive to outliers
    - can generate new data
### Bias vs. Variance
- $Bias(\hat{y}) = E[\hat{y}] - y$, distance away from average prediction
- $Variance(\hat{y}) = E[\hat{y}^2] - E[\hat{y}]^2$, data spread, expectation of squared biases
- $Error = Bias^2 + Variance + IrreducibleError$
### Loss Function
- **Goal**: find the right number of parameters and values (weights) to make the model predict accurately
- **Gradient Descent**:
    - loss $l(\omega)$
    - compute $\nabla_\omega$
    - find lowest error position
    - $\omega' = \omega - \eta \nabla_\omega l$
- $L_2$ error may penalize outliers too heavily, use $L_1$ otherwise
### Optimization
- aim to find $\arg \min\limits_w L(w)$
- Closed Form: linearize the problem, then solution is $\omega = (A^TA)^{-1} A^T y$
- Iterative Optimization: iterative until minimal loss, gradient descent
### Linear Algebra Review
- vector addition, matrix multiplication, matrix transpose, linear transformation, vector norm, projection, inverses, inner product
- column space: span of $A$'s column vectors
- null space: all $x$ such that $Ax = 0$
- linearly independent: $\alpha x_1 + \beta x_2 + \gamma x_3 = 0$ only if $\alpha = \beta = \gamma = 0$
- basis: for a vector space $V$, a linearly independent subset that spans $V$
- $\nabla f = \langle f_x, f_y, f_z \rangle$
- $\nabla_x(a^Tx) = a$
- $\nabla_x(x^T A x) = (A + A^T)x$
## Lecture 3: Content-Based Rec Sys and Word Embeddings
### Binary Representation Example
- represent items with a binary vector with each entry representing one feature
- recommend app with highest dot product (use dot product as similarity metric)
    - problem: an item with all the features is always the best match
- cosine similarity improves upon dot product, generalizes to higher dimensions
### Retreiving Information From Text
- length of vector = size of vocabulary
- simple approach: bag of words, each entry for word $i$ counts how many times word $i$ appears
    - problem: ignores word order, each word has equal importance
#### TF-IDF (Term Frequency - Inverse Document Frequency)
- diminishes weight of terms that occur in almost all documents
- increase weight of terms that occur rarely
- $tfidf(t,d,D) = tf(t,d) \cdot idf(t,D)*
##### Term Frequency
- for term $t$ in document $d$, $tf(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$
    - $f_{t,d}$ is the raw count of term $t$ in document $d$
    - denominator is length of document
        - normalize to represent the relative importance of a term in the document
- **Variations**
    - normalize by maximum count instead of total count
    - constant term for smoothing
        - small change in count should not result in large change in $tf$
        - double frequency does not mean double importance
    - $tf(t,d) = 0.5 + 0.5 \cdot \frac{f_{t,d}}{\max\{f_{t',d} : t' \in d\}}$
##### Inverse Document Frequency
- for term $t$ in document $d$:
- $idf(t,D) = \log \frac{N}{\mid \{d \in D : t \in d\} \mid}$
    - where $N$ is the total number of docs
    - denom. is the total number of documents that contain $t$
    - log for information theoretic reasons, specificity of a word is sublinear wrt. inverse occurence count
##### TF-IDF with Cosine Similarity
- generate a vector for each document
    - entry $t$ of document $d$: $tfidf(t,d,D)$
- apply cos. sim. to retrieve top matches
- assumption: distribution of terms represents distribution of topics
   - IDF ensures that stop words like "the", "is", etc. do not dominate
#### Embedding Matrix
- goal: build a nn. where word embeddings are learned in the form of weights
- embedding matrix $E: n \times d$ where $N$ is vocab size, $d$ the representation dimension
#### Word2Vec: Skip-Gram
- task: given center word, predict distribution of context words
- take dot product between center word and every possible context word as the predicted distribution (pre-softmax)
- optimize to model true distribution in corpus
- ex: word vector for ants $\times$ output weights for car $\rightarrow \frac{e^x}{\sum e^x} = $ probability that a random word near "ants" is "car"
- implementation: mutliplying center word embedding by $E^T$ gives desired dot products
- as a neural network: hidden layer is linear neurons (embedding dim.), output layer is softmax classifier
- want the *sum* of context word embeddings to be close to the center word embedding
##### Two Approaches
- CBOW (continuous bag of words)
    - $P(center word = w \mid context words)$
    - input weights are summed from surrounding words to be the weight for the center word
- Skip-Gram
    - $P(w $ in context words $\mid center word)$
    - input $w(t)$ is projected, output is surrounding words' weights
#### GloVe: Global Vectors for Word Representations
- goal: learn word vectors such that their dot product equals the logarithm of the words' probability of co-occurence
- can compute and $N \times N$ matrix of co-occurences from the corpus, where entry $i,j$ is the prob. of co-occurence of words $i$ and $j$
    - factoring this matrix into $EE^T$ gets us the word embeddings
        - gradient descent methods work
    - multiply words/features matrix by features/context matrix to obtain words/context matrix
#### Linear Substructures
- vector differences capture the relative meaning between words
    - e.g. queen is to woman what king is to man
    - ratio of co-occurence probabilities captures the difference in meaning
- ex: expect $P(k\mid \text{dry ice})/P(k\mid \text{carbon dioxide})$
    - in log space, $\log P(k\mid \text{dry ice}) - \log P(k\mid \text{carbon dioxide}) = \log P(k\mid \text{ice}) - \log P(k\mid \text{steam})$
    - satisfied by embeddings with $v_{\text{dry ice}} - v_{\text{CO2}} = v_{\text{ice}} - v_{\text{steam}}$
#### Other Similarity Metrics
- Euclidean, Cosine, Hamming, Manhattan, Minkowski, Chebyshev, Jaccard, Haversine, Sorensen-Dice
- Euclidean: $d(x,y) = \sqrt{\sum\limits_{i=1}^n(x_i - y_i)^2}$
- Ex: similarity in GloVe
    - an infrequent word might be more similar to another infrequent word, but the dot product never retreives this
        - solution: Euclidean distance/cosine similarity
    - cosine similarity ignores magnitude (words that are similar in meaning), Euclidean distance looks for similar magnitude (words that are similar in meaning and frequency)
- Manhattan Distance: $d(x,y) = \sum\limits_{i=1}^n |x_i - y_i|$
- Dice Coefficient (similarity between sets) $\frac{2|X\cap Y|}{|X| + |Y|}$
    - or, if represented with binary vectors, $\frac{2|a\cdot b|}{|a|^2 + |b|^2}$
- Jaccard Similarity
    - $J(A,B) = \frac{|A\cap B|}{|A\cup B|}$
    - $\frac{A\cdot B}{||A||^2 + ||B||^2 - A\cdot B}$