# csds440-f24-11

# CSDS440 Written Homework 1
**Instructions:** Each question is worth 10 points unless otherwise stated. Write your answers below the question. Each answer should be formatted so it renders properly on github. **Answers that do not render properly may not be graded.** Please comment the last commit with "FINAL COMMIT" and **enter the final commit ID in canvas by the due date.** 

When working as a group, only one answer to each question is needed unless otherwise specified. Each person in each group must commit and push their own work. **You will not get credit for work committed/pushed by someone else even if done by you.** Commits should be clearly associated with your name or CWRU ID (abc123). Each person is expected to do an approximately equal share of the work, as shown by the git logs. **If we do not see evidence of equal contribution from the logs for someone, their individual grade will be reduced.** 



1. Points are sampled uniformly at random from the interval $(0,1)^2$ so that they lie on the line $x+y=1$. Determine the expected squared distance between any two sampled points. 

**Answer:**


To determine the expected squared distance between two points sampled uniformly at random from the line $x + y = 1$ within the interval $(0,1)^2$, we need to find the dististance function and calculate its expected value:

- **Parametrize the Points on the line by solving for y and substituting y as a function of x**

Points on the line $x + y = 1$ can be parametrized as:

$$
(x, y) = (x, 1 - x) \quad \text{where } x \in [0, 1]
$$

- **consider Two Random Points on the Line and find the distance between the two points**

Let the two points be $(x_1, 1 - x_1)$ and $(x_2, 1 - x_2)$

The squared distance between the two points can be obtaind as follows:

$$
\text{Distance}^2 = (x_2 - x_1)^2 + ((1 - x_2) - (1 - x_1))^2 = 2(x_2 - x_1)^2
$$

- **Calculate the Expected Value of the Squared Distance**

Since $x_1$ and $x_2$ are independent and uniformly distributed over $[0, 1]$, we can calculate the expected value of $(x_2 - x_1)^2$ over the joint distribution of ($x_1,x_2$).

The joint distribution for continous uniform density can be written as:

$$
f_{x_1, x_2}(x_1, x_2) = 
\begin{cases}
1 & \text{if } 0 < x_1 < 1 \text{ and } 0 < x_2 < 1 \\
0 & \text{otherwise}
\end{cases}
$$

The expected squared distance between the two points is given by:

$$
\mathbb{E}[\text{Distance}^2] = 2 \mathbb{E}[(x_2 - x_1)^2]
$$


To find the expectation we have to integrate the product of the distance function $(x_2 - x_1)^2$ and the joint distribution over the domain $(0, 1) \times (0, 1)$:

$$
\mathbb{E}[(x_2 - x_1)^2] = \int_0^1 \int_0^1 (x_2 - x_1)^2 \cdot f_{x_1, x_2}(x_1, x_2) \. dx_1 \. dx_2
$$

Since $f_{x_1, x_2}(x_1, x_2) = 1$ within the interval $(0, 1) \times (0, 1)$, this simplifies to:

$$
\mathbb{E}[(x_2 - x_1)^2] = \int_0^1 \int_0^1 (x_2 - x_1)^2 \cdot 1 \. dx_1 \. dx_2
$$

This further simplifies to:

$$
\mathbb{E}[(x_2 - x_1)^2] = \int_0^1 \int_0^1 (x_2 - x_1)^2 \. dx_1 \. dx_2
$$

The integration limits from $0$ to $1$ indicate that we are considering the region where both $x_1$ and $x_2$ lie within $(0, 1)$, consistent with their joint distribution being $1$ over this region.
Th

** 5. Evaluate the Double Integral **

Simplify the integral:

$$
\mathbb{E}[(x_2 - x_1)^2] = \int_0^1 \int_0^1 (x_2^2 - 2x_1x_2 + x_1^2) \. dx_1 \. dx_2 
$$

Split the integrals:

$$
= \int_0^1 x_2^2 \, dx_2 \int_0^1 \, dx_1 - 2 \int_0^1 x_2 \, dx_2 \int_0^1 x_1 \, dx_1 + \int_0^1 x_1^2 \. dx_1 \int_0^1 \. dx_2 
$$

Evaluate the simpler integrals:

$$
= \left(\int_0^1 x_2^2 \. dx_2\right)(1) - 2 \left(\int_0^1 x_2 \. dx_2\right)\left(\int_0^1 x_1 \. dx_1\right) + \left(\int_0^1 x_1^2 \. dx_1\right)(1)
$$

$$
= \left[\frac{x_2^3}{3}\right]_0^1 - 2 \left[\frac{x_2^2}{2}\right]_0^1 \left[\frac{x_1^2}{2}\right]_0^1 + \left[\frac{x_1^3}{3}\right]_0^1 
$$

$$
= \frac{1}{3} - 2 \left(\frac{1}{2} \cdot \frac{1}{2}\right) + \frac{1}{3} 
$$

$$
= \frac{1}{3} - \frac{1}{2} + \frac{1}{3} = \frac{2}{3} - \frac{1}{2} = \frac{4}{6} - \frac{3}{6} = \frac{1}{6} 
$$

Thus:

$$
\mathbb{E}[(x_2 - x_1)^2] = \frac{1}{6} 
$$

Finally, we get the expected squared distance as:

$$
\mathbb{E}[\text{Distance}^2] = 2 \cdot \frac{1}{6} = \frac{1}{3}
$$

Therefore, we obtain the expected squared distance between any two randomly sampled points from the line $x + y = 1$ is $\boxed{\frac{1}{3}}$.



2. For any two random variables $X$ and $Y$, the conditional expectation of $X$ given $Y=y$ is defined by $E(X|Y=y)=\sum_x p_X(x|Y=y)$ for a fixed $y$. Show that, for any three random variables $A$, $B$ and $C$, $E(A+B|C=c)=E(A|C=c)+E(B|C=c)$.

**Answer:**
To prove that the conditional expectation of the sum of random variables is equal to the sum of their conditional expectations, i.e.,

$$
E(A + B \mid Y = y) = E(A \mid Y = y) + E(B \mid Y = y),
$$

we will use the linearity of expectation and the definition of conditional expectation.

### Proof:

#### We know that conditional expectation of a random variable \(X\) given another random variable \(C\) is defined as:

$$
E(X \mid Y = y) = \sum_{x} x \cdot P(X = x \mid Y = y),
$$

here the summation is over all possible values of \(X\).



 Let's expand the above definition of conditional expectation for the sum \(A + B\):

$$
E(A + B \mid C = c) = \sum_{a, b} (a + b) \cdot P(A = a, B = b \mid C = c).
$$

We can rewrite the RHS  by distributing the terms inside the sum:

$$
E(A + B \mid C = c) = \sum_{a, b} a \cdot P(A = a, B = b \mid C = c) + \sum_{a, b} b \cdot P(A = a, B = b \mid C = c).
$$

Now,  applying the summation seperately:

$$
E(A + B \mid C = c) = \sum_{a} a \left(\sum_b P(A = a, B = b \mid C = c)\right) + \sum_{b} b \left(\sum_a P(A = a, B = b \mid C = c)\right).
$$


We know that 
$$ 
\sum_b P(A = a, B = b \mid C = c)  = P(A = a \mid C = c)
$$ 
$$ \sum_a P(A = a, B = b \mid C = c)  = P(A = b \mid C = c)  $$ 
By marginalizing over the conditional probabilities:
$$
E(A + B \mid C = c) = \sum_{a} a \cdot P(A = a \mid C = c) + \sum_{b} b \cdot P(B = b \mid C = c).
$$

These are the definitions of the conditional expectations:

$$
E(A + B \mid C = c) = E(A \mid C = c) + E(B \mid C = c).
$$

### Conclusion:

$$
\boxed{E(A + B \mid C = c) = E(A \mid C = c) + E(B \mid C = c).}
$$

This proove that for any three random variables $A$, $B$ and $C$, $E(A+B|C=c)=E(A|C=c)+E(B|C=c)$. 



3. Describe two learning tasks that might be suitable for machine learning approaches. For each task, write down the goal, a possible performance measure, what examples you might get and what a suitable hypothesis space might be. Be original---don’t write about tasks discussed in class or described in the texts. Preferably select tasks from your research area (if any). Describe any aspect of the task(s) that may not fit well with the supervised learning setting and feature vector representation we have discussed. 

**Answer:**

**Task 1: HR Bot - Personalized Job Recommendations**

### Goal:
To develop a machine learning model that can provide personalized job recommendations to users based on their preferences, skills, and previous job search history or previous job applications. The system aims to enhance user engagement by tailoring job listings to individual profiles.

### Performance Measure:
A suitable performance measure could be **precision**, which evaluates how relevant the recommended jobs are. This can be determined through user post-interaction feedback or ratings. Another metric is the **click rate (CR)** on recommended jobs and number of job applications.

### Examples:
Input examples could include user profiles with features such as:
- Job search history
- Skills and qualifications
- Location preferences
- Industry preferences

Each example includes a user's employment history and job interactions, serving as labeled data for supervised learning.

### Hypothesis Space:
The **domain** of this space consists of the user profile and job feature data (search history, skills, job metadata). The **range** or output of this space is the set of job recommendations for each user (e.g., a ranked list of job IDs).

The hypothesis space might include:
- **Collaborative filtering models**
- **Content-based filtering**
- **Neural networks** for recommendation tasks

Hybrid models that combine user profile data with job metadata can be designed to enhance the personalization process.

### Challenge to Supervised Learning:
One challenge is the **cold-start problem**, where new users or jobs lack sufficient data for the model to make accurate recommendations. Traditional supervised learning won’t suffice without enough labeled data, so incorporating **unsupervised learning** or **reinforcement learning** might be necessary.

The feature vector representation can be very complex due to the diverse data types of user profiles and job descriptions.

- **User profiles** contain:
  - Structured data: skills, years of experience, location
  - Unstructured data: resumes or free-text inputs provided to the bot about job preferences
  - Behavioral data: job search history, clicks on jobs, applications

- **Job descriptions** contain:
  - Structured data: job title, qualifications, location, salary range
  - Unstructured data: paragraphs describing roles, responsibilities, and company culture

# Task 2: Mobile Face Recognition

## Goal:
To develop a machine learning model that can perform face recognition in real time, accurately, and quickly on mobile devices. This feature is used for authentication to log in to personal devices and verify identity for apps such as banking,finance, personal etc.

## Performance Measure:
Suitable performance measures include:
- **Accuracy**: How often the system correctly recognizes the authorized user.
- **False Positive Rate**: How often the model mistakenly recognizes unauthorized users as authorized.
- **False Negative Rate**: How often the system fails to recognize a legitimate user.
- **Processing Time**: The speed at which the system can recognize the user to ensure a seamless experience.

## Examples:
Input examples could include:
- Images captured of a person’s face using the mobile's camera.
- Data collection during user registration to train the model, including face images with different backgrounds and lighting conditions.
- The model must learn to identify the user while ensuring unauthorized users are not recognized.

## Hypothesis Space:
- The **domain** consists of all possible input values such as multiple poses, expressions, and different lighting conditions (pixel values of face images).
- The **range** is binary (0 or 1): whether the face matches the authorized user or not.

The hypothesis space might include:
- **Convolutional Neural Networks (CNNs)** or other **deep learning models** for feature extraction from face images.
- The domain represents continuous input (face images), and the output is a Boolean function indicating whether the face matches the authorized user or not.

## Challenges to Supervised Learning:
- **Real-time constraints** due to limited processing power on mobile devices.
- **Input variation**, including different lighting conditions, poses, and expressions, which can influence the model’s performance.

 
    



4. Consider a learning problem where the examples are described by $n$ Boolean attributes. Prove that the number of *distinct* decision trees that can be constructed in this setting is $2^{2^n}$. *Distinct* means that each tree must represent a different hypothesis in the space. \[Hint: Show that there is a bijection between the set of all Boolean functions over $n$ Boolean attributes and the set of all distinct trees.\]

**Answer:**
To prove that  the number of *distinct* decision trees that can be constructed in this setting is $2^{2^n}$ we have to show that there is a bijection between the set of decision trees of depth n and the set of all possible Boolean functions. 

We know that:-
- **Boolean Function:** A Boolean function of $n$ variables is a function $f: \{0,1\}^n \to \{0,1\}$.
- **Decision Tree:** A decision tree for $n$ variables is a binary tree where:
  - Each internal node corresponds to a test on one of the variables.
  - Each edge in the tree represnts a possible value of the variable (0 or 1).
  - Each leaf node represnt an output (0 or 1).

### Lets define following Sets

- $\mathcal{T}$ be the set of all decision trees for $n$ variables of depth n.
- $\mathcal{B}$ be the set of all Boolean functions with $n$ variables. The size of $\mathcal{B}$ is $2^{2^n}$ because there are $2^n$ possible inputs, each and every input to either zero(0) or one(1).

### To Prove: $\mathcal{T}$ is in bijection with $\mathcal{B}$.
 we have to show that the function is one to one and onto function which can be called Injectivity , surjectivity respectively.

#### To prove Injectivity [One to one] (Each Decision Tree Represents a Unique Boolean Function)

- **Define a Mapping:** Define a mapping $\phi: \mathcal{T} \to \mathcal{B}$ that assigns each decision tree $t \in \mathcal{T}$ to a Boolean function $f_t \in \mathcal{B}$.
   
- **Evaluating the Function $f_t$ from a tree $t \in \mathcal{T}$ :** For any input combination $(x_1, x_2, \ldots, x_n) \in \{0,1\}^n$:
   - Start at the root of the decision tree $t$. Traverse the tree from the root to the leaf by following edges corresponding to the values of the variables in $(x_1, x_2, \ldots, x_n)$.The output $f_t(x_1, x_2, \ldots, x_n)$ is the value of the leaf node reached.

**So for any input combination $(x_1, x_2, \ldots, x_n)$, a path from the root to the leaf in a decision tree t uniquely identifies a mapping from  $(x_1, x_2, \ldots, x_n)$ to the output $f_t(x_1, x_2, \ldots, x_n)$ of the boolean function**

3. **proof of Uniqueness:** Suppose two different decision trees $t_1$ and $t_2$ represent the same function $f$.
   - Since $t_1 \neq t_2$, there must be at least one input $(x_1, x_2, \ldots, x_n)$ where the traversal of $t_1$ and $t_2$ leads to different outputs, which contradicts the assumption that $t_1$ and $t_2$ represent the same function f as the function woulld have diffrent output that perticular combination $(x_1, x_2, \ldots, x_n)$ of input.
   - Therefore, $\phi$ is one to one or Injectivity.

#### Step 2: Surjectivity [Onto] (Every Boolean Function Can Be Represented by a Decision Tree)

1. **Construct a Decision Tree for Any Boolean Function $f \in \mathcal{B}$:**
   - Start with the root node. For each input variable $x_i$, create a branching node. Recursively create branches for each value (0 and 1) of $x_i$.
   - Continue this until all variables are used (depth of the tree is $n$). Assign the corresponding output $f(x_1, x_2, \ldots, x_n)$ at each leaf node based on the values of $(x_1, x_2, \ldots, x_n)$.

2. **Existence:** By the construction, every Boolean function $f \in \mathcal{B}$ has a corresponding decision tree $T \in \mathcal{D}$ that outputs $f$ for all inputs. Hence, $\phi$ is surjective.


### Conclusion
<img src="image-1.png" alt="Description" style="width: 50%; height: auto;">

We have shown mathematically that:

1. Each decision tree corresponds to a unique Boolean function (injectivity).
2. Every Boolean function can be represented by a decision tree (surjectivity).

Therefore, the set of decision trees $\mathcal{T}$ is in bijection with the set of Boolean functions $\mathcal{B}$.

### Finding Cardinality of set $\mathcal{B}$:-

- The set of all Boolean functions $\mathcal{B}$, consists of all possible mappings from the set of all combinations of input  $(x_1, x_2, \ldots, x_n)$ to a binary output (0 or 1).
- For $n$ Boolean attributes, there are $2^n$ possible input combinations, as each attribute can be either 0 or 1 and 2 possible output combination.

- Each of these $2^n$ input can map to either 0 or 1, which gives us 2 choices for each combination. Therefore, the total number of  mappings from the set of input combinations to the binary output is $2^{2^n}$. 
- Therefore, |$\mathcal{B}$| = $2^{2^n}$.

Finding Cardinality of set $\mathcal{T}$:-

#### Since there exist a Bijection mapping $\phi: \mathcal{T} \to \mathcal{B}$. The cardinality of both sets are same ie  $|\mathcal{T}| =  |\mathcal{B}|$=  $2^{2^n}$ 

**Therefore, this proves that distinct decision trees that can be constructed in a learning problem with $n$ Boolean attributes is $2^{2^n}$.**



5.	(i) Give an example of a nontrivial (nonconstant) Boolean function over $3$ Boolean attributes where $IG(X)$ would return zero for *all* attributes at the root. (ii) Explain the significance of this observation.

**Answer:**

#### (i) Nontrivial example of a Boolean Function Where $( IG(X) )$ is Zero for All Attributes

Consider the  XOR function of 3-input defined as:

$ f(A, B, C) = A \oplus B \oplus C $

This function outputs 1 if and only if odd number of inputs are 1.

**Truth Table:**

| A | B | C | $f(A, B, C)$ |
|---|---|---|------------------|
| 0 | 0 | 0 |        0         |
| 0 | 0 | 1 |        1         |
| 0 | 1 | 0 |        1         |
| 0 | 1 | 1 |        0         |
| 1 | 0 | 0 |        1         |
| 1 | 0 | 1 |        0         |
| 1 | 1 | 0 |        0         |
| 1 | 1 | 1 |        1         |



## Entropy Calculations

- ###  Calculate the Entropy of the Whole Dataset

For a binary output, the entropy is given by:

$$ H(Y) = - p_0 \log_2 p_0 - p_1 \log_2 p_1 $$

where $p_0$ and $p_1$ are the probabilities of the outcomes 0 and 1.

From the truth table, we can find the distribution of $f(A, B, C)$ is:

- $p_0$ (Probability of  0) = $\frac{4}{8} = 0.5$
- $p_1$ (Probability of  1) = $\frac{4}{8} = 0.5$

Thus:

$$ H(Y) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) $$

$$ H(Y) = - (0.5 \times (-1) + 0.5 \times (-1)) $$

$$ H(Y) = - (-1) $$

$$ H(Y) = 1 $$



- ### Information Gain on splitting on attribute $A$

**Entropy Calculation After Splitting on $A$:**

- **$A = 0$:**

| A | B | C | $f(A, B, C)$ |
|---|---|---|--------------|
| 0 | 0 | 0 |      0       |
| 0 | 0 | 1 |      1       |
| 0 | 1 | 0 |      1       |
| 0 | 1 | 1 |      0       |

For $A = 0$:
- Total number = 4
- Number of 0s = 2
- Number of 1s = 2
- $p_0 = 2/4 = 0.5$
- $p_1 = 2/4 = 0.5$

$$ H(Y | A = 0) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) $$
$$ H(Y | A = 0) = 1 $$

- **$A = 1$:**

| A | B | C | $f(A, B, C)$ |
|---|---|---|--------------|
| 1 | 0 | 0 |      1       |
| 1 | 0 | 1 |      0       |
| 1 | 1 | 0 |      0       |
| 1 | 1 | 1 |      1       |

For $A = 1$:
- Total number = 4
- Number of 0s = 2
- Number of 1s = 2
- $p_0 = 2/4 = 0.5$
- $p_1 = 2/4 = 0.5$

$$ H(Y | A = 1) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) $$
$$ H(Y | A = 1) = 1 $$

**Weighted Average Entropy After Splitting on $A$:**

Since the dataset is equally split between the outputs $A = 0$ and $A = 1$:

$$ H(Y | A) = 0.5 \times H(Y | A = 0) + 0.5 \times H(Y | A = 1) $$
$$ H(Y | A) = 0.5 \times 1 + 0.5 \times 1 $$
$$ H(Y | A) = 1 $$

**Information Gain for $A$:**

$$ IG(A) = H(Y) - H(Y | A) $$
$$ IG(A) = 1 - 1 $$
$$ IG(A) = 0 $$

- ###  Information Gain for Attribute $B$

**Entropy Calculation After Splitting on $B$:**

- **$B = 0$:**

| A | B | C | $f(A, B, C)$ |
|---|---|---|--------------|
| 0 | 0 | 0 |      0       |
| 0 | 0 | 1 |      1       |
| 1 | 0 | 0 |      1       |
| 1 | 0 | 1 |      0       |

For $B = 0$:
- Total number = 4
- Number of 0s = 2
- Number of 1s = 2
- $p_0 = 2/4 = 0.5$
- $p_1 = 2/4 = 0.5$

$$ H(Y | B = 0) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) $$
$$ H(Y | B = 0) = 1 $$

- **$B = 1$:**

| A | B | C | $f(A, B, C)$ |
|---|---|---|--------------|
| 0 | 1 | 0 |      1       |
| 0 | 1 | 1 |      0       |
| 1 | 1 | 0 |      0       |
| 1 | 1 | 1 |      1       |

For $B = 1$:
- Total number = 4
- Number of 0s = 2
- Number of 1s = 2
- $p_0 = 2/4 = 0.5$
- $p_1 = 2/4 = 0.5$

$$ H(Y | B = 1) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) $$
$$ H(Y | B = 1) = 1 $$

**Weighted Average Entropy After Splitting on $B$:**

Since the dataset is equally split between $B = 0$ and $B = 1$:

$$ H(Y | B) = 0.5 \times H(Y | B = 0) + 0.5 \times H(Y | B = 1) $$
$$ H(Y | B) = 0.5 \times 1 + 0.5 \times 1 $$
$$ H(Y | B) = 1 $$

**Information Gain for $B$:**

$$ IG(B) = H(Y) - H(Y | B) $$
$$ IG(B) = 1 - 1 $$
$$ IG(B) = 0 $$

- ### 3. Information Gain for Attribute $C$

**Entropy Calculation After Splitting on $C$:**

- **$C = 0$:**

| A | B | C | $f(A, B, C)$ |
|---|---|---|--------------|
| 0 | 0 | 0 |      0       |
| 1 | 0 | 0 |      1       |
| 0 | 1 | 0 |      1       |
| 1 | 1 | 0 |      0       |

For $C = 0$:
- Total number = 4
- Number of 0s = 2
- Number of 1s = 2
- $p_0 = 2/4 = 0.5$
- $p_1 = 2/4 = 0.5$

$$ H(Y | C = 0) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) $$
$$ H(Y | C = 0) = 1 $$

- **$C = 1$:**

| A | B | C | $f(A, B, C)$ |
|---|---|---|--------------|
| 0 | 0 | 1 |      1       |
| 1 | 0 | 1 |      0       |
| 0 | 1 | 1 |      0       |
| 1 | 1 | 1 |      1       |

For $C = 1$:
- Total number = 4
- Number of 0s = 2
- Number of 1s = 2
- $p_0 = 2/4 = 0.5$
- $p_1 = 2/4 = 0.5$

$$ H(Y | C = 1) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) $$
$$ H(Y | C = 1) = 1 $$

**Weighted Average Entropy After Splitting on $C$:**

Since the dataset is equally split between $C = 0$ and $C = 1$:

$$ H(Y | C) = 0.5 \times H(Y | C = 0) + 0.5 \times H(Y | C = 1) $$
$$ H(Y | C) = 0.5 \times 1 + 0.5 \times 1 $$
$$ H(Y | C) = 1 $$

**Information Gain for $C$:**

$$ IG(C) = H(Y) - H(Y | C) $$
$$ IG(C) = 1 - 1 $$
$$ IG(C) = 0 $$

### Conclusion

The information gain obtained after splitting on any attribute  in the XOR function is zero. 

### (ii) Significance of This Observation

- Balanced function: For balanced, symmetric functions like the XOR, splitting on any individual attribute does not reduce the entropy of the dataset. This occurs because the splits on any attribute produce subsets with equal numbers of 0s and 1s. 


- Independence of output from input Attributes: If the Information Gain for evry attributes at the root is zero, it means that no single attribute provides any information for predicting the output value. 

- Decision Tree Construction: In decision tree, all the attributes with zero Information Gain are not useful for splitting at that point in the tree. This means the decision tree algorithm giving O IG at the root node for all the attribute would not be a useful algorithm.


6. Estimate how many functions satisfying Q5 (i) could exist over $n$ attributes, as a function of $n$. 

**Answer:**
 
   - For a function $n$ Boolean attributes, the truth table has $2^n$ entries.
   - we select an attribute to split the table , lets assume that we choose $X_1$ to split on.
   - We split the truth table into two equal halves based on $X_1$:
     - Define split $S_0$: Inputs for which $X_1 = 0$, with $|S_0| = 2^{n}/2$ = $2^{n-1}$.
     - Define split $S_1$: Inputs for which $X_1 = 1$, with $|S_1| = 2^{n}/2$ = $2^{n-1}$.
 - Total number of boolean functions possible after the first split is all function from domain $2^{n-1}$ to co domain {0,1} whic is equal to $2^{2^{n-1}}$


**Now we have to assign Equal Output Distributions of 1 and 0 in both halves in $S_0$ and $S_1$:**

   - The distributions of 1 and 0 in $f$ has to be same for $S_0$ and $S_1$ must be identical so that that output is ballanced in each split.
   - This requires that the number of ones and zeros in $S_0$ and $S_1$ are the same. Ie half of  $S_0$ should have 1s and rest must be 0.
   - Flip the sign of output in the other half $S_1$
  - Total number of boolean functions possible with this split

**Assigning Outputs in $S_0$:**

   - We need to assign $2^{n-1}/2= 2^{n-2}$ ones and $2^{n-1}/2= 2^{n-2}$ zeros to the inputs in $S_0$ to make the 1st half **ballanced**.
   - The number of ways to select the inputs in $S_0$ map to 1 is:
     $$
     N_{S_0} = \binom{2^{n-1}}{2^{n-2}}
     $$
     This is because we are selecting $2^{n-2}$ positions out of total out of $2^{n-1}$ in S to assign a value of 1.

**Determining Outputs in $S_1$:**

   - To ensure identical distributions is maintained in the outputs of $S_1$, we can define the outputs in $S_1$ as the **complements** of the outputs in $S_0$.
  
     For the corresponding inputs of  $\mathbf{y} \in S_1$ where $\mathbf{y}$ differs from $\mathbf{x}$ only in the attribute  $X_1$ and rest of the attribute remain same:
     $$
     f(\mathbf{y}) = 1 - f(\mathbf{x})
     $$
     This ensures that the counts of ones and zeros in $S_1$ match those in $S_0$.

 **Total Number of Functions:**
   - There are also two constant functions where outut y = 0 or y =1 for all input.
   - Since the outputs of $S_0$  completely determines the  outputs in $S_1$, the total number of such functions is the number of ways we can assign outputs to $S_0$.
   - Therefore:
     $$
     N_{\text{functions}} = \binom{2^{n-1}}{2^{n-2}} +2
     $$
     This is the number of ways to select $2^{n-2}$ ones out of $2^{n-1}$ positions in $S_0$.

 **Approximation for Large $n$:**

   - The binomial coefficient $\binom{N}{N/2}$ for large $N$ can be approximated using Stirling's approximation:
     $$
     \binom{N}{N/2} \approx \frac{2^N}{\sqrt{\pi N}}
     $$
   - Applying this to $N = 2^{n-1}$:
     $$
     N_{\text{functions}} \approx \frac{2^{2^{n-1}}}{\sqrt{\pi \cdot 2^{n-1}}} +2
     $$
     $$
     N_{\text{functions}} \approx \frac{2^{2^{n-1}}}{2^{(n-1)/2} \sqrt{\pi}} +2
     $$ 
   - For large $n$ as $n$ tend to infinte , the numerator grows much more faster than the  denominator , so:
     $$
     N_{\text{functions}} \approx 2^{2^{n-1}} +2  \approx 2^{2^{n-1}}
     $$

7.	Show that for a continuous attribute $X$, the only split values we need to check to determine a split with max $IG(X)$ lie between points with different labels. (Hint: consider the following setting for $X$: there is a candidate split point $S$ in the middle of $N$ examples with the same label. To the left of $S$ are $n$ such examples. To the left of $N$, there are $L_0$ examples with label negative and the $L_1$ positive, and likewise $(M_0, M_1)$ to the right. Express the information gain of $S$ as a function of $n$. Then show that this function is maximized either when $n=0$ or $n=N$ with all else constant.) (30 points)

**Answer:**
![DocScanner 12 Sept 2024 1-54 pm_1](https://github.com/user-attachments/assets/49ab91a3-712f-4327-9d67-00b03f4d7a27)
![DocScanner 12 Sept 2024 1-54 pm_2](https://github.com/user-attachments/assets/2c4934f4-2334-42b3-8f2f-6d35eaef376f)
![DocScanner 12 Sept 2024 1-54 pm_3](https://github.com/user-attachments/assets/e31f4bd3-d699-4905-b916-d635ddaf1c6c)
![DocScanner 12 Sept 2024 1-54 pm_4](https://github.com/user-attachments/assets/c5c90993-0916-4ff1-b9d5-072d7d8a65b2)
![DocScanner 12 Sept 2024 1-54 pm_5](https://github.com/user-attachments/assets/ecacadfb-a655-4c7b-a3c0-642449a4bc13)

