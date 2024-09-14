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


We know that $$ \sum_b P(A = a, B = b \mid C = c)  = P(A = a \mid C = c)  $$ 
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

**Ans.4.**\
**Step 1: Boolean attributes and possible inputs**\
\
Let X={0,1}n denote the set of all possible inputs consisting of n Boolean attributes. Each element of X is an n-tuple of the form (x1,x2,…,xn), where xi∈{0,1} for i=1,2,…,n.\
\
The size of X, i.e., the total number of distinct inputs, is ∣X∣=2^n because each of the n attributes can take one of two values (either 0 or 1).\
\
**Step 2: Boolean functions**\
\
A Boolean function f:X→{0,1} maps each input x∈X to either 0 or 1. This means that for each of the 2^n possible inputs in X, the function f assigns one of two possible output values.\
\
The number of distinct Boolean functions is determined by the number of different ways to assign outputs to each of the 2^n inputs. Since there are two possible output values (either 0 or 1) for each input, the total number of distinct Boolean functions over n attributes is ∣F∣=2^2^n.\
\
This is because each of the 2^n input combinations can be mapped independently to 0 or 1, so we have 2 choices for each of the 2^n inputs, giving us 2^2^n possible Boolean functions.\
\
**Step 3: Decision trees and Boolean functions**\
\
A decision tree is a representation of a Boolean function. For each possible input x∈X, the decision tree traverses through internal nodes (each labeled by a Boolean attribute) to reach a leaf node that outputs either 0 or 1. The process of traversing the decision tree corresponds to evaluating the Boolean function.\
\
Thus, each decision tree defines a Boolean function, as it uniquely determines an output for each input x∈X. Since no two distinct decision trees can produce the same output for all inputs, each distinct decision tree corresponds to a unique Boolean function.\
\
**Step 4: Bijection between Boolean functions and decision trees**\
\
We now show that there is a bijection between the set of Boolean functions and the set of distinct decision trees.\
\
**Injectivity:** Suppose there are two distinct decision trees, T1 and T2. Since the structure of a decision tree completely determines the output for every possible input, if T1!=T2, there must be at least one input x∈X for which the outputs of T1 and T2 differ. Therefore, the Boolean functions represented by T1 and T2 are also distinct, implying that different trees correspond to different functions. Thus, the mapping from decision trees to Boolean functions is injective.\
**Surjectivity:** For every Boolean function f:X→{0,1}, there exists a decision tree that computes f. We can construct such a decision tree by recursively splitting on the attributes and assigning outputs at the leaves that match the function f's values for each possible input. Therefore, every Boolean function can be represented by a decision tree, proving that the mapping is surjective.\
\
Since the mapping is both injective and surjective, it is a bijection.\
\
**Step 5: Conclusion**
\
Since there is a bijection between the set of Boolean functions and the set of distinct decision trees, and since the number of Boolean functions is 2^2^n, the number of distinct decision trees is also 2^2^n.\
\
Thus, the number of distinct decision trees that can be constructed for n Boolean attributes is 2^2^n.\

5.	(i) Give an example of a nontrivial (nonconstant) Boolean function over $3$ Boolean attributes where $IG(X)$ would return zero for *all* attributes at the root. (ii) Explain the significance of this observation.
   

**Ans.5.**\
I take an example of the **parity function** on 3 Boolean attributes, which outputs 1 if the number of 1s in the input is **even**, and 0 if the number of 1s is **odd**. This function can be written as:
f(X1,X2,X3)=¬(X1⊕X2⊕X3).\
\
The truth table for this function is:

| X1 | X2 | X3 | f(X1,X2,X3) |
|----|----|----|-------------|
|  0 |  0 |  0 |           1 |
|  0 |  0 |  1 |           0 |
|  0 |  1 |  0 |           0 |
|  0 |  1 |  1 |           1 |
|  1 |  0 |  0 |           0 |
|  1 |  0 |  1 |           1 |
|  1 |  1 |  0 |           1 |
|  1 |  1 |  1 |           0 |


Truth Table shows,the output is equally likely to be 0 or 1 for any value of any attribute.\
For example, consider splitting on X1:\
\
• When X1=0, the subtable for (X2,X3) has 2 outcomes of 0 and 2 outcomes of 1, so the conditional entropy H(Y∣X1=0)=1.\
• When X1=1, the subtable for (X2,X3) has 2 outcomes of 0 and 2 outcomes of 1, so the conditional entropy H(Y∣X1=1)=1.\
\
\
Therefore, the conditional entropy is:\
\
H(Y∣X1)=1/2⋅1+1/2⋅1=1\
\
This holds for any attribute X1,X2,X3,because the function is symmetric in all attributes so the Information Gain for any attribute is:\
\
IG(Xi)=H(Y)−H(Y∣Xi)=1−1=0\
\
In this case, again, splitting based on any single attribute X1, X2, or X3 will not reduce the uncertainty of the output (entropy). \
\
Also, For each attribute, the output of the function is equally likely to be 0 or 1 regardless of the value of that attribute, so the Information Gain for any attribute will be zero.\
\
**(ii) Significance of this observation:**\
\
The fact that Information Gain is zero for all attributes at the root highlights that the **parity function is not linearly separable** and cannot be captured by a simple decision tree based on single-attribute splits. In other words, no single attribute provides any meaningful information about the output on its own because the function depends on a combination of attributes rather than on individual ones.\
\
This is significant because it shows the limitations of decision trees in handling functions like parity function. In order to capture the behavior of such a function, more complex models (e.g., those that can handle interactions between attributes) or deeper trees that consider combinations of attributes are necessary. It illustrates that decision trees can struggle with functions where the output is based on a non-obvious, non-linear combination of inputs.


6. Estimate how many functions satisfying Q5 (i) could exist over $n$ attributes, as a function of $n$. 

**Answer:**
 
7.	Show that for a continuous attribute $X$, the only split values we need to check to determine a split with max $IG(X)$ lie between points with different labels. (Hint: consider the following setting for $X$: there is a candidate split point $S$ in the middle of $N$ examples with the same label. To the left of $S$ are $n$ such examples. To the left of $N$, there are $L_0$ examples with label negative and the $L_1$ positive, and likewise $(M_0, M_1)$ to the right. Express the information gain of $S$ as a function of $n$. Then show that this function is maximized either when $n=0$ or $n=N$ with all else constant.) (30 points)

**Answer:**
![DocScanner 12 Sept 2024 1-54 pm_1](https://github.com/user-attachments/assets/49ab91a3-712f-4327-9d67-00b03f4d7a27)
![DocScanner 12 Sept 2024 1-54 pm_2](https://github.com/user-attachments/assets/2c4934f4-2334-42b3-8f2f-6d35eaef376f)
![DocScanner 12 Sept 2024 1-54 pm_3](https://github.com/user-attachments/assets/e31f4bd3-d699-4905-b916-d635ddaf1c6c)
![DocScanner 12 Sept 2024 1-54 pm_4](https://github.com/user-attachments/assets/c5c90993-0916-4ff1-b9d5-072d7d8a65b2)
![DocScanner 12 Sept 2024 1-54 pm_5](https://github.com/user-attachments/assets/ecacadfb-a655-4c7b-a3c0-642449a4bc13)

