# CSDS440 Written Homework 2
**Instructions:** Each question is worth 10 points unless otherwise stated. Write your answers below the question. Each answer should be formatted so it renders properly on github. **Answers that do not render properly may not be graded.** Please comment the last commit with "FINAL COMMIT" and **enter the final commit ID in canvas by the due date.** 

When working as a group, only one answer to each question is needed unless otherwise specified. Each person in each group must commit and push their own work. **You will not get credit for work committed/pushed by someone else even if done by you.** Commits should be clearly associated with your name or CWRU ID (abc123). Each person is expected to do an approximately equal share of the work, as shown by the git logs. **If we do not see evidence of equal contribution from the logs for someone, their individual grade will be reduced.** 



1.	Write a program to sample a set of $N$ points from $(−1,1)^2$. Label the points using the classifier $y=sign(0.5x_1+0.5x_2)$. Generate datasets from your program and use your ID3 code from Programming 1 to learn trees on this data (there is no need to do cross validation or hold out a test set). Plot a graph where the $x$-axis is the value of $N$, over $N={50, 100, 500, 1000, 5000}$, and the $y$-axis is the depth of the tree learned by ID3. Explain your observations. (20 points)

## Answer:

I have implemented the program in the Jupyter notebook at the location **"csds440-f24-11\Programming_1\notebooks\HW2.ipynb"**.

### The Program does the following:
- It samples a set of $N$ points from $(−1,1)^2$. Label the points using the classifier $y=sign(0.5x_1+0.5x_2)$. 
- Generate datasets from your program and use your ID3 code from Programming 1 to learn trees on this data
- Plot a graph where the $x$-axis is the value of $N$, over $N={50, 100, 500, 1000, 5000}$, and the $y$-axis is the depth of the tree learned by ID3.

![Decision Tree Depth vs N](images/image.png)

### Explanation of My Observations:

- The initial splits capture most of the decision structure early on, causing rapid growth in depth for small \( N \).
  
- For larger \( N \), additional data provides diminishing returns in terms of the information required to refine the boundary, leading to slower growth in depth.
  
- Overfitting prevention mechanisms and the nature of the data (e.g., linearly separable) contribute to a saturation point, where the tree stops growing significantly deeper.



2.	Show the decision boundaries learned by ID3 in Q1 for $N=50$ and $N=5000$ by generating an independent test set of size 100,000, plotting all the points and coloring them according to the predicted label from the $N=50$ and $N=5000$ trees. Explain what you see relative to the true decision boundary. What does this tell you about the suitability of trees for such datasets? (20 points)

## Answer:
I have implemented the program in the Jupyter notebook at the location *"csds440-f24-11\Programming_1\notebooks\HW2.ipynb"*.

## The Program Implements the Following:

1. *Sampling Data*: It samples $N$ points from the set  $(-1, 1)^2$ and lproduces labels these points using the classifier $y = \text{sign}(0.5x_1 + 0.5x_2)$.

2. *Learning with ID3*: It uses the ID3 decision tree algorithm that was implimentede in Programing Assignment 1 to learn trees on the generated datasets .

3. *Varying Dataset Sizes*: It produces a plots where the x-axis has the number of points $N$ (for $N = \{50, 100, 500, 1000, 5000\}$) and the y-axis has the depth of the decision tree .

4. *Plotting Decision Boundaries*: It generates an  test set of 100,000 independent points and plots how the trees trained on $N=50$ and $N=5000$ training points produces decision boundaries compared to the true decision boundary.
![Plotting Decision Boundaries](images/image-1.png)
### Explanation of Results:

#### Decision Boundary for $N = 50$:
- When training with only of 50 examples,the decision boundaries created by the tree is axis alligned staircase like structure that approximates the true linear boundary. 

- The tree *underfits* the data because it doesn't have enough data points to create precise splits.

#### Decision Boundary for $N = 5000$:
-The decision tree has enough data to create finer splits With 5000 examples. Because of this, it provides a much better approximationof the true decision boundary.
- However, the boundary is like a "staircase" pattern due to the axis-aligned nature of the splits. The tree cannot produce a perfectly diagonal boundary, despite the larger dataset,  

## Suitability of Decision Trees for Such Datasets:

#### 1. *Axis-Aligned Splits Lead to the "Staircase" Effect*
   - *Axis-aligned splits* It creates decision boundaries which are parallel to the feature axes like rectanges, because of which its difficult to learn decision boundaries which are not axis alligned.
  

#### 2. *Poor Generalization, Especially with Smaller Datasets*
   - For small datasets, decision trees doesnt *generalize* well to unseen data because of the few available splits  oversimplifies the decision boundary.


#### 3. *Scaling with Data Size*
   - With the large training set the decision tree improves its boundary and approximates the actual boundary  but still it has **axis-aligned splits**, making it hard to accurately capture linear boundaries .
   - With Larger datasets  underfitting is reduced, but it still suffer from **overfitting**, trying to fit the noise or small variations in the data.


3.	Consider the following table of examples over Boolean attributes, annotated with the target concept's label. Ignore the "Weight" column and use information gain to find the first split in a decision tree (remember that ID3 stops if there is no information gain). You can use your programming code/a numerical package like Matlab to do this, and just report the final result.(10 points)

|A1|	A2|	A3|	A4|	Label|	Weight|
|---|---|---|---|---|---|
|F|	F|	F|	F|	0	|1/256|
|F	|F	|F	|T|	0	|3/256|
|F	|F	|T	|F	|1	|3/256|
|F	|F	|T	|T	|1	|9/256|
|F	|T	|F	|F	|1	|3/256|
|F	|T	|F	|T	|1	|9/256|
|F	|T	|T	|F	|0	|9/256|
|F	|T	|T	|T	|0	|27/256|
|T	|F	|F	|F	|1	|3/256|
|T	|F	|F	|T	|1	|9/256|
|T	|F	|T	|F	|0	|9/256|
|T	|F	|T	|T	|0	|27/256|
|T	|T	|F	|F	|0	|9/256|
|T	|T	|F	|T	|0	|27/256|
|T	|T	|T	|F	|1	|27/256|
|T	|T	|T	|T	|1	|81/256|

### Answer:
- I have used the programming code from programming assignment 1 
- I have implemented the program in the Jupyter notebook at the location **"csds440-f24-11\Programming_1\notebooks\HW2.ipynb"**.

![alt text](images/image-2.png)


**Explaination:-**
- For all the attributes, each split on an attribute leads to a balanced distribution of the output, resulting in an information gain of 0 for all attributes. Therefore, we cannot split on any attribute


4.	Now from the same table, find another split using "weighted" information gain. In this case, instead of counting the examples for each label in the information gain calculation, add the numbers in the Weight column for each example. You can use your code/a numerical package like Matlab to do this, and just report the final result. (10 points)

### Answer:
- I have used the programming code from programming assignment 1 
- I have implemented the program in the Jupyter notebook at the location **"csds440-f24-11\Programming_1\notebooks\HW2.ipynb"**.

![alt text](images/image-3.png)
**Explaination:-**
- For the attribute A1, A2 and A3 we get the "weighted" information gain values as 0.0343, and for attribute A4 we get 0.
  

5.	There is a difference between the splits for Q3 and Q4. Can you explain what is happening? (10 points)

### Answer:
**Explanation of the Difference Between Splits:-**




 **Weighted Entropy:**

The weighted entropy formula takes into account the weights associated with each example. Instead of counting the examples for each class, we have to sum the weights of each examples.

$$
H_{weighted}(Y) = - \sum_{i=1}^{k} \left( \frac{W_i}{W_{total}} \right) \log_2 \left( \frac{W_i}{W_{total}} \right)
$$


 **Weighted Information Gain:**

Information gain is calculated as the difference between the entropy before the split and the weighted entropy after the split.

$$
IG_{weighted} = H_{weighted}(Y) - \sum_{j=1}^{m} \left( \frac{W_j}{W_{total}} \right) H_{weighted}(Y_j)
$$

**unweighted case:-**
- In the unweighted case, each example contributes equally to the calculation of entropy and information gain.

- For a balanced Boolean function, where there is an equal number of positive and negative examples in every split for each attribute (as given in the dataset), the information gain is zero for all attributes.

 **weighted case**:- 
- Examples with higher weights have higher contribution towards the entropy and information gain. If a perticular class in a split has a higher total weight, the splits try to reducing uncertainty around  higher-weighted class examples.

- Since the split tries to minimize the uncertainty for examples with higher weights, attributes which separates high-weighted examples from the rest will have a greater information gain. As a result, the split on attributes happens differently when compared to the unweighted case, prioritizing those features that provide better splits for the more "important" examples (those with higher weights).

-  In the weighted case, even if a dataset is balanced as per the class counts ie equal number of positive and negative examples in every split, if certain examples  in the dataset are weighted higher, the decision tree will splits in such a way to reduce entropy for those particular weighted examples.



6.	Person X wishes to evaluate the performance of a learning algorithm on a set of $n$ examples ( $n$ large). X employs the following strategy:  Divide the $n$ examples randomly into two equal-sized disjoint sets, A and B. Then train the algorithm on A and evaluate it on B. Repeat the previous two steps for $N$ iterations ( $N$ large), then average the $N$ performance measures obtained. Is this sound empirical methodology? Explain why or why not. (10 points)

### Answer: 

- Let’s assume D = data set with n being large
- In each iteration i, the dataset D is randomly partitioned into two disjoint subsets Ai and Bi, each of size n/2 , n/2.
- Each data point in D is likely to appear several times for both training and testing sets. Let say D=1,2,3,4,5,6,7,8,9,10. In first iteration A1 = 1,3,5,7,9 and   B1 = 2,4,6,8,10 both are disjoint. in iteration 2, A2 = 1,3,5,7,2 and B2 = 9,4,6,8,10 both are disjoint, A3 = 4,3,5,7,2 and B3 = 9,1,6,8,10 As we can see in each iteration most of training and testing data is same.
- The error estimate is dependent because the shared data points.
- 	Let us say Error estimate in each iteration Ej, and average error is sum(Ej) over iterations/N, where N iterations as given in question.
- Since, the Bi the training points is subset of D, the Bi is dependent on training points used in other iterations.
- The error estimate Ej will be underestimated because it is not independent variable because of data overlapping.
- Also, the training data is very less (n/2) which will affect the performance of model. If we train on less amounts of data there might be a chance that some examples it may not predict well.
-	overlapping data across iterations can lead to biased and overly optimistic estimates of the model's performance.
-	Increasing the N iterations ca reduce the variance and it can not eliminate the bias which is from the data dependency. For practical datasets bias and dependency remain significantly same. It won’t reduce over iterations.
-	So, given methodology is not very sounds. Why because it underestimates the Error rate Ej which will underestimate the variance also. Also misleading inference statistically because of the overlapping of data across the iterations.


7.	Two classifiers A and B are evaluated on a sample with P positive examples and N negative examples and their ROC graphs are plotted. It is found that the ROC of A dominates that of B, i.e. for every FP rate, TP rate(A) $\geq$ TP rate(B). What is the relationship between the precision-recall graphs of A and B on the same sample? (10 points)

### Answer: 
Given Two Classifiers **$A$ and $B$:**,The number of positive examples is denoted by $P$ (all positives) and the number of negative examples is denoted as $N$ (all negatives).

**ROC Curve has:**

- The X-axis represents the **False Positive Rate (FPR)**.
- The Y-axis represents the **True Positive Rate (TPR)**.



**Assumption:** Given that ROC curve of classifier $A$ dominates that of classifier $B$, meaning the ROC curve of $A$ lies above that of $B$.

- Therefore, for any given $FPR$, we have:  
  $TPR_A \geq TPR_B$.

---

### **Diagram of ROC Curve for Classifiers $A$ and $B$:**

The diagram shows the relationship between $FPR$ and $TPR$ for both classifiers $A$ and $B$:
![alt text](images/ROC.jpg)

- For any given $FPR$, $FPR_A \leq FPR_B$ and $TPR_A \geq TPR_B$.
- Alternatively, for any given $TPR$, $FPR_A \leq FPR_B$.

---



**We know that Precision is given by:**  

$$
\text{Precision} (P) = \frac{TP}{TP + FP}
$$

where:
- $TP$ is the number of true positives.
- $FP$ is the number of false positives.


### **Writing Precision as a function of TPR and FPR:**

**Express $TP$ in terms of $TPR$ and $P$:**  
We know that:

$$
TPR = \frac{TP}{\text{All Positives}} = \frac{TP}{P} \quad \text{(where $P$ is the total number of positives)}
$$

Thus, solving for $TP$:

$$
TP = TPR \times P \quad \text{(Equation 1)}
$$

**Express $FP$ in terms of $FPR$ and $N$:**  
Similarly, we know that:

$$
FPR = \frac{FP}{\text{All Negatives}} = \frac{FP}{N} \quad \text{(where $N$ is the total number of negatives)}
$$

Thus, solving for $FP$:

$$
FP = FPR \times N \quad \text{(Equation 2)}
$$

**Substituting into the Precision Formula:**  
Using Equations (1) and (2), we substitute $TP$ and $FP$ into the Precision formula:

$$
P = \frac{TP}{TP + FP} = \frac{TPR \times P}{(TPR \times P) + (FPR \times N)} \quad \text{(Equation 3)}
$$

---

**For a Precision-Recall Curve:**

- The X-axis represents Recall (which is the same as $TPR$).
- The Y-axis represents Precision $P$.

From the previous step, we have the expression for Precision as a function of $TPR$ and $FPR$. Now, we aim to compare the precision of classifiers $A$ and $B$.



 **Comparing Precision of Classifiers $A$ and $B$:**

 ![alt text](images/PVSRecall.jpg)

**Assume $TPR_A = TPR_B = TPR'$ and compare $FPR_A$ and $FPR_B$:**  
Given that classifier $A$'s ROC curve dominates that of classifier $B$, we know:
If 

$TPR_A = TPR_B = TPR'$, then $FPR_A \leq FPR_B$.

Since $FPR_A \leq FPR_B$, we can write:

$$
N \times FPR_A \leq N \times FPR_B \quad (\text{multiplying by $N$}).
$$

**Add $TPR' \times P$ to both sides of the denominator:**  
Now, adding $TPR' \times P$ (which corresponds to the true positives) to both sides in the denominator:

$$
P \times TPR' + N \times FPR_A \leq P \times TPR' + N \times FPR_B.
$$

Reciprocating and multiplying both sides by $P \times TPR'$:

$$
\frac{P \times TPR'}{P \times TPR' + N \times FPR_A} \geq \frac{P \times TPR'}{P \times TPR' + N \times FPR_B}.
$$

$$
P_A \geq P_B
$$

This shows that the precision of classifier $A$ is greater than or equal to that of classifier $B$.

---

### **Conclusion:**

Based on the analysis above, we can conclude that:

$$
P_A \geq P_B
$$

Thus, the Precision of classifier $A$ is greater than or equal to the Precision of classifier $B$, denoted as:

$$
\text{Precision}_A \geq \text{Precision}_B
$$




8.	Prove that an ROC graph must be monotonically increasing. (10 points)

### Answer:

To say a function $f(x)$ is said to be monotonically increasing

$$
\text{if }x_1 > x_2 \implies f(x_1) \geq f(x_2) \text{ for all x}
$$

and the ROC is said to be monotonically increasing,if the following is true:-

$$
\text{FPR}(t_2) > \text{FPR}(t_1) \implies \text{TPR}(t_2) \geq \text{TPR}(t_1)
$$

Where:

$$
\text{True Positive Rate (TPR)}(t) = \frac{\text{True Positives count at threshold } t}{\text{Actual Positivescount}} = \frac{\text{TP}(t)}{P}
$$

$$
\text{False Positive Rate (FPR)}(t) = \frac{\text{False Positives count at threshold } t}{\text{Actual Negatives count}} = \frac{\text{FP}(t)}{N}
$$

- $\text{TP}(t)$ and $\text{FP}(t)$ are the no. of true positives and false negatives at threshold $t$ repectively.
- $P$ and $N$ are the total no. of actual positive and total no. of actual negatives instances (a constant) respectively.


#### To Prove $\text{FPR}(t_2) > \text{FPR}(t_1) \implies t_2 < t_1$

A classifier gives a score to each instance. We classify an instance as positive if its score is higher than the threshold $t$.

If we are desreasing the thresold t it means we will get more instances for both "positive" and "negative".

Given that $\text{FPR}(t_2) > \text{FPR}(t_1)$, it must be that $\text{FP}(t_2) > \text{FP}(t_1)$ as $P$ and $N$ are constants, which implies more negative instances are classified as positive at $t_2$.

$$
\text{FP}(t_2) \geq \text{FP}(t_1) \implies t_2 < t_1
$$

Divide By N 

$$
\frac{\text{FP}(t_2)}{N} \geq \frac{\text{FP}(t_1)}{N} \implies t_2 < t_1
$$

$$
\text{FPR}(t_2) > \text{FPR}(t_1) \implies t_2 < t_1
$$

So, the one and only way for $\text{FPR}(t_2) > \text{FPR}(t_1)$ is if $t_2 < t_1$.

---

#### To Prove $t_2 < t_1 \implies \text{TPR}(t_2) \geq \text{TPR}(t_1)$

Similarly Lowering the threshold $t$ increases the number of positive instances correctly classified as positive.


$$
\text{if } t_2 < t_1 \implies \text{TP}(t_2) \geq \text{TP}(t_1)
$$

Divide by P 

$$
\text{TPR}(t_2) = \frac{\text{TP}(t_2)}{P} \geq \frac{\text{TP}(t_1)}{P} = \text{TPR}(t_1)
$$

$$
\text{if } t_2 < t_1 \implies \text{TPR}(t_2) \geq \text{TPR}(t_1)
$$

So, lowering the threshold ($t_2 < t_1$) implies that $\text{TPR}(t_2) \geq \text{TPR}(t_1)$.

---

Using the implications we have established:

$$
\text{FPR}(t_2) > \text{FPR}(t_1) \implies t_2 < t_1
$$

and

$$
t_2 < t_1 \implies \text{TPR}(t_2) \geq \text{TPR}(t_1)
$$

According to the transitive property :

$$
\text{FPR}(t_2) > \text{FPR}(t_1) \implies \text{TPR}(t_2) \geq \text{TPR}(t_1)
$$

Since ROC curve has FPR on the X-axis and TPR on the Y-axis and we have proven the following

$$
\text{FPR}(t_2) > \text{FPR}(t_1) \implies \text{TPR}(t_2) \geq \text{TPR}(t_1)
$$
 
 We can say the ROC curve is monotonically increasing with respect to the False Positive Rate.



9.	Prove that the ROC graph of a random classifier that ignores attributes and guesses each class with equal probability is a diagonal line. (10 points)

### Answer: 

To random classification the predicting p(positive) =0.5 and p(negative)=0.5 . It acts as a random variable S with values uniformly distributed between 0 and 1 for each instance. We can compute the TPR,FPR at different values of $t$ between 0 and 1.

The probability density function (PDF) of $S$ is:

$$
f_S(s) = \begin{cases}
1 & \text{if } 0 \leq s \leq 1 \\
0 & \text{otherwise}
\end{cases}
$$

The cumulative distribution function (CDF) of $S$ is:

$$
F_S(s) = P(S \leq s) = \begin{cases}
0 & \text{if } s < 0 \\
s & \text{if } 0 \leq s \leq 1 \\
1 & \text{if } s > 1
\end{cases}
$$

Calculate $P(\text{score} \geq t \mid \text{positive})$:


for positive instances also score distribution is same. So we have

$$
P(\text{score} \geq t \mid \text{positive}) = P(S \geq t)
$$

ByUsing the CDF of $S$

$$
P(S \geq t) = 1 - P(S < t) = 1 - F_S(t)
$$

Therefore:

$$
P(S \geq t) = 1 - t \quad \text{(since } F_S(t) = t \text{ for } t \in [0, 1])
$$

Calculate $P(\text{score} \geq t \mid \text{negative})$:

Similarly, for negative instances:

$$
P(\text{score} \geq t \mid \text{negative}) = P(S \geq t) = 1 - t
$$


Therefore, for both positive and negative instances

$$
P(\text{score} \geq t \mid \text{positive}) = P(\text{score} \geq t \mid \text{negative}) = 1 - t
$$

Probability Distributions:

- For positive 

  $P(\text{score} \geq t \mid \text{positive}) = 1 - t$

- For negative 

  $P(\text{score} \geq t \mid \text{negative}) = 1 - t$

-  because the probability that a random score exceeds $t$ is $1 - t$ and scores are uniformly distributed.



   At any threshold $t$:
     - $\text{TPR} = P(\text{predict positive} \mid \text{positive}) = 1 - t$
     - $\text{FPR} = P(\text{predict positive} \mid \text{negative}) = 1 - t$
   
   $\text{it means TPR} = \text{FPR for all points on the graph.} $
   
Because both are equal for all $t$, the points in x and y axis will be same and the will be a straight line and it will be be diagonal becasue the slope of line is 1 because TPR/FPR=1 and the graph line starts with (0,0) point.
 



