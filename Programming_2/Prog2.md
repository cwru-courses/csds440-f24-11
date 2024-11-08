# CSDS440 Programming Exercise 2 (120 points)
**Instructions:** Please log the last commit with "FINAL COMMIT" and **enter the final commit ID in canvas by the due date.** 

Each person in each group must commit and push their own work. **You will not get credit for work committed/pushed by someone else even if done by you.** Your commits should be clearly associated with your name or CWRU ID (abc123). Each person is expected to do an approximately equal share of the work, as shown by the git logs. **If we do not see evidence of equal contribution from the logs for someone, their individual grade will be reduced.** 

**Remember that submitting code which is not your own work constitutes an academic integrity violation.**

From canvas, you can download  the data files for the problems referred to in the questions below. Download 440data.zip and move the extracted folder (440data/) to the root of your repository. The provided gitignore will ignore this folder by default. Do not commit data files to GitHub.

Each datafile provided will have the following files: 
-- A “problem.names” file, which will list the attributes, their types and values. One attribute will be an “example-id” attribute which will be useful for identifying examples, but which you will not use when learning.
--	A “problem.data” file, which will list all the examples and their class labels.
--	A “problem.info” file, which gives additional information about the problem, such as how the data was generated. This is for your information only and does not affect the implementation in any way.

In the csds440-grouptemplate zip file on canvas, a basic organizational template has already been provided for you. Further, an abstract framework has been provided in Python which can parse the data files and set up a basic data structure. Python skeletons have also been provided for your implementation. There is a file called util.py where you should implement helper functions for cross validation, training and testing, computing metrics, and plotting graphs. This code will be reused for each assignment. 

Your programs must be written in Python. You may not use external libraries that implement any significant component of the problems. You may not reference open source code, or copy snippets from other code not written by you. Besides standard system libraries, you can use libraries that give you access to data structures or libraries for math operations (like numpy or scipy) or libraries for optimization (like cvxopt). Usable libraries have already been added to the requirements.txt file in csds440-grouptemplate. For any others, you must ask first before using them. 

Please ask me or the TAs for help if uncertain or stuck.
======================================================================

In this problem, you will implement and evaluate (i) naïve Bayes and (ii) logistic regression. 

## Naïve Bayes (30 points)

Implement the naïve Bayes algorithm discussed in class. Discretize continuous values. To do this, partition the range of the feature into k bins (value set through an option). To do this, divide the range of the feature into k equal-length disjoint intervals. The _j_<sup>th</sup> bin is the _j_<sup>th</sup> interval, so replace the original feature with a discrete feature that takes value _j_ if the original feature’s value maps to bin _j_. Use _m_-estimates to smooth your probability estimates. Use logs whenever possible to avoid multiplying too many probabilities together. Your main file should be called __nbayes__. Use the same framework/code provided in the first assignment. Your program should take four options:  

1.	The path to the data (see the first assignment).
2.	The –no-cv option (see the first assignment). 
3.	A positive integer (at least 2), which is the number of bins for any continuous feature.
4.	A real value _m_ for the _m_-estimate. If this value is negative, use Laplace smoothing. Note that _m_ = 0 is maximum likelihood estimation. The value of _p_ in the _m_-estimate should be fixed to $\frac{1}{v}$ for a variable with $v$ values. 

When your code is run, it should first construct 5 folds using stratified cross validation if this option is provided. To ensure repeatability, set the random seed for the PRNG to 12345. Then it should produce naïve Bayes models on each fold (or the sample according to the option) and report as in part (c).

## Logistic Regression (30 points)

Implement the logistic regression algorithm described in class. During learning, minimize the negative conditional log likelihood plus a constant ($\lambda$) times a penalty term, half of the 2-norm of the weights squared. You can use standard gradient descent for the minimization. Nominal attributes should be encoded as 1-of-N vectors. The main file should be called __logreg__.  It should take three options: the first two options above and a third which is a nonnegative real number that sets the value of the constant $\lambda$. The same notes about 5 fold stratified CV from above, etc. apply in this case.

## Metrics and Output format (20 points)

When either algorithm is run on any problem, it must produce output in exactly the following format:
```
Accuracy: 0.xyz 0.abc 
Precision: 0.xyz 0.abc
Recall: 0.xyz 0.abc
Area under ROC: 0.xyz 
```
For all metrics expect Area under ROC, “0.xyz” is the average value of each quantity over five folds. “0.abc” is the standard deviation. For Area under ROC, use the “pooling” method. Here, after running the classifier on each fold, store all the test examples' classes and confidence values in an array. After all folds are done, use this global array to calculate the area. To calculate the area under ROC, first calculate the TP and FP rates at each confidence level, using the numeric value output by the classifier as the confidence. Each pair of adjacent points is joined by a line, so the area under the curve is the sum over the areas of all trapezoids bounded by the FP rates of the adjacent points, the TP rates of the adjacent points, and the line joining the TP rates.

## Jupyter Notebook (40 points)

Prepare a jupyter notebook on your experiments. In the notebook, answer the following questions.
1. Create a table of the different metrics for naïve Bayes (_m_=0.01) and logistic regression ($\lambda$ =0.01) on the three datasets. Use a different cell for each dataset. For naïve Bayes, set the number of bins for continuous features to 10. (10 points)
2. Examine the effect of $\lambda$ on logistic regression. Do this by graphing area under ROC across λ=0, 0.001, 0.01, and 0.1 on the given datasets. (10 points)
3. __Research extension.__ Create one research hypothesis on naïve Bayes and/or logistic regression and evaluate it empirically. For example, you might test the effect of non-independent features, or different smoothing strategies for the probabilities, or see how accurately naïve Bayes can generate data/estimate densities etc. Your hypothesis may require you to implement other algorithms beyond the ones above. You will be evaluated on originality, technical strength, insightfulness of the observations you generate and the clarity of your discussion. (20 points)  

Place your code in the src/ directory and notebook in the notebooks/ directory and push to github. Include a short README file containing your experience with the skeleton/utility code provided and documentation, and anything you found confusing. Do not commit any other files. Note: jupyter notebooks do not work well with git, and merge conflicts can be an issue. It may be easier to put your response to each question above in a separate notebook and have only one person in group commit each notebook. 

## Grading Rubric

Your code must be efficient, cleanly written and easy to understand. For python, try to follow PEP8 as far as feasible. The code should handle errors and corner cases properly. You will lose points if the TAs cannot follow the logic easily or if your code breaks during testing. Note that we will test your code on data other than what is provided, and the TAs will not have time to debug your code if it breaks. Your code should work on the Python environment you were given. 
Generally, point deductions will follow these criteria:
-	Incomplete implementation/Not following assignment description: up to 100%
-   Syntax Errors/Errors causing the code to fail to compile/run:
    - Works with minor fix: 15%
    - Does not work even with minor fixes: 75%
-	Inefficient implementation: 15%
    -	Algorithm takes unreasonably long to run during grading: additional 10%
-   Poor code design: 20%
	- Hard-to-follow logic
    - Lack of modularity/encapsulation
    - Imperative/ad-hoc/"spaghetti" code
    - Duplicate code
-   Poor UI:
    - Bad input (inadequate exception handling, no `--help` flag, etc.): 10%
    - Bad output (overly verbose `print` debugging statements, unclear program output): 10%
-	Poor code style (substantially not PEP8): 5%
-	Poor documentation: 20%
-	Non-code commits: 2%
    - Committing data files
    - Committing non-source files (`.idea` files, `.iml` files, etc.)
    - Hint: use your .gitignore file!
-	Not being able to identify git contributor by their name or case ID: 5% per commit
-	Code not in src/: 2%
-	Notebook not in notebooks/: 2%

Bonus points may be awarded for the following:
-	Exceptionally well-documented code
-	Exceptionally well-written code
-	Exceptionally efficient code 
    -	Hint: use pure python (`for` loops, etc.) as minimally as possible!
