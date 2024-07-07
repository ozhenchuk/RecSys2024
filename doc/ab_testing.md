# A/B Testing framework

#### General Information
A/B testing helps test an online recommeder system based on agent/user interation with it. The main idea is to split the users into two groups: **control** and **treatment** group.
This is done randomly, without user knowledge.

#### Splitting
In case of MovieLens film recommender system, a **population split** is done via random sampling. In order to make the sampling more fair, we use the user's gender (if available) and use different weights for probabilities of getting into treatment group.

The splitting process itself is achieved via API server proxy, making the user's clients interact with different recommender systems depending on if they're in control or treatment group

#### Evaluation metrics

Since evaluation is done online, it is important to use approriate metrics. In this case, we can use **Mean reciprocal rank (MRR)**, which assigns a score $RR$ for each click on a recommendation from a user, but also considers the rank of the recommendation. The lower the recommendation on the list, the smaller is the metric. If a user ignores all recommendations, the score is equal to $0$. Then, the rank data is collected from each user and aggregated on a server, and the metric is calculated for the sample  $$ MRR = \frac{1}{n} \sum_{i=1}^{n}\frac{1}{RR_i}$$

#### Hypothesis testing

In order to make conclusions, a hypothesis has to be formed. In our case, we can compare the means of our metrics for both groups ($\mu_0, \mu_1$)and then do a test for their equality 

Since we do not know anything about the variances of the underlying distributions, we will use a **two-sided t-test** for equality of means

$H_0: \mu_0 \neq \mu_1$

$H_1: \mu_0 = \mu_1$

with significance level of 0.05