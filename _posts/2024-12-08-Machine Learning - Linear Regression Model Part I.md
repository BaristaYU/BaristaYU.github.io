---
title: "(ENG) Machine Learning - Linear Regression Model Part I"
categories:
  - Machine learning
  - Regression algorithm
tags:
  - AI
  - Andrew Ng
  - Supervised learning
---

Hello, my name is Yongkyun Yu.
I'm Korean, and I'm not good at English.
Therefore, I'm trying to improve my English skills by sharing some data science knowledge in English without translator.
You'll learn some knowledge, and I'll learn some English skills.
Quite cool, isn't it?
Thank you for your understanding, I'll try my best.

This post was written after attending Professor Andrew Ng's Supervised Machine Learning: Regression and Classification Course at Coursera.<br>
It'll be nice to understand this post if you know how to use numpy module in python and some mathmatical knowledge, but it's okay if not.

---

Today, We will learn about `Linear regression model` and its `Cost function`.

Before we begin, let's check about `Supervised learning` and `Unsupervised learning`.

What is `Supervised learning`:

- `Train` input $x$ and output $y$ to `predict` $y$ for new $x$.
- `Supervising` model to find `function` with training set $x$ and $y$, because $y$ is a `answer` of $x$.

What is `Unsupervised learning`:

- `Train` some data to `predict` some `pattern` or `structure`
- While trained data is `uncategorized`, using model to figure out its categories.

Now, let's learn about `Linear regression model`, the simplest `Regression model`, which is method of `Supervised learning`.

---

What is `Linear regression model`:

- An algorithm that has `linear relationship` between input $x$ and output $y$.
- `Regression` means a assumption that some data follows a particular `trend`.
  - For instance, if one family has 170 centimeters of average height, their son or daughter will also grow to about 170 centimeters.
  - This is because their heights try to `regress` 170 centimeters.

### Terminology note

Here's some note about terms in this post:

- `Scalar`: 1-Dimensional array
- `Vector`: n-Dimensional array
- `Training Set`, `Dataset`: a set of input data and output data to train model
- $x$, `Feature`: input data
- $y$, `Target`: output data
- $\hat{y}$, `y-hat`: predicted value of $y$
- `parameter`: editable value
- $w$, `weight`: parameter of model
- $b$, `bias`: parameter of model
- `Model`: a function that based on algorithm

Before learning `Multiple Linear Regression`, which has multiple $x$, let's learn about `Univariate Linear Regression`, which has a single $x$.

---

### Univariate Linear Regresssion

![image](https://github.com/user-attachments/assets/491911b5-9ccf-4fec-9e77-cccc4a52fd5c)
(This picture is little bit edited from the lecture.)

Above is a graph that indicates House sizes($x$) and House prices($y$) with `Univariate linear model`. <br>
If 3000 $\text{feet}^2$ size of house are selling, then what's the price of it?<br>
Referring to `Linear model`, it will price about 500 dollars.

Let's make a function from the graph.<br>
First, `Univariate linear model`'s function is:<br>
$$\hat{y}=wx+b$$<br>
This model's $y$ is `Predicted value`, so we should notate it $\hat{y}$. <br>
Referring to the graph, when $x$ is 0, then $\hat{y}$ is 0 too, so $b$ is 0 obviously. <br>
Next, when $x$ is 600, then $y$ is 100, so $w$ is $1/6$. <br> Finally, the function is shown below:<br>
$$\hat{y}=\frac{1}{6}x+0$$

Before making function, we predicted 3000 $\text{feet}^2$ size of house will price about 500 dollars. <br> According to the function, we predicted well.

---

### Cost Function

By `Univariate linear model`, we can `predict` house price.
Then, is the predicted value really accurate? we can check this `accuracy` as `Cost function`. Let's look at below: <br>
$$J(w,b)=\frac{1}{2m}\displaystyle\sum_{i=1}^m(\hat{y}^{(i)}-y^{(i)})^2$$<br>
Don't worry about math, we will learn not that deep.<br>
First, this function called `MSE`, `Mean Squared Error`, which is `one of popular function`.
Let's look carefully about the function.

- $J(w,b)$: `Cost function` $J$, which has parameter $w$, $b$.
- $\displaystyle\sum_{i=1}^m(\hat{y}^{(i)}-y^{(i)})^2$: $i$ goes 1 to $x$'s amount $m$, sum **sqaure of difference between each $i$th $\hat{y}$ and $y$**. **$(i)$ doesn't mean power of $i$.**
- $\frac{1}{2m}$: devide with $x$'s amount m, because it's **Mean** sqaured error. `2` in front of $m$ just exists for easier partial derivative, so `it doesn't effects to result`.

Now, check example to understand easily.
<br>Below is a sample data which related to $\hat{y}=\frac{1}{6}x+0$ model.

| Predicted price value $\hat{y}$ | Actual price value $y$ |
| :------------------------------ | ---------------------- |
| 100                             | 200                    |
| 400                             | 300                    |
| 500                             | 500                    |

$x$ has only one $y$, so amount of $x$, $m$ is 3.<br>
When $i$ is 1, difference of $\hat{y}$ and $y$ is -100. <br>We are following `MSE`, so we need sqaured value 10000. <br>When $i$ is 2, we have 100 and 10000, 0 and 0 for $i$ is 3. <br><br>
Did you noticed the meaning of square? If we don't square differences, sum of differences is -100 + 100 + 0 = 0. <br>However, predicted price values when $i$ is 1 and 2 are `not the same as actual value`. Therefore, it can't be 0 which is result of mean squared `error` function, so we sqaure differences to prevent 0.<br><br>There's one more question.<br>If we use square for prevent 0, then why don't we use `absolute value`?<br>That's right. We also have `MAE`, `Mean Absolute Error` function, which uses `absolute value` than squared value. <br> Then, why we use `MSE` more than `MAE`? Here's some reasons: <br>

1. `MSE` can penalize error more than `MAE`.
2. `MSE` is differentiable.<br>

Referring to number 1, perhaps `MSE` might weak for `outlier`. Actually, it is. <br> Importance of selecting `cost function` is should appropriate to `data` and `model`.<br> About number 2 and $\frac{1}{2m}$ of `MSE` function,<br> we will learn it when introducing `Gradient descent` in next post.

Return to the subject, our data don't look like having outlier, so we'll keep using `MSE` function: <br>
$$J(w,b)=\frac{1}{2m}\displaystyle\sum_{i=1}^m(\hat{y}^{(i)}-y^{(i)})^2$$ <br>
Sum of each sqaure of $i$th difference is 10000 + 10000 + 0 = 20000. <br>
When we divide to $\frac{1}{2m}$, the result of `MSE` is 20000 / (2 \* 3) = about 3333.3.
<br> As we noticed above, there's `no differences between predicted value and actual value when the result of cost function nears to 0`. <br><br> We can't manipulate data $x$ and $y$ to correct model. <br> Therefore, there's only two `parameter`, which are editable values, `weight` $w$ and `bias` $b$. <br> It's because $\hat{y}$ means $wx+b$. That's why the `cost function` is $J(w,b)$. <br><br> Now we have final question. <br>
How can we get appropriate $w$,$b$ that makes value of $J(w,b)$ near to 0? <br> We can figure out with `Gradient descent`, noticed above. <br> Let's learn it in next post.
<br>

---

Yongkyun Yu, 2024-12-08 <br>
Special thanks to Coursera and Professor Andrew Ng.
