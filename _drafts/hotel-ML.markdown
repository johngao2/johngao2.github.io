---
layout: post
title: "Uncensoring Hotel Data"
author: "John Gao"
categories: blog
tags: [RL]
---

# Hotel Data 

* Review importance of demand estimation in RM
* "remarkably simple model???"
* 

The process starts when a user reaches a webpage. While the page is loading, the following events happen in sequence:
1. The website sends impression info (user cookie ID, location, time, domain, URL, etc.) to a central auction exchange in the form of a **bid request**
2. Bots/agents from various companies receive this bid request and look at the info.
3. Each bot decides on a bid price and submits it. Each bot also submits the ad that it wishes to broadcast, should it win the auction.
4. The highest bidder gets their ad displayed on the webpage and pays a sum of money to the website. I'll focus on second-price auctions in this post, which means that the top bidder pays one cent above the second-highest bid regardless of how high the original top bid was. An alternative is first-price bidding, where the top bidder pays his original bid. I'm choosing this particular auction format because [Cai et al., 2017](https://arxiv.org/abs/1701.02490){:target="_ blank"} focus on it, and because it's far more common than first-price auctions.

Keep in mind that this process happens in the span of milliseconds, and repeats every time someone visits a page. To make this process clearer, here's a gif outlining the process for a hypothetical ad.

![Alt Text](../assets/img/rlb/rtb_process.gif){:height="700px" width="2000px"}
**Figure 1: RTB Process GIF**
{: style="text-align: center"}

The goal here is to design a bidding agent that maximizes some KPI under a budget constraint. There are various options for KPI, but we'll be using the most common one, which is the total number of clicks acquired.

# Background and Prior Work

The probability of a user clicking the ad after seeing it is called **click-through rate(CTR)**. In an ideal world, the cost of an impression would be:

$$
\text{CTR} \times \text{click value}
$$

Where click value is the average spend from a customer after clicking on an ad. This formula makes sense intuitively; the price of an ad is just the expected revenue from an impression. Finding the optimal bid price in this scenario is relatively straightforward. We would need to implement some supervised learning models to predict click value and CTR, and then submit their product as the bid price.

Unfortunately, we live in the real world, where things aren't so simple. The optimal price of an ad depends on many other factors, such as **market competition**, **remaining auction volume**, and **remaining campaign budget**. To bid optimally, agents need to model these variables to some extent. In the past, people have tried to fit static distributions to these variables, and then maximize some metric (e.g., total clicks or revenue) using these distributions. This method tended to not work well in practice, since market competition is very dynamic, which makes it very hard to have accurate static models.

## Past Bidding Agents

After receiving a bid request, an agent needs to do three things:
1. The agent first needs to estimate the *utility* of the impression (represented in this paper by CTR, but can use other metrics). This part is called the **utility estimation component**. The goal is to accurately predict how much benefit the company will receive from showing the ad, and is the most important decider of the bid price. For example, if an agent thinks that a particular user is unlikely to click on an ad from their company, then it'll bid lower, and vice versa.
2. The agent also needs to forecast the *cost distribution* of the ad. This is the **bid landscape forecasting component** of the agent. A cost distribution is needed to find the win probability for a given bid on an impression. This win probability is needed to plan out future budgeting decisions.
3. Given estimated utility, cost distribution, remaining budget, and remaining auction volume, the agent needs to decide on a final bid price for an impression. This is referred to as the **bid optimization component**. Past work on this particular component is pretty scarce, which is good since **this component is the main focus of this paper.**

**Utility Estimation Component:**  
The goal here is to estimate the probability of an event (e.g., user click) given ad and user info. This info is contained in what we refer to as the **feature vector** of an impression. As the name suggests, the feature vector contains features (or inputs) for our estimator. These probabilities are relatively stationary, which makes utility estimation a fairly typical supervised learning problem. Therefore, any old supervised learning model will do here.

In practice, companies tend to use [logistic regression](http://wnzhang.net/share/rtb-papers/cvr-est.pdf){:target="_ blank"} or gradient boosted trees for this task. There are also cases of more niche models that perform weight updates automatically, such as [Bayesian probit regression](http://quinonero.net/Publications/AdPredictorICML2010-final.pdf){:target="_ blank"} or ["Follow the Regularized Leader" (FTRL) logistic regression](http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf){:target="_ blank"}. Since utility estimation isn't the focus, the paper covered in this post ([Cai et al., 2017](https://arxiv.org/abs/1701.02490){:target="_ blank"}) uses FTRL logistic regression to predict CTR.

**Cost Estimation Component (a.k.a. Bid Landscape Forecasting):**  
The bidding agent also needs to model the price distribution for each ad. This distribution is needed to find win probability, which is the c.d.f. of the distribution at some bid price. Unfortunately, this isn't a typical supervised learning problem like utility estimation.

The main problem here is that the price is **censored** for most ads. Specifically, it's **right-censored**, which means that sometimes we don't know how the high the market price of a bid can be. This is because the agent can only ob