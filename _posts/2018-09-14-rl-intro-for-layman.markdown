---
layout: post
title: "An Absolute Beginner's Intro to Reinforcement Learning (Part 1)"
author: "John Gao"
categories: journal
tags: [RL]
---
In this **relatively long** tutorial series, I'll go over some of the basic fundamental theory behind reinforcement learning (RL). I'll try to do it in a way that should be accessible to anyone with basic math knowledge; as such, **the technical difficulty of this series will be relatively low**. The focus will be on diving deeply into the RL problem framework and one specific RL technique, with particular emphasis on the intuition behind them. Breadth is not the focus here, but if you want a really comprehensive view of RL techniques then I highly recommend reading this [blog post](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#markov-decision-processes){:target="_ blank"}. I'm hoping to keep the math in this post as light as possible, and focus on visuals.

* The generated Toc will be an ordered list
{:toc}

# Introduction
---
This two-part tutorial series will heavily focus on presenting the fundamental ideals of classic RL. These concepts are also very important for modern RL (e.g. Deep RL), but I won't be covering the modern stuff too deeply here. After reading this series, you should be able to:
* Have a general, high-level understanding of what constitutes RL
* Be able to convert normal word problems into the RL problem format
* Know at least one RL algorithm you can use to solve the RL problem, and understand how that algorithm works

The scroll bar on your right might look frighteningly small, but don't worry; a lot of this blog post is comprised of pictures, so it's not as long as you think. There will also be some exercise problems. The answers will be right after the problem, and separated by something like this:  

**ANSWERS BELOW**

---

Of course, try not to scroll down to the answers until you've at least tried the question.

**Acknowledgements:**  
Most of the work in this post comes from Sutton & Bartok's [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf){:target="_ blank"}. It's genuinely amazing and decently accessible, so I would highly recommend to everyone who has an interest in getting started with RL. Also, much of the content here comes from a presentation I did while I was interning at TD Bank, so I'd like to thank them for giving me time to work on this.

# Chapter 1: An Overview of Reinforcement Learning (RL)
___

## What is RL?
Reinforcement learning has exploded in popularity in the past few years. Now it receives a staggering amount attention from both industry and academia, whereas 20 years ago it was a niche research field. I won't get much into the history of RL here, but it's actually super interesting, so you should read about it [here](http://incompleteideas.net/book/ebook/node12.html){:target="_ blank"}. Usually when an idea suddenly gets a large amount of traction, a lot of information often gets lost in the process. As a result, there's still a lot of confusion about what constitutes RL, so I wanted to take this first section to provide a unified definition for reinforcement learning.

**Reinforcement learning is not just a single technique;** it's a class of algorithms. Here's the official definition, from one of the most famous people in the field, Richard Sutton.

>"We define reinforcement learning as any effective way of solving reinforcement learning problems."

This definition might feel like a cop-out answer at first glance, but it's the only one that works. This is because the field of RL is so broad and the techniques so diverse that the only viable option is to define reinforcement learning in terms of its problem. It's also important to note that when people refer to "reinforcement learning", they can either be referring to the RL problem, or the RL techniques used to solve that problem.

The RL problem looks like this:

![image](../assets/img/intro_to_rl/simple_agent_env.png)

This diagram shows the basic RL problem framework. In every iteration of this loop, we have an agent that does some action which presumably affects the environment. After that action, the agent receives two things from the environment: a reward, and an updated observation of the environment. The observation of the environment is not perfect; it's simply whatever information the agent can sense. This observation of the environment's current state is usually just called the "state".

When an agent commits an action, that action will result in some immediate reward and possibly a state change. The reward that the agent receives **depends on its action, its current state, and its next state**. This means that an action that works well in a certain state may be disastrous if tried in a different state. Furthermore, future states will affect future rewards, and there can be states that tend to give more rewards than others. Consequently, when deciding if an action is best, agents need to consider both the immediate reward of the action as well as how it affects the state. An action that provides good immediate reward but negatively affects the state may be worse than one that gives little immediate reward but results in a good state transition. The "best" action will yield the greatest total reward, which is the sum of immediate reward and expected future rewards. To sum up, the reinforcement learning problem is simply this: **what is the best action for each possible state?**

## Traits of RL algorithms
Although Sutton's definition of RL is concise and technically accurate, I want to list some traits that are usually used to distinguish RL algorithms from non-RL algorithms. The main trait that distinguishes RL is its unique problem format, but there are also some other things that do a pretty good job of highlighting the important ideas behind RL. Here are some of them:

1. **Delayed consequences:**  
  Reinforcement learning algorithms consider not just the immediate consequences of an action, but also the future consequences. I'll go over this more later.
2. **Learning through trial-and-error:**  
  This is pretty distinctive of RL, as there are not many other techniques that try stuff out at random. However, it's important to note that not all RL algorithms do this, and that there exist non-RL algorithms such as evolutional strategies which do learn by trial-and-error.

      *An aside for those who know about supervised learning:*  
      Algorithms that learn by **error** get often get mixed up with those that learn by **trial-and-error**. A good amount of supervised learning algorithms use some kind of error function (sometimes called a "cost function") to learn. One well-known example is the artificial neural network. However, neural nets, along with other error-based techniques, don't reduce their error by trying out new weights at random. They will instead use some derivative of the error, or another deterministic mathematical method to update their weights. Even when randomness is introduced to these algorithms, the randomness doesn't affect the learning process (e.g. in stochastic gradient descent, the neural nets still update weights the same way; it's only the input data that gets randomized). This "trying out random things" aspect is the key distinction between techniques that use **error** vs those that use **trial-and-error**.  
      I brought this up because this particular confusion caused research into actual trial-and-error algorithms to dry up in the 60's and 70's. In fact, some researchers and textbooks still make this mistake today.
3. **Exploration-exploitation tradeoff.**  
  This trait is a bit less self-explanatory than the other ones. Basically, agents will always attempt to maximize long-term reward, but they'll generally have incomplete information about the consequences of actions. However, an agent will know which action has historically yielded the best results. If an agent desires more complete information on an action, it has no choice but to try that action out (this is a good example of the trial-and-error). Hence the tradeoff: should an agent **exploit** the action that has worked best historically, or should the agent **explore** and get information on other actions which may prove to be even better? This is a very interesting problem that's unique to reinforcement learning. However, not all RL algorithms need to deal with this.

Now that you have a general idea of what reinforcement learning is, let's dive into the reinforcement learning problem format more deeply.

# Chapter 2: The Reinforcement Learning Problem
---
As stated in the introduction, I'll be going deeply into the reinforcement learning problem format so that hopefully you can start applying this problem format to real-world problems. Remember, it's only after you convert a problem into the RL format that you can solve it using RL algorithms.

## Basic Problem Framework
Let's go back to the reinforcement learning problem diagram, except now we'll look at it more formally.

![image](../assets/img/intro_to_rl/advanced_agent_env.png)

There are 5 variables we need to keep track of for now:
* $$t$$: This is called the **stage**, and keeps track of how many times the agent has interacted with the environment. At every stage, the agent will commit an action and receive a reward as well as an updated state. For example, if a bot needs to make a decision on where to go every second, then each second is a stage. You can think of this like a time-step if it's more intuitive that way, but just keep in mind that not all stage variables are based on time.
* $$a_t$$: This represents the **action** taken at stage $$t$$. The set of all possible actions is $$A$$, so every $$a_t$$ has to be taken from this set (in fancy math notation: $$a_{t} \in A$$)
* $$s_t$$: This is the **state** variable. It basically represents what the agent sees at stage $$t$$. This variable will change based on which action the agent takes, and how the states transition from one to another in a particular environment. Like actions, there is a finite set of possible states. So $$s_t$$ is equal to some $$s$$ where $$s \in S$$.
* $$r_t$$: This is the **reward**. Each stage, the agent will receive some kind of reward based on it's current action, current state, and subsequent state. As such, the reward can be written as a function of these three things:  

$$r_{t}=r(s_{t}, a_{t}, s_{t+1})
$$

An agent can usually be represented by the following sequence:

$$s_0 \rightarrow a_0 \rightarrow r_1 \rightarrow s_1 \rightarrow a_1 \rightarrow r_2
\rightarrow s_2 \rightarrow a_2 \rightarrow r_3 \rightarrow \ldots
$$

Where the agent sees a state, does an action, receives a reward, sees the next state, does the next action, etc...

Reinforcement learning problems can also be **episodic** or **continuous**. Episodic problems will have the stage and reward reset after a certain number of stages. The agent will try to maximize its reward in each "episode" given it's limited stages. Each episode is independent, which means that actions and states in one episode will not affect the next; think of this as a "hard reset". Continuous problems, on the other hand, will theoretically go on forever with $$t$$ going to infinity. **In this series, we'll mainly focus on episodic problems** since *a)* they're easier to comprehend and *b)* they're more common in real life.

*An aside for notation of $$t$$:*  
I still haven't yet found a unified view from the RL community for how to notate $$t$$. A lot of people call $$t$$ the "time step", but I find this inaccurate since $$t$$ might not represent time in certain problems. Some researchers also consider $$t$$ just another state variable that the agent needs to consider, which is technically correct. However, I felt that this interpretation could lead to confusion. In the end, I decided to use the word "stage", which is common in engineering and operations research.

## Formulating the RL problem as a Markov Decision Process
The basic problem formulation above is a bit abstract, and hard to work with. This is why pretty much everyone will formulate an RL problem as a Markov Decision Process (MDP). Here's a diagram of an example arbitrary MDP:

![image](../assets/img/intro_to_rl/mdp_example.png){:height="700px" width="700px"}
**Figure 1: Diagram for Example MDP**
{: style="text-align: center"}

This diagram might seem quite complex initially, but it uses basically the same variables as we discussed before. In our example MDP, there are 3 states (S1, S2, and S3) and two choices for action (A1, A2). These are represented by the blue and red circles respectively. At any stage, the agent will be in one of the three possible states, and from there, it will need to select from one of the two possible actions. Sorry for abusing the notation here; I realize now that S1 and $$s_1$$ can be confusing, but please try to remember that they are different things. S1 refers to one of the possible state values ($$s \in \{S1, S2, S3\}$$), while $$s_1$$ refers to the state that the agent is in during stage $$t=1$$.

The rewards are the orange arrows. For instance, if an agent starts in state S3, takes action A1, and lands back in S3, then the agent gets a reward of 8. Another way of stating this is $$r(S3, A1, S3)=8$$. Keep in mind that the order of the reward function inputs do matter, since $$r(S1,A2,S2)$$ is not equal to $$r(S2,A2,S1)$$ in general.

There's one variable that we didn't mention before, which is the state transition variable. This refers to the probability of moving from some state $$s$$ to some state $$s'$$ if the agent takes some action $$a$$. In more formal notation, this is $$P(s' \vert s,a)$$. For those that aren't familiar, the " $$\vert$$ " character just means "given", so that formula can be translated to "the probability of the next state being $$s'$$ given state $$s$$ and action $$a$$". State transitions are represented by the thin black arrows in the diagram. For instance, if an agent starts in state S1 and takes action A2, then its probability of landing in S2 is 0.3. This is shown in Figure 1 as $$P(S2 \vert A2, S1)=0.3$$

If at this point you don't have a clear grasp of the MDP variables, don't worry, since there will be plenty of examples to flesh them out further.

**Exercise 1: MDP basics**  
Use the MDP diagram above to answer the following questions:  
*a):* $$P(S3 \vert A1, S2)=?$$  
*b):* $$r(S3, A1, S1)=?$$  
*c):* $$P(S1 \vert A2, S1)=?$$  
**ANSWERS BELOW**  

---

**Solutions for Exercise 1**  
*a):* $$P(S3 \vert A1, S2)=0.2$$   
*b):* $$r(S3, A1, S1)=-2$$  
*c):* $$P(S1 \vert A2, S1)=0$$. It's convention to not draw the arrow at all if there's a 0 probability of a state transition.

We mentioned that agents can be represented with a sequence like this:

$$s_0 \rightarrow a_0 \rightarrow r_1 \rightarrow s_1 \rightarrow a_1 \rightarrow r_2
\rightarrow s_2 \rightarrow a_2 \rightarrow r_3 \rightarrow \ldots
$$

To give you a better idea of how this applies to Markov Decision Processes, here's an example of an agent moving through the first 3 stages of an MDP, ending at $$t=4$$. The yellow dot represents the agent. Pay particular attention to how each of the four variables change, and how those changes affect the movement of the agent.

The agent starts out every stage in a state node, and based on the action they choose, they will end up in the corresponding action node. Then, based on the state transition probabilities, they will end up some state. This gif only shows one simulated path that the agent could take. The movement of the agent could be completely different if this experiment was run again, due to the effect of random chance on the state transitions.

![Alt Text](../assets/img/intro_to_rl/mdp_movement.gif){:height="700px" width="2000px"}
**Figure 2: MDP Movement GIF**
{: style="text-align: center"}

Now that you hopefully understand MDP's a bit better, it's important to keep in mind that MDP's are still just a framework to make problems more workable. Therefore, this framework is no use to us if we don't know how to convert regular problems into MDP format. The next section will focus on exactly that.

## Translating word problems into the MDP format

In the last section, we showed that Markov Decision Processes are the best way to represent RL problems. Therefore, if we want to make a word problem into an RL problem, all we need to do is to redefine our word problem in terms of the MDP variables. Unfortunately, this is more art than science, so this section will mainly focus on providing examples and exercises.

Before diving into examples, here are some rules of thumb that may help:
* **Episode**: We'll only be looking at episodic problems, but even so, we'll still need to determine what constitutes an episode, and what indicates the end of an episode. This is actually easier than it seems; just find a point where the agent "resets", and that'll most likely indicate the end of an episode. E.g. for an agent that plays tic-tac-toe, each game would be an episode since everything resets at the end of a game.
* **Stage**: The stage can be difficult to determine. Remember that during each stage, an action needs to be made. Therefore, if you can figure out at what points an agent will need to do an action, then those points are the stage. E.g. an agent that plays tic-tac-toe needs to take an action each turn, which means that each turn is a stage.
* **Action**: The action is usually not hard to determine. This is simply the set of options that an agent can take during each stage. E.g. in tic-tac-toe, an agent can place a marker on any empty space on the board during a turn. Each empty space represents a possible action.
* **State**: The state tends to be pretty hard to determine. Think of the state as the all of the relevant information the agent is able to receive during each state. In tic-tac-toe, the state would usually be the position of every X, O, and blank space on the board. In this case, the state perfectly represents the environment, which means that the agent has perfect information. If we handicapped the agent and only allowed it to sense the top 2 rows of the board, then its state would be the position of every X, O, and blank space in the top 2 rows. Also, there can be multiple state variables, each tracking a different, independent type of information (though you usually want to combine state variables if possible, since more state variables is generally not good for computational reasons).
* **Reward**: The reward is unique in that we don't determine it from the problem. Instead, us engineers must design a reward function ourselves, which means that there can be multiple right answers. We need to define a numeric reward (can be zero or negative) for every $$(s', a, s)$$ combination. For the sake of demonstration, we'll just talk about reward in general terms and won't actually go through each possible state-action-state combination. When designing a reward, we need to be sure to only reward the agent for results that we want. E.g. we shouldn't reward a chess agent for taking pieces, since it's not always necessarily a good thing. Instead, we should reward it for winning games. Designing good reward functions tends very difficult in modern RL with complicated problems, and is a huge blocker that prevents RL from being useful in practice. In our simple example problems, however, it should be fairly easy.
* **State transitions**: Depending on which RL algorithm you use, knowing the state transition probabilities might be optional. The algorithm we focus on will require at least either knowing or predicting the state transitions, but a lot of RL algorithms **allow you to ignore the state transitions completely**. We'll go into more detail on this later. Anyways, determining the state transition probabilities in a general sense is usually pretty intuitive as long as you know what the states are. Finding exact numbers for them is harder, but for the sake of demonstration (and time) we won't do that for every example in this post.

Now that we know how to find our MDP variables, let's look at some examples!

**Example 1: Recycling Bot**
*This example is from Sutton and Bartok (2017)*

Imagine a busy startup with cans of Red Bull littering the floor. To remedy this dangerous tripping hazard, HR has decided to purchase a recycling bot. The robot's internal AI will be trained using an RL algorithm. The robot's AI only receives one piece of information, which comes from a sensor that provides the current battery level. There are three possible battery levels: high, low, and none. Each hour, the robot must take one of the following actions:
1. **Search**:  
The robot roams the house randomly (like a Roomba) in search of items to recycle. On average, it finds 3 cans per hour. However, searching has a 50% chance to lower the battery level.
2. **Wait**:
The robot waits for cans to be brought to it. However, most people prefer to litter their cans on the ground, so the bot usually only receives 1 can per hour if it waits. Waiting will not reduce battery level.
3. **Recharge**:
The robot can decide to recharge by itself without a human manually charging it. However, if it decides to use an hour to recharge, then it cannot recycle any cans during that hour.

At the end of every work day, the robot will automatically recharge itself, so the bot starts each day with high battery level. Also, the robot is active only during work hours.

**RL Problem Statement:** What actions should the robot take, given its battery level, to recycle the most cans possible each day?

Knowing all this info, we can determine all the MDP variables we need. I'll list them below, but try to think about them a bit before scrolling down.

* **Episode**: Each episode is a 'workday', since at the end of each day, the robot 'resets' by being charged automatically.
* **Stage**: From the info above we know that the robot needs to make a decision each hour. Therefore, each hour is a stage. Knowing that each day is an episode, we can also determine that there are 8 stages per episode (assuming an 8 hour workday).
* **Action**: During each stage, the robot can search, wait, or recharge. These are the 3 actions.
* **State**: We're told that the the robot's AI only gets one piece of info, which is battery level. Therefore, the battery level is the state. This highlights an important idea in that the state variable can be sometimes counterintuitive. One might be tempted to think that the environment is the office space, and that the state would be related to that somehow. However, the state is defined as the information that the agent is able to receive. Since our bot doesn't have any sensors that perceive the office space, the only state variable comes from the one sensor that we do have, which is the battery sensor. The three possible states are high battery, low battery, and no battery.
* **Reward**: Since there are multiple 'correct' reward functions, here's one example reward function that I designed which should work pretty well. Since we know how many cans on average get picked up from each action, we can make each can equal to a reward of 1. That means that if the robot recycles 3 cans in an hour, then it gets a reward of 3. We don't want the robot to run out of battery during the day, since one of the busy workers would need to waste time charging the bot manually. Therefore, we'll provide a reward of -10 if the robot's battery level goes to "none".
* **State transitions**: Since the battery level is the state, then the state transition would simply be the probability of the battery level changing. For example, searching would have a 50% chance of reducing the battery, so $$P(low \vert search, high)=0.5$$ and $$P(none \vert search, low)=0.5$$. This logic also applies to the other actions, with waiting keeping battery level the same, and recharging increasing battery level.

Now that we've defined all the variables we need, here's a diagram for our recycling bot.

![Alt Text](../assets/img/intro_to_rl/recycling_diag_1.png){:height="400px" width="900px"}
**Figure 3: Recycling Bot MDP**
{: style="text-align: center"}

Notice something interesting here: if the robot has low battery, does a search, and is unlucky enough to end up in the 'no battery' state, then it will **always** get back to a 'high battery' state and will **always** get a total reward of -7 (+3 from the cans it finds, and -10 from needing to be manually recharged). Therefore, we can redraw the diagram and get rid of the 'no battery' state.

![Alt Text](../assets/img/intro_to_rl/recycling_diag_2.png){:height="400px" width="900px"}
**Figure 4: Recycling Bot MDP (concise)**
{: style="text-align: center"}

And we're done formulating the problem as an MDP! Note that we haven't actually solved this RL problem, since we still don't know what the best actions for each state are. Don't worry about this though, I'll cover solving the MDP in part 2 of this tutorial. For now, focus on formulating word problems into MDP's. I'll give you one more example, then we'll do 3 exercise questions.

**Example 2: Blackjack**

![Alt Text](../assets/img/intro_to_rl/blackjack.png){:height="200px" width="300px"}

Suppose we're trying to train a blackjack agent using RL. In blackjack, you start out with 2 cards, and every 'turn' you can choose between getting another card or keeping your current cards. Each card has a numeric value (including face cards), and the goal is to get as close to 21 as possible without going over. Your only opponent is the dealer, and each 'hand' ends when either one of you goes over 21 or both you and the dealer stops getting more cards. The dealer cannot choose whether it wants more cards or not; if it has less than a total of 17, it must get more cards. If the dealer has more than 17, it must stop getting more cards. I want to keep this simple, so we'll ignore advanced rules such as splitting and doubling up. More detailed info on this extremely popular card game can be found on Google. Let's formulate this in terms of MDP variables.

* **Episode**: Each hand of blackjack is an episode, since that's when everything gets reset.
* **Stage**: The agent needs to make a decision after every card is dealt. Therefore, each 'card' is a stage. The first stage would be at the beginning, when the agent has 2 cards and needs to decide if it wants a third card. Then, if it takes the third card, the next stage would occur when the agent needs to decide whether or not it wants a fourth card, and so on.
* **Action**: During each stage, the agent can decide to hit (get another card) or stay (keep the current cards and end the hand).
* **State**: For this problem, at the very least, the agent needs to know it's own card total, and the dealer's total. We can create 2 state variables to represent these two totals. Additional states can be added to improve agent performance at the cost of computation time, such as a state variable that represents which cards are no longer in the deck (this is essentially card counting). The agent is considered to win if the dealer state goes over 21, or if the agent state is higher than the dealer state at the end of the hand. It's a draw if the agent state = dealer state.
* **Reward**: The benefit of 1v1 games is that the reward function is usually trivially easy to design. We can give the agent a reward of 1 if it wins the hand, 0 if there's a draw (a.k.a 'push'), and -1 if the agent loses a hand. In all other cases, give the agent a reward of zero, since the only states that matter are the win, draw, and lose states.
* **State transitions**: The state transitions are simply the probabilities of the card totals changing. For example, if the agent hits, then the state transition probabilities will depend on which cards remain in the deck. However, if the agent stays, then the agent card total will not change.

Hopefully that blackjack example helped you. Now we'll do 3 exercise questions.

**Exercise 2: Pole-cart**  

![Alt Text](../assets/img/intro_to_rl/pole_cart.png){:height="200px" width="500px"}

Here's a famous hello-world to reinforcement learning. Basically, we have a cart which can move on a rail. On this cart is a pole, which it needs to balance vertically by using side-to-side movement. The cart starts in the middle, and there's a limit to how far the cart can move on either side. Assume the cart can move at a constant speed left or right, or not move at all. Every time the pole falls down, it gets manually reset to a nearly vertical position and the cart gets put back in the middle. Then, the agent will try again. The RL problem is essentially this: how should the cart move such that the pole stays balanced as long as possible?

Please determine the following variables using the information given:
* **Episode**:  
* **Stage**:
* **Action**:  
* **State**: (hint: there is one state that's necessary, but there can be more)  
* **Reward**:  
* **State transitions**:

**ANSWERS BELOW**  

---

**Solutions for Exercise 2**  
* **Episode**: Each time the pole falls, it represents an episode, since the pole and cart both get reset.
* **Stage**: The agent will need to make decisions pretty much constantly, so the stage is just some time interval. This can be every millisecond, every second, every half second, etc., as long as you stick with one. Which time interval you use is up to you, but keep in mind that the smaller the time interval, the better the agent will tend to perform, at the cost of increased computing requirements.
* **Action**: During each stage, the cart can decide to use that time interval to move left, right, or stay in the same place.
* **State**: There is one state that is absolutely necessary, which is the angle of the pole. After all, we're trying to balance the pole, so we need to know how the pole is leaning in order to to counterbalance it with movement. We can also add states such as cart position (since the range of movement is limited), and angular velocity of the pole.
* **Reward**: Since we want to reward the agent for keeping the pole from falling for as long as possible, we can give it a reward of 1 for every stage that the pole doesn't fall. This reward total will reset each episode.
* **State transitions**: The state transitions will be different for each state variable. For pole angle, the state transition will be how much the pole angle changes should the cart move in a particular direction. I'm sure there's a formula for this somewhere in a classical mechanics textbook.

**Exercise 3: Knapsack Problem**  

![Alt Text](../assets/img/intro_to_rl/knapsack.png){:height="400px" width="700px"}

This is a well-known resource-allocation/combinatorial optimization problem which is commonly seen in fields such as computer science and applied math. Basically, imagine you're going camping and you have a backpack. You have a list of items that you would like to bring with you. Unfortunately, bag space is limited, so you can't bring all of them. Each item has a utility value, and will take up a certain amount of space. You'll need to go down this list of items, and for each item, decide if you want to pack it in your bag. To make the problem simpler, we'll assume that once you make a decision on an item, you can't go back (although it can be proven that the order in which you look at the items doesn't matter). The RL problem is this: how do you choose items such that you bring the most total utility given limited bag space?

Please determine the following variables using the information given:
* **Episode**:  
* **Stage**:
* **Action**:  
* **State**:
* **Reward**:  
* **State transitions**:

**ANSWERS BELOW**  

---

**Solutions for Exercise 3**  
* **Episode**: Each bag is an episode. Technically, you only have one bag, but you can repeat the problem by packing it, unpacking it, and repacking it multiple times, which results in multiple episodes. This way, the agent can try multiple times to get the best combination of items.
* **Stage**: As you go down the list of items, each item is a stage. For example, stage 1 is item 1, stage 2 is item 2, etc.
* **Action**: During each stage (item), you can choose between packing it and not packing it.
* **State**: The state is the amount of space remaining. It is **not** the total utility in the bag, since that shouldn't affect your future decisions. An agent should be trying to maximize utility no matter what, so the utility that it currently has shouldn't be a factor. However, the agent will need to consider the space remaining, since that will determine how liberally it can accept items (e.g. if the agent is reaching the end of the list but still has a ton of space, it should accept pretty much every item)
* **Reward**: The reward should be the total utility at the end of each episode. We're trying to maximize utility, so we can just make it the reward function directly.
* **State transitions**: Since the state is the amount of space remaining, the state transition is how much the remaining space decreases if an item is packed (this is given by the space requirement of each item). If an item is not packed, then no space is used up so the state stays the same. Unlike some of our previous examples, the state transitions here are **deterministic**, which means that there is no random chance in this particular RL problem.


**Exercise 4: Chess**  

![Alt Text](../assets/img/intro_to_rl/chess.png){:height="200px" width="200px"}

If you don't know the rules for chess, you can skip this one, or try to learn them through Google. Otherwise, imagine you're creating an RL chess agent.

Please determine the following variables using your knowledge of chess:
* **Episode**:  
* **Stage**:
* **Action**:  
* **State**:
* **Reward**:  
* **State transitions**:

**ANSWERS BELOW**  

---

**Solutions for Exercise 4**  
* **Episode**: Each game is an episode, since everything resets from game to game.
* **Stage**: Every time it's the agent's turn, it needs to decide on an action. Therefore, each of the agent's turns is a stage.
* **Action**: The action is a chess move.
* **State**: The state is the position of all the pieces right before the agent makes its move.
* **Reward**: As in blackjack, the reward should be 1 for a win, 0 for a draw, and -1 for a loss. One common misconception is to reward the agent for taking pieces. This is usually a bad idea since it can encourage the agent to make suboptimal moves in order to take a piece. Only reward agents for things that are always desirable; taking a piece is not always desirable.
* **State transitions**: The state transition is how the board changes between the agent's moves. This is technically unknown since we can't know how an opponent will behave. However, we don't need to know the state transitions exactly. As long as we have some idea of how the states change (e.g. which moves are legal for the opponent), then we can predict the state transitions and use those predictions to train the agent. Even if the predictions are bad, the agent should still continue to improve.

# Conclusion

This brings us to the end of part 1. Hopefully you now have more knowledge on:
* The general ideas underpinning reinforcement learning
* Translating word problems into RL problems

In this post, we put a lot of focus on the RL problem itself, but we didn't spend much time solving that problem. For example, we still don't know the best way to play blackjack, or the best moves for chess, or the best way to decide which items to bring on a camping trip. Worry not, however, as we'll be solving the MDP in part 2 of this tutorial series.

See you then!

---

If you notice any mistakes, have any feedback, or would just like to discuss this topic, send me a message!  

**Part 2 is currently being written; stay tuned!**
