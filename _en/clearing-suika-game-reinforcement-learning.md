---
title: "Clearing the Suika Game with Reinforcement Learning"
permalink: /en/clearing-suika-game-reinforcement-learning/
date: 2026-09-28T00:00:00+09:00
categories:
  - work
tags:
  - AI
  - reinforcement learning
  - games
ref: clearing-suika-game-reinforcement-learning
---

Remember the Suika Game, the watermelon game? While watching Korean streamer Chimchakman play it, I thought it would be fun to train it with reinforcement learning. It happened to be the Chuseok holiday, so I decided to build it myself with Claude, partly as a way to study reinforcement learning.

To do reinforcement learning, you first need an environment. So I started by building the game that would serve as the environment. I used Pymunk as the physics engine and Pygame for rendering.

The rules are simple. The agent picks one of 32 slots to drop a fruit into. It waits until the fruit settles, then drops the next one. When two identical fruits touch, they merge into the next larger fruit, and the game ends when fruit crosses the line. In the end, all the agent learns is which of the 32 slots to drop the fruit into.

<figure>
<img src="/assets/2026-09-28-suika-reinforcement-learning/game-screen.png" alt="The Suika game we built. Melon, pineapple, apples and other fruit are stacked at the bottom of the box, with a dekopon waiting at the top to be dropped">
<figcaption markdown="span">The Suika game I built. The red dashed line is the limit line, and the right side shows the score, the next fruit, and the order of the 11 fruits from cherry to watermelon.</figcaption>
</figure>

The first algorithm I used was PPO. PPO stands for Proximal Policy Optimization, a widely used policy gradient algorithm in reinforcement learning. Policy gradient methods directly update the policy's parameters in the direction that increases expected reward.

I shrank the game screen and fed it to a convolution-based policy network, giving a reward every time fruits merged. Dropping fruit at random scores about 1,500 points, and the trained agent also stayed around 1,500. A failure.

<figure>
<img src="/assets/2026-09-28-suika-reinforcement-learning/observation.png" alt="The same scene as an RGB image shrunk to 72×96, as an image with fruit information split into channels, and each of those three channels shown in grayscale">
<figcaption markdown="span">What the policy network sees. From the left: the screen shrunk to 0.2× (72×96) in RGB, the observation I switched to later, and that observation's three channels. The channels mark the fruit level (brightness), fruit of the same kind as the one about to drop, and fruit of the same kind as the next one.</figcaption>
</figure>

I changed the hyperparameters, changed the observations, and changed the policy network's architecture. None of it made much difference. If anything, the simple strategy of dropping every fruit in the middle held up pretty well.

So I changed approach. I built a kind of 1-step greedy strategy that, every turn, drops the fruit into each of the 32 positions and picks the one with the highest immediate score. It went up to 4,400 points.

Next, I trained a neural network to copy the roughly 40,000 moves this strategy produced. That's Behavioral Cloning (BC). Even without reinforcement learning, it averaged about 3,390 points.

This is where a problem showed up. Over the course of a game, about half of the 32 positions lead to nearly the same outcome no matter which you choose. The moves that really matter come up only occasionally, and their signal is easily buried under the big scores from chain merges. In other words, not every move mattered equally.

I put PPO back on top of BC, but it had little effect. So I redesigned the reward. I took off 300 points when fruit crossed the line and added a little extra reward as the pile got lower.

That raised the average score by 538 points over BC. Comparing the play, it hadn't learned to merge fruit better; it had learned to survive longer. Even when fruit piled up near the line, it merged the right fruit to bring the pile down and escaped the crisis. That's behavior hard to learn from BC, which copied a greedy strategy that only looks at the immediate score one move ahead.

The biggest gain came from outside of training. A trained policy network comes with a value function attached, which predicts how good the current state is. Armed with that value function, every turn I actually dropped the fruit into each of the 32 positions, then added the immediate score to the predicted future value and picked the best spot. Adding the future value as is actually lowered the score. The 32 candidates were so similar to each other that the move that looked best was usually the one where the prediction error spiked the most. Giving the future value a weight of just 0.05 brought the average to 4,953 points. That was nearly 1,000 points higher than the policy playing on its own.

<figure>
<img src="/assets/2026-09-28-suika-reinforcement-learning/search-candidates.png" alt="The results of dropping a dekopon into eight different slots from the same state. Five slots score 0, slot 28 scores 15, slots 30 and 32 score 24, and slot 30 is outlined in red">
<figcaption markdown="span">What search does in one turn. Eight of the results from copying the same state and dropping the dekopon into different slots. The numbers are the slot number and the score gained right away. Slots 30 and 32 had the same immediate score, and slot 30 won on the future value predicted by the value function.</figcaption>
</figure>

Then why not teach the policy the moves the search made? I had the search agent play 550,000 moves and trained the policy to copy them. I tried doubling the screen resolution, splitting fruit types into separate channels, and even replacing the screen entirely with the exact coordinates of each fruit fed into a Transformer. Everything stalled around 4,000 points. The rate of matching the move the search chose never got past 15%. The policy never learned where a fruit would roll and what it would hit.

<figure>
<img src="/assets/2026-09-28-suika-reinforcement-learning/scores.svg" alt="Bar chart of mean score per agent. Random 1,463, PPO from scratch 1,648, BC 3,390, PPO on BC with reward shaping 3,913, policy imitating search 4,050, 1-step greedy 4,391, search 4,953">
<figcaption markdown="span">Mean score per agent. Each played the same 100 games with the same fruit order. The policies imitating search were nearly identical with a convolution backbone (4,058) and a Transformer (4,042), so they're shown as their average.</figcaption>
</figure>

The reason lies in the physics. Whether a fruit touches or misses by a few pixels, and whether a chain merge follows, is sensitive to initial conditions. Getting it right in one shot from a single frame is a very hard problem. The search policy doesn't try to get it right. It rolls things out and checks every time, and maximizes the score. In the end, what made the difference wasn't the model's architecture, but whether it could ask the simulator before deciding.

<figure>
<video src="/assets/2026-09-28-suika-reinforcement-learning/agents-compare.mp4" poster="/assets/2026-09-28-suika-reinforcement-learning/agents-compare-poster.jpg" controls muted playsinline preload="metadata"></video>
<figcaption markdown="span">Six agents playing one game with the same fruit order (6× speed). From the top left: random, PPO trained from scratch, BC, PPO trained on top of BC with the redesigned reward, a Transformer imitating search, and search. When a game ends, the fruit turns gray from the bottom up. Out of 100 games, I picked one where search made three watermelons. Search scored 10,250 here, twice its average, and apart from random and PPO swapping places, the rest follow the order of the average scores.</figcaption>
</figure>

I learned two things from this little project. That the strategy an agent learns changes completely depending on what reward you give it. And that if you have a model of the world, it's better to pull it out when you need it than to cram it into your head. Maybe both are obvious.

Anyway, it was fun. Try implementing reinforcement learning yourself sometime when you're bored.
