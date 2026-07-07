---
title: "What It Means to Predict the Next Word"
permalink: /en/what-it-means-to-predict-the-next-word/
date: 2026-07-07T00:00:00+09:00
categories:
  - thoughts
tags:
  - ai
  - language-models
  - prediction
ref: what-it-means-to-predict-the-next-word
---

A recent paper makes an interesting claim.[^paper] The training objective of an LLM is to predict the probability distribution of a single next word. Looking at this objective alone, the model feels like a device that sees only one step ahead. An LLM is a machine that looks at the sentence given so far and picks the word likely to come next. Many of the misunderstandings and criticisms surrounding LLMs start from this intuition. How could a model that merely guesses the next word plan, reason, and understand a whole context? The question arises almost inevitably.

From the standpoint of probability distributions, however, this intuition shifts a little. The probability distribution over an entire sentence can always be expressed as a product of next-token conditional distributions. The distribution over whole sentences and the distributions over next tokens are not completely separate things. To preserve the plausibility of the sentence as a whole, the next-token distribution at each step must carry more than a merely local judgment. This is the point the authors dig into: the next-token logits may not simply be values expressing how plausible the next word is, but a compressed interface for maintaining the distribution over the entire sequence.

This sounds a little strange. How can predicting the next word be connected to beholding the whole sentence? Intuitively, it seems the meaning and completeness of a sentence can only be evaluated after reading it to the end. Which sentence is good, which answer is appropriate, which chain of reasoning is correct — these seem knowable only by going all the way to the end. So a model that evaluates whole sentences and a model that predicts words one at a time look like entirely different kinds of machine.

The authors compare these two by casting them as an autoregressive model and an energy-based model. The autoregressive model is the language model we all know: it looks at the preceding context, predicts the next token, and repeats the process to generate a sentence. The energy-based model, by contrast, assigns a single score to an entire sentence. It evaluates at the sequence level whether a whole answer is good or bad, plausible or less plausible. The former advances one token at a time along the flow of time; the latter looks like a model that beholds the completed whole at once.

Probabilistically, however, any joint distribution can be decomposed into a product of conditional distributions. This is a simple but powerful fact. Handling the probability of a whole sentence directly, and composing the probability of a whole sentence by multiplying next-token probabilities, are in principle two ways of expressing the same object. The paper goes further, claiming that in function space there exists an explicit one-to-one correspondence between autoregressive models and energy-based models. That is, under ideal conditions, a model that assigns energy to whole sentences can be converted into a next-token prediction model, and a next-token prediction model can likewise be interpreted as an energy-based model defining a distribution over whole sentences.

What matters is what lives inside the next-token logits. We usually understand a logit as something like a score for the next word. But from this paper's perspective, a logit is not merely the score of the word that feels natural right now. It must reflect, to some degree, the possibilities of every future continuation that could follow if this word is chosen. In other words, folded into the score of a single token is the soft value of the future sentences that will unfold after it.

At this point the meaning of next-token prediction goes beyond intuition. The model really does output only one token at a time. But the value needed to choose that one token is, ideally, a compressed evaluation of all the paths that could come after it. When you play a stone in Go, that move is not merely a spot that looks good on the current board. A good move implies the countless variations that will unfold afterward. Likewise, an LLM's next token is not simply one next word — it can implicate the entire branching of future sentences that choosing that word opens up.

This changes the very way we look at language. We read language from left to right. A word appears, then the next word, then the one after that. Language seems to flow linearly. But meaning does not arise only from left to right. A single word is at once the consequence of past context and a compression of future sentences not yet written.

For example, the moment you write "therefore," a future opens up in the direction of summarizing or concluding the preceding discussion. The moment you write "but," a future of reversal, opposition, and revision opens. Write "perhaps," and a space of uncertainty opens; write "necessarily," and a space of strong determination opens. A single word is not just a single word — it operates like an operator that changes the very topology of future sentences.

The same goes for sentences. A sentence is not the sum of the words that have already appeared. A sentence is the tension between the words that have appeared and the words that have not yet appeared. A good sentence is not merely one in which the words so far are beautifully arranged, but one that organizes well the space of possibilities that can follow. Style, perhaps, is not a habit of word choice but a way of compressing and branching future possibility.

This may also be why the first sentence of a novel is so powerful. A first sentence is not strong because it carries a great deal of information. It creates the distribution of the entire world that will unfold afterward. The power of a philosophical sentence is similar. Introduce a single concept, and the paths of thought possible from then on change.

This perspective does not stay confined to language. Generalize just a little, and it makes us see prediction on all sequential data differently. Language is only one kind of sequential data. Film, music, gesture, stock prices, climate, biosignals, sensor data, gameplay, logs of human behavior — all of these unfold over time. We read such data in temporal order. A sentence flows from word to word, a film from frame to frame, music from note to note, a time series from step to step.

From the standpoint of prediction, the next moment is not simply the next moment. The next moment is the cross-section where the whole future trajectory appears under present conditions. A good predictive model is not a model that looks one step ahead, but a model that compresses the structure of all possible paths into the distribution one step ahead.

Consider predicting film frames. Predicting the next frame sounds a bit silly, because it seems like the task of guessing the pixels just 1/24 of a second later. But good next-frame prediction is not just getting small object motions right. If a person is opening a door in the current frame, the next frame is connected to that person's intention, the laws of physics, the movement of the camera, the lighting, the context, and the entire range of actions that could follow. A single next frame is a very short future, yet it reveals the link that can connect it to the entire scene.

Music is the same. Predicting the next note is not merely picking a sound likely to follow the previous one. A good next note carries within it the tension of the preceding melody, the harmonic progression, the rhythmic structure, genre conventions, the listener's expectations, and the direction of a destination not yet reached. The next note is a stepping stone that reveals the possibilities of the entire piece.

Time-series data is no different. Predicting the next moment's temperature, stock price, heart rate, or traffic is, on the surface, a matter of getting one value $x_{t+1}$ right. But to get that value right, you have to learn the long-range structure of the data-generating process. Seasonality, cycles, trends, shocks, recoveries, lag effects, feedback loops — all of these are reflected in the probability distribution of that one next value. So one-step prediction is really not the problem of guessing the next moment, but the problem of compressively estimating the state of the entire dynamical system.

Seen this way, prediction is not mere guessing. Prediction is the compression of a world model. The model does not explicitly unroll and compute every possible future — in most cases that is impossible. Instead, it learns the recurring structure in past data and compresses the distribution of possible futures, under present conditions, into a small output. The next word, the next frame, the next note, the next value: all are the surface of that compression.

Of course, theoretical possibility and actual capability are different things. That a perfect correspondence exists in function space does not mean that an actual Transformer or time-series model learns that correspondence perfectly. Real models have finite parameters, are trained on limited data, and their optimization is imperfect. So the question is not whether next-step prediction is fundamentally myopic, but rather how well an actual model can compress the value of the future into the distribution of the next step.

This difference matters. The former is a claim about structural impossibility; the latter is a matter of learning and representational capacity. If next-token prediction or next-frame prediction were myopic in principle, then the reasoning and planning abilities of such models would be close to illusions from the start. But if next-step prediction is one way of representing the distribution over whole sequences, the limits lie elsewhere. How can sufficiently complex future value be folded into the small output of the present? What data and what training processes make such compression possible? And when does that compression fail?

This perspective also changes the way we look at data. Sequential data is not simply an array of values laid out in time. It is a probability distribution over the space of possible paths. The actual sequence we observe is one path realized from that distribution. A sentence, a film scene, a musical phrase, a change in heart rate, a movement of the market — each is one trajectory among countless possibilities. And to predict the next moment is not to look only at the part of the trajectory just ahead, but to estimate the entire field of possibility not yet realized.

The next moment, then, is small but never simple. Folded into the next word is the whole sentence not yet revealed; folded into the next frame is the whole scene not yet filmed; folded into the next note is the whole piece not yet played. Folded into the next value is the state of a system not yet revealed.

We experience sequences as the flow of time. But what makes that flow possible is the vast distribution lying behind every moment. A one-step-ahead prediction does not point only one step ahead. It briefly reveals the shadow of the entire future on the surface of the present.

So the statement that an LLM predicts the next word sounds a little different than it used to. It is not merely a word-guessing game. It is the act of compressing the space of possible sentences and drawing it out, at each moment, as a single distribution. And if we extend this thought beyond language, prediction turns out to be a way of handling every sequential world.

We say the model predicts the next moment, but what the model actually learns is not the next moment itself. What the model learns is the structure of the world folded into the next moment. Beneath the surface of the next moment hides the entire context not yet unfolded.

[^paper]: ["Autoregressive Language Models are Secretly Energy-Based Models"](https://arxiv.org/abs/2512.15605), arXiv:2512.15605 (2025).
