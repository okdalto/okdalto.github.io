---
title: "What People Get Wrong About the Word \"Artificial Intelligence\""
date: 2024-11-18T11:34:30+09:00
categories:
  - thoughts
tags:
  - artificial intelligence
  - aura
ref: misunderstanding-the-word-ai
---
### This is a piece I originally posted on Facebook back in 2020.

The dramatic leap in the performance of artificial neural networks has made "artificial intelligence" a phrase you can now hear everywhere, even if you're not an engineer. But people only hear the words often; among the general public, AI seems to be understood as something rather different from what it actually is. Clearing up that kind of misunderstanding ought to be the job of experts, yet lately I get the sense that there are people who deliberately exploit and inflate it for particular ends.

To borrow Walter Benjamin's vocabulary, the phrase "artificial intelligence" carries an "aura." It gives the impression of a being on par with—or even superior to—a human, one that thinks and acts with intent. In a similar way, words like "neural network," "learning," and "prediction" easily create the feeling that the network thinks like a human and acts with a personality and intentions of its own. But is that really the case?

Well, I suspect people feel that way precisely because they don't really know how deep-learning-based AI works. An actual network is, at its core, just layers of matrix multiplications and additions stacked on top of one another, with activation functions sprinkled in to provide nonlinearity. During training, the network uses differentiation to adjust the numbers it multiplies by and the numbers it adds; once training is finished, the network operates using nothing but these operations. Of course, depending on the network, any number of other complex operations can be added, but the most basic form of a network can have a structure this simple. Once you've heard this much, a network looks less like a person and more like a calculator. The many networks I've trained never tried to kill me or acted on some hidden agenda—they simply, diligently computed outputs from inputs.

## On the Intelligence of Networks

So should we conclude that networks have no intelligence? Not that either. In my view, networks do have intelligence. But here's the thing you mustn't misunderstand: the bar I set for intelligence is extremely low. In Rolf Pfeifer and Josh Bongard's **"How the Body Shapes the Way We Think,"** there's an anecdote about the "Swiss robot."

The Swiss robot is a very simple machine with wheels and two antennae. It runs on a simple algorithm: it moves forward, and whenever either antenna bumps into anything, it turns and moves the other way. Yet if you place this robot in a room cluttered with boxes and let time pass, it gradually starts clustering the boxes together, appearing to behave intelligently. If you then slightly change the angle of the robot's antennae, this phenomenon can no longer be observed.

**So where does the intelligence reside?**  
Is it in the angle of the antennae? Or does the system as a whole possess the intelligence?

In a similar vein, we can bring up John Searle's famous thought experiment, the **"Chinese Room."**  
There's a person in a room who doesn't understand a word of Chinese. This person's job is this: when someone comes and hands over a note with a question written in Chinese, the person looks at the shapes of the Chinese characters and produces the predetermined "correct" Chinese response. Can we then say that this person understands Chinese?

I think intelligence can be found in both cases. The reason is simple: the system appears to behave intelligently. After all, we don't even clearly know what intelligence is or where it comes from, and it's impossible to confirm whether another person has the same kind of inner self that I do.

## The Difference Between Intelligence and Personhood

Personally, I believe—much as Marvin Minsky argued in **`The Society of Mind`**—that human intelligent behavior, too, is carried out mechanically through the interaction of simple modules. Each module, like a network, knows nothing whatsoever about the higher-level concept, and yet the whole, assembled together, constitutes a single intelligence.

This view seems to contradict what I said earlier about networks having no personhood, so let me put it plainly: I think networks do have intelligence, but to such a faint degree that it's hard to claim any meaningful difference from a calculator, and we're still a very long way from talking about personhood.

There's one more point worth making, though: even if something doesn't think like a human, it can still be more than enough to replace a human. The work we do is, in many cases, less complex than we like to think. For example, even without thinking like a human, a system is perfectly capable of classifying digits, and the simpler and more computerizable a task is, the more easily it can be replaced by narrow AI.

## Artificial Intelligence and Art

For all I've said, I remain very interested in art that uses artificial intelligence stripped of any personhood-aura. It's not that I'm interested in its personhood or its intelligence—rather, I'm interested in the styles it produces. GAN-generated images are especially intriguing to me: the very parts an engineer's eye would dismiss as artifacts (those error-like blemishes that give an unintended, artificial feel) strike me instead as fun and as a fresh kind of style. Maybe I feel this way because I studied design rather than fine art.

On top of this, I'm also very interested in visualizing networks in an artistic context. I've worked on showing the degree of a network's activations, or visualizing the training process to reveal what's actually happening—work meant to take apart the idea of artificial intelligence and convey its true nature. I believe such attempts help dispel the pervasive misunderstandings and narrow the gap between experts and non-experts.
