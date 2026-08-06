---
title: "On Learning and What Does Not Change"
permalink: /en/learning-and-what-does-not-change/
date: 2026-08-06T00:00:00+09:00
categories:
  - thoughts
tags:
  - AI
  - learning
  - education
ref: learning-and-what-does-not-change
---

Two things I've felt while giving guest lectures here and there recently:

1. To reach a certain level, you have no choice but to clear the barriers that come with it.
2. Hard things can only be learned the hard way.

Since AI arrived, studying can look like it has become completely pointless. What I actually feel is somewhat different.

One student in my OpenGL course said they wanted to interpolate between two images. I couldn't properly understand the request. It was far too abstract.

What makes it abstract? Let's take an example. In which space is this interpolation supposed to happen?

Pixel space? RGB space? HSV or HSL space? A parameter space of the image's position, scale, and rotation? An optical flow space that accounts for correspondences between pixels? The Fourier domain? A manifold?

And what path do we take within that space? Do we follow a straight line between the two points? A geodesic? Something like a Bezier curve?

At what speed does this position change? Do all components move at the same speed? Does the speed differ from moment to moment? If so, how?

Depending on which of these you choose, the visual result turns out completely different. So unless you understand these concepts and can also visualize them in your mind, it is hard to produce good results even with the same tools. You are left to rely on luck.

In the age of agents, to make something you have to be able to explain it. And a good explanation ultimately requires an understanding of the problem itself.

![Your professor's aura of logical distortion — you understand while the professor is nearby, and stop understanding the moment they leave the room](/assets/2026-08-06-learning-and-what-does-not-change/aura-of-logical-distortion.jpg)

*Image credit: [PHD Comics](https://phdcomics.com), Jorge Cham*

There is a meme about the "aura of understanding" that surrounds a professor. When the professor comes over and explains something, in that moment you feel like you understand it completely. But once the professor walks away, you can no longer understand the very same concept. It was the illusion of understanding. Something similar happens when using AI. I'm guilty of it myself these days: when I run into a difficult concept, I ask the AI to explain it at an elementary-school level, skim the answer, and move on.

I think that fully understanding a concept means being able to apply it to other problems. But after learning this carelessly, far from applying it, I find myself asking the AI again the next time I face the exact same problem.

The brain has a property called neuroplasticity. Thanks to it, we can rewire our brain's circuits — which is what learning is. There are many ways to change those circuits, but one that has stayed with me is stress.

This stress does not mean pushing yourself to your limits. It is closer to an appropriate cognitive tension: the discomfort of facing what you don't know, the time spent struggling to come up with an answer on your own, the process of retracing methods that failed. We begin to rewire our circuits not while listening to an explanation, but in the moments we struggle to think it through again without one.

In that sense, AI does not tear down the barriers to learning.

It used to be that finding information, memorizing syntax, and figuring out how to implement things were the big barriers. Now AI handles much of that quickly. In exchange, other abilities have become more important: precisely defining what to build, judging whether a given answer is sound, choosing the right option among many, and finding the cause when the result is wrong.

The request to interpolate two images is the same. Generating the code has become easier than before, but deciding what to interpolate, and in which space, is still hard. With more options available, it may even have become harder. AI can implement RGB interpolation, optical flow, or a transformation in the Fourier domain. But which method suits the present purpose is a judgment that must be made by someone who understands that purpose.

The problem is that AI gives the feeling of having acquired this judgment too. Because its explanations are smooth and its answers plausible, we mistake the experience of having read something for the experience of having understood it. Just as you seem to know everything while listening to the professor's explanation, and can do nothing again the moment the professor leaves.
In the end, to reach a certain level, you must clear the barriers that match it. And truly hard things can still only be learned the hard way.
