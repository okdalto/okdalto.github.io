---
title: "Art and Virus"
permalink: /en/art-and-virus/
date: 2026-07-01T00:00:00+09:00
categories:
  - thoughts
tags:
  - art
  - information-theory
  - artificial-intelligence
ref: art-as-virus
---

Humans take in the world through their sense organs, but sensation is not the same as perception of the world itself. Inside us there is a model that interprets the world. We recognize a face as a face, classify a landscape as a landscape, infer emotion from tone of voice, and read intent from an image. Most of what we call seeing is in fact the result of prediction, correction, and compression carried out by our internal model.

When we look at someone's face, we do not compute the pixels of the eyes, nose, and mouth one by one. We use a model of the face we already know. We read one expression as sadness, one posture as fatigue, one palette as warmth, one composition as stability. This is not because the world is originally shaped that way, but because we have learned to compress the world that way.

From this view, understanding is not simply a state of knowing a great deal of information. Understanding is closer to the ability to compress a complex world into shorter rules. To find a pattern in what looks disordered, to bind together events that seem far apart into a single structure, to keep only the important differences within a flood of sensory data. To understand the world is, in the end, to rewrite it into a form we can handle.

From the perspective of information theory, art is an interpolation between information and noise. What does this mean?

In information theory, information is what reduces uncertainty. When you learn that some event has occurred, if the number of possible worlds you had been imagining shrinks, then that much information has been conveyed. Conversely, a fact you already believed to be nearly certain carries little information even when it is newly announced. This is why "the sun will rise tomorrow" is hardly surprising.

Written as a formula, the information carried by an event $x$ that occurs with probability $p(x)$ is defined as:

$$I(x) = -\log p(x)$$

A near-certain event ($p \to 1$) gives $I(x) \to 0$, carrying almost no information, while the rarer an event is ($p \to 0$), the greater its information. And in information theory, entropy is defined as the average uncertainty in a state where the outcome is not yet known — that is, the expected value of this information:

$$H(X) = -\sum_{x} p(x)\log p(x) = \mathbb{E}[-\log p(X)]$$

Entropy is at its maximum when all outcomes are equally plausible, and approaches zero when the outcome is all but fixed to a single possibility.

A work of art also serves as a medium that conveys the artist's intent or emotion. And a work of art often carries high interpretive uncertainty. But that uncertainty is different from the uncertainty of pure randomness. The uncertainty of a good work is not scattered noise; it arises around a structure the viewer can grasp.

I think the criterion that best distinguishes a work of art from a mathematical formula is the way information is conveyed. In a work of art, intent or emotion is (for the most part) not conveyed directly. Artists mix their intent or emotion with noise in various ways, obstructing the precise transmission of information and allowing the viewer to interpret the work at will. This can be read as an act of increasing interpretive uncertainty. Because the information is not conveyed clearly, a diversity of readings by the viewer becomes possible. If a mathematical formula compresses information in the direction of minimizing misunderstanding as much as possible, a work of art instead compresses information in a way that leaves behind a certain amount of misreading, delay, and ambiguity.

Yet it is not the case that anything with a lot of uncertainty becomes art. A completely random noise image is hard to predict. It is also hard to compress. In that sense its entropy can be high. But we generally do not receive it as a deep artistic experience, because it lacks any interpretable structure. Conversely, an image that is too obvious has structure but low information content, so we can process it easily within the model we already hold.

Art occurs between these two. It is neither so unfamiliar that we can grasp nothing, nor so familiar that it fails to convey any information. A good work is familiar enough to connect to the model the viewer already has, yet unfamiliar enough to make that model fail.

And this failure sometimes brings about a change in the model we hold. Some works are not simply understood and finished. Rather, they keep working longer after they are understood. Once you have grasped their structure, the world afterward is seen again through that structure. In other words, they revise our model for interpreting the world.

Ted Chiang's short story "Understand" shows this problem in an extreme form. In the story, the protagonist's intelligence rises sharply after an accident. The world appears at a different resolution, and human behavior, language, and desire no longer manifest the way they used to.

But in "Understand," understanding is not a safe ability. The protagonist models systems — including himself — so precisely that he comes to understand an image, created by another intelligent being, designed to collapse a system, and meets his end. He met his end precisely because he was intelligent enough to understand that image.

Art, too, works in a similar way, at a layer deeper than the message. An ordinary message conveys some content. But strong art does not stop at conveying content. It changes the rules by which subsequent content is interpreted. Rather than telling the viewer to think this way, a work intervenes in the very way the viewer thinks.

Interpreting this phenomenon from the perspective of information theory, we could say that art disturbs the viewer's prior distribution and, by generating certain prediction errors, changes the way they interpret the world.

If we call the belief the viewer holds before seeing the work the prior $p(\theta)$, and the belief after taking in the work as an observation $x$ the posterior $p(\theta \mid x)$, the relation between them can be written with Bayes' theorem:

$$p(\theta \mid x) = \frac{p(x \mid \theta)\, p(\theta)}{p(x)}$$

How much a work has changed the viewer's model — the amount of updating from prior to posterior — can be measured by the Kullback–Leibler divergence between the two distributions:

$$D_{\mathrm{KL}}\!\left(p(\theta \mid x)\,\big\|\,p(\theta)\right) = \sum_{\theta} p(\theta \mid x)\log\frac{p(\theta \mid x)}{p(\theta)}$$

For an overly obvious work, $p(x\mid\theta)$ fits the prior well, so this value is close to zero (the model does not change); for overly unfamiliar noise, the prediction error is large but there is no structure to grasp, so no updating takes place. Art occurs in the region where this divergence is meaningfully large yet still within what the viewer can bear.

A prior is the expectation we already hold before we look at the world. A person will look like this. Love will be an emotion like this. A landscape should be beautiful like this. An image should be clean. Errors should be removed. A story should end in a certain way.

Good art attacks precisely these expectations. But it does not destroy them completely. Leaving behind just enough familiarity to remain accessible, it makes the predictions go astray from within. And that deviation induces the viewer's internal model to revise itself. Just as a virus infiltrates a host's system and executes its own code, a work operates within the viewer's internal model. And once a work begins to run, it goes on to change even the world outside the work.

The power of art, therefore, does not lie simply in beauty. The power of art lies in the fact that it remains even after viewing and intervenes in the way we interpret the world. Some works grow stronger after time passes than they were right after being seen. A scene you did not understand at first connects one day with another experience, and in that moment the work runs again. Art is less a closed file than a process running in the background.

From this view, appreciation is not safe consumption. We think we are looking at the work, but at the same time the work is tuning us. We interpret the work, but the work revises our interpreting apparatus. The moment we understand a work, it becomes, within us, a new rule for understanding the world again.

Art enters the human internal system and changes the rules by which the world is interpreted. So art is not decoration but infiltration, not expression but infection, and appreciation can be seen as a slow hack.
