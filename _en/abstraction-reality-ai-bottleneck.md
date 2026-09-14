---
title: "Abstraction and Reality: The Bottleneck of the AI Era"
permalink: /en/abstraction-reality-ai-bottleneck/
date: 2026-09-14T00:00:00+09:00
categories:
  - thoughts
tags:
  - AI
  - mathematics
  - science
ref: abstraction-reality-ai-bottleneck
---

Recently, [OpenAI announced that it had found an answer to the Navier–Stokes Millennium Prize Problem](https://openai.com/index/navier-stokes-solution/). More precisely, it did not derive a general solution. It offered an answer to the long-standing question of whether a solution that begins with smooth initial conditions always remains smooth, showing instead that a singularity can form. Navier–Stokes is about equations that describe fluids in the real world, but it is also a highly abstract mathematical problem. The problem is clearly defined, what must be proved is fixed, and whether a submitted proof is right or wrong can be verified.

<figure>
<img src="/assets/2026-09-14-abstraction-reality-ai-bottleneck/navier-stokes-equations.jpg" alt="The Navier–Stokes equations describing the motion of fluids">
<figcaption markdown="span">The Navier–Stokes equations describe the motion of fluids. Whether their three-dimensional solutions always exist and remain smooth is one of the [Millennium Prize Problems](https://www.claymath.org/millennium/navier-stokes-equation/). Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Navier-Stokes_equations.jpg), Sander Bais and Gijs Mathijs Ontwerpers</figcaption>
</figure>

This event, which feels as though it could become an AlphaGo moment for AI, reminded me of an xkcd comic called ['Purity'](https://xkcd.com/435/). It lines up sociology, psychology, biology, chemistry, physics, and mathematics, then jokes about which field is more "pure." In the comic, sociology is applied psychology, psychology is applied biology, and chemistry is ultimately applied physics. At the far end of purity is mathematics: the most abstract discipline, and the one farthest removed from reality.

<figure>
<img src="/assets/2026-09-14-abstraction-reality-ai-bottleneck/xkcd-purity.png" alt="The xkcd comic Purity, arranging fields from sociology to mathematics by increasing purity">
<figcaption markdown="span">A comic arranging disciplines by their distance from reality, or their "purity." Source: [xkcd 435, Purity](https://xkcd.com/435/), Randall Munroe ([CC BY-NC 2.5](https://xkcd.com/license.html))</figcaption>
</figure>

Then it occurred to me that the AlphaGo moments for these fields might arrive in the reverse order.

Mathematics is a clean environment for AI. Inputs can be formalized, rules can be stated explicitly, and answers can be verified. A true statement such as 1+1=2 will not become false when we look at it again tomorrow. A failed approach can be discarded at any time and restarted. Given enough compute, hypothesis generation and verification can be repeated at tremendous speed.

But the closer we get to reality, the more the situation changes. It isn't simply that there are more variables. Two fundamental problems emerge.

First, the world itself changes while we interact with it. Organisms adapt to their environments, people hear predictions and alter their behavior, and societies respond to new technologies and institutions. The moment we think we have found a rule, the conditions under which that rule holds may change. Especially in fields that deal with people and society, the observer and the observed cannot be completely separated. The act of AI explaining and intervening in the world becomes part of the world in turn.

Second, seeing the consequences of those changes takes time. In mathematics, we can form a hypothesis, test it, and, if it fails, immediately try another in the same environment. In reality, cells actually have to grow, people actually have to live, and real time has to pass before the effects of a policy appear. Even if AI can produce a million hypotheses in an hour, it cannot observe a million outcomes that only become known ten years later within that same hour.

From this perspective, AI's AlphaGo moments may arrive in the order of academic purity: beginning with mathematics, moving through physics, chemistry, and biology, and reaching psychology and sociology later. This doesn't mean that pure disciplines are easier. It means that the more abstract the world, the more readily the problem and its verification can be enclosed in a closed loop.

The closer we get to reality, the more the bottleneck moves outside that loop. Making AI think faster and making the world answer faster are two different problems. The most important resource in the future may therefore be not compute itself, but the speed at which we can interact with reality. This is why automated laboratories, robots, high-speed simulations, and digital twins matter. They aren't merely technologies that make AI smarter. They are technologies that make reality answer AI faster.

The final bottleneck of the AI era, then, may not be intelligence, but the ever-changing world itself.
