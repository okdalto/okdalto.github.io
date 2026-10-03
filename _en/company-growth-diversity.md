---
title: "Company Growth and Diversity"
permalink: /en/company-growth-diversity/
date: 2026-10-04T00:00:00+09:00
categories:
  - thoughts
tags:
  - business
  - diversity
  - creativity
ref: company-growth-diversity
---

A company that once grew from strength to strength sometimes stops growing. Revenue stagnates, and although it keeps releasing new products, they no longer feel as fresh. It has far more people and capital than before and is hiring better talent, yet no clear breakthrough seems to emerge.

This is especially common in industries such as IT, design, content, and fashion, where creating something new is a constant necessity. The company initially succeeded by making something that hadn't existed before, but strangely, the larger it grows, the more it begins making similar things over and over.

Why does this happen?

While thinking about why companies lose their innovative edge as they grow, it occurred to me that this might be viewed as a problem of space, through the lens of manifold theory.

Borrowing the manifold hypothesis, which holds that high-dimensional data actually lies on or near a much lower-dimensional manifold, let's assume that people's thoughts and tastes are also vectors in some high-dimensional space.

$$
\mathbf{x}_i \in \mathbb{R}^D
$$

And let's suppose that the thoughts people can actually have do not fill the entire space uniformly, but cluster on a much lower-dimensional structure.

$$
\mathbf{x}_i \in \mathcal{M} \subset \mathbb{R}^D,
\qquad
\dim(\mathcal{M}) \ll D
$$

Here, think of $\mathcal{M}$ as the manifold of thoughts people can produce through their lives.

People who have had similar experiences in the same industry will naturally gather in similar regions of this space. The ways people in fashion, software, and architecture view the world will each have a somewhat localized distribution.

The emergence of a new company resembles the appearance of a new position in this space. This is especially true in industries where creating something new matters.

Let the founder's position be $\mathbf{x}\_f$, and the average position of the existing industry be $\boldsymbol{\mu}\_{industry}$. A founder who succeeds with a new idea may be someone situated some distance from the industry's average position.

$$
\left\|
\mathbf{x}_f-\boldsymbol{\mu}_{industry}
\right\| > \epsilon
$$

If that position is sufficiently novel and people find value in it, products begin to sell. As more people embrace them, the company grows.

But as a company grows, people who share its thoughts and tastes begin to gather around it. Those who already like the company are more likely to apply, and the company tries to hire people who fit its culture. This is what we commonly call culture fit.

In a very simplified model, the probability of being hired might look something like this.

$$
P(\text{hire}\mid\mathbf{x})
\propto
\exp\left(
-\frac{\|\mathbf{x}-\mathbf{x}_f\|^2}{2\sigma^2}
\right)
$$

The idea is that the closer someone's thinking is to the founder or the existing organization, the more likely they are to join. Over time, the company repeatedly samples the neighborhood of the position it originally established. As a result, the range of ideas the organization explores becomes increasingly similar.

At first, this will be an advantage. People who think alike make decisions quickly and easily agree on what is good. It also helps preserve the company's distinctive character.

The problem is that as this process repeats, the thinking within the organization becomes increasingly alike.

If we define the center of the current organization as

$$
\boldsymbol{\mu}_{company}
=
\mathbb{E}_{\mathbf{x}\sim p_{company}}[\mathbf{x}]
$$

we can also view the company's success and reinforcement of its culture as a process in which its members gather around this center.

$$
\mathbb{E}
\left[
\|\mathbf{x}-\boldsymbol{\mu}_{company}\|^2
\right]
\downarrow
$$

A paradox emerges: the company initially succeeded by being different from others, but after succeeding, only people similar to itself remain.

In this state, gathering more people to generate new ideas is unlikely to produce significantly different results.

For example, suppose we combine two people's thoughts, $\mathbf{x}\_A$ and $\mathbf{x}\_B$, to create a new idea.

$$
\mathbf{x}_{new}
=
(1-\alpha)\mathbf{x}_A
+
\alpha\mathbf{x}_B,
\qquad
0\leq\alpha\leq1
$$

This is a form of interpolation.

But if $\mathbf{x}\_A$ and $\mathbf{x}\_B$ are already very close,

$$
\|\mathbf{x}_A-\mathbf{x}_B\| \approx 0
$$

no amount of repeated interpolation will take the result far beyond their neighborhood.

Even if the number of people grows from ten to a thousand, the area they can explore does not expand if all thousand occupy a similar region of the space.

I think this is one reason successful companies eventually stop growing. The company has grown, but the distribution of its thinking has actually narrowed.

How, then, can we find a completely new position? It is tempting to think that we must extrapolate from our existing positions.

$$
\mathbf{x}_{new}
=
\mathbf{x}_A
+
\lambda(\mathbf{x}_A-\mathbf{x}_B),
\qquad
\lambda>0
$$

We move beyond what we know, using a direction we have already identified.

But extrapolation is difficult because we don't know what lies beyond what we know. An organization that has spent a long time exploring only one particular region may not even know which direction to take.

That doesn't mean there is no way forward. Another approach is to bring in someone situated far from the existing organization.

We deliberately bring into the organization someone whose distance from its current center, $\boldsymbol{\mu}\_{company}$, satisfies

$$
\left\|
\mathbf{x}_{new}-\boldsymbol{\mu}_{company}
\right\| \gg 0
$$

this condition.

For example, instead of continuing to recruit only within fashion, a fashion company could bring in an architect. A software company could add a psychologist rather than gathering only developers and product planners.

What matters is less the profession itself than how different the person's position is from the existing organization's when they look at a problem.

Then something interesting happens.

Let the existing organization be $A$ and the newcomer be $B$. A new path appears:

$$
\mathbf{x}(\alpha)
=
(1-\alpha)\mathbf{x}_A
+
\alpha\mathbf{x}_B
$$

This gives us a new route.

If $B$ is sufficiently far away, this interpolation passes through regions the organization has never explored before.

The interpolation between these two points does not necessarily have to stay within the local region of the manifold occupied by the existing organization. Connecting two sufficiently distant positions can instead take us outside the region the organization has occupied. This is precisely the region that interests me here.

In other words,

$$
\begin{gathered}
\text{Extrapolation from } A
\\
\Downarrow
\\
\text{Interpolation between } A \text{ and } B
\end{gathered}
$$

this transformation becomes possible.

<figure>
<img src="/assets/2026-10-04-company-growth-diversity/exploration-space.svg" alt="Conceptual comparison of a narrow interpolation range among similar members and a wider exploration path created by connecting a distant newcomer">
<figcaption markdown="span">Top: the narrow exploration range produced by interpolation among similar members. Bottom: a new path becomes available by connecting a distant $B$. The gray curve represents a manifold; the dashed line shows an extrapolation direction from $A$ and a nearby member. This is a simplified two-dimensional illustration of a space of thoughts, not actual data. A straight interpolation path is not guaranteed to stay on the manifold or correspond to viable ideas.</figcaption>
</figure>

A region that represented extrapolation from $A$ becomes a region of interpolation the moment we bring $B$ into the organization.

We don't necessarily have to create something new from nothing. If we can find two points far enough apart, we can turn regions that represented extrapolation from either point into interpolation between the two. Instead of finding a new point directly, we create a new space to explore by connecting points that already exist but were far apart.

This resembles my earlier thought that creativity consists of connecting concepts that are far apart. Seen this way, the kinds of people a company needs may change over its life cycle.

An early-stage company may actually need similar people. If everyone in a small organization looks in completely different directions, nothing gets made. Once a new position has been found, it matters that people who strongly share the founder's thinking gather and move quickly in one direction.

But continuing to hire in the same way after a company has grown sufficiently creates risks. Adding more people like ourselves amounts to sampling the local region of the manifold the organization currently occupies more densely. It is a more precise exploration of an already well-explored space.

When a company faces a crisis, the person it needs may be someone farthest from the existing organization, rather than someone similar to its best current member.

Simply hiring a diverse range of people, however, will not resolve this problem on its own.

If we bring in someone at a new position, $\mathbf{x}\_B$, and then continually evaluate them by the existing company's standards,

$$
\mathbf{x}_B^{(t)}
\rightarrow
\boldsymbol{\mu}_{company}
$$

this convergence may occur.

As judgments such as "We've always done it this way here" and "That doesn't fit our brand" repeat, the newcomer may converge toward the organization's average, or fail to converge and leave the company.

For diversity to create a new space for exploration, it isn't enough to include people from different positions. They must be able to influence decisions while maintaining those positions.

From this perspective, diversity may be more than an ethical issue.

Especially in industries that must continually produce new images and ideas, it is directly connected to the size of the space a company can explore, and may therefore be a strategy for survival.

Successful companies grow by repeating the methods that brought them success. But if this repetition continues long enough, the organization becomes increasingly well fitted to the small region it occupies.

Perhaps a company's crisis is a form of overfitting. It has learned the small region that produced its past success so thoroughly that it can no longer account for the distribution outside it. If so, the solution isn't to find a better sample at the same position. At some point, what the company needs may be someone more similar, but someone more distant.
