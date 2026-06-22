---
title: "Love and the Slot Machine"
date: 2026-06-19T10:00:00+09:00
categories:
  - thoughts
tags:
  - love
  - probability
  - decision-making
ref: love-and-the-slot-machine
---

In English, a slot machine used to be called a "one-armed bandit." The single lever on its side looked like an arm, and because it kept taking your money it was a thief — a bandit. So a multi-armed bandit means a thief with many arms, or a situation with several slot machines lined up. The player has to choose one arm at every moment, without knowing which one will pay out the most.

<figure>
<img src="/assets/2026-06-19-love-and-slot-machine/slot-machine.jpg" alt="The Liberty Bell, the first slot machine, built by Charles Fey in 1899">
<figcaption markdown="span">The "Liberty Bell," the first slot machine, built by Charles Fey in 1899. The lever on its side looked like an arm, which is how it earned the name "one-armed bandit." Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Liberty_bell.jpg)</figcaption>
</figure>

The problem is simple. There are several slot machines. Each pays out with a different probability. But we don't know which one is good. We can pull only one lever at a time, and we see the reward that follows. The goal is to figure out which machine is best while, in the very process of figuring it out, collecting as much reward as possible.

This is where the difficulty arises. Should you keep pulling the machine you don't know well, or stick with the one that has looked best so far? The former is called exploration, the latter exploitation. Without exploration you never discover the better option. Without exploitation you never reap enough from the good option you've already found. The multi-armed bandit problem is, in the end, about how to balance exploration and exploitation in an uncertain world.

There are a few well-known strategies for this. The simplest is the ε-greedy strategy. It picks the option that has given the best result so far, but with a small probability tries some other option at random. Most of the time it follows the familiar best; once in a while it leaves the door open to an unfamiliar possibility.

$$a_t = \begin{cases} \displaystyle\arg\max_{a} \hat{Q}_t(a), & \text{with prob. } 1-\varepsilon \\[6pt] \text{a random choice}, & \text{with prob. } \varepsilon \end{cases}$$

Here $\hat{Q}_t(a)$ is the average reward option $a$ has paid out so far, and $\varepsilon$ is the probability of exploring at random. If $\varepsilon = 0$ it is pure exploitation, forever following the familiar best; if $\varepsilon = 1$ it is pure exploration, yanking a random lever every time. Usually you use a small value in between.

In terms of love, an ε-greedy person mostly follows their own taste. They trust the type they're drawn to, the relational style they find comfortable, the traits of people they've repeatedly liked. But every so often they doubt that taste. They strike up a conversation with someone they'd normally never meet, accept tenderness offered in an unfamiliar form, and test a possibility that lies outside the criteria they've built.

The strength of this strategy is its simplicity. A person can't compute every possibility each time. In the end we rely on rules of thumb. The judgment that "I'm drawn to people like this" or "this kind of relationship suits me" is forged through plenty of trial and error. ε-greedy doesn't throw that rule of thumb away, yet it keeps it from hardening into prejudice.

But the weakness is just as clear. Random exploration is, literally, random. Leaving the door open to an unfamiliar possibility now and then is necessary, but you can't know whether that possibility leads anywhere good. In love, random exploration sometimes becomes needless chaos, and to someone else it can read as an irresponsible signal. A person who keeps pulling levers — even though they already have a perfectly good relationship — on the grounds that maybe there's something else out there, ends up unable to deepen any relationship at all.

The second strategy is UCB, the Upper Confidence Bound. It doesn't look only at the average reward so far. It also factors in the uncertainty of options you haven't tried enough. Put simply, it gives points not just to options that look high-performing, but also to options worth investigating precisely because you don't know them well yet.

$$a_t = \arg\max_{a}\left[\, \hat{Q}_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \,\right]$$

The first term, $\hat{Q}_t(a)$, is the average reward so far — exploitation. The second term is an uncertainty bonus that grows the fewer times $N_t(a)$ that option has been pulled — exploration. A lever you've pulled only a little gets bonus points just for being still-unknown, and the more you pull it the smaller that bonus becomes. The constant $c$ sets the size of the bonus, i.e. how aggressively you explore.

In love, UCB is close to the attitude of not trusting first impressions alone. Some people aren't intense from the start. The conversation opens slowly, the appeal shows up late, the sense of stability reveals itself only over time. UCB-style love gives such a person room. It doesn't close off a possibility merely because you don't know it well yet. For someone you have few samples of, it takes the very scarcity of samples into account.

The strength of this strategy is that it curbs hasty judgment. In love, first impressions are powerful but often inaccurate. A strong rush doesn't guarantee a good relationship, and an awkward first meeting doesn't spell a bad one. UCB looks at the mean and the uncertainty together. If the signals so far aren't terrible and there's still a lot you don't know, it decides there's reason to look a little further.

But UCB has its weakness too. Uncertainty is seductive. A person you don't know yet can look like a bigger possibility than they really are. Precisely because you don't know them well, they get idealized. The thought that "if I just look a little more, there might be something there" makes a relationship more careful — and, at the same time, can make you defer the decision indefinitely. In love, uncertainty is a reason to explore, but it is sometimes just another name for lingering attachment.

The third strategy is Thompson Sampling. Here you hold a probabilistic belief about how good each option is, and update that belief every time a new result comes in. Rather than choosing from a fixed scorecard, you act while continually updating a distribution of imperfect belief.

$$\theta_a \sim p(\theta_a \mid \mathcal{D}), \qquad a_t = \arg\max_{a}\, \theta_a$$

You hold your belief about each option's reward probability $\theta_a$ in the form of a distribution, then each turn you draw one sample from each distribution and pick the option that produced the highest value. And whenever new data $\mathcal{D}$ arrives, you update that belief by Bayes' rule.

$$p(\theta_a \mid \mathcal{D}) \;\propto\; p(\mathcal{D} \mid \theta_a)\, p(\theta_a)$$

That is, you multiply your prior belief $p(\theta_a)$ by the newly observed evidence $p(\mathcal{D} \mid \theta_a)$ to form the posterior belief. (When rewards are success/failure, you commonly use a $\text{Beta}(\alpha_a, \beta_a)$ distribution, incrementing $\alpha_a$ on a success and $\beta_a$ on a failure.)

The strategy that resembles love most may be this one. We don't come to know someone all at once. Instead we watch small signals. How they keep their promises, their bearing after a conflict, the way they speak to other people, how they handle our vulnerability, their consistency in recurring situations. As that data piles up, we shift the belief in our hearts little by little. Is this person trustworthy? Does this relationship leave me in a better state? Is this tenderness momentary, or lasting?

The strength of Thompson-Sampling-style love is flexibility. You aren't locked into your first judgment. As good signals accumulate you open up more; as bad signals repeat you scale your belief back. It treats love not as a binary of conviction versus giving up, but as the updating of a possibility. Because real relationships mostly unfold amid ambiguous information, this approach is rather human.

But this strategy isn't safe either. Belief isn't updated by data alone. Desire, loneliness, expectation, and old wounds all intervene. The same behavior is read differently when it comes from someone you like. A late reply could be indifference or just busyness, but a person whose heart has already tilted tends to read it in their own favor. In love we are not good Bayesians. We often see only the evidence we want to see, and fail to update at the very moment we should.

These three strategies are quite useful for explaining love. ε-greedy shows the balance between familiar taste and unfamiliar possibility. UCB explains the room you can give to someone you don't yet know well. Thompson Sampling resembles the process of updating your belief about someone through small experiences. Together they show that love isn't an entirely mysterious event. Love, too, has a structure of choice, a scarcity of information, and a cost of regret.

And yet — love is not a multi-armed bandit problem.

In a multi-armed bandit, the levers don't change. The slot machine doesn't remember the player. The machine's reward distribution doesn't shift depending on the order in which I pulled the levers, the hopes I carried, or the manner in which I approached. The lever is simply pulled, and a reward is given. The player observes and learns, but the object does not learn.

Love is different. The object of your love is not a lever. They observe me, interpret me, and respond. The way I approach changes them, and their response changes me in turn. A person who was clumsy at first can grow tender within trust, and a relationship that was attractive at first can be worn down by recurring anxiety. The reward of a relationship isn't drawn from a fixed probability distribution; it keeps changing within an environment the two people build together.

So the important question in love is not which lever pays out the most. The more accurate question is closer to: with whom do we change each other's reward distributions for the better? Good love may not be about discovering a fully-formed optimal option from the start. It may instead be the process by which two imperfect people become a stable environment for each other and create rewards that weren't there at the beginning.

Of course, this doesn't mean you should endure just any relationship. Some relationships hand you bad data again and again. Anxiety, contempt, avoidance, violence, deceit — these aren't mere variance; they are important signals. The fact that love changes both people is a romantic thing to say, but it doesn't mean every change is for the better. Some encounters widen a person, and some make a person smaller. So even in love, observation is needed, updating is needed, and the decision to leave is needed.

But what remains at the end is not a matter of algorithms. The multi-armed bandit is a problem of maximizing reward. Love, though, is a problem of receiving reward and, at the same time, of creating it. We don't merely go around searching for a pre-set probability of happiness. By spending time with someone, we change our tone, lower our fears, build trust, and create new options in each other's worlds.

So likening love to a multi-armed bandit is useful, but in the end you have to drop the analogy. Love has exploration and exploitation. There's the strategy of following your taste, the strategy of giving uncertainty a chance, and the strategy of updating belief through experience. But the object of your love is not a lever, and a relationship is not a fixed reward distribution.

Love is not a slot machine. We choose among many possibilities, but after choosing we are not people who simply wait for a reward. We become people who change that possibility. And someone changes us the same way. This is why love is harder than calculation. Love is not the problem of finding the optimal lever, but the question of what kind of world we can each become for the other.
