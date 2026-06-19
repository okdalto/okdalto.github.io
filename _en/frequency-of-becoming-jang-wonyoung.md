---
title: "The Frequency That Turns You Into Jang Wonyoung"
date: 2026-06-03T11:00:00+09:00
categories:
  - work
tags:
  - AI
  - Generative
  - Fourier
ref: frequency-of-becoming-jang-wonyoung
---
<blockquote class="instagram-media" data-instgrm-permalink="https://www.instagram.com/p/DZC3gFgp4oQ/" data-instgrm-version="14"></blockquote>
<script async src="//www.instagram.com/embed.js"></script>

There's a meme going around about a "frequency that steals Jang Wonyoung's face." Play it while you sleep, the story goes, and you can take Jang Wonyoung's face for yourself; the longer you play it the higher your odds, and there's even a dire warning attached that once you've listened, you can never go back to your own face. It has relatives too, like the "frequency that steals Lee Jae-yong's fortune" and the "frequency of seductive charm."

The moment I saw this meme, the Fourier transform popped into my head. Couldn't you actually become Jang Wonyoung through frequencies? With the Fourier transform, that is.

### The Fourier transform, and the 1D version ###

The Fourier transform lets you break any signal down into its various frequency components, that is, into a sum of sines and cosines. Extend this to two dimensions and you can think of each frequency component as a single circular motion, with many circular motions superimposing to form a single shape. That familiar video where rotating arrows spin around and trace out a contour, that's exactly this.

I had actually made a "frequency that turns you into Jang Wonyoung" once before, using this approach. I traced the outline from a photo of Jang Wonyoung into one long curve, then ran a Fourier transform on that curve to draw her contour as the sum of countless circular motions.

<blockquote class="instagram-media" data-instgrm-permalink="https://www.instagram.com/p/DVN1ouCCZnH/" data-instgrm-version="14"></blockquote>
<script async src="//www.instagram.com/embed.js"></script>

But the instant you turn a photo into a curve, the image has already become a mere "line." You throw away all the shading, texture, and surfaces of Jang Wonyoung's face and keep only the outline. So this time I extended it to two dimensions. The plan was to decompose not a line, but the image itself.

### Extending to two dimensions ###

If the Fourier transform of a 1D curve splits a signal into a sum of circular motions, the Fourier transform of a 2D image splits the image into a sum of wave patterns. Each frequency component you extract this way isn't a circle but a black-and-white stripe pattern (a plane wave) flowing in a particular direction. Written as an equation, it's a single sheet of stripes like this:

$$
\cos\big(2\pi(u\,s + v\,t) + \phi\big)
$$

Here $s = x/W$ and $t = y/H$ are normalized coordinates within the image, and $(u, v)$ are integers indicating how many times the pattern oscillates horizontally and vertically. A large $u$ means tightly packed horizontally, a large $v$ means tightly packed vertically. In the end they're just stripes differing only in direction and tightness.

<figure>
  <img src="/assets/2026-06-03-jangwonyoung-frequency/gratings.png" alt="Examples of plane waves">
  <figcaption>2D plane waves (stripes) differing in direction and tightness. The larger (u, v) is, the tighter the stripes.</figcaption>
</figure>

The 2D discrete Fourier transform decomposes the image into the strength and phase of these stripes.

$$
F(u,v) = \sum_{x=0}^{W-1}\sum_{y=0}^{H-1} I(x,y)\, e^{-2\pi i\left(\frac{ux}{W} + \frac{vy}{H}\right)}
$$

The magnitude $\lvert F(u,v) \rvert$ of each $F(u,v)$ tells you the brightness (amplitude) of that stripe, and the phase $\angle F(u,v)$ tells you where to shift the stripe to. On the left below is the original, and on the right is the resulting map of frequencies (the log-magnitude spectrum). The bright point in the center is low frequency, and the further out you go, the higher the frequency.

<figure>
  <img src="/assets/2026-06-03-jangwonyoung-frequency/spectrum.png" alt="Original and spectrum">
  <figcaption>The original image (left) and its log-magnitude spectrum (right). The center is low frequency, the outside is high frequency.</figcaption>
</figure>

So I applied a 2D Fourier transform to a photo of Jang Wonyoung, obtaining roughly 43,000 frequency components, and lined them up by radial frequency $r = \sqrt{u^2 + v^2}$ from smallest to largest, that is, from low to high frequency. Low frequencies handle the broad blocks of light and shadow, while high frequencies handle the details like strands of hair and the shape of the eyes.

Indeed, if you add them up one by one starting from the low frequencies, you can watch a blurry smudge gradually turn into a sharp Jang Wonyoung.

<figure>
  <img src="/assets/2026-06-03-jangwonyoung-frequency/reconstruction.png" alt="Cumulative reconstruction from low to high frequency">
  <figcaption>The result of reconstruction by accumulating frequencies from low to high. A blurry smudge gradually sharpens into focus.</figcaption>
</figure>

In the end, this stacking is just this sum:

$$
I(s,t) = \sum_{(u,v)} A_{u,v}\, \cos\big(2\pi(u\,s + v\,t) + \phi_{u,v}\big)
$$

### The process of stacking up into Jang Wonyoung ###

Just merging everything at once would be no fun, so I built it so you can watch the stacking process. The screen consists of two planes.

The top one is the accumulation plane. This is where the Jang Wonyoung reconstructed from all the frequencies added so far gradually comes into focus. The bottom one is the staging plane. The single frequency about to be added next surfaces here as a black-and-white stripe, then rises up and slowly seeps into the accumulation plane before disappearing. And then the next frequency immediately appears.

My favorite part of the implementation is that when adding a frequency, I don't draw and combine the stripes one by one in spatial space. I plant the corresponding component (and its conjugate) as a point in the frequency domain and run the inverse Fourier transform just once, and out pops the reconstructed image right away. Thanks to this, turning on all 40,000-plus frequencies isn't the slightest bit heavy. In the world of Fourier, adding a single sheet of stripes is just flipping on a single point.

### Wrapping up ###

If in 1D we drew contours with circular motion, in 2D we fill surfaces with waves. It amounts to transcribing the same person again, with one extra dimension added.

The "frequency that turns you into Jang Wonyoung" floating around the internet has no scientific basis, but these frequencies here, at least, really do turn into Jang Wonyoung the more they stack up. Literally, I've made a frequency that becomes Jang Wonyoung.

That said, playing it won't turn my face into Jang Wonyoung's. Sadly.
