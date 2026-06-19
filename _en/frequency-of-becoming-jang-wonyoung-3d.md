---
title: "The Frequency of Becoming Jang Wonyoung, in 3D"
date: 2026-06-07T18:00:00+09:00
categories:
  - work
tags:
  - AI
  - Generative
  - Fourier
  - 3D
ref: frequency-of-becoming-jang-wonyoung-3d
---
<blockquote class="instagram-media" data-instgrm-permalink="https://www.instagram.com/p/DZR41f0pgov/" data-instgrm-version="14"></blockquote>
<script async src="//www.instagram.com/embed.js"></script>

I've made "the frequency of becoming Jang Wonyoung" twice now. The first time, I drew her outline as a sum of circular motions; the next time, I [built her photo out of a sum of ripples](/en/frequency-of-becoming-jang-wonyoung/). One dimension, then two. Which means the next question was already settled: would it work in three dimensions too?

### A world without plane waves ###

Up through two dimensions, the Fourier transform just worked, because the plane the image sat on stayed perfectly flat all the way out. On a flat plane, a plane wave is defined the same way everywhere, and those stripes form a complete basis.

But a head is a curved surface. On a curved surface there are no coordinates $(s,t)$ to plug into $\cos\big(2\pi(u\,s+v\,t)\big)$ in the first place. It even becomes ambiguous what it would mean for a stripe starting at the crown to flow "straight" until it comes around past the chin. I wanted to decompose the shape, but I first had to go find the very waves I would decompose it with.

### The head first ###

I needed something to decompose to begin with. Using a model that reconstructs a full 3D head mesh from a single photo ([PanoHead](https://github.com/SizheAn/PanoHead)), I generated a full-head mesh of Jang Wonyoung — a closed surface, sealed all the way around to the back of the head, made of about 40,000 vertices.

### Spherical harmonics ###

The most famous Fourier-on-a-surface is spherical harmonics. So at first I mapped the head onto a sphere and decomposed it with spherical harmonics.

But a face mesh isn't a sphere, so when you force the mesh onto one, protruding parts like the nose and ears get badly stretched, and that distortion carried straight through into the reconstruction. Keeping only the low frequencies didn't make things smoothly rounded — it made them droop. So I scrapped this approach.

### A surface has its own waves ###

Sprinkle sand on a drumhead and set it vibrating, and the sand gathers along the nodal lines to form patterns. These are Chladni patterns. They appear because the shapes the membrane can ring in — its eigenmodes — are already fixed by the membrane's own form.

The same is true for any curved surface. If you solve for the eigenfunctions of the Laplace–Beltrami operator on a surface $M$,

$$
\Delta_M\,\varphi_k = -\lambda_k\,\varphi_k
$$

these $\varphi_k$ become the "sine waves" intrinsic to that surface — the so-called manifold harmonics. A small eigenvalue $\lambda_k$ corresponds to a gentle wave sweeping broadly across the surface; a large one, to fine ripples. Solve this problem on a sphere and you get exactly the spherical harmonics back. In other words, it's a generalization of that earlier basis, but with one decisive difference: this time it isn't a sphere that determines the basis, but the shape of Jang Wonyoung's head itself.

On a mesh, this becomes the problem of finding the eigenvectors of the cotangent Laplacian. I computed the first 3,000 modes of the head mesh, starting from the lowest frequencies.

<figure>
  <img src="/assets/2026-06-07-jangwonyoung-frequency-3d/modes.png" alt="Eigenmodes of the head surface">
  <figcaption>Eigenmodes of the head surface. Low modes are waves sweeping broadly across the head; high modes are fine ripples.</figcaption>
</figure>

### Decomposing the shape ###

With the image, I decomposed the brightness of the pixels; here, I decompose the vertex coordinates themselves. Any function on the surface can be projected onto this basis, and the $x$, $y$, and $z$ coordinates are themselves functions defined on the surface.

$$
\mathbf{x} \;\approx\; \sum_{k=1}^{K} \langle \mathbf{x},\, \varphi_k \rangle\, \varphi_k
$$

The low-frequency modes carry the large-scale bulk; the high-frequency modes handle the detail. Add up all 3,000 and it converges to Jang Wonyoung's head.

<figure>
  <img src="/assets/2026-06-07-jangwonyoung-frequency-3d/buildup.png" alt="Shape accumulating from low to high frequency">
  <figcaption>The result of accumulating modes from low frequency upward. From potato to Jang Wonyoung.</figcaption>
</figure>

### Showing the build-up ###

Out of the photo, a sparse, low-frequency head emerges, and the modes are added in band by band. Each band doesn't just snap into place — it sloshes about in a damped oscillation before settling, so the surface rings once every time new detail arrives. I like that the fact that the shape is a sum of waves also reveals itself through motion. In the corner of the screen I've parked a small head displaying the standing-wave pattern of the mode currently being added. It's the 3D version of a Chladni pattern.

### Wrapping up ###

In one dimension I expressed a line as circular motion, in two dimensions a surface as waves, and in three dimensions a shape as natural vibrations. These are the frequencies of becoming Jang Wonyoung in each dimension.

Of course, my face still won't turn into Jang Wonyoung's just because I leave it playing. This is as far as I'll take the bit.
