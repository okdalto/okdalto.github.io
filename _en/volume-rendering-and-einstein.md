---
title: "Volume Rendering and Einstein"
date: 2025-10-22T10:00:00+09:00
categories:
  - thoughts
tags:
  - rendering
  - physics
ref: volume-rendering-and-einstein
---
> This is adapted from something I wrote on my Instagram story in October 2025.

When you render a cloud of particles — like an actual cloud — in volume rendering, the usual approach is to treat the cloud as a uniform volume and model the local variations inside it as scattering particles. It has been proven that, if you do the math correctly, the result you get from this volume-based calculation is identical to what you'd get by computing the interaction between light and every single particle in the cloud individually.

<figure>
<img src="/assets/2026-06-07-volume-rendering/cloud.jpg" alt="A cumulus cloud">
<figcaption markdown="span">What volume rendering is trying to imitate — a real cumulus cloud. Instead of computing each of the countless water droplets one by one, we model it as a single uniform volume filled with scattering particles. Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Cumulus_cloud_above_Lechtaler_Alps_at_tannheim,_Austria.jpg)</figcaption>
</figure>

And that proof? Einstein did it. Yes, *that* guy — the everything-is-relative one. So you could say Einstein made a contribution to volume rendering.

The paper is titled *Theorie der Opaleszenz von homogenen Flüssigkeiten und Flüssigkeitsgemischen in der Nähe des kritischen Zustandes* ([Annalen der Physik, 1910](https://onlinelibrary.wiley.com/doi/abs/10.1002/andp.19103381612) · [English translation](https://einsteinpapers.press.princeton.edu/vol3-trans/245)).

So from now on, whenever you're playing Red Dead Redemption 2 and you see those clouds rendered with volume rendering, take a moment to thank Einstein.

I posted this because I found it funny that Einstein popped up out of nowhere while I was studying. But when you think about it, optics is part of physics too, so it's only natural that physicists have had an enormous influence on rendering. Newton himself made things like prisms, after all... Thank you, Mr. Newton.
