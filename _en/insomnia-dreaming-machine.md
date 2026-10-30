---
title: "RIIZE Insomnia — Dreams of an Insomniac Machine"
permalink: /en/insomnia-dreaming-machine/
date: 2026-10-30T00:00:00+09:00
categories:
  - work
tags:
  - AI
  - EEG
  - Media Art
ref: insomnia-dreaming-machine
---

Reading the lyrics of Insomnia, I wondered how to turn sleepless nights and dreams into images. How would I go about turning a dream into an image? Research on generating images from EEG suddenly came to mind.

DreamDiffusion proposes a method for generating images from EEG recorded while a person looks at images. Rather than translating EEG into sentences to use as prompts, it directly uses features extracted from EEG to condition image generation.

But EEG is noisy and varies considerably from person to person. There is also a shortage of data pairing EEG with images. In this study, the encoder is first pretrained by masking parts of EEG signals and learning to reconstruct the missing portions. Before being shown images, it learns the patterns of the signals themselves. It is then trained to bring the EEG embeddings closer to the CLIP embeddings of the images the person was viewing. These features are passed to a pretrained Stable Diffusion model to generate images. [DreamDiffusion paper](https://arxiv.org/abs/2306.16934)

<figure>
<img src="/assets/2026-10-30-insomnia/dreamdiffusion-overview.png" alt="EEG encoder pretraining and image generation in DreamDiffusion" loading="lazy">
<figcaption markdown="span">DreamDiffusion architecture. Source: [official repository](https://github.com/bbaaii/DreamDiffusion) · Paper: [Bai et al., 2023](https://arxiv.org/abs/2306.16934)</figcaption>
</figure>

More recently, researchers have reconstructed movies that mice were watching from their brain activity. A study published in 2026 measured neuronal activity in the mouse visual cortex and reconstructed ten-second movies at 30 frames per second that had not been used in training. This did not use EEG sensors placed on the scalp, however. It used two-photon calcium imaging to optically observe neuronal activity. [Paper](https://elifesciences.org/articles/105081)

<figure>
<img src="/assets/2026-10-30-insomnia/mouse-reconstruction.jpg" alt="Method and evaluation of movie reconstruction from mouse visual cortex activity" loading="lazy">
<figcaption markdown="span">Reconstructing movies from mouse visual cortex activity. Source: [Bauer et al., 2026, Figure 1](https://elifesciences.org/articles/105081/figures#fig1), CC BY 4.0.</figcaption>
</figure>

These studies deal with images and movies viewed while awake. But what would happen if I fed EEG recorded during sleep into a generative model? I was curious.

So I bought an EEG headset. A two-photon microscope was beyond the budget of a broke graduate student.

<figure>
<img src="/assets/2026-10-30-insomnia/eeg-headset.png" alt="Wearing the forehead sensor and ear clip, with the face pixelated." loading="lazy">
<figcaption markdown="span">Wearing the MindWave Mobile 2.</figcaption>
</figure>

The MindWave Mobile 2 I used records a single channel at Fp1, on the left side of the forehead. Compared with research equipment that records EEG from multiple locations, it provides limited information, and signals such as eye blinks get mixed in. I could hardly expect the performance reported in the papers, but I decided to collect my own data and connect it to a generative model.

To collect data, I built a program that displayed a sequence of different images while recording EEG. All I had to do was attach the sensor and watch the screen. It sounds simple, but the readings changed considerably with sensor contact and electrical noise in the surroundings. I readjusted the sensor and ear clip, unplugged the charger, and moved away from power outlets. Even collecting the data was less straightforward than I had expected. Being broke is rough.

<figure>
<video controls muted playsinline preload="metadata" poster="/assets/2026-10-30-insomnia/collection-poster.jpg">
<source src="/assets/2026-10-30-insomnia/eeg-collection.mp4" type="video/mp4">
</video>
<figcaption markdown="span">Collecting EEG while viewing images. No audio.</figcaption>
</figure>

Using the collected data, I trained a model to connect EEG patterns with image features. CLIP extracts an embedding from the image being viewed, and the EEG encoder learns to predict that embedding. I then fed in EEG recorded while sleeping with Insomnia playing and passed the predicted features to an open-source generative model to create new images.

At first, similar images kept appearing. The predicted features were clustered within a narrow range. Keeping the recorded EEG as the source of the values, I adjusted the distribution and variation of the embeddings to increase image diversity.

<figure>
<div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px;">
<img src="/assets/2026-10-30-insomnia/generated-01.png" alt="Fox" loading="lazy" style="width: 100%; margin: 0;">
<img src="/assets/2026-10-30-insomnia/generated-02.png" alt="Jellyfish" loading="lazy" style="width: 100%; margin: 0;">
<img src="/assets/2026-10-30-insomnia/generated-05.png" alt="Swimming pool" loading="lazy" style="width: 100%; margin: 0;">
<img src="/assets/2026-10-30-insomnia/generated-15.png" alt="Empty room" loading="lazy" style="width: 100%; margin: 0;">
<img src="/assets/2026-10-30-insomnia/generated-23.png" alt="Cheetah" loading="lazy" style="width: 100%; margin: 0;">
<img src="/assets/2026-10-30-insomnia/generated-30.png" alt="Rock formations" loading="lazy" style="width: 100%; margin: 0;">
</div>
<figcaption markdown="span">Images generated using embeddings predicted from the recorded EEG.</figcaption>
</figure>

I adjusted the intervals between images to follow changes across sections of the music, and had objects and spaces transform into one another rather than simply overlap and fade away. I then stacked the video frames in chronological order. Time passing in the video became depth in space. I volume-rendered this mass using ray marching. The current image sits at the front, with earlier images remaining behind it. They become more transparent toward the back, with parts also opening up according to brightness.

I drew the recorded EEG as a scrolling waveform at the bottom of the screen. The embeddings used for image generation were projected into three dimensions using PCA and connected into a curve. I used this curve as a light source, allowing light to spread and be occluded inside the volume of stacked images.

<figure>
<video controls muted playsinline preload="metadata" poster="/assets/2026-10-30-insomnia/volume-poster.jpg" style="max-width: 460px; margin: 0 auto;">
<source src="/assets/2026-10-30-insomnia/insomnia-volume.mp4" type="video/mp4">
</video>
<figcaption markdown="span">RIIZE Insomnia Emotional Video. No audio.</figcaption>
</figure>

What you are seeing, then, is the result of a machine translating EEG recorded during sleep into images. I cannot know how closely it resembles the dream I actually had. It is closer to a dream made jointly by my EEG and the images the generative model knows.

To put it romantically, perhaps it is a dream dreamed by an insomniac machine.
