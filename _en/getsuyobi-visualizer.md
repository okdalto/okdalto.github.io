---
title: "Making the 'G e t s u yo b i' Visualizer"
date: 2025-12-03T15:05:14+09:00
categories:
  - work
tags:
  - AI
  - Visualizer
ref: getsuyobi-visualizer
---
<iframe width="560" height="315" src="https://www.youtube.com/embed/FrdYPi9tJ1w?si=nJSh_wXHo2cZf-4T" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


A great opportunity recently came my way: I got to make the visualizer for 'G e t s u yo b i', a track written by suiimeetsyou. I happened to be thirsty to create something at the time, so I decided to let that creative urge run wild, and I had a wonderful time working on it.

For this project I used the thing everyone's been raving about lately, the one and only 'AI'. That's right, I am, in fact, an AI artist. These days there are an awful lot of people who define themselves as AI artists. And yet, work that actually interrogates AI as a medium in its own right is rare. AI is usually treated as just a tool for creation, not as the subject of the work itself. That has always been a mystery to me. If simply using AI as a tool makes you an AI artist, then why don't we call people Photoshop artists or paintbrush artists? Is it like calling someone a cellist because they play the cello: you used AI, so you're an AI artist?

That's why I think the term "AI artist" as we use it now is a transitional one. Back when digital work was still unfamiliar, the term "digital artist" carried real meaning because digital work was special. But now it's harder to find a creator who *doesn't* use digital tools, so almost nobody uses that label just to mean an artist who works on a computer. In the same way, once AI becomes a commonplace creative tool, using AI in your work will be taken for granted, and the term "AI artist" will quietly disappear.

What matters to me in my own work is directly exploring the structure of the technology we call deep learning, the properties of latent space, the hidden formal elements within it, and then taking the inner workings themselves as raw material for creation. In that sense this visualizer isn't a piece that simply ends at "I used AI." It's closer to a work that probes the materiality and formal language of the neural network as a medium. So this is... AI art, in the proper sense.

This making-of essay started off with a bit of bragging, but to explain *how* I explored AI as a medium, I'm going to work through it step by step, starting from the most basic concepts.

### Generative AI

What is a computer? "Compute" means "to calculate." Right: a computer is fundamentally a calculator. That's why the same input always produces the same output. Have you ever typed 1+1 into a calculator and gotten 3? Once the input is fixed, the network's output is effectively already decided.

This is a far cry from what we normally mean by generation. To generate is to make something that didn't exist before, and whatever does the generating ought to be able to produce a variety of results even from slightly varied inputs. But a computer is, at its core, a deterministic calculator. And yet today's generative models seem to have the ability to conjure something out of nothing. What's going on here?

### Noise

To make a computer appear to create something new, we feed it a raw ingredient called uncertainty. Enter noise. What we call noise is a set of random values that follow a particular probability distribution. There are many kinds of noise. There's the familiar Uniform noise, where every value is equally likely, and there's Salt-and-Pepper noise, where only extreme values occasionally spike. But in generative models, the one used most heavily by far is Gaussian noise, sampled from a Gaussian distribution. Why Gaussian noise of all things?

The Gaussian distribution is the distribution that countless natural phenomena follow. Phenomena that arise from many factors each contributing a little, such as people's heights, exam scores, sensor errors, and wind strength, tend to take on the shape of a Gaussian distribution. The more layered and complex the causes, the closer the distribution of the outcome gets to a Gaussian. Why does it work out that way? Because that's just how the world is built. The proof of this is Gauss's Central Limit Theorem.

The Gaussian distribution is also extremely convenient to work with computationally. It's smooth, and its shape doesn't break down badly even after linear transformations, which makes it easy to handle as deep learning pushes signals through many layers. In short, the Gaussian distribution is natural, easy to work with, and stable. As a result, using Gaussian noise has settled into a kind of standard in generative models.

### Method

So we've established that we use Gaussian noise. But what's the best way to structure the network used for generation, or to train it? At its simplest, you could use an MLP that takes noise as input, progressively grows the size of the vector, and then reshapes the result into an image. Simple as that approach is, the results won't be good.

A more advanced option is to use convolution. Convolution is the operation that multiplies and sums the values of neighboring pixels to extract features from an image. In generating an image you do the reverse: you use transposed convolution to expand low-resolution feature information into progressively higher-resolution images. Because this approach takes neighboring pixel values into account while generating an image, it works noticeably better at image generation than simply chaining MLPs together.

When you graft the training scheme of a GAN (Generative Adversarial Networks) onto this, you get a DCGAN (Deep Convolutional Generative Adversarial Networks). The idea behind GANs is simple but powerful: you set up a generator and a discriminator that judges what the generator produces, and train them together. By analogy, the generator is a counterfeiter and the discriminator is an examiner trying to tell real money from fake. The generator produces ever more convincing images to fool the discriminator, and the discriminator becomes ever stricter to catch the generator's tricks. As these two networks compete and grow against each other, the model gradually becomes able to produce more and more plausible images, that is, images that closely approximate the original data distribution. That's the core idea of a GAN.

The DCGAN was treated as the canonical convolution-based GAN, but the image quality still wasn't all that satisfying. And then StyleGAN arrived.

### StyleGAN

In the older DCGAN, you simply fed Gaussian noise into the very front of the network, and that information was pushed down through the entire generation process. In StyleGAN, by contrast, a stack of MLPs is used to transform z into a latent space. This is described as disentangling, because you can think of it as unraveling (disentangling) the space of noise expressed as a Gaussian distribution into a space the generator can interpret more easily. The vector obtained this way is called w, and this w is injected like a control signal at each resolution stage of the network. Injecting the signal by using w as the parameters of a normalization step is called Adaptive Instance Normalization, or AdaIN. Through this structure, the overall structure of the image, its mid-level details, and its fine textures could each be controlled as independent styles, and image generation quality also leaped dramatically.

### StyleGAN2

Later, StyleGAN2 arrived. StyleGAN2 did away with AdaIN and instead adopted an approach that modulates the weights directly via w, suppressing the excessive normalization of the original StyleGAN and improving image quality. After that came StyleGAN3, which applied antialiasing, but when it comes to raw image generation quality it wasn't easy to beat StyleGAN2. For this project I used exactly that, StyleGAN2. The reason is simple: I'd taken it apart from the ground up before, so I'm familiar with it.

### Training

Per suiimeetsyou's request that fruit make an appearance, I devised a way to collect fruit images. First, I used GPT to write up a list of various fruits. Using that list I built prompts for fruit set against a white background, fed those prompts into an open-source image generation model, and produced an enormous number of fruit images to train on.

I trained StyleGAN on this data. Training is simple. I rented a runpod server, ran code I'd written in torch to carry out the training, and used a pretrained checkpoint to do transfer learning, which cut down the training time.

### Latent Travel

Feed a latent vector into the network and an image comes out. In this setup the network plays the role of a mapping function that corresponds a point in latent space to a point in image space. If you move a point smoothly through latent space, the corresponding image moves too via the mapping, and that motion is deeply uncanny. This seems to be called latent travel.

The visual trick I used at the beginning of the visualization is based on this: moving the latent vector w in time with the music. You can get a similar visual effect by moving z, but the disentangled w is more expressive, so I went with that.

In any case, the point is that I moved a point in latent space (whether it was z or w). And here a problem arises. Moving w is all well and good, but along what path should it move?

### Curve

At its simplest, you could try this: generate several positions (w) within latent space, connect them with straight-line paths, and have w move linearly along those paths. It's very simple and easy to build, but this approach has a problem. Because the segments are straight lines, the images before and after change abruptly wherever two paths meet (at the points where the path bends). So to get smooth changes, you have to use curves.

There are many kinds of curves. Anyone who's properly studied computer graphics will probably have a slew of curve names spring to mind: Hermite Curve, Bezier Curve, Catmull-Rom Spline, and so on. Of these, I chose the Catmull-Rom Spline. The reason is simple: it produces a smooth curve even when you give it only positions and no tangents. Accordingly, I gave it n values of w, built a looping Catmull-Rom Spline, and wrote a class that samples w along that curve, end to end, using a t value from 0 to 1 on an infinite curve whose ends are joined. Using values sampled this way, the image changes smoothly, and because the start and end of the curve are connected, the front and back of the video link up infinitely. If you then use the amplitude extracted from the music to control the motion along this curve (that is, the change in t), you can easily pull out video of fruit shifting in time with the music.

### Feature

What is an image feature? Besides w, StyleGAN needs another value to generate an image: the noise image used at the start of the convolution stages. StyleGAN applies many rounds of upsampling and modulated convolution to this small noise, progressively growing the image size, and the intermediate results produced along the way are called features. If you twist these features this way and that, the edits show up in the final generated result too. What I tried in the latter half of this project was to cut off the upsampling stages, pull out the intermediate feature, distort that feature in various ways, and then let it pass through the rest of the generation process. Things like applying a slit-scan to the feature, rotating it, or splitting it into multiple pieces and moving each one up and down. Here too I could control the motion using the amplitude extracted from the music. Motion can ultimately be expressed as numbers, so the amplitude value, being a number itself, can be multiplied and added into that motion in all sorts of ways.

The output that came out of all this is what fills this visualizer. Now that you know the principles, it's pretty simple, isn't it? Technical elements like the structure of GANs, the properties of latent space, and feature-level manipulation may look hard on the surface, but the moment you understand the principles, they actually become material for creation. AI isn't merely a convenient automation tool; its very structure is itself another raw material, and depending on how the creator handles that material, they can make wildly different forms of expression.
