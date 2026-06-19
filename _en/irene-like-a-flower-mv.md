---
title: "Making the VFX for IRENE's 'Like A Flower' Music Video"
date: 2024-12-08T15:34:30+09:00
categories:
  - dev
tags:
  - IRENE
  - vfx
ref: irene-like-a-flower-mv
---
<iframe width="992" height="458" src="https://www.youtube.com/embed/KdOF5-h4qpw?start=79" title="IRENE 아이린 &#39;Like A Flower&#39; MV" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

Starting from 1:19 in the video.

---

A studio reached out and asked me to make a video in which faces are generated using AI. They sent over some references, and it looked like they were after something built with StyleGAN. StyleGAN is actually 2019-era technology, and I've been doing this kind of work for a long time. SUHO's [Lights](https://www.youtube.com/watch?v=aExqq6s2lJ8) music video is a good example of it.

## GAN

So what is StyleGAN, exactly? To understand it, you first need to know about GANs (Generative Adversarial Networks). A GAN is a type of generative model in which two neural networks — a generator and a discriminator — are set up to learn by competing against each other. The generator's objective is to produce data that resembles the real thing and fool the discriminator, while the discriminator tries to tell whether what it's looking at is real or fake. By repeating this process, you end up with a generator that produces ever more refined data.

### Enter StyleGAN

StyleGAN is a model that builds on the GAN architecture, proposed by Tero Karras, a ridiculously smart researcher at NVIDIA. Beyond his work on StyleGAN, Karras is a versatile researcher who has also published papers on facial capture and rendering. There are two main ways StyleGAN differs from a conventional GAN.

1. Instead of feeding the simple input noise vector directly into the network the way earlier GANs did, StyleGAN passes the noise through a stack of multi-layer perceptrons (MLPs) and converts it into a style vector. This style vector is what controls the "style" at each layer. This is called disentanglement: unlike the old approach, which had to cram information into a Gaussian noise space, this process lets the generator do its work in a more meaningful space.

2. To implement how the style affects the image, StyleGAN uses a technique called AdaIN (Adaptive Instance Normalization). AdaIN adjusts the per-channel mean and standard deviation (std) of the input image to values derived from the style vector, effectively controlling the image's stylistic attributes. It's essentially a learned normalization technique, and it lets the style vector control fine-grained properties of the image. Below is some example AdaIN code.

```python

class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, content, style):
        # Compute content mean and std
        content_mean = content.mean(dim=(2, 3), keepdim=True)
        content_std = content.std(dim=(2, 3), keepdim=True) + self.eps # prevent division by zero

        # Compute style mean and std
        style_mean = style.mean(dim=(2, 3), keepdim=True) if style.dim() == 4 else style.unsqueeze(-1).unsqueeze(-1)
        style_std = style.std(dim=(2, 3), keepdim=True) + self.eps if style.dim() == 4 else style.unsqueeze(-1).unsqueeze(-1)

        # Normalize content to zero mean and unit variance
        normalized_content = (content - content_mean) / content_std

        # Scale and shift to match style statistics
        stylized = normalized_content * style_std + style_mean

        return stylized
```


# StyleGAN2

StyleGAN2 is an improved version that came along to address several limitations discovered in the original StyleGAN. It dramatically improved quality for high-resolution image generation and focused on fixing StyleGAN's existing shortcomings. The AdaIN used in StyleGAN was an efficient way to apply a style vector to an image, but it suffered from interference between the fine structure of the generated image and its stylistic attributes. The downside was that adjusting the style could distort the overall shape of the image or make it unstable. So StyleGAN2 introduced Weight Demodulation. In this approach, the style vector is applied directly to the layer's weights, which improves stability.

## So what does StyleGAN have to do with making VFX?

As I mentioned above, if you interpolate between two vectors in W+ space, you can produce an image that changes continuously. That's exactly how I made the video for this project. But what are W+ space and these vectors? The names sound intimidating, but what they mean is simple. We live in a three-dimensional space, so if you express your position as numbers, you can represent it with three values. Add a time axis and it becomes four, and so on. (Strictly speaking, a vector and a position are different things, but let's not get into that.) A StyleGAN style vector has 512 dimensions. In W+ space it would be 8×512 dimensions.

## Interpolation

So how do you interpolate this style vector? The simplest option is linear interpolation. The problem with linear interpolation is that at the moment one image transitions into another, the direction changes abruptly and the motion feels jerky and choppy.

So instead of linear interpolation I used splines. There are many kinds of splines — Bézier, Hermite, Catmull-Rom, and so on — each with its own pros and cons. Among them I chose the Catmull-Rom spline, because it produces a smooth, natural curve from nothing more than the position values as input.

![Catmull-Rom spline](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-12-08-IRENE%20%EC%95%84%EC%9D%B4%EB%A6%B0%20Like%20A%20Flower%20MV%20%EC%A0%9C%EC%9E%91%EA%B8%B0/Catmull-Rom_Spline.png?raw=true)

Sample positions along this curve, feed them into the StyleGAN generator as input, and — ta-da — you get a video of a face morphing smoothly. Below is some example Catmull-Rom spline code.

```python

def catmull_rom_spline(P0, P1, P2, P3, num_points=100):
    t = torch.linspace(0, 1, num_points)
    t2 = t * t
    t3 = t2 * t

    f1 = -0.5 * t3 + t2 - 0.5 * t
    f2 = 1.5 * t3 - 2.5 * t2 + 1.0
    f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    f4 = 0.5 * t3 - 0.5 * t2

    return f1[:, None] * P0 + f2[:, None] * P1 + f3[:, None] * P2 + f4[:, None] * P3

```

## W Space Encoding

But these generated faces couldn't just be anything I pleased. IRENE's face had to show up, Damon's face had to show up, and so on. That was a problem. The solution is simple: map IRENE's face into W+ space. It sounds complicated, but since a face image is normally produced from a vector sampled in W+ space, you just run that in reverse — extracting a W+-space vector from an image.

The simplest approach is to optimize the w value directly, but you can also use an encoder like [pSp](https://github.com/eladrich/pixel2style2pixel). In my case I went with optimization. A plain mean-squared-error (MSE) loss didn't fit well, so I added a perceptual loss using a pretrained [VGG16](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html).

![irene](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-12-08-IRENE%20%EC%95%84%EC%9D%B4%EB%A6%B0%20Like%20A%20Flower%20MV%20%EC%A0%9C%EC%9E%91%EA%B8%B0/irene.jpeg?raw=true)

## The Interface

Because I'd been asked to create the computer screen that IRENE operates, I had to bring in other interface elements on top of the face video. It turns out the Ubuntu UI can be used without any copyright issues, so I went with the Ubuntu UI — familiar to anyone who knows their way around computers, but probably foreign-looking to the general public. In the video you can see two windows side by side, and I needed something that looked vaguely like coding. So in the left window I used print statements to output the actual latent vectors used when generating the StyleGAN images (look closely and you'll see tensor values being printed). The right window shows [nvitop](https://pypi.org/project/nvitop/0.2.5.1/), which anyone doing deep learning will recognize. nvitop is a tool for monitoring the status of NVIDIA GPUs.

## Wrapping Up

The StyleGAN family of techniques is already five years old. And yet it's still going strong in the field of image generation. What matters isn't whether you're using the newest model. You can only find good answers if you understand what problem you're trying to solve and what lies at the root of that problem.

---

I've put the code up on [GitHub](https://github.com/okdalto/mv_stylegan). The StyleGAN2 base model code is borrowed from [Seonghyeon Kim's code](https://github.com/rosinality/stylegan2-pytorch) — another Kim Seonghyeon, over at Naver. Let me take this chance to thank him. I owe you a great deal...
