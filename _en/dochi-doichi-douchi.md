---
title: "Making the Video for the DOECHII Name Vote"
date: 2025-06-04T10:34:30+09:00
categories:
  - work
tags:
  - AI
  - Generative
ref: dochi-doichi-douchi
---
![image](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2025-06-04-%EB%8F%84%EC%B9%98%20%EB%8F%84%EC%9D%B4%EC%B9%98%20%EB%8F%84%EC%9A%B0%EC%B9%98/doecii.gif?raw=true)

When you talk about the pop stars dominating the charts overseas right now, DOECHII is impossible to leave out. If you spend any time watching Shorts, you've almost certainly run into her signature track, "Anxiety," at least once.

<iframe width="560" height="315" src="https://www.youtube.com/embed/riCP9x31Kuk?si=fb00Er7BmvK3poNP" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

But how exactly should that name, DOECHII, be written in Korean? Dochi? Doichi? Douchi?

According to the National Institute of Korean Language's rules for transliterating foreign words, "Dochi" is the correct form. But isn't language ultimately whatever the people using it decide it is?

<blockquote class="instagram-media" data-instgrm-permalink="https://www.instagram.com/p/DJ0sJBiyHYb/" data-instgrm-version="14"></blockquote>
<script async src="//www.instagram.com/embed.js"></script>

So Universal Music Korea cooked up a fun [marketing campaign](https://www.instagram.com/p/DJ0sJBiyHYb/?utm_source=ig_web_copy_link&igsh=MWxyOHBqYjhscmhkZw==): let the public vote on which of Dochi, Doichi, or Douchi should become the official spelling. With a presidential election right around the corner, you couldn't ask for a more timely little event.

My job here was to make the video that would play at the "name-deciding party." What kind of video should it be? For starters, the typography itself, the words "Dochi," "Doichi," "Douchi," had to be the centerpiece. And since it was going to play at a party, the characters needed personality, something eye-catching enough to hold attention. What immediately came to mind was generating images of bizarre, distorted typefaces learned by a model, and creating a sense of motion through interpolating between them.

The concept of training a model on typefaces to spit out strange letterforms was actually something I'd tried to attempt a few times before, but every attempt had fizzled out. I'd always regretted that, so this felt like the perfect chance to finally make it happen.

I started with the most important part of any training run: preparing the data. There seem to be two broad approaches to generating typefaces, an image-based approach and a vector-based approach. The difference is whether you train on a glyph that's been rasterized from a vector into an image, or whether you train on the vector itself. It's probably easiest to think of these as the domains of Photoshop and Illustrator, respectively. I haven't studied typeface generation in any great depth, so I can't say for sure, but if you handle typefaces as images, an image-generation model like StyleGAN or Stable Diffusion would make a suitable backbone, whereas if you handle them as vectors, a backbone that's good at predicting sequential data, like a Transformer, seems like the better fit.

For this project I decided to handle the typefaces as images. Working with vectors means you also have to write a rasterizer to render them, and study the OTF or TTF file specs on top of that, so there's just a lot more to keep track of.

For the image generator, I used StyleGAN2. It can be trained even on a modest 24GB of RAM, it generates quickly, and most importantly, I've used it so much that I know its entire architecture inside out.

Because I was treating typefaces as images, my training data needed to be individual glyph images rendered from a wide variety of fonts. I threw together a simple rendering script in Python, but ran into a problem: no matter how I centered things, the glyphs wouldn't sit in the middle of the image.

Why is it a problem if the glyph isn't centered? Honestly, since I wasn't trying to predict the exact form of a typeface, it's not a huge deal. It does affect generation quality, though. As the StyleGAN3 paper pointed out, StyleGAN2's architecture produces great output, but because of the padding in the convolution process, that quality ends up being position-dependent. And, more to the point, the off-center images bugged me to no end, so I started thinking about how to align the glyphs.

![image](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2025-06-04-%EB%8F%84%EC%B9%98%20%EB%8F%84%EC%9D%B4%EC%B9%98%20%EB%8F%84%EC%9A%B0%EC%B9%98/position_map.gif?raw=true)

The simple solution that came to mind was to multiply the binarized (0 or 1) pixel values of the text by the position corresponding to each pixel in the image (see the figure above), then take the average over x and y. This way, from an image with a black background, you can find the center of the white region (the parts that are 1, i.e. where the glyph is). As a bonus, computing the standard deviation tells you the glyph's width and height.

I went ahead and trained on the data I'd gathered this way. The results were so-so, but because StyleGAN's default output is 3 channels, the lettering, which should have been grayscale, ended up with different values per channel, causing the colors to separate.

```python
# To RGB class 
# Source : https://github.com/rosinality/stylegan2-pytorch
class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

```

I simply modified the to_rgb class to turn the 3-channel output into a single channel. The catch is that doing this makes the tensor shapes mismatch the already-trained checkpoint, so the checkpoint won't load. And since starting from an already-trained checkpoint is important, otherwise you easily run into things like mode collapse, this was a problem I had to solve.

The fix for this is simple too. A checkpoint loaded with torch.load is just a dictionary, so you find the entries related to to_rgb and trim them down, and it works fine. Reducing the channels this way not only eliminates the per-channel color separation in the glyphs, but, since the loss is now computed over a single channel, I could see that training went a bit more smoothly as well.

With training done, all that was left was to generate the glyphs as several sequences and lay them out nicely in a row. I built the sequences from images generated by moving StyleGAN's latent code around in various directions. This technique is apparently called "latent traveling." If you're not familiar with image-generation models, this probably sounds like gibberish. Latent traveling in StyleGAN means moving from one point in the latent space to another, interpolating the values in between to produce images. To put it simply: StyleGAN is a structure that takes a single 512-dimensional vector and produces an image. So if you create a vector that wanders around in that 512-dimensional vector space, and line up the images it produces, you get a sequence with a strange, flowing quality to it.

I wanted to lay out three of these generated sequences side by side, but a problem came up. "Doichi" and "Douchi" are three syllables each, so no issue there, but "Dochi" is only two. In other words, to render the two-syllable word, the middle sequence had to contain an empty blank.

The solution is simple. Create a latent code that produces a black image, and randomly steer the latent code toward it. But how do you obtain a latent code that produces black? That's simple too: optimize the latent code. To explain in more detail, you make a black image, compare StyleGAN's output against it to compute a loss, calculate the gradient of the latent code, and then move the latent code in the opposite direction of that gradient.

One thing to watch out for here is that StyleGAN's final activation function is tanh. So the value range is between -1 and 1, which means a black image has to be filled not with 0 but with -1.

When I laid out the glyphs I'd generated this way in a row, I didn't like how uneven the spacing between them was. In particular, when the middle glyph was a blank (black), the gaps between letters got way too wide. To fix this, I reused the center-and-width code from earlier, and I was able to lay the glyph images out in a neat, even row.

I shared this first cut of the video. The feedback was that the variety of glyphs was nice, but it would be better if the words "Dochi Doichi Douchi" themselves showed up more often.

So I decided to make the three words appear more frequently. The idea was to fix the first and last glyphs as "do" and "chi" respectively. I generated 1,000 random latent codes and the 1,000 corresponding images, and collected the ones that looked like "do" and "chi." By using the latent codes that produced those images, I could make sure "do" or "chi" always came out.

And that's how I pulled out the final result. The result itself looks fairly simple, but it actually took quite a bit of effort. Next time, I'd love to try generating the glyphs in real time in sync with music, or experiment with other visual compositions beyond just lining the glyphs up in a row. There's a bit of lingering regret, but all in all it was a fun project!
