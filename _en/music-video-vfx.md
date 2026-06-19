---
title: "Making the VFX for a Music Video"
date: 2023-06-14T11:34:30+09:00
categories:
  - work
tags:
  - Woo Won-jae
  - face swap
  - deep learning
ref: music-video-vfx
---
This past May, Anigma Technologies had the rare opportunity to take part in producing the music video for "Ransome," a new song by the rapper Woo Won-jae.


<iframe width="560" height="315" src="https://www.youtube.com/embed/PRtHZvclTsg?si=aPLiQiE315aEZGO6" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


Anigma is a startup founded by master's and PhD students from KAIST's Visual Media Lab. Lately, drawing on our expertise in facial animation, we've been focused on building AI-driven lip-sync technology. This music video turned out to be the perfect moment to put that technology to use.

In an age like ours, swapping someone's face with AI is hardly a surprise anymore. Deepfakes have practically become synonymous with face manipulation, and even before they came along, papers like Face2Face were being published.

But here's something worth pausing on: face swap and face reenactment look remarkably alike yet are fundamentally different concepts. Even researchers in the field tend to conflate the two. A 2020 paper from Disney Research points this out in Section 2.4. The problem we're actually solving isn't face swap—replacing a source face with a target face—but rather something closer to face reenactment: driving a target face with the expressions of a source face.

Beyond deepfakes, there are countless face swap/reenactment techniques out there. At a glance they may all seem to work well, but the story changes the moment you try to use them in an actual production. In many cases they suffer from a range of problems: low resolution, the person ending up looking like someone else, or a failure to faithfully track the shape of the person's mouth.

We drew on research such as [Deferred Neural Rendering](https://arxiv.org/abs/1904.12356). The idea behind that paper is simple but effective: use a 3D model as a kind of intermediate medium—an intermediate representation—and leave the final rendered result to a neural network.

To use this technique, we had to fit a 3D model to the video of Woo Won-jae's face. For this we used a 3D Morphable Model, or 3DMM. You can think of a 3DMM as a statistics-based model capable of representing a wide variety of faces. Combining neural rendering with a 3DMM like this is very common in face research.

On top of that, we added a technique that lets the 3D face generate expressions from speech. This sort of technique is generally called speech-to-animation. There's a great deal of research in this area too—NVIDIA's 2017 work, JALI, VOCA, MakeItTalk, and many others. It also happens to be one of the main research areas of our lab.

There were plenty of bumps along the way in adapting our technology to a real production, but the experience was all the more valuable for it. Once again, my thanks to director Yoon Jun-hee for the opportunity. :)
