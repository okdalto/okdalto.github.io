---
title: "Deep Learning, Folding Space, and Creativity"
date: 2024-11-18T11:34:30+09:00
categories:
  - thoughts
tags:
  - deep learning
  - creativity
ref: deep-learning-folding-space
---
The way deep learning works is by folding space. And in my view, creativity is also a matter of folding space. You might be thinking, "What nonsense is this?" — but hear me out.

To explain, I first need to walk through how deep learning works. There's a lot one could say, but to put it very simply, deep learning ultimately rests on a simple function: $y=Ax+b$. The trouble is that no matter how many functions of this form you stack together, they can only solve linear problems. This is where the activation function comes in. There are many kinds of activation functions, but in the end they all serve the same purpose: to add nonlinearity to $Ax+b$. The ReLU function, for example, creates nonlinearity in a wonderfully simple way — it turns any value below zero into zero and lets anything above zero pass through unchanged. Written as an equation, it looks like this:


$$
y = ReLU(Ax+b)\\
$$


Now, this kind of nonlinearity ends up squashing or folding space. If this isn't your field, your head might be starting to spin. So let me put it more simply with the abs function as an example. The abs function is what you use to take the absolute value of a number. Feed it -3, for instance, and out comes $\|-3\|$ — that is, 3. How might we describe this from the perspective of coordinates? You could say that it takes coordinates that used to sit on the negative side and brings them over to the positive side — in other words, it "folds" the space of numbers! In fact, on sites like Shadertoy, this very "folding" function is commonly used to fold three-dimensional space and create fractals, and it even goes by the name "folding." [(reference)](http://roy.red/posts/folding-the-koch-snowflake/)

But what does any of this have to do with creativity? What could squashing and folding space this way and that possibly have to do with being creative?

According to recent neuroscience research, even abstract concepts are stored in the human brain as positional information. Place cells and grid cells are responsible for this, and the scientists who discovered them were awarded the Nobel Prize in 2014 for the work. In other words, information is stored in our brains as three-dimensional coordinates. Remarkably, we can find a similar idea in deep learning: the latent space. The latent space is the multidimensional space a neural network has learned. In this space, similar features end up located close to one another. By exploiting this property, a neural network can perform all sorts of tasks — classifying or generating input data, and more.

Still, an unresolved question remains. What does all of this have to do with creativity?

The popular streamer "Chimchakman" once coined a delightfully novel expression on his broadcast: "shampoo broth." To roughly explain the context — he was talking about how, in the shower, he lathers shampoo into his hair and then uses the runoff "broth" to wash the rest of his body as a bonus.

"Shampoo" and "broth" are presumably stored somewhere in our heads as positional values. If you imagine drawing a map of the average person's mind, it's easy to predict that "shampoo" and "broth" would sit quite far apart. And yet, by "folding" the space so that the two concepts are brought close together, the witty and creative expression "shampoo broth" is born. This is precisely why I argue that "folding" space is creativity! And it's why I believe artificial intelligence can be creative.

Examples of creativity through this kind of "folding" can be found everywhere. Think about the iPhone. Many people hold it up as the very symbol of innovation, but you could also see it as a combination of things that already existed. Those things came together and produced a new object: the iPhone.

You can easily find similar cases in research, too. The Diffusion Model, so much in the spotlight lately, is essentially the Langevin equation from physics applied to the problem of image generation. NeRF, likewise, can be seen as taking the volume rendering that already existed and replacing its sampling function with a neural network.

What I came to feel while doing research in graduate school is that almost nothing is entirely new. Most papers include a Related Work section. As someone once said, we stand on the shoulders of giants. And I believe that depending on what we look at from up on those shoulders, and from which angle, we'll have the chance to discover something new.
