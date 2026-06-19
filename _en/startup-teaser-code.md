---
title: "Recreating the Code in the Teaser for tvN's New Drama 'Start-Up'"
date: 2020-09-05T15:34:30+09:00
categories:
  - dev
tags:
  - Start-Up
  - drama
ref: startup-teaser-code
---
<iframe width="560" height="315" src="https://www.youtube.com/embed/QLiAdBBAVxI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


tvN just unveiled a new drama. They were marketing it as a "coding romance," so my curiosity got the better of me and I watched the teaser—and wouldn't you know it, around the 7-second mark some code shows up.

![A still from the teaser showing code on screen](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2020-09-05-tvN%20%EC%83%88%20%EB%93%9C%EB%9D%BC%EB%A7%88%20'%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85'%20%ED%8B%B0%EC%A0%80%EC%97%90%20%EB%93%B1%EC%9E%A5%ED%95%98%EB%8A%94%20%EC%BD%94%EB%93%9C%20%EB%94%B0%EB%9D%BC%ED%95%98%EA%B8%B0/KakaoTalk_20200905_023416044.jpg?raw=true)

And this code—when you zoom in, you can actually make some of it out.

![A close-up of the code from the teaser](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2020-09-05-tvN%20%EC%83%88%20%EB%93%9C%EB%9D%BC%EB%A7%88%20'%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85'%20%ED%8B%B0%EC%A0%80%EC%97%90%20%EB%93%B1%EC%9E%A5%ED%95%98%EB%8A%94%20%EC%BD%94%EB%93%9C%20%EB%94%B0%EB%9D%BC%ED%95%98%EA%B8%B0/close_up.png?raw=true)

Anyone can tell this is C++ code.

It looked like fun, so I went ahead and reimplemented it exactly as shown.

![My reimplementation of the code from the teaser](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2020-09-05-tvN%20%EC%83%88%20%EB%93%9C%EB%9D%BC%EB%A7%88%20'%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85'%20%ED%8B%B0%EC%A0%80%EC%97%90%20%EB%93%B1%EC%9E%A5%ED%95%98%EB%8A%94%20%EC%BD%94%EB%93%9C%20%EB%94%B0%EB%9D%BC%ED%95%98%EA%B8%B0/code1.png?raw=true)

Once I got it working, it turned out to be a simple bit of code that counts how many times string B appears in string A.

That piqued my interest, so I dug around and found something called the [KMP algorithm](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm), which lets you implement the same thing in O(N+M) time instead of O(N*M).

![A KMP-based implementation of the string-counting code](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2020-09-05-tvN%20%EC%83%88%20%EB%93%9C%EB%9D%BC%EB%A7%88%20'%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85'%20%ED%8B%B0%EC%A0%80%EC%97%90%20%EB%93%B1%EC%9E%A5%ED%95%98%EB%8A%94%20%EC%BD%94%EB%93%9C%20%EB%94%B0%EB%9D%BC%ED%95%98%EA%B8%B0/code2.png?raw=true)

Then I scoured Stack Overflow and managed to get the same functionality in a very short piece of code.

![A very compact version of the string-counting code](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2020-09-05-tvN%20%EC%83%88%20%EB%93%9C%EB%9D%BC%EB%A7%88%20'%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85'%20%ED%8B%B0%EC%A0%80%EC%97%90%20%EB%93%B1%EC%9E%A5%ED%95%98%EB%8A%94%20%EC%BD%94%EB%93%9C%20%EB%94%B0%EB%9D%BC%ED%95%98%EA%B8%B0/code3.png?raw=true)

That was fun!
