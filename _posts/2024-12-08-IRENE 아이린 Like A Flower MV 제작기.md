---
title: "아이린 'Like A Flower' 뮤직비디오 VFX 제작기"
date: 2024-12-08T15:34:30+09:00
categories:
  - 개발
tags:
  - 아이린
  - vfx
---

<iframe width="992" height="458" src="https://www.youtube.com/embed/KdOF5-h4qpw" title="IRENE 아이린 &#39;Like A Flower&#39; MV" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

한 스튜디오로부터 AI를 활용해 얼굴이 생성되는 영상을 제작해달라는 의뢰를 받았다. 요청 내용을 보아하니 StyleGAN을 이용해 작업해달라는 의도인 것 같다. 사실 StyleGAN은 이미 꽤 오래된 기술이고, 나는 이런 작업을 오래전부터 해왔다. 예를 들면, 수호의 [Lights](https://www.youtube.com/watch?v=aExqq6s2lJ8) 뮤직비디오 작업이 바로 그 대표적인 사례다.

## StyleGAN
근데, StyleGAN이 뭔가? 이를 이해하려면 먼저 GAN(Generative Adversarial Network)에 대해 알아야 한다. GAN은 생성 모델의 일종으로, 두 개의 신경망인 **생성자(Generator)**와 **판별자(Discriminator)**가 서로 경쟁하며 학습하도록 짜여진 것을 뜻한다. 생성자의 목적 함수는 실제 데이터와 유사한 데이터를 만들어 판별자를 속이는 것이고, 판별자는 그것이 진짜인지 가짜인지 구분하려고 한다. 이 과정을 반복하면서 생성자는 점점 더 정교한 데이터를 만들어내는 생성자를 얻을 수 있다.

### StyleGAN의 등장
StyleGAN은 GAN의 구조를 발전시킨 모델로, NVIDIA의 Tero Karras라는 끝내주게 똑똑한 연구자가 제안했다. StyleGAN이 기존의 GAN과 다른 점은 크게 두 가지이다.

1. StyleGAN은 기존 GAN에서 사용하던 단순한 입력 노이즈 벡터를 그대로 사용하는 대신, 노이즈를 여러 개의 **다층 퍼셉트론(MLP, Multi-Layer Perceptron)**에 통과시켜 스타일 벡터로 변환한다. 이 스타일 벡터는 각 계층의 "스타일"을 조절하는 역할을 한다. 이를 disentangle이라 부르는데 Gaussian noise space에 정보를 우겨넣어야 했던 기존 방법과 달리 이 과정을 통해 Generator가 좀 더 의미 있는 공간에서 생성을 할 수 있게 된다. 

2. StyleGAN은 스타일이 이미지에 주는 영향을 구현하기 위해 **AdaIN(Adaptive Instance Normalization)**이라는 방법을 사용한다. AdaIN은 입력 이미지의 채널별 평균(mean)과 표준편차(std)를 스타일 벡터에서 유도된 값으로 조정하여, 이미지의 스타일 속성을 효과적으로 제어한다. 일종의 학습된 Normalization 테크닉인 셈인데, 이를 통해 스타일 벡터가 이미지의 세부적인 속성을 제어할 수 있게 된다.

# StyleGAN2

StyleGAN2는 StyleGAN에서 발견된 몇 가지 한계점을 해결하며 등장한 개선된 버전이다. 이 모델은 고해상도 이미지 생성에서의 품질을 크게 향상시켰으며, StyleGAN의 기존 문제점을 보완하는 데 초점을 맞췄다. StyleGAN1에서 사용되던 AdaIN은 스타일 벡터를 이미지에 적용하는 효율적인 방법이었지만, 생성된 이미지의 세부적인 구조와 스타일 속성이 서로 간섭(interference)하는 문제가 있었다. 이는 스타일을 조절하면서 이미지의 전체적인 형태가 왜곡되거나 불안정해질 수 있다는 단점을 초래했다. 따라서 StyleGAN2는 Weight Demodulation(가중치 비정규화) 방식을 도입했다. 이 방식에서는 스타일 벡터를 레이어의 가중치에 직접 적용해서 안정성을 높였다.

## 그래서 StyleGAN이 VFX 제작과 무슨 상관인가?

위에서 언급했던 것과 같이, 스페이스 내의 벡터 두 개를 보간하면 연속적으로 변화하는 이미지를 만들어낼 수 있다. 이 작업에서도 그렇게 영상을 제작했다. 스페이스와 벡터는 무엇인가? 이름은 어려워 보이지만 의미하는 건 간단하다. 우리는 3차원 스페이스에 살고 있다. 따라서 우리의 위치를 숫자로 표현하면, 세 개의 값으로 나타낼 수 있을 것이다. 시간축을 추가하면 네 개, 이런 식이다. (사실 엄밀하게 따지면 벡터와 위치는 다르다. 그렇지만 넘어가자) 스타일갠의 스타일 벡터는 512차원의 값을 가진다. W+ 스페이스라면 8*512차원일 것이다. 

## 보간

이 스타일 벡터를 그럼 어떻게 보간할 것인가? 간단하게는 선형 보간(Linear interpolation)을 사용할 수 있다. 그런데 단순한 선형 보간을 사용하면 한 이미지에서 다른이미지로 변화하는 순간에 방향 전환이 급격하게 일어나면서 동작이 뚝뚝 끊어지는 느낌이 들게 된다.

따라서 나는 선형 보간 대신 스플라인을 사용했다. 스플라인의 종류는 아주 많다. 베지어(Bezier), 에르미트(Hermite), 캣멀-롬(Catmull-Rom), 등등. 각각의 스플라인은 각각의 장단점이 있다. 나는 이 중에서 캣멀-롬 스플라인을 선택했다. 위치 값만 인풋으로 줘도 부드럽고 자연스러운 곡선을 생성할 수 있기 때문이다.

이 곡선을 따라 위치를 샘플링하고, 그걸 StyleGAN Generator에 입력으로 주면? 짜잔, 부드럽게 얼굴이 변화하는 영상이 만들어진다.

## W Space Encoding

그런데 이렇게 생성한 얼굴들, 그냥 아무거나 생성하면 안 됐다. 아이린 얼굴도 나와줘야 하고, 데이먼의 얼굴도 나와야 하고. 문제가 있었다. 해결책은 간단하다. 아이린 얼굴을 W 공간으로 매핑하는 것이다. 말이 어렵지만, 원래 W 스페이스에서 샘플링한 벡터로 얼굴 이미지가 만들어지니, 이를 거꾸로 이미지에서 W 스페이스의 벡터로 추출하는 것이다.

단순하게는 W값을 최적화(Optimize)하는 방법을 사용해도 되고, [pSp](https://github.com/eladrich/pixel2style2pixel)와 같은 인코더를 사용해도 된다. 나의 경우는 최적화 방법을 선택했는데, 단순 평균제곱오차(MSE) 손실으로는 피팅이 잘 안 되어서, 미리 학습된 [VGG16](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html)을 사용해 Perceptual 손실을 추가했다.

![irene](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-12-08-IRENE%20%EC%95%84%EC%9D%B4%EB%A6%B0%20Like%20A%20Flower%20MV%20%EC%A0%9C%EC%9E%91%EA%B8%B0/irene.jpeg?raw=true)

## 인터페이스

아이린씨가 조작하는 컴퓨터 화면을 제작해달라고 요청받았기 때문에, 영상 외에도 다른 인터페이스를 가져다 붙여야 했다. 찾아보니 우분투 UI는 저작권 문제 없이 사용해도 된다고 했다. 그래서 컴퓨터를 잘 아는 사람들이라면 친숙할, 하지만 일반인에게는 낯설 모습일 우분투 UI를 가져왔다. 영상에는 좌우로 창 두 개가 보이는데, 뭔가 코딩스러운 모습이 필요했다. 그래서 왼쪽 창에는 print 문을 사용해 실제로 StyleGAN 이미지를 뽑을 때의 레이턴트 벡터를 출력했다(자세히 보면 Tensor 인스턴스가 출력되는 걸 확인할 수 있다.). 오른쪽 창은 딥러닝을 하는 사람이라면 누구나 친숙할 nvitop을 출력했다. nvitop은 NVIDIA GPU의 상태를 모니터링하기 위한 도구이다. 

## 마무리

StyleGAN 계열 기술은 이미 5년 전의 기술이다. 그런데도 이미지 생성 분야에서는 여전히 활약 중이다. 중요한 건 최신 모델을 쓰느냐가 아니다. 어떤 문제를 풀고자 하는지, 그리고 그 문제의 근본에 무엇이 깔려 있는지를 이해해야만 진짜 답을 찾을 수 있다. 결국, 핵심은 거기에 있다.

---

코드는 [깃헙](https://github.com/okdalto/mv_stylegan)에 올려놓았다. 베이스 모델이 되는 StyleGAN2 코드는 네이버에 계신 또다른 [김성현님의 코드](https://github.com/rosinality/stylegan2-pytorch)를 빌려왔다. 이자리를 빌어 감사드린다.