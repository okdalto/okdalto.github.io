---
title: "RIIZE Insomnia — 불면의 기계가 꾸는 꿈"
permalink: /work/insomnia-dreaming-machine/
date: 2026-10-30T00:00:00+09:00
categories:
  - 작업
tags:
  - AI
  - EEG
  - 미디어아트
ref: insomnia-dreaming-machine
---

Insomnia의 가사를 읽으며 잠들지 못하는 밤과 꿈을 이미지로 옮길 방법을 고민했다. 꿈을 이미지로 옮기려면 어떻게 해야 할까? 문득, 뇌파로 이미지를 생성하는 연구가 떠올랐다.

DreamDiffusion은 사람이 이미지를 볼 때 측정한 뇌파를 입력으로 받아 이미지를 생성하는 방법을 제안한다. 뇌파를 문장으로 번역해 프롬프트로 사용하는 것이 아니라, 뇌파에서 추출한 특징을 이미지 생성의 조건으로 직접 사용하는 방식이다.

그런데 뇌파는 잡음이 많고 사람마다 차이도 크다. 이미지와 짝지어진 데이터도 충분하지 않다. 이 연구에서는 먼저 뇌파 신호의 일부를 가리고, 가려진 부분을 복원하도록 인코더를 사전학습시킨다. 이미지를 알려주기 전에 뇌파 신호 자체의 패턴부터 배우게 하는 것이다. 이후에는 뇌파에서 추출한 임베딩이, 보고 있던 이미지의 CLIP 임베딩과 가까워지도록 학습시킨다. 이렇게 얻은 특징을 이미 학습된 Stable Diffusion에 전달해 이미지를 생성한다. [DreamDiffusion 논문](https://arxiv.org/abs/2306.16934)

<figure>
<img src="/assets/2026-10-30-insomnia/dreamdiffusion-overview.png" alt="DreamDiffusion의 EEG 인코더 사전학습과 이미지 생성 과정" loading="lazy">
<figcaption markdown="span">DreamDiffusion의 전체 구조. 출처: [공식 저장소](https://github.com/bbaaii/DreamDiffusion) · 논문: [Bai et al., 2023](https://arxiv.org/abs/2306.16934)</figcaption>
</figure>

최근에는 쥐가 보고 있던 영상을 뇌 활동으로부터 재구성한 연구도 나왔다. 2026년 발표된 연구에서는 쥐의 시각피질 신경세포 활동을 측정해, 학습에 사용하지 않은 10초짜리 영상을 초당 30프레임으로 재구성했다. 다만 이때 사용한 것은 두피에 붙이는 뇌파 센서가 아니라, 신경세포의 활동을 광학적으로 관찰하는 이광자 칼슘 이미징이었다. [논문](https://elifesciences.org/articles/105081)

<figure>
<img src="/assets/2026-10-30-insomnia/mouse-reconstruction.jpg" alt="쥐의 시각피질 활동으로부터 영상을 재구성하는 방법과 평가 결과" loading="lazy">
<figcaption markdown="span">쥐의 시각피질 활동으로부터 영상을 재구성하는 과정. 출처: [Bauer et al., 2026, Figure 1](https://elifesciences.org/articles/105081/figures#fig1), CC BY 4.0.</figcaption>
</figure>

이 연구들은 깨어 있는 동안 보고 있던 이미지와 영상을 다룬다. 그런데, 만약 잠을 잘 때 기록한 뇌파를 생성 모델에 넣으면 무엇이 나올까? 궁금해졌다.

그래서 뇌파 측정기를 구입했다. 가난한 대학원생의 예산으로는 이광자 현미경까진 구비할 수 없었다.

<figure>
<img src="/assets/2026-10-30-insomnia/eeg-headset.png" alt="이마 센서와 귀 클립을 착용한 모습. 얼굴은 모자이크 처리했다." loading="lazy">
<figcaption markdown="span">MindWave Mobile 2를 착용한 모습.</figcaption>
</figure>

내가 사용한 MindWave Mobile 2는 왼쪽 이마의 Fp1 위치에서 한 채널만 측정하는 장비다. 여러 위치에서 뇌파를 측정하는 연구용 장비에 비해 얻을 수 있는 정보가 제한적이고, 눈 깜빡임 같은 신호도 섞인다. 논문에서 제시한 성능을 그대로 기대하기는 어렵지만, 직접 데이터를 모아 생성 모델에 연결해보기로 했다.

데이터 수집을 위해 다양한 이미지를 연달아 보여주는 동시에 뇌파를 저장하는 프로그램을 만들었다. 내가 할 일은 센서를 붙이고 화면을 바라보는 것이었다. 간단해 보이지만, 센서 접촉 상태와 주변의 전기적 잡음에 따라 측정값이 꽤 달라졌다. 센서와 귀 클립을 다시 맞추고, 충전기를 빼고, 콘센트에서 떨어져 앉기도 했다. 데이터를 모으는 것부터 생각만큼 간단하지는 않았다. 돈이 없으니 서럽다.

<figure>
<video controls muted playsinline preload="metadata" poster="/assets/2026-10-30-insomnia/collection-poster.jpg">
<source src="/assets/2026-10-30-insomnia/eeg-collection.mp4" type="video/mp4">
</video>
<figcaption markdown="span">이미지를 보면서 뇌파 데이터를 수집하는 과정. 무음.</figcaption>
</figure>

이렇게 수집한 데이터로 뇌파의 패턴과 이미지의 특징을 연결하도록 모델을 학습시켰다. 보고 있던 이미지에서 CLIP으로 embedding을 추출하고, 뇌파 인코더가 그 embedding을 예측하도록 학습시키는 방식이다. 이후 Insomnia를 들으며 잠을 자는 동안 기록한 뇌파를 입력하고, 예측된 특징을 오픈소스 생성 모델에 전달해 새로운 이미지들을 만들었다.

처음에는 비슷한 이미지가 반복해서 나왔다. 예측된 특징값들이 좁은 범위에 몰려 있었기 때문이다. 기록한 뇌파에서 나온 값은 유지하되, embedding의 분포와 변동 폭을 조정해 이미지의 다양성을 늘렸다.

<figure>
<div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px;">
<img src="/assets/2026-10-30-insomnia/generated-01.png" alt="여우" loading="lazy" style="width: 100%; margin: 0;">
<img src="/assets/2026-10-30-insomnia/generated-02.png" alt="해파리" loading="lazy" style="width: 100%; margin: 0;">
<img src="/assets/2026-10-30-insomnia/generated-05.png" alt="수영장" loading="lazy" style="width: 100%; margin: 0;">
<img src="/assets/2026-10-30-insomnia/generated-15.png" alt="빈 방" loading="lazy" style="width: 100%; margin: 0;">
<img src="/assets/2026-10-30-insomnia/generated-23.png" alt="치타" loading="lazy" style="width: 100%; margin: 0;">
<img src="/assets/2026-10-30-insomnia/generated-30.png" alt="암석 지형" loading="lazy" style="width: 100%; margin: 0;">
</div>
<figcaption markdown="span">기록한 뇌파에서 예측한 embedding을 조건으로 생성한 이미지들.</figcaption>
</figure>

음악의 구간별 변화에 맞춰 이미지가 등장하는 간격을 조절하고, 장면이 단순히 겹쳐 사라지기보다는 사물과 공간의 형태가 변하며 이어지도록 했다. 그다음에는 영상의 프레임들을 시간 순서대로 쌓았다. 영상에서 지나간 시간이 공간 안에서는 깊이가 되도록 한 것이다. 이 덩어리를 레이마칭으로 볼륨 렌더링했다. 앞에는 현재의 이미지가 있고, 뒤에는 조금 전의 이미지들이 남는다. 뒤로 갈수록 투명해지고, 밝기에 따라서도 일부가 비어 보이도록 했다.

실제로 기록한 뇌파는 화면 아래에 흐르는 파형으로 그렸다. 이미지 생성에 사용한 임베딩은 PCA로 3차원에 옮겨 곡선으로 연결했다. 이 곡선을 광원으로 사용해, 이미지들이 쌓인 볼륨 안에서 빛이 퍼지고 가려지도록 했다.

<figure>
<video controls muted playsinline preload="metadata" poster="/assets/2026-10-30-insomnia/volume-poster.jpg" style="max-width: 460px; margin: 0 auto;">
<source src="/assets/2026-10-30-insomnia/insomnia-volume.mp4" type="video/mp4">
</video>
<figcaption markdown="span">RIIZE Insomnia Emotional Video. 무음.</figcaption>
</figure>

그러니까 지금 보는 영상은, 잠든 동안의 뇌파를 기계가 이미지로 옮긴 결과다. 내가 본 꿈과 얼마나 닮았는지는 알 수 없다. 내 뇌파와 생성 모델이 알고 있는 이미지들이 함께 만든 꿈에 가깝다.

낭만적으로 표현하면, 불면의 기계가 꾸는 꿈이라고 볼 수도 있겠다.
