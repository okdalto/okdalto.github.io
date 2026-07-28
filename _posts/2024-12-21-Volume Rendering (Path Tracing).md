---
title: "Volume Rendering, Path Tracing 버전"
permalink: /work/volume-rendering-path-tracing/
date: 2024-12-21T10:34:30+09:00
categories:
  - 작업
tags:
  - rendering
  - volumetric
  - cloud
ref: volume-rendering-path-tracing
---
![이미지](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-12-21-Volume%20Rendering%20(Path%20Tracing)/volume_rendering.jpg?raw=true)

이전 글에서 Ray integration을 단순화하여 Ray marching으로 Volume rendering을 구현하는 방법에 대해 다뤘다. 
해당 구현에서는 단 하나의 방향에서만 들어오는 빛을 가정하였으나, 실제로는 각 산란 지점마다 구면의 모든 방향으로 들어오는 빛을 고려하는 것이 좀 더 물리적으로 정확하다 할 수 있겠다.

## 광선 적분 ##

더 자세한 설명을 위해 다시 Ray Integration을 들여다보자. 광선 적분의 일반적인 형태는 다음과 같다.

$$C(t) = \int_{t_{\text{near}}}^{t_{\text{far}}} T(t) \cdot \sigma(t) \cdot \left[c(t) + L_{\text{ext}}(t)\right] \, dt$$

각 변수 및 함수의 의미는 다음과 같다. 

$C(t)$: 최종적으로 얻어지는 광선의 색상 또는 밝기(에너지). 이는 광선 적분 결과로, 매질을 통과한 광선이 얼마나 축적되었는지 나타낸다.

$t_{near}$, $t_{far}$: 광선이 매질과 교차하는 시작점과 끝점.
광선 적분은 광원이 발사한 빛이 매질과 처음 만나기 시작한 지점 $t_{near}$ 에서 매질을 통과하며 끝나는 지점 $t_{far}$ 까지 수행된다.

$T(t)$는 투과도(transmittance)로, 물체를 통과하는 동안 빛의 강도가 약화되는 정도를 설명하는데, 다음과 같이 정의된다.

$$T(t) = \exp\left(-\int_{t_{\text{near}}}^{t} \sigma(s) \, ds\right)$$

위 식은 [Beer-Lambert Law](https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law)의 일반화된 형태이다. 빛은 에너지를 가지고 있다. 빛이 매질(볼륨)을 통과하면서 이 에너지는 흡수되거나 산란된다. 이 중에서 흡수를 설명하는 것이 Beer's law이다. 식을 잘 살펴보면, $t_{\text{near}}$에서 ${t}$까지 빛이 이동했을 때, 누적된 소멸 계수에 따라 빛이 얼마만큼 살아남는지를 나타내는 값이라는 것을 알 수 있다.

$\sigma(t)$: 흡수와 산란의 총합. 소멸 계수(extinction)라고 부른다.

$c(t)$: 매질의 고유 색상(Color) 및 강도(Intensity) 정보. 특정 지점에서 매질이 가진 고유한 색과 밝기를 나타내며, 광선 적분에 기여한다.

$L_{\text{ext}}$: 외부에서 들어오는 빛

이 글에서 가장 중요하게 다루는 부분은 해당 부분이다. 일반적으로 $L_{\text{ext}}$를 나타내기 위해서는 아래와 같이 모든 방향에서 들어오는 빛을 고려해야 한다.

$$L_{\text{ext}}(t) = \int_{S^2} L(\omega) \cdot p(\omega \cdot \omega') \, d\omega$$

표면과 달리 매질에서는 빛이 뒤에서 앞으로도 지나갈 수 있기 때문에, 적분 범위는 반구가 아니라 구면 전체 $S^2$, 즉 입체각 $4\pi$ 전부다.
$p$는 위상 함수(phase function)로, 들어온 방향 $\omega$의 빛이 나가는 방향 $\omega'$로 얼마나 꺾이는지를 나타내는 확률 밀도다.

## 이전의 방법과 차이점 ##

이전 구현은 광선을 일정한 간격으로 잘라 걸으면서, 각 스텝마다 밀도에 비례하는 확률로 산란 여부를 결정했다.
직관적이긴 하지만 이 방식에는 문제가 있다. 스텝 크기를 얼마로 잡느냐에 따라 결과가 달라진다는 점이다.
스텝을 촘촘하게 하면 느려지고, 성기게 하면 얇은 디테일을 통째로 건너뛰면서 그림이 밝아지거나 어두워진다.
샘플을 아무리 많이 쌓아도 "정답"으로 수렴하지 않고, 스텝 크기가 만들어낸 다른 그림으로 수렴한다.

이번 구현에서는 스텝이라는 개념을 아예 없앴다.
빛이 매질 안에서 얼마나 날아가다 부딪히는지를 확률 분포에서 직접 뽑고, 그 지점에서 위상 함수를 따라 방향을 꺾은 뒤, 광원을 직접 조준해 기여를 모은다.
이 세 가지가 각각 델타 트래킹(delta tracking), Henyey-Greenstein 샘플링, 그리고 NEE(next event estimation) + MIS다.
하나씩 살펴보자.

## 자유 행로를 직접 뽑기: 델타 트래킹 ##

밀도가 $\sigma$로 균일한 매질이라면, 빛이 충돌 없이 거리 $t$만큼 날아갈 확률은 $e^{-\sigma t}$다.
여기서 충돌 거리를 뽑는 건 쉽다. 균등 난수 $\xi$를 하나 뽑아서 $t = -\ln(\xi) / \sigma$ 하면 끝이다.

문제는 우리 매질의 밀도가 균일하지 않다는 것이다. 프랙탈 SDF의 내부 깊이로 밀도를 만들기 때문에 위치마다 제멋대로다.
그래서 쓰는 것이 델타 트래킹이다. 아이디어는 이렇다.

1. 공간 전체에서 실제 밀도보다 항상 크거나 같은 상한값 $\bar\sigma$(majorant)를 하나 정한다.
2. 매질이 $\bar\sigma$로 꽉 차 있다고 **가정하고** 충돌 거리를 뽑는다.
3. 그 지점에서 실제 밀도를 재서, $\sigma(x) / \bar\sigma$의 확률로 "진짜 충돌"로 인정한다.
4. 인정되지 않으면 아무 일도 없었다는 듯 그 자리에서 다시 1번으로 돌아간다.

가짜 충돌(null collision)로 밀도를 메꿔서 균일한 매질인 척한 다음, 확률적으로 가짜를 걸러내는 것이다.
놀랍게도 이렇게 하면 어떤 편향도 없이 정확한 자유 행로 분포가 나온다. 스텝 크기라는 파라미터 자체가 사라진다.

```glsl
float t = tStart;

for (int eventIndex = 0; eventIndex < MAX_TRACKING_EVENTS; ++eventIndex)
{
    float xi = max(randomFloat01(rngState), 1e-7);
    t += -log(xi) / SIGMA_MAJORANT;

    if (t >= tEnd)
    {
        return false;
    }

    vec3 position = rayOrigin + rayDirection * t;
    float acceptanceProbability = extinctionAt(position) / SIGMA_MAJORANT;

    if (randomFloat01(rngState) < acceptanceProbability)
    {
        collisionDistance = t;
        collisionPosition = position;
        return true;
    }
}
```

여기서 지켜야 하는 약속이 딱 하나 있다. **$\bar\sigma$가 진짜로 상한이어야 한다**는 것.
어딘가에서 $\sigma(x) > \bar\sigma$가 되어버리면 위 확률이 1을 넘어가고, 그 지점의 밀도는 조용히 과소평가된다.
그래서 밀도 함수에서 상수 밀도를 더하는 짓을 하지 않고, 아예 `MAX_DENSITY`로 잘라버렸다.

```glsl
float sampleDensity(vec3 p)
{
    float signedField = sceneField(p);
    float interiorDepth = max(-signedField, 0.0);

    // 여기서 상수 밀도를 더하지 않는다. 그래야
    // extinctionAt(p)가 SIGMA_MAJORANT를 절대 넘지 않는다.
    return clamp(interiorDepth * DENSITY_SCALE, 0.0, MAX_DENSITY);
}
```

## 그림자도 확률로: 비율 트래킹 ##

산란 지점에서 광원까지 빛이 얼마나 살아남는지, 즉 투과도 $T$도 구해야 한다.
이걸 밀도 적분으로 정직하게 계산하려면 결국 다시 스텝을 밟아야 한다.

그래서 델타 트래킹을 살짝 비틀어 쓴다. 충돌 지점을 뽑는 것까지는 똑같은데, 거기서 멈추는 대신 $1 - \sigma(x)/\bar\sigma$를 계속 곱해 나가는 것이다.
가짜 충돌일 확률을 누적하는 셈인데, 이 값의 기댓값이 정확히 $T$가 된다. 이것을 비율 트래킹(ratio tracking)이라고 부른다.
충돌 하나로 그림자를 끊어버리는 것보다 분산이 훨씬 낮고, 무엇보다 결과가 부드럽다.

```glsl
float transmittance = 1.0;

for (int eventIndex = 0; eventIndex < MAX_SHADOW_EVENTS; ++eventIndex)
{
    float xi = max(randomFloat01(rngState), 1e-7);
    t += -log(xi) / SIGMA_MAJORANT;

    if (t >= tEnd)
    {
        break;
    }

    vec3 position = rayOrigin + rayDirection * t;
    float localExtinction = extinctionAt(position);

    transmittance *= max(1.0 - localExtinction / SIGMA_MAJORANT, 0.0);

    if (transmittance < 1e-4)
    {
        return 0.0;
    }
}
```

## 위상 함수: Henyey-Greenstein ##

진짜 충돌이 일어났다면 이제 방향을 꺾어야 한다. 이때 쓰는 것이 Henyey-Greenstein 위상 함수다.

$$p(\cos\theta) = \frac{1}{4\pi} \cdot \frac{1 - g^2}{\left(1 + g^2 - 2g\cos\theta\right)^{3/2}}$$

$g$는 비대칭 파라미터로, 산란의 성격을 조정한다.
$g > 0$이면 빛이 전방으로 산란되고, $g < 0$이면 후방으로 산란되며, $g = 0$이면 등방성 산란을 나타낸다.
구름처럼 강한 전방 산란을 원해서 이 구현에서는 $g = 0.65$를 썼다.

이전 구현에서는 $z$축 기준으로 방향을 만든 다음 Rodrigues 회전 공식으로 기준 방향에 맞춰 정렬했는데,
회전축과 회전 행렬을 만드는 이 과정은 사실 필요가 없다. 진행 방향을 축으로 하는 정규직교 기저를 하나 세우고 그 위에서 바로 벡터를 조립하면 된다.
행렬 곱도 없고, 두 벡터가 거의 평행할 때 터지는 특이점도 없다.

```glsl
vec3 sampleHenyeyGreenstein(vec3 forwardDirection, float g, inout uint rngState)
{
    float xi1 = randomFloat01(rngState);
    float xi2 = randomFloat01(rngState);

    float cosTheta;

    if (abs(g) < 1e-3)
    {
        cosTheta = 1.0 - 2.0 * xi1;
    }
    else
    {
        float ratio = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi1);
        cosTheta = (1.0 + g * g - ratio * ratio) / (2.0 * g);
        cosTheta = clamp(cosTheta, -1.0, 1.0);
    }

    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = c_twopi * xi2;

    vec3 w = normalize(forwardDirection);
    vec3 helper = abs(w.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);

    vec3 tangent = normalize(cross(helper, w));
    vec3 bitangent = cross(w, tangent);

    return normalize(
        tangent * (sinTheta * cos(phi)) +
        bitangent * (sinTheta * sin(phi)) +
        w * cosTheta
    );
}
```

$\cos\theta$를 뽑는 저 식은 HG 분포의 누적 분포 함수를 역으로 푼 결과다.
$g$가 0에 가까우면 분모가 0으로 가면서 수치가 폭발하기 때문에, 그때는 그냥 등방성으로 처리한다.

이렇게 위상 함수를 **정확히** 중요도 샘플링했기 때문에 좋은 일이 하나 생긴다.
경로 가중치에 들어가야 할 $p(\omega)/\text{pdf}(\omega)$가 1로 딱 상쇄되어 버리는 것이다.
그래서 산란이 일어날 때 throughput에 곱해줄 것은 단일 산란 알베도(single-scattering albedo)뿐이다. 흡수는 여기에 들어 있다.

```glsl
throughput *= MEDIUM_ALBEDO;
```

## 광원을 직접 조준하기: NEE와 MIS ##

이제 산란 지점에서 빛을 모아야 한다. 그냥 방향을 꺾어서 날려 보내고 운 좋게 광원에 맞기를 기다릴 수도 있다.
하지만 광원이 작으면 그 확률은 처참하게 낮고, 화면은 소금 뿌린 것처럼 노이즈로 뒤덮인다.

그래서 산란이 일어날 때마다 광원 표면 위의 한 점을 **직접** 골라서, 거기까지의 투과도를 재고 기여를 더한다.
이것이 NEE(next event estimation)다. 이 구현의 광원은 위쪽에 떠 있는 발광하는 박스이고, 여섯 면의 넓이에 비례해서 점을 고른다.

여기서 면적에 대한 확률 밀도를 입체각에 대한 확률 밀도로 바꿔줘야 한다. 변환은 다음과 같다.

$$p_\omega = p_A \cdot \frac{d^2}{\cos\theta_l}$$

$d$는 산란 지점에서 광원까지의 거리, $\cos\theta_l$은 광원 표면의 법선과 광선이 이루는 각도다.
멀수록, 그리고 비스듬히 볼수록 그 광원은 작게 보인다는 당연한 이야기를 식으로 쓴 것뿐이다.

문제는 이제 같은 빛을 두 가지 경로로 세게 된다는 점이다. 광원을 직접 조준해서 한 번, 위상 함수를 따라 날아가다 우연히 광원에 부딪혀서 또 한 번.
그대로 두면 빛이 두 배로 새어 나온다.
그렇다고 후자를 막아버리면, 매질이 옅어서 광택처럼 좁게 산란하는 상황에서는 오히려 위상 함수 쪽이 훨씬 좋은 추정치인데 그걸 버리게 된다.

MIS(multiple importance sampling)는 둘 다 살리되 가중치를 나눠 갖는 방법이다. 여기서는 power heuristic을 썼다.

$$w_A = \frac{p_A^2}{p_A^2 + p_B^2}$$

두 추정치의 가중치를 합치면 정확히 1이 되므로 에너지가 새지 않는다.
게다가 자기가 잘하는 상황에서 자기 가중치가 커지도록 알아서 갈린다. 작은 광원이면 광원 샘플링 쪽이, 넓게 퍼진 밝은 면이면 위상 샘플링 쪽이 이긴다.

```glsl
float phasePdf = henyeyGreensteinPhase(
    clamp(dot(pathDirection, lightDirection), -1.0, 1.0),
    PHASE_G
);

float lightPdfOmega = lightSample.pdfArea * distanceSquared / cosOnLight;
float misWeight = powerHeuristic(lightPdfOmega, phasePdf);

return LIGHT_BOX_EMISSION * transmittance * phasePdf * misWeight /
    max(lightPdfOmega, 1e-8);
```

반대편, 그러니까 위상 함수를 따라 날아간 광선이 광원에 부딪혔을 때도 짝이 되는 가중치를 곱해줘야 한다.
이때는 인자의 순서가 뒤집힌다.

```glsl
if (rayWasPhaseSampled)
{
    float lightPdfOmega =
        (1.0 / emissiveBoxArea()) *
        (lightDistance * lightDistance) / cosOnLight;

    emissionWeight = powerHeuristic(previousPhasePdf, lightPdfOmega);
}

radiance += throughput * LIGHT_BOX_EMISSION * emissionWeight;
```

카메라에서 출발한 첫 광선이 곧바로 광원에 닿은 경우에는 `rayWasPhaseSampled`가 false라서 가중치를 곱하지 않는다.
그 경로는 NEE와 겹칠 일이 없기 때문에 온전히 다 받아야 한다.

## 언제 멈출 것인가: 러시안 룰렛 ##

산란을 몇 번까지 따라갈 것인가. 그냥 8번에서 자르면 그만큼의 에너지가 사라지고, 두꺼운 부분이 실제보다 어두워진다.
러시안 룰렛은 경로를 확률적으로 죽이되, 살아남은 경로의 기여를 그 확률로 나눠서 부풀린다.
평균적으로는 에너지가 보존되면서 경로 길이만 짧아지는, 공짜에 가까운 최적화다.

```glsl
if (bounce >= 2)
{
    float survivalProbability = clamp(maxComponent(throughput), 0.05, 0.95);

    if (randomFloat01(rngState) > survivalProbability)
    {
        break;
    }

    throughput /= survivalProbability;
}
```

이미 어두워진 경로일수록 일찍 죽는다. 밝은 경로는 오래 산다. 합리적이다.

## 누적과 톤매핑 ##

몬테카를로 추정치는 샘플 하나로는 노이즈 덩어리다. 결국 많이 쌓아야 한다.
그래서 Buffer A에 프레임을 누적한다. 알파 채널에 샘플 개수를 세어두고, $1/N$ 가중치로 이전 결과와 섞는 방식이다.
카메라가 움직이거나 스페이스바를 누르면 누적을 리셋한다.

```glsl
sampleCount = min(previousFrame.a + 1.0, 8192.0);
float blendWeight = 1.0 / sampleCount;

accumulatedRadiance = mix(previousFrame.rgb, currentSample, blendWeight);
```

픽셀 좌표에도 매 프레임 서브픽셀 지터를 준다. 누적이 쌓이면서 안티에일리어싱이 공짜로 따라온다.

마지막으로 Image 패스에서는 선형 공간에 쌓인 radiance에 노출을 곱하고, ACES 필믹 톤매핑을 거쳐 sRGB로 인코딩한다.
광원의 밝기가 18 같은 값이라 1.0을 한참 넘어가는데, 이걸 그냥 clamp해버리면 하이라이트가 종잇장처럼 뭉개진다.

```glsl
float exposure = 1.25;
vec3 color = acesFilm(linearRadiance * exposure);
color = linearToSRGB(color);
```

이런 식으로 빛이 단순히 하나의 방향으로 들어오는 것이 아니라, 실제로 산란하며 매질과 상호작용하는 것까지 고려해서 렌더하도록 만들 수 있다.
스텝 크기를 튜닝하며 눈치를 보는 대신, 샘플을 더 쌓으면 정직하게 정답으로 수렴한다. 참 쉽죠?
전체 코드는 [쉐이더토이에서](https://www.shadertoy.com/view/lfyyWt) 살펴볼 수 있다.
