---
title: "GLSL Volumetric rendering (좀 더 엄밀하고 어려운 버전)"
date: 2024-12-21T10:34:30+09:00
categories:
  - 작업
tags:
  - rendering
  - volumetric
  - cloud
---

## 광선 적분 ##

![이미지](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-11-25%20Volumetric%20rendering/3D_representations.jpg?raw=true)

이전 글에서 Ray integration을 단순화하여 Ray marching으로 Volume rendering을 구현하는 방법에 대해 다뤘다. 
해당 구현에서는 단 하나의 방향에서만 들어오는 빛을 가정하였으나, 실제로는 각 Ray step마다 구면의 모든 방향으로 들어오는 빛을 고려하는 것이 좀 더 물리적으로 정확하다 할 수 있겠다.
더 자세한 설명을 위해 다시 Ray Integration을 들여다보자. 광선 적분의 일반적인 형태는 다음과 같다.

$$C(t) = \int_{t_{\text{near}}}^{t_{\text{far}}} T(t) \cdot \sigma(t) \cdot \left[c(t) + L_{\text{ext}}(t)\right] \, dt$$

각 변수 및 함수의 의미는 다음과 같다. 

$C(t)$: 최종적으로 얻어지는 광선의 색상 또는 밝기(에너지). 이는 광선 적분 결과로, 매질을 통과한 광선이 얼마나 축적되었는지 나타낸다.

$\sigma(t)$: 흡수와 산란의 총합.

$c(t)$: 매질의 고유 색상(Color) 및 강도(Intensity) 정보. 특정 지점에서 매질이 가진 고유한 색과 밝기를 나타내며, 광선 적분에 기여한다.

$t_{near}$, $t_{far}$: 광선이 매질과 교차하는 시작점과 끝점.
광선 적분은 광원이 발사한 빛이 매질과 처음 만나기 시작한 지점 $t_{near}$ 에서 매질을 통과하며 끝나는 지점 $t_{far}$ 까지 수행된다.

$T(t)$는 투과도(transmittance)로, 물체를 통과하는 동안 빛의 강도가 약화되는 정도를 설명하는데, 다음과 같이 정의된다.

$$T(t) = \exp\left(-\int_{t_{\text{near}}}^{t} \sigma(s) \, ds\right)$$

위 식은 [Beer-Lambert Law](https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law)의 일반화된 형태이다. 빛은 에너지를 가지고 있다. 빛이 매질(볼륨)을 통과하면서 이 에너지는 흡수되거나 산란된다. 이 중에서 흡수를 설명하는 것이 Beer's law이다. 식을 잘 살펴보면, $t_{\text{near}}$에서 ${t}$까지 빛이 이동했을 때, 누적된 소멸 계수에 따라 빛이 얼마만큼 살아남는지를 나타내는 값이라는 것을 알 수 있다.

일반적으로 $L_{\text{ext}}$를 나타내기 위해서는 아래와 같이 모든 방향에서 들어오는 빛을 고려해야 한다.

$$L_{\text{ext}}(t) = \int_{\Omega} T(t) \cdot I(\omega) \cdot p(\omega, t) \, d\omega$$

## 이전의 방법과 차이점 ##

코드에서 구현된 Ray Integration은 주어진 수식을 이산 형태(Discrete Form)로 근사하여 계산한다. 
여러 가지 기능이 구현되어 있지만, 이전 구현과의 차이는 간단하다. 
외부에서 들어오는 빛을 단일 방향으로 근사하는 것이 아니라, 실제 물리적 성질을 바탕으로 빛이 여러 방향으로 반사/굴절되는 현상을 시뮬레이션하는 것이다.

따라서 나는 매 step마다 각 ray에서 빛을 발사하고, 그 빛이 확률적으로 다른 곳으로 Scattering하도록 만들었다.
산란 확률은 Beer's law에 따라서 아래와 같이 결정된다.

$$P(\text{scatter}) = 1.0 - e^{-\sigma_s \cdot \Delta x}$$

코드로 표현한다면 다음과 같다.

```glsl
float calculateScatterProbability(float sigma_s, float delta_x) {
    return 1.0 - exp(-sigma_s * delta_x);
}
```

Scattering확률을 구하고, 만약 Scattering한다고 결정된다면, 그 빛의 방향은 랜덤하게 결정되는 것이 아니라 Henyey-Greenstein Phase Function에 따라 샘플링된다.
그 방식은 좀 복잡한데 먼저 코드를 살펴보면 다음과 같다.


```glsl

float samplePhi(float xi2) {
    return c_twopi * xi2;
}

vec3 computeNewDirection(float cosTheta, float phi) {
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return vec3(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta
    );
}

vec3 alignToDirection(vec3 newDir, vec3 omegaPrime) {
    vec3 zAxis = vec3(0.0, 0.0, 1.0);
    vec3 v = cross(zAxis, omegaPrime);
    float s = length(v);
    float c = dot(zAxis, omegaPrime);
    mat3 rotation;

    if (s < 0.00001) { // 거의 0에 가까울 때는 회전 없이 반환
        return newDir;
    }

    float invsSq = 1.0 / (s * s);
    mat3 vSkew = mat3(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0
    );

    // Rodrigues' rotation formula: R = I + vSkew + vSkew^2 * (1 - c) / s^2
    mat3 vSkewSq = vSkew * vSkew;
    rotation = mat3(1.0) + vSkew + vSkewSq * (1.0 - c) * invsSq;

    return rotation * newDir;
}

vec3 sampleHenyeyGreenstein(vec3 omegaPrime, float g, uint rngState) {
    float xi1 = randomFloat01(rngState);
    float xi2 = randomFloat01(rngState);
    float cosTheta = sampleCosTheta(g, xi1);
    float phi = samplePhi(xi2);
    vec3 newDir = computeNewDirection(cosTheta, phi);
    return alignToDirection(newDir, omegaPrime);
}

float calculateScatterProbability(float sigma_s, float delta_x) {
    return 1.0 - exp(-sigma_s * delta_x);
}

float rand(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

```

sampleHenyeyGreenstein의 사용

```glsl

float shadowScatterProbability = calculateScatterProbability(shadowDensity, LIGHT_STEP);
if (shadowScatterProbability > randomFloat01(rngState)) {
    lightDir = sampleHenyeyGreenstein(lightDir, 0.6, rngState);
}

```

`sampleHenyeyGreenstein` 함수는 Henyey-Greenstein 위상 함수를 사용하여 주어진 방향에서 산란된 새로운 방향 벡터를 샘플링한다. 
이 함수는 먼저 난수를 생성한다. 첫 번째 난수는 Henyey-Greenstein 분포에서 $\cos\theta$를 샘플링하는 데 사용되고, 두 번째 난수는 방위각 $\phi$를 결정한다.
$\cos\theta$는 주어진 비대칭 파라미터 $g$를 사용해 샘플링되며, $g$ 값은 산란의 특성을 조정한다.
$g > 0$이면 빛이 전방으로 산란되고, $g < 0$이면 후방으로 산란되며, $g = 0$이면 등방성 산란을 나타낸다.
이후 $\phi$는 $2\pi$를 곱해 방위각을 결정하며, $\cos\theta$와 $\phi$를 사용해 새로운 방향 벡터를 계산한다. 이 벡터는 기본적으로 $z$축 기준으로 정의된다.
이렇게 생성된 벡터는 `alignToDirection` 함수를 통해 기준 방향으로 정렬된다. 
이 과정은 회전축과 회전 행렬을 계산해 벡터를 회전시키는 방식으로 이루어지며, 최종적으로 기준 방향과 정렬된 새로운 벡터가 반환된다.

전체 코드는 [쉐이더토이에서](https://www.shadertoy.com/view/lfyyWt) 살펴볼 수 있다.
