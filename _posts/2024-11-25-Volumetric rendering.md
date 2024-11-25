---
title: "Volumetric rendering"
date: 2024-11-25T10:34:30+09:00
categories:
  - 작업
tags:
  - rendering
---

물체를 렌더하기 위해서는 물체를 어떻게 표현할 것인지 그 표현(Representation) 방법에 대해서 고려해야만 한다. 
일반적인 방법은 Polygon을 이용한 Representation이다.

이 방식은 물체의 형태를 정의하는 데 효과적이지만, 다음과 같은 이유로 볼륨 형태의 물체를 표현하는 데는 부적합하다.

1. 폴리곤 렌더링은 주로 물체의 외곽 경계를 정의하며, 내부의 밀도나 상태 변화는 표현할 수 없다. 
예를 들어, 구름이나 연기처럼 내부가 균일하지 않은 구조물은 폴리곤으로 적절히 표현하기 어렵다.

2. 구름이나 연기처럼 복잡한 형태를 폴리곤으로 표현하려면 수많은 폴리곤을 생성해야 하며, 이는 계산 비용과 메모리 사용량을 급격히 증가시킨다. 
이로 인해 사실적인 표현이 비효율적이고 제한적이다.

3. 폴리곤 렌더링은 표면과 빛의 상호작용을 기반으로 동작한다. 
그러나 볼륨 형태의 물체는 빛이 내부를 통과하며 흡수, 산란, 굴절되는 복잡한 상호작용을 수반한다. 
이러한 과정을 폴리곤으로 시뮬레이션하기에는 근본적인 제약이 따른다.

그러나, Representation에는 Polygon만 있는 것이 아니다. Voxel이나 Splat, SDF, Neural volume 등 다양한 Representation이 존재한다. 
볼륨 렌더링에서도 여러 Representarion을 사용할 수 있겠으나, 본 예제에서는 Voxel representation을 사용할 것이다.
Voxel은 "Volume"과 "Pixel"의 합성어로, 3D 공간에서의 격자 구조를 의미한다. 마인크래프트를 생각하면 좋다. 
각 Voxel은 물리적 특성(예: 밀도, 색상, 투명도 등)을 나타내며, 이는 물체 내부의 복잡한 상태를 표현하는 데 적합하다. 
이 Representation을 기반으로, 공간 내의 모든 데이터가 빛과 어떻게 상호작용하는지를 계산하여 화면에 시각화하는 방법을 알아보자.

## 볼륨 렌더링의 기본 원리 ##

## 광선 적분(Ray Integration) ##

광원에서 발사된 빛은 3D 공간 내에서 광선(ray) 형태로 이동하며 매질과 상호작용한다. 
이 과정은 광선 적분이라는 방정식을 통해 계산된다.
이는 1984년에 Siggraph에서 James T. Kajiya가 발표한 논문, 'Ray Tracing Volume Densities'에서 처음 소개되었다.
광선 적분은 빛이 물체 내부를 통과하면서 축적되는 색상과 밝기(에너지)를 계산하며, 매질의 흡수(Absorption), 산란(Scattering), 방출(Emission) 효과를 반영한다. 다만 본 예제에서 우리는 Emission을 고려하지 않는다.

$$C(t) = \int_{t_{near}}^{t_{far}} T(t) \cdot \sigma(t) \cdot c(t) \, dt$$

여기에서 $C(t)$는 $t$ 시점에서의 광선의 누적 색상, $\sigma(t)$는 산란계수(Scattering coefficient), $c(t)$는 색상을 나타낸다. 

이 식은 광선이 $t_{near}$에서 $t_{far}$까지 통과하는 동안의 색상을 계산한다. 이를 통해 물체 내부의 색상과 밝기를 계산할 수 있다.

$T(t)$는 투과도(transmittance)로, 물체를 통과하는 동안 빛이 얼마나 흡수되는지를 나타내는데, 다음과 같이 정의된다.

$$T(t) = \exp\left(-\int_{t_{\text{near}}}^{t} \sigma(s) \, ds\right)$$

위 식은 Beer's law의 일반화된 형태이다.

빛은 에너지를 가지고 있다. 빛이 매질(볼륨)을 통과하면서 이 에너지는 흡수되거나 산란된다. 이 중에서 흡수를 설명하는 것이 Beer's law이다. 

식을 잘 살펴보면, $t_{\text{near}}$에서 ${t}$까지 빛이 이동했을 때, 누적된 소멸 계수에 따라 빛이 얼마만큼 살아남는지를 나타내는 값이라는 것을 알 수 있다.

## Ray Marching ##

Ray Integration의 Analytic한 해를 얻는 것은 거의 불가능하다. 
우리의 목표는 물리적으로 정확하진 않더라도 그럴 듯한 결과물을 실시간으로 렌더하는 것이고, 따라서 위 적분식을 Riemann sum으로 근사해서 구현하기 위해 Ray marching이 사용된다. 
이는 Nelson Max가 1995년에 발표한 논문 'Optical Models for Direct Volume Rendering'에서 처음 소개되었다.
Ray marching은 말 그대로 광선이 카메라에서 출발해서 화면 안으로 일정 Step만큼 '행진하는' 것이라 상상하면 된다. 
카메라에서 출발한 광선이 3D 공간을 일정 간격으로 샘플링하며, 매질의 속성을 누적하는 것이다. 
이 과정에서 각 지점의 밀도와 색상 값을 합산하여 최종적으로 픽셀의 색상을 결정한다.
이를 식으로 표현하면 다음과 같다.

$$C(t) \approx \Delta t \sum_{k=0}^{M-1} T(t_k) \cdot \sigma(t_k) \cdot c(t_k)$$

여기에서 $\Delta t = \frac{t_{\text{far}} - t_{\text{near}}}{M}$ 는 샘플링 간격, $t_k = t_{\text{near}} + k \cdot \Delta t$는 $k$ 번째 샘플링 지점, $M$ 은 이산화된 샘플링 개수를 나타낸다.

## NeRF ##

사족을 달자면 NeRF 논문을 들여다보면 동일한 Ray integration 식이 등장하는 것을 확인할 수 있다. 
NeRF 또한 볼륨 샘플링 함수를 뉴럴 네트워크로 대체했을 뿐이지, 여전히 Volume rendering을 다루고 있기 때문이다.

![이미지](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-11-25%20Volumetric%20rendering/nerf.jpg?raw=true)
<div style="text-align: center;">
*NeRF 논문에서 등장하는 Ray Integration 식*
</div>

## Scattering ##

빛은 매질을 통과하며 산란한다. 
산란에는 여러 종류가 있지만, 자연에서 쉽게 관측할 수 있는 산란은 Rayleigh scattering과 Mie scattering이다.

Rayleigh scattering은 매우 작은 입자, 예를 들면 공기 분자와 같은 입자를 빛이 통과하며 발생한다. 
하늘이 파란 이유는 바로 이러한 Rayleigh scattering 때문이다. 빛의 파장이 짧은 파란색 빛은 쉽게 산란하는데, 이것이 우리 눈에 들어오기 때문이다.
반면에 저녁이 되면 태양이 지평선으로 가까워지면서 더 먼 거리를 통과하게 되는데, 이때는 파장이 긴 빛이 더 많이 산란되어 빨간 빛이 우세해진다. 
이 현상을 한 단어로 노을이라고도 한다.

Mie scattering은 Rayleigh scattering과 달리 더 큰 입자(예: 먼지, 연기, 구름)에 의해 발생한다. 
이러한 산란은 빛의 파장에 관계없이 모든 파장의 빛을 산란하며, 따라서 구름이나 입김은 하얗게 보인다.

## 외부 빛의 기여 (Direct lighting) ##

위 식에서 우리가 하나 빼먹은 것이 있다. 바로 외부 빛의 기여이다.
물체의 색상은 물체 자체의 색상과 외부 빛의 색상이 결합된 결과이다.
따라서 물체의 색상을 계산할 때에는 물체 자체의 색상과 외부 빛의 색상을 모두 고려해야 한다.

외부 조명까지 고려한다면 Ray integration을 다음과 같이 표현할 수 있다.

$$C(t) = \int_{t_{\text{near}}}^{t_{\text{far}}} T(t) \cdot \sigma(t) \cdot \left[c(t) + L_{\text{ext}}(t)\right] \, dt$$

일반적으로 $L_{\text{ext}}$를 나타내기 위해서는 아래와 같이 모든 방향에서 들어오는 빛을 고려해야 한다.

$$L_{\text{ext}}(t) = \int_{\Omega} T(t) \cdot I(\omega) \cdot p(\omega, t) \, d\omega$$

그러나, 이는 계산 비용이 매우 높기 때문에, 우리는 딱 하나의 방향($\omega$)만을 고려할 것이다.

$$L_{\text{ext}}(t) \approx \Delta s \cdot \sum_{k=0}^{M-1} T(s_k) \cdot \sigma(s_k) \cdot I(\omega_d) \cdot p(\omega_d, s_k)$$

각 변수들의 의미는 다음과 같다.

$M$: 샘플링 횟수

$\Delta s$: 샘플 간격

$s_k$: 샘플링 지점

$T(s_k)$: 투과도

$\sigma(s_k)$: 매질의 흡수계수

$I(\omega_d)$: $\omega_d$ 방향에서의 빛의 세기

$p(\omega_d, s_k)$: 샘플 지점 $s_k$에서 $\omega_d$으로 빛이 산란되는 확률을 나타내는 Phase Function

사실, 이것은 외부 조명을 고려하지 않은 C(t)를 구하는 것과 거의 유사하다. Phase function만 빼고.

## Phase function ##

Phase function은 관찰자의 시점에서 빛이 입자에 의해 특정 방향으로 산란되는 정도를 나타내는 함수이다. 
우리는 구름 볼륨 내에서의 Mie scattering을 다루고 있기 때문에, Mie scattering의 Phase function을 사용할 것이다.
이는 입사한 빛이 산란되는 각도에 따라 빛의 강도가 어떻게 분포되는지를 설명하며, 구체적으로는 입사 방향과 산란 방향 사이의 각도 $\theta$에 의존한다.
구름과 같은 볼륨 내에서는 빛이 물방울과 같은 입자들에 의해 여러 번 산란되며, 이 과정에서 Phase function은 전방 산란(Forward scattering)과 후방 산란(Backward scattering)의 상대적 비율을 결정한다.
전방 산란은 빛이 입자와 상호작용한 후에도 원래의 진행 방향과 유사한 방향으로 계속 이동하는 현상을 의미한다. 
이는 구름과 같은 매질에서 주요한 특징으로, 물방울의 크기가 빛의 파장보다 큰 경우 전방 산란이 우세하게 나타난다. 
전방 산란이 일어나면 빛이 매질을 통과하면서 비교적 적은 산란각으로 진행하게 되어, 구름이 상대적으로 투명하게 보이거나, 태양 주변의 밝은 후광 효과(Halo effect)를 생성하는 데 기여한다.

반면 후방 산란(Backward scattering)은 빛이 입자와 상호작용한 후 원래의 진행 방향과 반대 방향으로 반사되거나 크게 굴절되는 현상이다. 
이는 상대적으로 드물게 발생하며, 빛이 구름을 통과하지 못하고 되돌아오는 과정에서 관찰된다. 
후방 산란은 구름의 불투명도와 빛의 산란된 경로를 시각적으로 부드럽게 만드는 데 기여한다.

Mie scattering의 Phase function은 특히 입자의 크기와 빛의 파장에 민감하게 반응하며, 구름처럼 비교적 큰 입자가 포함된 매질에서는 전방 산란이 우세한 특성을 보인다. 
따라서 Phase function은 빛과 입자의 상호작용을 정확히 모델링하여 구름 내부의 광학적 성질과 시각적 특성을 결정하는 데 중요한 역할을 한다.
그런데, Mie scattering의 Phase function은 계산하기에 매우 비용이 많이 든다. 
따라서, 일반적으로는 Mie scattering의 Phase function을 Henyey-Greenstein phase function으로 근사하여 사용한다.
Henyey-Greenstein phase function은 다음과 같이 정의된다.

$$p(\cos\theta) = \frac{1}{4\pi} \cdot \frac{1 - g^2}{(1 + g^2 - 2g\cos\theta)^{3/2}}$$

여기서 $g$는 전방 산란과 후방 산란의 상대적 비율을 나타내는 매개변수이다.

$g = 0$의 경우 등방성 산란(Isotropic scattering)으로, 모든 방향으로 균등하게 산란된다.

$g > 0$의 경우 Forward scattering이 우세하다.

반대로, $g < 0$의 경우엔 Backward scattering이 우세하다고 볼 수 있다.


## Volumetric rendering의 구현 ##

다음은 위에서 설명한 내용으로 GLSL로 구현된 Volume rendering의 예제이다. 
본 코드가 어떻게 동작하는지는 [Shadertoy](https://www.shadertoy.com/view/MfKcWc)에서 확인할 수 있다.

![이미지](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-11-25%20Volumetric%20rendering/cloud.jpg?raw=true)
*<div style="text-align: center;">
Shadertoy 예제
</div>*


```glsl
#define FOWARD 0.8 // 전방 산란 계수
#define BACKWARD -0.2 // 후방 산란 계수
#define RAY_ITER 120 // Ray marching 반복 횟수
#define LIGHT_ITER 16 // 조명 계산 샘플 반복 횟수
#define LIGHT_ATTEN 64.0 // 빛 감쇠 계수
#define RAY_STEP_SIZE 0.01 // Ray marching 단계 크기

// 축 회전을 위한 함수
void rotate(inout vec3 z, vec3 axis, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    // 축 회전을 위한 회전 행렬 계산
    mat3 rot = mat3(
        c + axis.x * axis.x * (1.0 - c),       axis.x * axis.y * (1.0 - c) - axis.z * s, axis.x * axis.z * (1.0 - c) + axis.y * s,
        axis.y * axis.x * (1.0 - c) + axis.z * s, c + axis.y * axis.y * (1.0 - c),       axis.y * axis.z * (1.0 - c) - axis.x * s,
        axis.z * axis.x * (1.0 - c) - axis.y * s, axis.z * axis.y * (1.0 - c) + axis.x * s, c + axis.z * axis.z * (1.0 - c)
    );
    z = rot * z; // 벡터에 회전 적용
}

// 절차적 프랙탈 형태를 계산하는 함수
float fractal(vec3 p) {
    for (int i = 0; i < 8; i++) {
        // 시간에 따라 회전하는 프랙탈
        rotate(p, vec3(1.0, 0.0, 0.0), iTime * 0.2);
        rotate(p, vec3(0.0, 1.0, 0.0), iTime * 0.1);
        // 반사 대칭
        if (p.x + p.y < 0.0) p.xy = -p.yx;
        if (p.y + p.z < 0.0) p.yz = -p.zy;
        if (p.z + p.x < 0.0) p.zx = -p.xz;
        p -= 0.06; // 축소 및 이동
    }
    return length(p) - 0.15; // 최종 거리 계산
}

// SDF(거리 함수)로 프랙탈 활용
float sdf(vec3 p) {
    return fractal(p);
}

// Henyey-Greenstein Phase Function
float HenyeyGreenstein(float sundotrd, float g) {
    float gg = g * g;
    return (1. - gg) / pow(1. + gg - 2. * g * sundotrd, 1.5);
}

// 산란 계산 (전방 및 후방 산란 혼합)
float getScattering(float sundotrd) {
    return mix(HenyeyGreenstein(sundotrd, FOWARD), HenyeyGreenstein(sundotrd, BACKWARD), 0.5);
}

// 밀도 샘플링 (절차적 밀도 생성)
float sampleDensity(vec3 p) {
    return pow(max(-sdf(p), 0.0), 1.3) * 10.0; // SDF 기반 밀도 및 증폭
}

// 빛의 위치를 Lissajous 곡선으로 계산
vec3 lightPosLissajous(float t) {
    float A = 1.5;  // x축 진폭
    float B = 1.2;  // y축 진폭
    float C = 1.1;  // z축 진폭
    float a = 3.1;  // x축 주파수
    float b = 2.2;  // y축 주파수
    float c = 4.3;  // z축 주파수
    float delta = 0.2; // 위상 차이

    float x = A * sin(a * t + delta);
    float y = B * sin(b * t);
    float z = C * sin(c * t);

    return vec3(x, y, z); // 빛의 동적 위치 반환
}

// 메인 렌더링 함수
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // 정규화된 픽셀 좌표 [-1, 1]
    vec2 uv = fragCoord / iResolution.xy;
    uv = (uv - 0.5) * 2.0;

    vec3 col = vec3(0.0); // 초기 색상 값

    vec3 camPos = vec3(0.0, 0.0, -2.0); // 카메라 위치
    vec3 rayPos = camPos; // 광선 시작점
    vec3 rayDir = normalize(vec3(uv, 0.0) - camPos); // 광선 방향
    float time = iTime * 0.2; // 동적 시간
    vec3 lightPos = lightPosLissajous(time); // 빛의 위치 계산

    float transmittance = 1.0; // 초기 투과도

    rayPos += rayDir; // 광선 이동 시작
    for (int i = 0; i < RAY_ITER; i++) {
        rayPos += rayDir * RAY_STEP_SIZE; // 광선 전진
        float density = sampleDensity(rayPos); // 현재 위치의 밀도 계산
        if (density <= 0.0) {
            continue; // 밀도가 없으면 다음 반복
        }
        vec3 lightDir = lightPos - rayPos; // 빛 방향
        float lightDistance = length(lightDir); // 빛 거리
        lightDir = lightDir / lightDistance; // 단위 벡터로 정규화
        float lightStep = lightDistance / float(LIGHT_ITER); // 조명 단계 크기
        float sundotrd = dot(rayDir, -lightDir); // 광선과 빛 방향의 내적
        float scattering = getScattering(sundotrd); // 산란 계산
        vec3 lightRayPos = rayPos; // 그림자 계산용 광선 위치
        float shadowDensity = 0.0; // 그림자 밀도 초기화
        for (int j = 0; j < LIGHT_ITER; j++) {
            shadowDensity += sampleDensity(lightRayPos) * lightStep; // 그림자 밀도 누적
            lightRayPos += lightDir * lightStep; // 빛 방향으로 전진
        }
        vec3 externalLight = vec3(exp(-shadowDensity * LIGHT_ATTEN) * scattering); // 외부 빛 계산
        col += transmittance * externalLight * density; // 누적된 색상
        transmittance *= exp(-density * RAY_STEP_SIZE * LIGHT_ATTEN); // 투과도 갱신
        if (transmittance < 0.01) break; // 투과도가 낮으면 조기 종료
    }

    col = pow(col, vec3(1.0 / 2.2)); // 감마 보정
    fragColor = vec4(col, 1.0); // 최종 색상 출력
}
```

만약 TouchDesigner에서 구현한다면, 다음과 같을 것이다.


```
#define FOWARD 0.8 // 전방 산란 계수
#define BACKWARD -0.2 // 후방 산란 계수
#define RAY_ITER 120 // Ray marching 반복 횟수
#define LIGHT_ITER 16 // 조명 계산 샘플 반복 횟수
#define LIGHT_ATTEN 64.0 // 빛 감쇠 계수
#define RAY_STEP_SIZE 0.01 // Ray marching 단계 크기

uniform float iTime;

out vec4 fragColor;

// 축 회전을 위한 함수
void rotate(inout vec3 z, vec3 axis, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    // 축 회전을 위한 회전 행렬 계산
    mat3 rot = mat3(
        c + axis.x * axis.x * (1.0 - c),       axis.x * axis.y * (1.0 - c) - axis.z * s, axis.x * axis.z * (1.0 - c) + axis.y * s,
        axis.y * axis.x * (1.0 - c) + axis.z * s, c + axis.y * axis.y * (1.0 - c),       axis.y * axis.z * (1.0 - c) - axis.x * s,
        axis.z * axis.x * (1.0 - c) - axis.y * s, axis.z * axis.y * (1.0 - c) + axis.x * s, c + axis.z * axis.z * (1.0 - c)
    );
    z = rot * z; // 벡터에 회전 적용
}

// 절차적 프랙탈 형태를 계산하는 함수
float fractal(vec3 p) {
    for (int i = 0; i < 8; i++) {
        // 시간에 따라 회전하는 프랙탈
        rotate(p, vec3(1.0, 0.0, 0.0), iTime * 0.2);
        rotate(p, vec3(0.0, 1.0, 0.0), iTime * 0.1);
        // 반사 대칭
        if (p.x + p.y < 0.0) p.xy = -p.yx;
        if (p.y + p.z < 0.0) p.yz = -p.zy;
        if (p.z + p.x < 0.0) p.zx = -p.xz;
        p -= 0.06; // 축소 및 이동
    }
    return length(p) - 0.15; // 최종 거리 계산
}

// SDF(거리 함수)로 프랙탈 활용
float sdf(vec3 p) {
    return fractal(p);
}

// Henyey-Greenstein Phase Function
float HenyeyGreenstein(float sundotrd, float g) {
    float gg = g * g;
    return (1. - gg) / pow(1. + gg - 2. * g * sundotrd, 1.5);
}

// 산란 계산 (전방 및 후방 산란 혼합)
float getScattering(float sundotrd) {
    return mix(HenyeyGreenstein(sundotrd, FOWARD), HenyeyGreenstein(sundotrd, BACKWARD), 0.5);
}

// 밀도 샘플링 (절차적 밀도 생성)
float sampleDensity(vec3 p) {
    return pow(max(-sdf(p), 0.0), 1.3) * 10.0; // SDF 기반 밀도 및 증폭
}

// 빛의 위치를 Lissajous 곡선으로 계산
vec3 lightPosLissajous(float t) {
    float A = 1.5;  // x축 진폭
    float B = 1.2;  // y축 진폭
    float C = 1.1;  // z축 진폭
    float a = 3.1;  // x축 주파수
    float b = 2.2;  // y축 주파수
    float c = 4.3;  // z축 주파수
    float delta = 0.2; // 위상 차이

    float x = A * sin(a * t + delta);
    float y = B * sin(b * t);
    float z = C * sin(c * t);

    return vec3(x, y, z); // 빛의 동적 위치 반환
}

// 메인 렌더링 함수
void main() {
    // 정규화된 픽셀 좌표 [-1, 1]
    uv = (vUV.st - 0.5) * 2.0;

    vec3 col = vec3(0.0); // 초기 색상 값

    vec3 camPos = vec3(0.0, 0.0, -2.0); // 카메라 위치
    vec3 rayPos = camPos; // 광선 시작점
    vec3 rayDir = normalize(vec3(uv, 0.0) - camPos); // 광선 방향
    float time = iTime * 0.2; // 동적 시간
    vec3 lightPos = lightPosLissajous(time); // 빛의 위치 계산

    float transmittance = 1.0; // 초기 투과도

    rayPos += rayDir; // 광선 이동 시작
    for (int i = 0; i < RAY_ITER; i++) {
        rayPos += rayDir * RAY_STEP_SIZE; // 광선 전진
        float density = sampleDensity(rayPos); // 현재 위치의 밀도 계산
        if (density <= 0.0) {
            continue; // 밀도가 없으면 다음 반복
        }
        vec3 lightDir = lightPos - rayPos; // 빛 방향
        float lightDistance = length(lightDir); // 빛 거리
        lightDir = lightDir / lightDistance; // 단위 벡터로 정규화
        float lightStep = lightDistance / float(LIGHT_ITER); // 조명 단계 크기
        float sundotrd = dot(rayDir, -lightDir); // 광선과 빛 방향의 내적
        float scattering = getScattering(sundotrd); // 산란 계산
        vec3 lightRayPos = rayPos; // 그림자 계산용 광선 위치
        float shadowDensity = 0.0; // 그림자 밀도 초기화
        for (int j = 0; j < LIGHT_ITER; j++) {
            shadowDensity += sampleDensity(lightRayPos) * lightStep; // 그림자 밀도 누적
            lightRayPos += lightDir * lightStep; // 빛 방향으로 전진
        }
        vec3 externalLight = vec3(exp(-shadowDensity * LIGHT_ATTEN) * scattering); // 외부 빛 계산
        col += transmittance * externalLight * density; // 누적된 색상
        transmittance *= exp(-density * RAY_STEP_SIZE * LIGHT_ATTEN); // 투과도 갱신
        if (transmittance < 0.01) break; // 투과도가 낮으면 조기 종료
    }

    col = pow(col, vec3(1.0 / 2.2)); // 감마 보정
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}

```