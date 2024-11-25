---
title: "GLSL Volumetric rendering (좀 더 친절하고 쉬운 버전)"
date: 2024-11-25T10:34:30+09:00
categories:
  - 작업
tags:
  - rendering
  - volumetric
  - cloud
---

- 좀 더 엄밀하고 어려운 버전은 [여기](https://okdalto.github.io/%EC%9E%91%EC%97%85/GLSL-Volumetric-rendering-(%EC%A2%80-%EB%8D%94-%EC%97%84%EB%B0%80%ED%95%98%EA%B3%A0-%EC%96%B4%EB%A0%A4%EC%9A%B4-%EB%B2%84%EC%A0%84)/)를 참고하세요

## 물체를 표현하는 방법: Polygon과 Voxel Representation ##

3D 공간에서 물체를 렌더링하려면 물체를 어떻게 표현할지(Representation)가 중요합니다. 일반적으로 가장 널리 쓰이는 방식은 폴리곤(Polygon) 기반 표현입니다. 폴리곤은 물체의 형태를 정의하는 데 유용하지만, 볼륨 형태의 물체를 표현하는 데는 한계가 있습니다. 그 이유는 다음과 같습니다:

1. 내부 정보 부족:
폴리곤은 주로 물체의 외곽 경계만 정의하며, 내부의 밀도나 상태 변화를 표현하지 못합니다.
예: 구름, 연기 같은 내부가 복잡한 구조물.

2. 복잡도 증가:
이런 복잡한 형태를 폴리곤으로 표현하려면 수많은 폴리곤이 필요합니다. 이는 계산 비용과 메모리 사용량을 크게 증가시켜 비효율적입니다.

3. 빛 상호작용의 제약:
폴리곤은 표면에서 빛이 반사되고 굴절되는 것을 표현하지만, 빛이 물체 내부를 통과하며 발생하는 흡수, 산란 같은 현상을 다루는 데 한계가 있습니다.

## 다른 Representation 방식 ##

폴리곤 외에도 다양한 방법이 존재합니다:

- Voxel: 3D 격자로 구성된 방식. (마인크래프트처럼 격자로 나눈 블록 형태)
- SDF(Signed Distance Function): 거리 기반으로 물체를 표현.
- Splat: 점을 중심으로 물체를 표현.
- Neural Volume: 뉴럴 네트워크로 표현.

여기서는 Voxel Representation을 사용해 볼륨 형태의 물체를 다룹니다. Voxel은 "Volume"과 "Pixel"의 합성어로, 3D 공간을 격자로 나눠 각 격자(Voxel)가 물리적 특성(밀도, 색상, 투명도 등)을 가집니다. 이를 이용하면 물체 내부의 복잡한 상태를 표현하기 쉽습니다.

![이미지](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-11-25%20Volumetric%20rendering/3D_representations.jpg?raw=true)
*다양한 Shape representation. 좌측 위부터 시계방향으로 SDF, Voxel, Polygon, Splat.*

## 빛은 어떻게 표현할까? ##

구름처럼 내부가 복잡한 물체는 빛이 단순히 반사되거나 굴절되지 않고, 물체를 통과하며 산란되고 흡수됩니다.
이를 계산해서 화면에 보이는 모습을 만들어내야 합니다.

## Ray Marching: 빛의 경로를 따라가 보기 ##

빛은 카메라에서 출발해 물체를 향해 쭉 나아갑니다.
이 빛의 경로를 작은 간격으로 나눠서 물체 내부를 하나씩 확인해 보는 방식이 Ray Marching입니다.

1. 빛의 경로를 따라가며 확인: 카메라에서 출발한 광선이 물체 내부를 지나면서 각 블록의 색깔, 밀도, 투명도를 누적합니다.

2. 결과 색상 계산: 누적한 정보를 바탕으로 빛이 얼마나 흡수되고, 얼마나 투명하며, 어떤 색깔을 가지는지 계산합니다.

## 구름을 더 사실적으로: 빛의 산란 ##

빛은 물체 내부에서 산란됩니다. 쉽게 말하면, 빛이 물체에 부딪힌 후 여러 방향으로 퍼져 나갑니다.

전방 산란(Forward Scattering)
빛이 부딪힌 뒤 원래 방향과 비슷하게 나아가는 경우.
→ 구름 속 빛의 투명한 느낌을 만듭니다.

후방 산란(Backward Scattering)
빛이 부딪힌 뒤 반대로 튕겨 나오는 경우.
→ 구름의 뭉개진 가장자리 표현에 기여합니다.

구름은 전방 산란이 많아서 태양 근처에서 환하게 빛나는 후광 효과를 만듭니다.

## Volumetric rendering의 구현 ##

다음은 위에서 설명한 내용으로 GLSL로 구현된 Volume rendering의 예제입니다. 
본 코드가 어떻게 동작하는지는 [Shadertoy](https://www.shadertoy.com/view/MfKcWc)에서 확인할 수 있습니다.

![이미지](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-11-25%20Volumetric%20rendering/cloud.jpg?raw=true)
*Shadertoy 예제*


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

만약 TouchDesigner에서 구현한다면, 다음과 같을 것입니다.


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