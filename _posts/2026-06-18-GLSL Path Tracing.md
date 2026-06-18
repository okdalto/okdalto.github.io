---
title: "GLSL Path Tracing"
date: 2026-06-18T10:34:30+09:00
categories:
  - 작업
tags:
  - rendering
  - path tracing
  - GLSL
---

<figure>
<img src="/assets/2026-06-18-path-tracing/fractal-render.jpg" alt="이 글의 GLSL path tracer로 렌더링한 금속 fractal">
<figcaption markdown="span">이 글에서 뜯어볼 GLSL path tracer로 렌더링한 결과. 공간을 접어 만든 금속 fractal이 환경을 비추고, thin lens 카메라가 얕은 피사계 심도를 만든다. 전체 코드는 [Shadertoy](https://www.shadertoy.com/view/dstyzH)에서 직접 돌려볼 수 있다.</figcaption>
</figure>

[빛과 렌더링](/생각/빛과-렌더링/)에서는 빛이 물질과 어떻게 상호작용하는지를 물리적으로 살펴봤고, [Volume Rendering, Path Tracing 버전](/작업/Volume-Rendering-%28Path-Tracing%29/)에서는 그 빛을 매질 속에서 Monte Carlo 방식으로 추적하는 방법을 다뤘다. 이번 글에서는 매질이 아니라 표면(surface)을 다룬다. SDF로 정의된 장면에 빛을 쏘고, 표면에서 반사·확산되는 광선을 끝까지 따라가 한 장의 이미지를 만들어내는 GLSL Path Tracer를 처음부터 끝까지 뜯어보자.

## 렌더링 방정식 ##

물리 기반 렌더링은 결국 하나의 식을 푸는 일이다. 1986년 James Kajiya가 전설적인 논문 [The Rendering Equation](https://dl.acm.org/doi/10.1145/15922.15902)에서 정리한 렌더링 방정식은 다음과 같다.

$$L_o(\mathbf{p}, \omega_o) = L_e(\mathbf{p}, \omega_o) + \int_{\Omega} f_r(\mathbf{p}, \omega_i, \omega_o)\, L_i(\mathbf{p}, \omega_i)\, (\omega_i \cdot \mathbf{n})\, d\omega_i$$

어떤 점 $\mathbf{p}$에서 방향 $\omega_o$로 나가는 빛 $L_o$는, 그 점이 스스로 방출하는 빛 $L_e$와, 반구 $\Omega$의 모든 방향 $\omega_i$에서 들어온 빛 $L_i$가 표면에서 반사된 양의 합이다. $f_r$은 입사 방향과 반사 방향의 관계를 정의하는 BRDF(Bidirectional Reflectance Distribution Function)이고, $(\omega_i \cdot \mathbf{n})$은 빗겨 들어온 빛이 표면에 덜 기여하는 Lambert cosine 항이다.

<figure>
<img src="/assets/2026-06-18-path-tracing/brdf-diagram.svg" alt="BRDF의 입사 방향, 출사 방향, normal vector 다이어그램">
<figcaption markdown="span">한 점에서의 빛의 관계. 들어오는 빛 ωᵢ, 나가는 빛 ωₒ, 그리고 normal vector n. BRDF $f_r$는 ωᵢ로 들어온 빛이 ωₒ로 얼마나 반사되는지를 정의한다. 출처: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:BRDF_Diagram.svg)</figcaption>
</figure>

문제는 우변의 $L_i$ 안에 다시 똑같은 적분이 들어있다는 점이다. 한 점에 들어오는 빛은 다른 표면에서 나간 빛이고, 그 빛은 또 다른 표면에서 온 빛이다. 이 재귀를 닫힌 형태로 푸는 것은 불가능하다.

그래서 [볼륨 렌더링 글](/작업/Volume-Rendering-%28Path-Tracing%29/)에서와 똑같은 전략을 쓴다. 반구 전체를 적분하는 대신, 방향을 무작위로 하나 골라 빛을 따라가고, 그것을 여러 번 반복해 평균낸다. Monte Carlo 적분이다.

$$L_o \approx L_e + \frac{1}{N}\sum_{k=1}^{N} \frac{f_r\, L_i\, (\omega_i \cdot \mathbf{n})}{p(\omega_i)}$$

여기서 $p(\omega_i)$는 그 방향을 뽑을 확률밀도다. 분모에 확률을 나눠주는 것이 핵심인데, 잘 안 뽑히는 방향이 우연히 선택되면 그만큼 크게 보정해줘야 추정값이 한쪽으로 치우치지 않기 때문이다. 이 보정항을 매 반사마다 곱해 나가는 변수를 보통 throughput이라 부른다. 코드 전체가 사실상 이 한 줄을 광선 한 가닥에 대해 구현한 것이다.

## 장면을 거리로: SDF와 Ray Marching ##

장면은 폴리곤이 아니라 SDF(Signed Distance Function)로 정의한다. 공간의 한 점을 넣으면 가장 가까운 표면까지의 거리를 돌려주는 함수다. 바닥은 단순한 평면이다.

```glsl
float sdPlane(vec3 p, vec3 n, float h)
{
    return dot(p, n) + h;
}
```

흥미로운 것은 주인공인 fractal이다. 공간을 접고(fold), 회전시키고, 이동시키는 과정을 세 번 반복해 distance field를 만든다.

```glsl
vec2 foldPair(vec2 v)
{
    float s = v.x + v.y;
    float d = v.x - v.y;
    return 0.5 * vec2(abs(s) + d, abs(s) - d);
}

float fractal(vec3 p)
{
    for (int i = 0; i < 3; i++)
    {
        rotX(p, iTime * 0.1);
        rotY(p, iTime * 0.2);
        rotZ(p, iTime * 0.3);

        p.xy = foldPair(p.xy);
        p.yz = foldPair(p.yz);
        p.zx = foldPair(p.zx);

        p -= 0.2;
    }

    return length(p) - 0.3;
}
```

`foldPair`는 평면을 대각선 기준으로 접는 연산이다. 마지막에 그냥 구의 distance field `length(p) - 0.3`을 반환하지만, 그 앞에서 공간을 여러 번 접어 두었기 때문에 하나의 구가 접힌 만큼 복제되어 복잡한 fractal 형태로 나타난다.

이렇게 정의한 distance field를 따라 광선을 전진시키는 것이 Ray Marching이다. 현재 위치에서 표면까지의 거리만큼은 무엇과도 부딪히지 않는다는 것이 보장되므로, 그 거리만큼 안전하게 점프하며 표면에 닿을 때까지 나아간다.

<figure>
<img src="/assets/2026-06-18-path-tracing/ray-marching.png" alt="sphere tracing으로 표면까지의 거리만큼 전진하는 과정 시각화">
<figcaption markdown="span">Ray Marching(sphere tracing). 각 지점에서 표면까지의 거리를 반지름으로 삼으면 그 원 안에서는 절대 부딪히지 않으니, 그만큼 안전하게 전진한다. 표면에 가까워질수록 걸음이 촘촘해진다. 출처: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Visualization_of_SDF_ray_marching_algorithm.png)</figcaption>
</figure>

```glsl
bool RayMarch(vec3 ro, vec3 rd, out vec3 hitPos, out HitInfo hit)
{
    float t = 0.0;

    for (int i = 0; i < MAX_MARCH_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        hit = mapScene(p);

        if (hit.dist < SURF_EPS)
        {
            hitPos = p;
            return true;
        }

        if (t > MAX_DIST)
        {
            break;
        }

        t += max(hit.dist * 0.8, SURF_EPS);
    }

    hitPos = ro + rd * t;
    return false;
}
```

공간을 접어 만든 distance field는 실제 거리를 약간 과대평가하는 경우가 있어 그대로 점프하면 표면을 뚫고 지나갈 수 있다. 그래서 거리의 80%(`hit.dist * 0.8`)만큼만 조심스럽게 전진한다.

## thin lens camera ##

가장 단순한 카메라는 한 점(pinhole)에서 광선을 쏜다. 그러면 모든 것이 또렷하게 찍히지만, 우리 눈이나 실제 렌즈가 만드는 out-of-focus(피사계 심도)는 표현할 수 없다. 그래서 thin lens 모델을 쓴다.

```glsl
vec3 pinholeDir = normalize(
    forward * focalLength +
    right * screen.x +
    up * screen.y
);

float focusDist = 1.4;
float lensRadius = 0.05;

vec3 focusPoint = camPos + pinholeDir * focusDist;

vec2 lens = RandomInDisk(rngState) * lensRadius;

vec3 ro = camPos + right * lens.x + up * lens.y;
vec3 rd = normalize(focusPoint - ro);
```

원리는 이렇다. 먼저 pinhole 카메라라면 광선이 향했을 방향(`pinholeDir`)을 따라 초점 거리(`focusDist`)만큼 떨어진 곳에 초점 평면 위의 한 점(`focusPoint`)을 잡는다. 그다음 광선의 출발점을 한 점이 아니라 반지름 `lensRadius`짜리 원판 위의 무작위 점으로 흩뜨린다(`RandomInDisk`). 출발점은 흔들리지만 모든 광선이 똑같은 `focusPoint`를 향하므로, 초점 평면 위의 물체는 항상 또렷하게 모이고 그보다 가깝거나 먼 물체는 흩어져 흐려진다. `lensRadius`를 키우면 조리개를 연 것처럼 흐림이 강해진다.

<figure>
<img src="/assets/2026-06-18-path-tracing/dof.svg" alt="조리개 크기에 따른 피사계 심도 변화 다이어그램">
<figcaption markdown="span">렌즈 원판(조리개)을 통과한 광선이 초점 평면 위에서는 한 점에 모이지만, 그 앞뒤의 점은 흩어져 흐려진다. 조리개를 키울수록(`lensRadius`↑) 흐림이 강해진다. 출처: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Lens_out_of_focus_DOF_diagram.svg)</figcaption>
</figure>

## 광선을 따라가기 ##

이제 본체인 경로 추적 루프다. 광선 한 가닥이 장면 안에서 최대 `MAX_BOUNCES`번 튕기며 빛을 모은다.

```glsl
vec3 radiance = vec3(0.0);
vec3 throughput = vec3(1.0);

for (int bounce = 0; bounce < MAX_BOUNCES; bounce++)
{
    vec3 hitPos;
    HitInfo hit;

    bool didHit = RayMarch(ro, rd, hitPos, hit);

    if (!didHit)
    {
        radiance += throughput * SampleEnvironment(rd);
        break;
    }

    vec3 n = getNormal(hitPos);

    if (dot(n, rd) > 0.0)
    {
        n = -n;
    }

    Material mat = hit.mat;

    radiance += throughput * mat.emission;

    ro = hitPos + n * SURF_EPS * 4.0;
```

`radiance`는 지금까지 모은 빛, `throughput`은 앞서 설명한 보정항의 누적 곱이다. 광선이 아무것도 맞히지 못하고 장면을 빠져나가면 environment map(HDRI)을 샘플링해 그 빛을 `throughput`만큼 실어 더하고 끝낸다. 결국 이 path tracer에서 빛은 전적으로 환경에서 들어온다.

표면에 닿으면 먼저 normal vector를 구한다. distance field에서 normal vector는 각 축 방향으로 거리의 gradient를 구해 얻는다.

```glsl
vec3 getNormal(vec3 p)
{
    vec2 e = vec2(NORMAL_EPS, 0.0);

    return normalize(vec3(
        sdf(p + e.xyy) - sdf(p - e.xyy),
        sdf(p + e.yxy) - sdf(p - e.yxy),
        sdf(p + e.yyx) - sdf(p - e.yyx)
    ));
}
```

normal vector가 광선과 같은 쪽을 보고 있으면 뒤집어 항상 광선을 마주보게 만들고(`dot(n, rd) > 0.0`), 표면이 스스로 빛을 낸다면 그 발광(emission)을 더한다. 마지막으로 다음 광선의 출발점을 표면에서 normal vector 방향으로 살짝 띄운다. 이렇게 하지 않으면 새 광선이 방금 맞힌 표면에 곧바로 다시 부딪히는 self-intersection이 생긴다.

## Fresnel과 재질 ##

표면에 닿은 빛이 반사될지 확산될지를 결정하는 것이 재질이다. [빛과 렌더링](/생각/빛과-렌더링/)에서 다뤘듯, 매질의 경계에서 빛이 얼마나 반사되는가는 굴절률(IOR)로 정해지고, 그 비율은 보는 각도에 따라 달라진다. 정면에서 볼 때의 반사율 $F_0$는 다음과 같다.

$$F_0 = \left(\frac{n_1 - n_2}{n_1 + n_2}\right)^2$$

```glsl
float F0FromIOR(float n1, float n2)
{
    float r0 = (n1 - n2) / (n1 + n2);
    return r0 * r0;
}
```

각도에 따른 변화는 매번 정확히 계산하기엔 비싸서, 보통 Schlick 근사를 쓴다.

$$F(\theta) = F_0 + (1 - F_0)\,(1 - \cos\theta)^5$$

표면을 스치듯 비스듬히 볼수록($\cos\theta \to 0$) 반사율이 1에 가까워진다. 이 현상 때문에 물웅덩이를 정면에서 보면 바닥이 비치지만, 멀리 비스듬히 보면 하늘이 거울처럼 비친다.

<figure>
<img src="/assets/2026-06-18-path-tracing/fresnel-plot.svg" alt="입사각에 따른 Fresnel 반사율 곡선 (n=1 → 1.5)">
<figcaption markdown="span">입사각에 따른 Fresnel 반사율(공기→유리, n=1→1.5). 정면(0°)에서는 몇 %만 반사되지만, 스치듯 비스듬히 볼수록(90°에 가까울수록) 반사율이 100%로 치솟는다. 출처: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Fresnel_reflection.svg)</figcaption>
</figure>

```glsl
float cosTheta = clamp(dot(n, -rd), 0.0, 1.0);

float dielectricF0 = F0FromIOR(1.0, mat.ior);
vec3 F0 = mix(vec3(dielectricF0), mat.albedo, mat.metallic);

float x = clamp(1.0 - cosTheta, 0.0, 1.0);
vec3 F = F0 + (1.0 - F0) * x * x * x * x * x;
```

여기서 금속/비금속을 하나의 식으로 통합하는 흔한 기법이 보인다. 비금속(절연체)은 회색빛의 약한 반사($F_0 \approx 0.04$)를 내고 색은 diffuse에서 나온다. 반면 금속은 diffuse가 없고 반사색 자체가 금속의 색이다. 그래서 `metallic` 값으로 $F_0$를 절연체의 값과 물체 고유색(`albedo`) 사이에서 섞어, 금속일수록 반사에 색이 입혀지도록 한다.

## 반사냐 확산이냐: 확률적 선택 ##

빛이 한 번 튕길 때마다 반사(specular)와 diffuse 두 경로로 갈라진다. 둘 다 추적하면 광선이 매 반사마다 두 배로 불어나 폭발한다. 그래서 둘 중 하나만 확률적으로 고른다.

```glsl
vec3 specularWeight = F;
vec3 diffuseWeight = mat.albedo * (1.0 - mat.metallic) * (1.0 - F);

float specularProb = max(specularWeight.r, max(specularWeight.g, specularWeight.b));
specularProb = clamp(specularProb, 0.0, 1.0);

bool chooseSpecular = RandomFloat01(rngState) < specularProb;
```

반사로 갈 확률을 Fresnel 반사율 $F$에 비례하게 잡는다. 비스듬히 보는 표면일수록 $F$가 커지므로 자연스럽게 반사 쪽을 더 자주 고른다. 이렇게 중요한 방향을 더 자주 뽑는 것이 중요도 표본추출(importance sampling)이고, 표본을 적게 써도 노이즈가 덜한 그림을 얻는 비결이다.

```glsl
if (chooseSpecular)
{
    vec3 reflected = reflect(rd, n);

    float a = mat.roughness * mat.roughness;

    rd = normalize(mix(
        reflected,
        RandomCosineHemisphere(reflected, rngState),
        a
    ));

    throughput *= specularWeight / max(specularProb, 1e-4);
}
else
{
    rd = RandomCosineHemisphere(n, rngState);

    throughput *= diffuseWeight / max(1.0 - specularProb, 1e-4);
}
```

반사를 골랐다면 완벽한 거울 반사 방향(`reflect`)을 기준으로, 거칠기(roughness)에 따라 방향을 흩뜨린다. 거칠기가 0이면 순수한 거울, 1에 가까우면 반사 방향 주변으로 넓게 퍼지는 광택 표면이 된다. 미세면(microfacet) 기반의 GGX 같은 정식 모델은 아니고, 거울 반사 방향과 cosine 분포 사이를 선형 보간하는 가벼운 근사지만, 적은 비용으로 그럴듯한 광택을 만들어낸다.

<figure>
<img src="/assets/2026-06-18-path-tracing/reflection-types.svg" alt="거울 반사(specular)와 diffuse 비교 다이어그램">
<figcaption markdown="span">거울 반사(왼쪽)와 diffuse(오른쪽). roughness가 0이면 거울처럼 한 방향으로, 1에 가까우면 모든 방향으로 흩어지는 diffuse가 된다. 코드는 이 둘 사이를 거칠기로 보간한다. 출처: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Specular_And_Diffuse_Reflection.svg)</figcaption>
</figure>

diffuse를 골랐다면 normal vector를 중심으로 한 cosine 가중 반구에서 방향을 뽑는다. Lambert 표면이 정확히 이 분포를 따르기 때문에, 앞서 본 Monte Carlo 식의 $\cos\theta$ 항과 확률 $p(\omega_i)$가 깔끔하게 약분된다.

두 경우 모두 `throughput`에 각 경로의 가중치를 곱하고, 그 경로를 고른 확률로 나눈다. 반사를 `specularProb` 확률로 골랐으니 `specularProb`로 나누고, diffuse는 `1 - specularProb`로 나눈다. 이 나눗셈이 있어야 둘 중 하나만 추적하더라도 평균적으로는 둘 다 추적한 것과 같은(편향 없는) 결과가 나온다.

## Russian Roulette ##

여러 번 튕긴 광선은 throughput이 작아져 최종 그림에 거의 기여하지 않게 된다. 그렇다고 무작정 끝까지 추적하면 낭비고, 일정 횟수에서 잘라버리면 그만큼 빛을 잃어 그림이 어두워진다. 둘 사이의 타협이 Russian Roulette이다.

```glsl
if (bounce > 1)
{
    float p = max(throughput.r, max(throughput.g, throughput.b));
    p = clamp(p, 0.05, 0.95);

    if (RandomFloat01(rngState) > p)
    {
        break;
    }

    throughput /= p;
}
```

throughput이 작을수록 큰 확률로 광선을 죽인다. 단, 살아남은 광선은 `throughput /= p`로 다시 키워준다. 이 보정 때문에 기댓값은 그대로 유지되어, 광선을 중간에 끊어도 결과에 편향이 생기지 않는다. 계산량은 줄고 밝기는 보존된다.

## 점진적 누적 ##

광선 한 가닥, 한 프레임의 결과는 노이즈투성이다. Path tracing은 본질적으로 여러 표본의 평균이 필요하다. 그래서 프레임마다 결과를 이전 프레임과 섞어 쌓아간다.

<figure>
<img src="/assets/2026-06-18-path-tracing/samples.png" alt="픽셀당 표본 수가 늘수록 노이즈가 줄어드는 비교 이미지">
<figcaption markdown="span">픽셀당 표본 수에 따른 노이즈. 왼쪽 위가 1 표본이고 칸마다 두 배씩 늘어 오른쪽 아래는 32,768 표본이다. 표본이 쌓일수록 노이즈가 사라지며 매끈한 이미지로 수렴한다. 출처: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Path_tracing_sampling_values.png)</figcaption>
</figure>

```glsl
vec3 previous = texture(iChannel0, uv).rgb;

vec3 col;
if (iFrame == 0)
{
    col = radiance;
}
else
{
    col = mix(previous, radiance, 0.05);
}
```

## 직접 돌려보기 ##

지금까지 뜯어본 코드 전체는 [Shadertoy](https://www.shadertoy.com/view/dstyzH)에 올려두었다. 마우스로 드래그해 시점을 바꾸거나, 프레임이 쌓이며 노이즈가 줄어드는 누적 과정을 실시간으로 볼 수 있다.

## 정리 ##

코드를 한 바퀴 돌고 나면 path tracing의 골격이 그대로 보인다. 빛을 거꾸로(카메라에서 장면으로) 쏘고, 표면에 닿을 때마다 Fresnel로 반사/diffuse를 확률적으로 갈라 한 방향만 따라가며, 그 선택의 확률로 보정한 throughput을 곱해 나가다, environment map에서 빛을 받아오거나 Russian Roulette으로 끝낸다. 그리고 noisy한 결과를 프레임마다 쌓아 평균낸다.

[빛과 렌더링](/생각/빛과-렌더링/)에서 다룬 물리가 이 짧은 쉐이더 안에서 그대로 코드가 된다. 닫힌 형태로 풀 수 없는 렌더링 방정식을 Monte Carlo 표본추출과 누적 평균으로 근사하는 것이 path tracing의 핵심이다.
