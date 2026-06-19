---
title: "Volume Rendering, the Path Tracing Version"
date: 2024-12-21T10:34:30+09:00
categories:
  - work
tags:
  - rendering
  - volumetric
  - cloud
ref: volume-rendering-path-tracing
---
![Image](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-12-21-Volume%20Rendering%20(Path%20Tracing)/volume_rendering.jpg?raw=true)

In an earlier post I covered how to implement volume rendering with ray marching by simplifying the ray integral. That implementation assumed light arriving from a single direction only. In reality, though, it's more physically accurate to account for light arriving from every direction on the sphere at each ray step.

## The ray integral ##

Let's look at the ray integral again so I can explain things in more detail. The general form of the ray integral is as follows.

$$C(t) = \int_{t_{\text{near}}}^{t_{\text{far}}} T(t) \cdot \sigma(t) \cdot \left[c(t) + L_{\text{ext}}(t)\right] \, dt$$

Here's what each variable and function means.

$C(t)$: the final color or brightness (energy) of the ray. This is the result of the ray integral, representing how much has accumulated as the ray passes through the medium.

$t_{near}$, $t_{far}$: the entry and exit points where the ray intersects the medium. The ray integral runs from $t_{near}$, the point where light emitted by the source first meets the medium, to $t_{far}$, where it finishes passing through.

$T(t)$ is the transmittance, which describes how much the light's intensity is attenuated as it passes through the object. It's defined as follows.

$$T(t) = \exp\left(-\int_{t_{\text{near}}}^{t} \sigma(s) \, ds\right)$$

This is a generalized form of the [Beer-Lambert Law](https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law). Light carries energy. As it passes through a medium (a volume), that energy is either absorbed or scattered. The part that describes absorption is Beer's law. Looking closely at the equation, you can see it expresses how much of the light survives—based on the accumulated extinction coefficient—after the light has traveled from $t_{\text{near}}$ to ${t}$.

$\sigma(t)$: the sum of absorption and scattering.

$c(t)$: the medium's intrinsic color and intensity. It represents the color and brightness the medium has at a given point, and it contributes to the ray integral.

$L_{\text{ext}}$: light arriving from the outside.

This is the part this post focuses on most. In general, to express $L_{\text{ext}}$ you have to account for light arriving from every direction, as shown below.

$$L_{\text{ext}}(t) = \int_{\Omega} T(t) \cdot I(\omega) \cdot p(\omega, t) \, d\omega$$

Looking at the equation, you can see that it accounts for light arriving from every direction over the hemisphere defined as $\Omega$.

## How this differs from the previous approach ##

The ray integral implemented in the code is computed by approximating the given equation in discrete form. A number of features were added, but the difference from the previous implementation is simple. Instead of approximating incoming light as a single direction, we simulate—based on real physical properties—the phenomenon of light being reflected and refracted in many directions. Sending light in literally every direction is hard, so we'll send as many rays as we can in random directions to approximate the actual integral. To put it fancily, you could say we use the [Monte Carlo Method](https://ko.wikipedia.org/wiki/%EB%AA%AC%ED%85%8C%EC%B9%B4%EB%A5%BC%EB%A1%9C_%EB%B0%A9%EB%B2%95).

To do this, at every step I fire light from each ray and let that light scatter elsewhere probabilistically. The scattering probability is determined by Beer's law as follows.

$$P(\text{scatter}) = 1.0 - e^{-\sigma_s \cdot \Delta x}$$

In code, this looks like the following.

```glsl
float calculateScatterProbability(float sigma_s, float delta_x) {
    return 1.0 - exp(-sigma_s * delta_x);
}
```

Once we compute the scattering probability, if scattering is decided to happen, the light's direction isn't chosen at random—it's sampled according to the Henyey-Greenstein phase function. The method is a bit involved, so let's look at the code first.

```glsl

// Using sampleHenyeyGreenstein

float shadowScatterProbability = calculateScatterProbability(shadowDensity, LIGHT_STEP);
if (shadowScatterProbability > randomFloat01(rngState)) {
    lightDir = sampleHenyeyGreenstein(lightDir, 0.6, rngState);
}

```
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

    if (s < 0.00001) { // When nearly zero, return without rotation
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

The `sampleHenyeyGreenstein` function uses the Henyey-Greenstein phase function to sample a new scattered direction vector from a given direction. It first generates two random numbers. The first is used to sample $\cos\theta$ from the Henyey-Greenstein distribution, and the second determines the azimuthal angle $\phi$. $\cos\theta$ is sampled using the given asymmetry parameter $g$, and the value of $g$ tunes the character of the scattering. When $g > 0$ light scatters forward, when $g < 0$ it scatters backward, and when $g = 0$ it represents isotropic scattering. Then $\phi$ is determined by multiplying by $2\pi$, and a new direction vector is computed using $\cos\theta$ and $\phi$. This vector is initially defined relative to the $z$ axis. The vector generated this way is then aligned to the reference direction through the `alignToDirection` function. This is done by computing a rotation axis and rotation matrix and rotating the vector, and in the end a new vector aligned with the reference direction is returned.

This way, instead of light simply arriving from a single direction, we can render it accounting for how it actually scatters and interacts with the medium. Easy, right? You can check out the full code [on Shadertoy](https://www.shadertoy.com/view/lfyyWt).
