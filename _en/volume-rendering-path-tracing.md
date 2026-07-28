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

In an earlier post I covered how to implement volume rendering with ray marching by simplifying the ray integral. That implementation assumed light arriving from a single direction only. In reality, though, it's more physically accurate to account for light arriving from every direction on the sphere at each scattering point.

## The ray integral ##

Let's look at the ray integral again so I can explain things in more detail. The general form of the ray integral is as follows.

$$C(t) = \int_{t_{\text{near}}}^{t_{\text{far}}} T(t) \cdot \sigma(t) \cdot \left[c(t) + L_{\text{ext}}(t)\right] \, dt$$

Here's what each variable and function means.

$C(t)$: the final color or brightness (energy) of the ray. This is the result of the ray integral, representing how much has accumulated as the ray passes through the medium.

$t_{near}$, $t_{far}$: the entry and exit points where the ray intersects the medium. The ray integral runs from $t_{near}$, the point where light emitted by the source first meets the medium, to $t_{far}$, where it finishes passing through.

$T(t)$ is the transmittance, which describes how much the light's intensity is attenuated as it passes through the object. It's defined as follows.

$$T(t) = \exp\left(-\int_{t_{\text{near}}}^{t} \sigma(s) \, ds\right)$$

This is a generalized form of the [Beer-Lambert Law](https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law). Light carries energy. As it passes through a medium (a volume), that energy is either absorbed or scattered. The part that describes absorption is Beer's law. Looking closely at the equation, you can see it expresses how much of the light survives—based on the accumulated extinction coefficient—after the light has traveled from $t_{\text{near}}$ to ${t}$.

$\sigma(t)$: the sum of absorption and scattering. This is called the extinction coefficient.

$c(t)$: the medium's intrinsic color and intensity. It represents the color and brightness the medium has at a given point, and it contributes to the ray integral.

$L_{\text{ext}}$: light arriving from the outside.

This is the part this post focuses on most. In general, to express $L_{\text{ext}}$ you have to account for light arriving from every direction, as shown below.

$$L_{\text{ext}}(t) = \int_{S^2} L(\omega) \cdot p(\omega \cdot \omega') \, d\omega$$

Unlike at a surface, light inside a medium can pass through from behind as well as from the front, so the domain of integration isn't a hemisphere—it's the full sphere $S^2$, all $4\pi$ steradians of it. $p$ is the phase function, the probability density describing how much light arriving from direction $\omega$ gets bent toward the outgoing direction $\omega'$.

## How this differs from the previous approach ##

The previous implementation chopped the ray into fixed-size steps and, at each step, decided whether to scatter with a probability proportional to the density. It's intuitive, but there's a problem with it: the result depends on what you pick for the step size. Make the steps fine and it gets slow; make them coarse and you skip straight over thin details, so the image gets brighter or darker. No matter how many samples you accumulate, it doesn't converge to the "right answer"—it converges to some other image that the step size manufactured.

In this implementation I got rid of the notion of a step entirely. I draw how far light flies before it collides directly from a probability distribution, bend the direction at that point according to the phase function, then aim straight at the light source to gather its contribution. Those three things are delta tracking, Henyey-Greenstein sampling, and NEE (next event estimation) + MIS, respectively. Let's take them one at a time.

## Drawing the free path directly: delta tracking ##

If a medium has uniform density $\sigma$, then the probability that light flies a distance $t$ without colliding is $e^{-\sigma t}$. Drawing a collision distance from that is easy: pull one uniform random number $\xi$ and compute $t = -\ln(\xi) / \sigma$. Done.

The problem is that our medium isn't uniform. Density comes from the interior depth of a fractal SDF, so it's all over the place from point to point. That's what delta tracking is for. The idea goes like this.

1. Pick an upper bound $\bar\sigma$ (the majorant) that is always greater than or equal to the actual density everywhere in space.
2. **Pretend** the medium is packed uniformly at $\bar\sigma$ and draw a collision distance.
3. Measure the real density at that point and accept it as a "real collision" with probability $\sigma(x) / \bar\sigma$.
4. If it isn't accepted, act like nothing happened and go back to step 1 from right there.

You pad the density with fake collisions (null collisions) so the medium can pretend to be uniform, then probabilistically filter the fakes back out. Remarkably, this yields the exact free-path distribution with no bias at all. The step-size parameter simply ceases to exist.

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

There's exactly one promise you have to keep here: **$\bar\sigma$ really has to be an upper bound.** If $\sigma(x) > \bar\sigma$ anywhere, the probability above exceeds 1 and the density at that point gets quietly underestimated. So instead of doing anything like adding a constant density inside the density function, I clamp it outright with `MAX_DENSITY`.

```glsl
float sampleDensity(vec3 p)
{
    float signedField = sceneField(p);
    float interiorDepth = max(-signedField, 0.0);

    // No constant density is added here. This guarantees that
    // extinctionAt(p) never exceeds SIGMA_MAJORANT.
    return clamp(interiorDepth * DENSITY_SCALE, 0.0, MAX_DENSITY);
}
```

## Shadows by probability too: ratio tracking ##

From a scattering point, we also need to know how much light survives the trip to the light source—the transmittance $T$. Computing that honestly as a density integral means going back to stepping again.

So we twist delta tracking slightly. Drawing the collision points works exactly the same, but instead of stopping there, we keep multiplying by $1 - \sigma(x)/\bar\sigma$. In effect we accumulate the probability that the collision was fake, and the expected value of that product is exactly $T$. This is called ratio tracking. It has far lower variance than cutting the shadow off at the first collision, and above all the result is smooth.

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

## The phase function: Henyey-Greenstein ##

If a real collision happened, now we have to bend the direction. That's what the Henyey-Greenstein phase function is for.

$$p(\cos\theta) = \frac{1}{4\pi} \cdot \frac{1 - g^2}{\left(1 + g^2 - 2g\cos\theta\right)^{3/2}}$$

$g$ is the asymmetry parameter, and it tunes the character of the scattering. When $g > 0$ light scatters forward, when $g < 0$ it scatters backward, and when $g = 0$ it represents isotropic scattering. I wanted strong forward scattering like a cloud, so this implementation uses $g = 0.65$.

The previous implementation built the direction relative to the $z$ axis and then aligned it to the reference direction with Rodrigues' rotation formula. Building a rotation axis and a rotation matrix like that is actually unnecessary. Set up an orthonormal basis around the propagation direction and assemble the vector directly in it. No matrix multiply, and no singularity blowing up when the two vectors are nearly parallel.

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

That expression for $\cos\theta$ is the result of inverting the cumulative distribution function of the HG distribution. When $g$ is near zero the denominator heads for zero and the numbers explode, so in that case we just treat it as isotropic.

Because the phase function is importance-sampled **exactly**, one nice thing happens: the $p(\omega)/\text{pdf}(\omega)$ that should enter the path weight cancels cleanly to 1. So the only thing left to multiply into the throughput when scattering occurs is the single-scattering albedo. Absorption is baked into that.

```glsl
throughput *= MEDIUM_ALBEDO;
```

## Aiming straight at the light: NEE and MIS ##

Now we have to gather light at the scattering point. We could just bend the direction, fire it off, and wait to get lucky and hit the light source. But if the light is small, that probability is dismal, and the screen ends up covered in noise like scattered salt.

So every time scattering occurs, we pick a point on the light's surface **directly**, measure the transmittance out to it, and add the contribution. This is NEE (next event estimation). The light in this implementation is an emissive box floating overhead, and points are chosen in proportion to the areas of its six faces.

At that point we have to convert a probability density over area into a probability density over solid angle. The conversion is as follows.

$$p_\omega = p_A \cdot \frac{d^2}{\cos\theta_l}$$

$d$ is the distance from the scattering point to the light, and $\cos\theta_l$ is the angle between the light surface's normal and the ray. It's just the obvious observation—the farther away and the more obliquely you view it, the smaller the light appears—written as an equation.

The problem is that we now count the same light along two different paths: once by aiming at the light directly, and once by flying along the phase function and happening to hit the light. Leave it alone and twice as much light leaks out. But block the second one and, in situations where the medium is thin and scatters in a narrow, glossy lobe, you throw away the phase-function estimate that was actually far better.

MIS (multiple importance sampling) is how you keep both while splitting the weight between them. Here I used the power heuristic.

$$w_A = \frac{p_A^2}{p_A^2 + p_B^2}$$

The weights of the two estimators sum to exactly 1, so no energy leaks. On top of that, the split happens on its own so that each strategy gets a bigger weight in the situations it's good at. For a small light, light sampling wins; for a broad bright surface, phase sampling wins.

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

The other side—when a ray that flew along the phase function hits the light—needs its matching weight too. This time the order of the arguments flips.

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

When the very first ray out of the camera lands on the light directly, `rayWasPhaseSampled` is false, so no weight is applied. That path can never overlap with NEE, so it should receive the whole thing.

## When to stop: Russian roulette ##

How many scattering events should we follow? Just cutting it off at 8 throws away that much energy, and thick regions come out darker than they should. Russian roulette kills paths probabilistically but inflates the contribution of the survivors by dividing by that same probability. On average energy is conserved while only the path length shrinks—an optimization that's close to free.

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

The darker a path has already become, the sooner it dies. Bright paths live long. Reasonable.

## Accumulation and tone mapping ##

A Monte Carlo estimate is a lump of noise if you only have one sample. In the end you have to pile them up. So I accumulate frames into Buffer A, counting the sample number in the alpha channel and blending with the previous result at a weight of $1/N$. Moving the camera or pressing the spacebar resets the accumulation.

```glsl
sampleCount = min(previousFrame.a + 1.0, 8192.0);
float blendWeight = 1.0 / sampleCount;

accumulatedRadiance = mix(previousFrame.rgb, currentSample, blendWeight);
```

The pixel coordinate gets a subpixel jitter every frame too. As the accumulation builds up, antialiasing comes along for free.

Finally, the Image pass multiplies the radiance accumulated in linear space by an exposure, runs it through ACES filmic tone mapping, and encodes it to sRGB. The light's emission is a value like 18, well past 1.0, and simply clamping that would flatten the highlights like a sheet of paper.

```glsl
float exposure = 1.25;
vec3 color = acesFilm(linearRadiance * exposure);
color = linearToSRGB(color);
```

This way, instead of light simply arriving from a single direction, we can render it accounting for how it actually scatters and interacts with the medium. Instead of tuning a step size and reading the tea leaves, piling on more samples converges honestly to the right answer. Easy, right? You can check out the full code [on Shadertoy](https://www.shadertoy.com/view/lfyyWt).
