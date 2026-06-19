---
title: "Light and Rendering"
date: 2025-10-20T10:34:30+09:00
categories:
  - thoughts
tags:
  - Rendering
  - Light
  - Physics
ref: light-and-rendering
---
## What is light? ##

Light is an electromagnetic transverse wave. "Electromagnetic" means that light is made up of an electric field and a magnetic field; "transverse" means that the direction in which the wave travels and the direction in which it oscillates are perpendicular to each other.

<figure>
<img src="/assets/2025-10-20-light-rendering/emwave.svg" alt="Diagram of an electromagnetic wave">
<figcaption markdown="span">Light is an electromagnetic transverse wave in which the electric field (E) and magnetic field (B) oscillate perpendicular to the direction of travel. Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Electromagnetic_wave_EN.svg)</figcaption>
</figure>

We can classify light by its wavelength. The spectrum is enormous, ranging from gamma rays with extremely short wavelengths of 0.01 nanometers all the way up to ELF (Extremely Low Frequency) waves whose wavelengths exceed 10,000 kilometers. Within all of this, the band that humans can actually see is very narrow: visible light, roughly between 400 and 700 nanometers.

How does light interact with matter? Let's look at the atomic scale. The protons sit at the center of the atom as a positively charged cluster, while the electrons surround it as a fast-moving, negatively charged cloud.

As I described above, light (an electromagnetic wave) is a wave made of an electric field and a magnetic field. Of these, the electric field exerts a force on positive charges in the direction of the field and on negative charges in the opposite direction.

The accelerated charges then emit new electromagnetic waves in all directions. Part of the energy of the incident light is absorbed by the material, and the rest is re-emitted as electromagnetic waves. This whole process is called scattering.

When the scattered light has the same wavelength as the original light, we call it elastic scattering. If the light contains several frequency components, each frequency interacts with the material independently. Apart from inelastic interactions such as fluorescence, no interference occurs between different frequency components.

An isolated molecule behaves like an oscillating dipole, scattering light in all directions, but the radiation from a single molecule is strongest in the direction perpendicular to its oscillation axis (the lateral direction). In a real medium containing many molecules, however, interference effects between the scattered light produce relatively strong scattering along the direction of propagation (forward and backward).

Some molecules can absorb the electromagnetic energy of the incident light. A molecule that absorbs light transitions to a higher energy level, a condition known as the excited state. Most of the absorbed energy is converted into molecular vibration or rotation and ultimately into heat, but some of it is re-emitted as radiation (fluorescence, phosphorescence) or can drive a chemical reaction. This absorption is closely tied to the wavelength of the light: because each molecule has its own characteristic energy-level differences, it selectively absorbs only light of specific wavelengths.

In rendering, we generally deal with large numbers of molecules, and many molecules interact with light differently than a single isolated one does. Molecules that are close together scatter waves coherently, meaning their wave phases line up with one another in some fashion. As a result, the waves produced by the interaction with light can become stronger than the simple sum of the original waves (constructive interference) or, conversely, weaker (destructive interference).

In an ideal gas, the molecules do not influence one another, so their relative positions are completely random and uncorrelated. Because of this, the phase differences between nearby scattered waves are randomized, making them incoherent with one another. No interference occurs, and the overall scattering looks similar to the scattering observed from individual molecules.

In very small liquid or solid particles, the molecules are packed densely within a cluster smaller than the wavelength of light. In this case, the wavelengths of light scattered by each molecule are in phase and interfere constructively, which strengthens the energy of the scattered wave. In other words, even when the molecular density per unit volume is the same, the intensity of scattered light is far greater when the molecules are clumped together into particles. This is why clouds and fog scatter light so strongly.

As long as the size of each particle stays below roughly one-tenth of the wavelength of light, a collection of such particles arranged at random scatters light in much the same way as an ideal gas made of individual molecules. This kind of scattering is called Rayleigh scattering. It is the reason the midday sky looks blue and the sky turns red at sunrise and sunset.

<figure>
<img src="/assets/2025-10-20-light-rendering/rayleigh.svg" alt="Diagram of Rayleigh scattering">
<figcaption markdown="span">Rayleigh scattering — shorter wavelengths (blue) scatter more, so the midday sky looks blue, while the setting sun's light, which travels a long path through the atmosphere, looks red. Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Rayleigh_sunlight_scattering.svg)</figcaption>
</figure>

Once the particle size exceeds about one-tenth of the wavelength of light (a rough figure), the interference becomes more complicated and is no longer constructive. As a result, the angular and wavelength dependence of the scattering gradually diverges from Rayleigh scattering. This type of scattering is called Mie scattering, and it is most commonly observed in particle aggregates such as fog, clouds, and smoke. It is why an overcast sky looks gray rather than blue.

## Homogeneous media ##

Now that we've looked at particles, let's turn to a larger-scale aggregate of matter: a homogeneous medium (a volume filled with identical molecules spaced at uniform intervals).

In truth, if we look at things atom by atom, no homogeneous medium can exist in the world. But because assuming a homogeneous medium makes the problem far simpler, optics frequently simplifies things this way.

As described earlier, light is an electromagnetic wave, so the optical properties of an object are deeply connected to its electrical properties. Based on their electrical properties, materials can be divided into three categories: dielectrics, metals, and semiconductors.

Semiconductors have complex physical properties and are rarely dealt with in rendering, so we'll skip them and look only at dielectrics and metals.

In a dielectric, all of the electrons are tightly bound to the atomic nuclei. So if you apply a constant electric field, the positively charged nucleus shifts slightly in the direction of the field and the negatively charged electrons shift slightly in the opposite direction, but the two remain bound to each other.

Now suppose that instead of a constant electric field we apply an electromagnetic wave (which, as we've established, is what light is). In this case, the waves scattered from neighboring molecules undergo coherent interference and also interfere with the incident wave. If the molecules are identical and arranged at uniform intervals (a homogeneous medium), constructive interference occurs in the forward direction of the original wave, while destructive interference occurs in every other direction.

The result is that the original wave travels in the same direction but with a changed speed. This change in wave speed — the ratio of the original wave speed to the new wave speed — is one of the medium's optical properties, called the index of refraction (IOR).

In a metal, the core electrons are strongly bound to the nuclei and don't move, but the outer valence electrons are weakly bound and can move freely throughout the metal. These conduction electrons exist collectively within the metal, and when an external constant electric field is applied, they move in the direction opposite the field to form an electric current.

When an electromagnetic wave strikes a chunk of metal, on the other hand, the wiggling electric field inside it drives the valence electrons to form eddy currents that cancel out the original electromagnetic wave. As a result, the strength of the electromagnetic wave decays rapidly inside the metal. The wave therefore cannot penetrate deep into the metal, and most of it is reflected at the surface.

The index of refraction also applies to metals. During the short distance the electromagnetic wave briefly penetrates before dying out inside the metal, its propagation speed changes. There is also a physical quantity that describes how strongly the wave decays exponentially with distance as it travels through the metal, called the attenuation index. These two values — the index of refraction and the attenuation index — generally depend on the wavelength of light.

When these two numbers are combined, they fully define how a medium — whether dielectric or metal — affects light as it passes through. They are often combined into a single complex number, called the complex index of refraction. The complex index of refraction is an important physical quantity that simultaneously describes how much light slows down (refraction) and how much it weakens (absorption) as it passes through a medium. It is extremely useful because it lets us treat the medium as a single volume without worrying about each individual molecule.

As noted earlier, most media are not perfectly homogeneous. For metals, this isn't much of a problem, because light can't penetrate deep into the metal anyway. For dielectrics, though, the situation is different. If a dielectric were perfectly homogeneous, it would simply transmit incoming light, like transparent water or glass. So for a dielectric to be opaque or colored, its interior must be heterogeneous, or it must absorb light — that is, there must be differences in composition or a distribution of particles inside it. Fortunately, such heterogeneous dielectrics can be modeled as a homogeneous medium containing particles.

The destructive interference that suppresses scattering in all directions arises from the uniformity of the molecules and the resulting coherency of the scattered waves. But if this uniformity is broken locally — for example, if a small cluster of different molecules, air, bubbles, or density variations appear inside the medium — these changes disrupt the destructive interference pattern, allowing the scattered light waves to propagate in other directions. Such localized inhomogeneity can be modeled as particles. A gas, too, instead of modeling every individual particle, can be modeled as a homogeneous volume containing these scattering particles.

Done correctly, this modeling produces exactly the same scattering as treating each molecule as an individual scatterer. An amusing fact is that Albert Einstein discovered this around 1910.

Naturally, the appearance of a material is affected differently depending on the type of particles it contains. For example, some particles, such as dyes, don't scatter light strongly but absorb light of certain wavelengths. This property produces a colored transparency effect.

Conversely, there are particles, such as those in milk, that don't absorb light but scatter it strongly. This scattering is usually Mie scattering, though in some cases — opals, for instance — Rayleigh scattering occurs.

In general, the apparent color of a medium is determined by the combined effect of its scattering and absorption properties. Because scattering is often less wavelength-dependent than absorption, scattering generally determines the overall opacity of the medium while absorption determines its color.

Both scattering and absorption depend strongly on scale. For example, at small scales of a few meters, water shows almost no absorption, and clear air looks transparent. But as the scale grows — that is, as the viewing distance increases — both phenomena become very pronounced. This is why the open sea looks blue from a distance, why far-off mountains take on a faint bluish hue, and why light appears to gradually weaken over long distances through the atmosphere. All of this stems from the scale dependence of absorption and scattering.

## Object surfaces ##

From an optical standpoint, the surface of an object is a two-dimensional boundary separating two media (volumes) with different indices of refraction. In most common rendering situations, the outer region is air, and since air's index of refraction is very close to 1, it is usually approximated as exactly 1 for simplicity. The inner region's index of refraction depends on the material the object is made of, and this index is generally expressed as a complex number, as follows:

$$η+iκ.$$

Here η represents how much the light is refracted, and κ represents how much the light is absorbed as it travels through the medium.

From an optical standpoint, a surface has two important aspects. The first is the surface's material property, which determines the index of refraction. The second is the surface's geometric shape, which affects how light interacts with the surface.

Let's first focus on the surface's material properties, assuming the simplest possible surface geometry: a perfectly flat plane. As we saw earlier, when light hits a boundary where the index of refraction changes abruptly, scattering occurs and the light splits into multiple directions. A flat surface is a special case of this phenomenon, so let's look at exactly which new directions the light splits into.

According to the boundary conditions for the electric and magnetic fields, incident light scatters into exactly two directions. One is the reflected direction, the other the transmitted direction. The reflected light makes the same angle with the surface normal as the incident wave, while the transmitted light is refracted at a different angle depending on the index of refraction (IOR). When κ is zero (that is, for a dielectric), the equation for computing the refraction angle is known as Snell's Law.

<figure>
<img src="/assets/2025-10-20-light-rendering/snell.svg" alt="Diagram of reflection and refraction (Snell's Law)">
<figcaption markdown="span">At a flat boundary, light splits into reflection and refraction (transmission), and the refraction angle follows Snell's Law. Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Snell%27s_Law..svg)</figcaption>
</figure>

Now that we've looked at the surface's material properties, let's talk about its geometric shape. The key point here is how the roughness of the surface relates to the wavelength of light. Whether the surface is rough or smooth, and how the scale of that roughness compares to the wavelength of light, is what matters.

This relationship determines how light is reflected or scattered, and ultimately determines the gloss, matteness, or texture of an object as we perceive it.

<figure>
<img src="/assets/2025-10-20-light-rendering/reflection.svg" alt="Diagram of specular and diffuse reflection">
<figcaption markdown="span">Specular reflection from a smooth surface (left) and diffuse reflection from a rough surface (right). How the surface roughness compares to the wavelength of light determines the form of the reflection. Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Specular_And_Diffuse_Reflection.svg)</figcaption>
</figure>

Surface irregularities much smaller than the wavelength of light have no optical effect and can be ignored. Conversely, irregularities much larger than the wavelength of light merely tilt the surface and don't affect its local flatness. In other words, only surface detail comparable to the wavelength of visible light (roughly within the 1× to 100× range) reflects light differently than a flat plane does and influences the surface's optical behavior.

As a result, almost every shading model used in film and game production relies on one of two premises: that the surface detail is much larger than the wavelength of light, or that it is much smaller than the wavelength of light.

Thanks to these assumptions, we can simplify what is in reality complex microstructure and efficiently compute the reflection and diffusion of light.

This assumption — that the surface has no meaningful detail at the scale of an individual wavelength of light — is the fundamental basis of geometric optics. Geometric optics is a theory that, with rare exceptions, has been almost universally used in computer graphics for the past 40 years. In geometric optics, light is modeled not as a wave but as a ray. When a ray of light intersects a surface, the surface near that intersection is treated as a plane. This is identical to the flat-surface case we used when discussing reflection and transmission earlier. In other words, the geometric-optics approach ignores the wave nature of light and treats the surface simply as a boundary off which rays bounce and change direction, allowing complex light interactions to be computed efficiently.

Geometric optics can also handle surface irregularities much larger than an individual wavelength of light. That is, macroscopic shapes (the curvature of an object or the tilt of a face, for example) can be adequately represented within the geometric-optics model. But when these irregularities are too small to render individually — in other words, at a scale smaller than a single pixel on screen — we call this microgeometry. This microgeometry is not modeled directly or visually resolved, yet it has a large effect on the surface's visual properties such as reflection, diffusion, and gloss. To represent it in a statistical or probabilistic way, computer graphics uses approaches like the microfacet model.

As we saw earlier, both the reflected direction and the transmitted direction depend on the surface normal. So when microgeometry changes this normal at different points across the surface, the directions of the reflected and transmitted light change accordingly. Each point on the surface reflects light in only one specific direction, but a single pixel contains countless tiny surface fragments that reflect light in many directions. The final rendering we see is therefore determined by the overall sum of all these different reflection directions.

Most surfaces have isotropic microgeometry — that is, they have rotational symmetry and no particular directionality. But some surfaces have anisotropic microgeometry, taking the form of a pattern aligned in a particular direction. This causes reflections or highlights to stretch in one direction, or to blur in one direction.

At the macroscopic level, we don't model microgeometry explicitly. Instead, we treat it statistically, regarding the surface as a cone that reflects and refracts light in many directions. The rougher the surface, the wider the cone of reflected and refracted directions, and the more the light spreads out into a diffuse-looking reflection.

So far we've looked at the visual effects of reflected light. So what about transmitted light? That depends on the material of the object. As we saw earlier, metals absorb transmitted light immediately, so we don't need to consider any light interaction beneath the surface. A homogeneous dielectric, such as glass or clean water, is transparent, so the transmitted light simply travels in a straight line until it reaches the opposite surface of the object. But with heterogeneous dielectrics, things get much more interesting. In this case, a participating medium exists beneath the surface, and the refracted light undergoes scattering and absorption as it passes through that medium. These interactions determine the object's internal color, translucency, and subsurface scattering effect, and they play a key role in the realistic depiction of materials such as skin, wax, and marble.

After undergoing absorption and scattering, some of the transmitted light is scattered back out of the surface (backscattered). This re-emitted light exits the surface at various distances from the point of entry. The distribution of distances between the entry point and exit point depends on the density of the scattering particles and other physical properties. If the pixel size or shading sample area is larger than this entry-to-exit distance, the shading computation can treat the distance as effectively zero. That is, when the difference between the entry and exit points can be ignored, all shading can be computed at a single point, and that point's color is affected only by the light reaching that location.

It's convenient to separate these two different light-matter interactions into distinct shading terms. The term arising from surface reflection is called specular. The term that arises from the process of refraction, absorption, scattering, and re-refraction is called diffuse. In other words, specular represents light reflected directly off the surface, while diffuse represents light that emerges after being scattered inside the material. Conversely, when the pixel size is smaller than the entry-to-exit distance, the shading cannot be treated as happening at a single point. This calls for special subsurface scattering rendering techniques.

In fact, local diffuse shading and nonlocal subsurface scattering (i.e., diffusion) arise from physically the same phenomenon. The only difference is the relationship between resolution and scattering distance. This is often thought of as a difference in material properties — distinguishing, for example, plastic as a diffuse material and skin as a subsurface-scattering material.

But in reality, the distance from the camera is what decides this. Plastic simply looks like a diffuse material from a distance, but in close-up it shows clear subsurface scattering. Conversely, skin shows noticeable scattering up close, but from a distance it can be represented adequately with simple local diffuse shading alone.
