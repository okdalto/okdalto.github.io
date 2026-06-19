---
title: "GLSL Volumetric Rendering (The Gentler, Easier Version)"
date: 2024-11-25T10:34:30+09:00
categories:
  - work
tags:
  - rendering
  - volumetric
  - cloud
ref: glsl-volumetric-rendering-gentle
---
- For the more rigorous, more difficult version, see [here](/en/glsl-volumetric-rendering-rigorous/).

## Ways to Represent an Object: Polygon and Voxel Representation ##

To render an object in 3D space, how you choose to represent it matters a great deal. The most widely used approach by far is the polygon-based representation. Polygons are great for defining the shape of an object, but they run into limits when it comes to representing volumetric objects. Here's why:

1. Lack of interior information:
Polygons mainly define the outer boundary of an object, and they can't capture changes in density or state inside it.
Example: structures with complex interiors, like clouds or smoke.

2. Growing complexity:
Representing such intricate shapes with polygons requires an enormous number of them. This drives up computational cost and memory usage dramatically, making it inefficient.

3. Constraints on light interaction:
Polygons can represent light reflecting and refracting at a surface, but they struggle to handle phenomena like absorption and scattering that occur as light passes through the interior of an object.

## Other Representation Methods ##

Beyond polygons, there are a variety of approaches:

- Voxel: a representation built from a 3D grid. (Block shapes carved into a grid, like in Minecraft.)
- SDF (Signed Distance Function): represents objects based on distance.
- Splat: represents an object around points.
- Neural Volume: represents an object with a neural network.

Here we'll use the Voxel representation to handle volumetric objects. "Voxel" is a portmanteau of "Volume" and "Pixel": you divide 3D space into a grid, and each cell (voxel) carries physical properties such as density, color, and opacity. This makes it easy to represent the complex internal state of an object.

![Various shape representations](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-11-25%20Volumetric%20rendering/3D_representations.jpg?raw=true)
*Various shape representations. Clockwise from the top left: SDF, Voxel, Polygon, Splat.*

## How Do We Represent Light? ##

For an object with a complex interior, like a cloud, light doesn't simply reflect or refract; it passes through the object, scattering and being absorbed along the way.
We need to compute all of this to produce the image that ends up on screen.

## Ray Marching: Following the Path of Light ##

Light sets out from the camera and travels straight toward the object.
Ray marching is a technique that splits this light path into small intervals and checks the interior of the object one step at a time.

1. Following and checking along the light path: the ray that left the camera passes through the interior of the object, accumulating the color, density, and opacity of each block as it goes.

2. Computing the resulting color: based on the accumulated information, we compute how much of the light is absorbed, how transparent it is, and what color it takes on.

## Making Clouds More Realistic: Light Scattering ##

Light scatters inside an object. Put simply, after light strikes the object it spreads out in many directions.

Forward Scattering:
When light keeps traveling in roughly the same direction it was going after the collision.
→ This creates the translucent feel of light within a cloud.

Backward Scattering:
When light bounces back in the opposite direction after the collision.
→ This contributes to the soft, fuzzy edges of a cloud.

Clouds exhibit a lot of forward scattering, which produces the bright halo effect you see near the sun.

## Implementing Volumetric Rendering ##

Below is an example of volume rendering implemented in GLSL using everything described above.
You can see how this code behaves on [Shadertoy](https://www.shadertoy.com/view/MfKcWc).

![Shadertoy example](https://github.com/okdalto/okdalto.github.io/blob/master/assets/2024-11-25%20Volumetric%20rendering/cloud.jpg?raw=true)
*The Shadertoy example*


```glsl
#define FOWARD 0.8 // Forward scattering coefficient
#define BACKWARD -0.2 // Backward scattering coefficient
#define RAY_ITER 120 // Number of ray marching iterations
#define LIGHT_ITER 16 // Number of lighting sample iterations
#define LIGHT_ATTEN 64.0 // Light attenuation coefficient
#define RAY_STEP_SIZE 0.01 // Ray marching step size

// Function for rotation about an axis
void rotate(inout vec3 z, vec3 axis, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    // Compute the rotation matrix for rotation about the axis
    mat3 rot = mat3(
        c + axis.x * axis.x * (1.0 - c),       axis.x * axis.y * (1.0 - c) - axis.z * s, axis.x * axis.z * (1.0 - c) + axis.y * s,
        axis.y * axis.x * (1.0 - c) + axis.z * s, c + axis.y * axis.y * (1.0 - c),       axis.y * axis.z * (1.0 - c) - axis.x * s,
        axis.z * axis.x * (1.0 - c) - axis.y * s, axis.z * axis.y * (1.0 - c) + axis.x * s, c + axis.z * axis.z * (1.0 - c)
    );
    z = rot * z; // Apply the rotation to the vector
}

// Function that computes a procedural fractal shape
float fractal(vec3 p) {
    for (int i = 0; i < 8; i++) {
        // Fractal that rotates over time
        rotate(p, vec3(1.0, 0.0, 0.0), iTime * 0.2);
        rotate(p, vec3(0.0, 1.0, 0.0), iTime * 0.1);
        // Reflective symmetry
        if (p.x + p.y < 0.0) p.xy = -p.yx;
        if (p.y + p.z < 0.0) p.yz = -p.zy;
        if (p.z + p.x < 0.0) p.zx = -p.xz;
        p -= 0.06; // Scale down and translate
    }
    return length(p) - 0.15; // Compute the final distance
}

// Use the fractal as an SDF (distance function)
float sdf(vec3 p) {
    return fractal(p);
}

// Henyey-Greenstein Phase Function
float HenyeyGreenstein(float sundotrd, float g) {
    float gg = g * g;
    return (1. - gg) / pow(1. + gg - 2. * g * sundotrd, 1.5);
}

// Scattering computation (mix of forward and backward scattering)
float getScattering(float sundotrd) {
    return mix(HenyeyGreenstein(sundotrd, FOWARD), HenyeyGreenstein(sundotrd, BACKWARD), 0.5);
}

// Density sampling (procedural density generation)
float sampleDensity(vec3 p) {
    return pow(max(-sdf(p), 0.0), 1.3) * 10.0; // SDF-based density with amplification
}

// Compute the light position along a Lissajous curve
vec3 lightPosLissajous(float t) {
    float A = 1.5;  // x-axis amplitude
    float B = 1.2;  // y-axis amplitude
    float C = 1.1;  // z-axis amplitude
    float a = 3.1;  // x-axis frequency
    float b = 2.2;  // y-axis frequency
    float c = 4.3;  // z-axis frequency
    float delta = 0.2; // Phase difference

    float x = A * sin(a * t + delta);
    float y = B * sin(b * t);
    float z = C * sin(c * t);

    return vec3(x, y, z); // Return the dynamic light position
}

// Main rendering function
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Normalized pixel coordinates [-1, 1]
    vec2 uv = fragCoord / iResolution.xy;
    uv = (uv - 0.5) * 2.0;

    vec3 col = vec3(0.0); // Initial color value

    vec3 camPos = vec3(0.0, 0.0, -2.0); // Camera position
    vec3 rayPos = camPos; // Ray starting point
    vec3 rayDir = normalize(vec3(uv, 0.0) - camPos); // Ray direction
    float time = iTime * 0.2; // Dynamic time
    vec3 lightPos = lightPosLissajous(time); // Compute the light position

    float transmittance = 1.0; // Initial transmittance

    rayPos += rayDir; // Start advancing the ray
    for (int i = 0; i < RAY_ITER; i++) {
        rayPos += rayDir * RAY_STEP_SIZE; // Advance the ray
        float density = sampleDensity(rayPos); // Compute the density at the current position
        if (density <= 0.0) {
            continue; // If there is no density, go to the next iteration
        }
        vec3 lightDir = lightPos - rayPos; // Light direction
        float lightDistance = length(lightDir); // Light distance
        lightDir = lightDir / lightDistance; // Normalize to a unit vector
        float lightStep = lightDistance / float(LIGHT_ITER); // Lighting step size
        float sundotrd = dot(rayDir, -lightDir); // Dot product of the ray and light directions
        float scattering = getScattering(sundotrd); // Compute scattering
        vec3 lightRayPos = rayPos; // Ray position for shadow computation
        float shadowDensity = 0.0; // Initialize shadow density
        for (int j = 0; j < LIGHT_ITER; j++) {
            shadowDensity += sampleDensity(lightRayPos) * lightStep; // Accumulate shadow density
            lightRayPos += lightDir * lightStep; // Advance along the light direction
        }
        vec3 externalLight = vec3(exp(-shadowDensity * LIGHT_ATTEN) * scattering); // Compute external light
        col += transmittance * externalLight * density; // Accumulated color
        transmittance *= exp(-density * RAY_STEP_SIZE * LIGHT_ATTEN); // Update transmittance
        if (transmittance < 0.01) break; // Early termination if transmittance is low
    }

    col = pow(col, vec3(1.0 / 2.2)); // Gamma correction
    fragColor = vec4(col, 1.0); // Output the final color
}
```

If you were to implement it in TouchDesigner, it would look like this.


```glsl
#define FOWARD 0.8 // Forward scattering coefficient
#define BACKWARD -0.2 // Backward scattering coefficient
#define RAY_ITER 120 // Number of ray marching iterations
#define LIGHT_ITER 16 // Number of lighting sample iterations
#define LIGHT_ATTEN 64.0 // Light attenuation coefficient
#define RAY_STEP_SIZE 0.01 // Ray marching step size

uniform float iTime;

out vec4 fragColor;

// Function for rotation about an axis
void rotate(inout vec3 z, vec3 axis, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    // Compute the rotation matrix for rotation about the axis
    mat3 rot = mat3(
        c + axis.x * axis.x * (1.0 - c),       axis.x * axis.y * (1.0 - c) - axis.z * s, axis.x * axis.z * (1.0 - c) + axis.y * s,
        axis.y * axis.x * (1.0 - c) + axis.z * s, c + axis.y * axis.y * (1.0 - c),       axis.y * axis.z * (1.0 - c) - axis.x * s,
        axis.z * axis.x * (1.0 - c) - axis.y * s, axis.z * axis.y * (1.0 - c) + axis.x * s, c + axis.z * axis.z * (1.0 - c)
    );
    z = rot * z; // Apply the rotation to the vector
}

// Function that computes a procedural fractal shape
float fractal(vec3 p) {
    for (int i = 0; i < 8; i++) {
        // Fractal that rotates over time
        rotate(p, vec3(1.0, 0.0, 0.0), iTime * 0.2);
        rotate(p, vec3(0.0, 1.0, 0.0), iTime * 0.1);
        // Reflective symmetry
        if (p.x + p.y < 0.0) p.xy = -p.yx;
        if (p.y + p.z < 0.0) p.yz = -p.zy;
        if (p.z + p.x < 0.0) p.zx = -p.xz;
        p -= 0.06; // Scale down and translate
    }
    return length(p) - 0.15; // Compute the final distance
}

// Use the fractal as an SDF (distance function)
float sdf(vec3 p) {
    return fractal(p);
}

// Henyey-Greenstein Phase Function
float HenyeyGreenstein(float sundotrd, float g) {
    float gg = g * g;
    return (1. - gg) / pow(1. + gg - 2. * g * sundotrd, 1.5);
}

// Scattering computation (mix of forward and backward scattering)
float getScattering(float sundotrd) {
    return mix(HenyeyGreenstein(sundotrd, FOWARD), HenyeyGreenstein(sundotrd, BACKWARD), 0.5);
}

// Density sampling (procedural density generation)
float sampleDensity(vec3 p) {
    return pow(max(-sdf(p), 0.0), 1.3) * 10.0; // SDF-based density with amplification
}

// Compute the light position along a Lissajous curve
vec3 lightPosLissajous(float t) {
    float A = 1.5;  // x-axis amplitude
    float B = 1.2;  // y-axis amplitude
    float C = 1.1;  // z-axis amplitude
    float a = 3.1;  // x-axis frequency
    float b = 2.2;  // y-axis frequency
    float c = 4.3;  // z-axis frequency
    float delta = 0.2; // Phase difference

    float x = A * sin(a * t + delta);
    float y = B * sin(b * t);
    float z = C * sin(c * t);

    return vec3(x, y, z); // Return the dynamic light position
}

// Main rendering function
void main() {
    // Normalized pixel coordinates [-1, 1]
    uv = (vUV.st - 0.5) * 2.0;

    vec3 col = vec3(0.0); // Initial color value

    vec3 camPos = vec3(0.0, 0.0, -2.0); // Camera position
    vec3 rayPos = camPos; // Ray starting point
    vec3 rayDir = normalize(vec3(uv, 0.0) - camPos); // Ray direction
    float time = iTime * 0.2; // Dynamic time
    vec3 lightPos = lightPosLissajous(time); // Compute the light position

    float transmittance = 1.0; // Initial transmittance

    rayPos += rayDir; // Start advancing the ray
    for (int i = 0; i < RAY_ITER; i++) {
        rayPos += rayDir * RAY_STEP_SIZE; // Advance the ray
        float density = sampleDensity(rayPos); // Compute the density at the current position
        if (density <= 0.0) {
            continue; // If there is no density, go to the next iteration
        }
        vec3 lightDir = lightPos - rayPos; // Light direction
        float lightDistance = length(lightDir); // Light distance
        lightDir = lightDir / lightDistance; // Normalize to a unit vector
        float lightStep = lightDistance / float(LIGHT_ITER); // Lighting step size
        float sundotrd = dot(rayDir, -lightDir); // Dot product of the ray and light directions
        float scattering = getScattering(sundotrd); // Compute scattering
        vec3 lightRayPos = rayPos; // Ray position for shadow computation
        float shadowDensity = 0.0; // Initialize shadow density
        for (int j = 0; j < LIGHT_ITER; j++) {
            shadowDensity += sampleDensity(lightRayPos) * lightStep; // Accumulate shadow density
            lightRayPos += lightDir * lightStep; // Advance along the light direction
        }
        vec3 externalLight = vec3(exp(-shadowDensity * LIGHT_ATTEN) * scattering); // Compute external light
        col += transmittance * externalLight * density; // Accumulated color
        transmittance *= exp(-density * RAY_STEP_SIZE * LIGHT_ATTEN); // Update transmittance
        if (transmittance < 0.01) break; // Early termination if transmittance is low
    }

    col = pow(col, vec3(1.0 / 2.2)); // Gamma correction
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}

```
