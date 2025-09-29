#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    glm::vec3 minCorner = glm::vec3(-0.5f, -0.5f, -0.5f);
    glm::vec3 maxCorner = glm::vec3(0.5f, 0.5f, 0.5f);

    Ray localRay;
    localRay.origin = glm::vec3(box.inverseTransform * glm::vec4(r.origin, 1.0f));
    localRay.direction = glm::vec3(box.inverseTransform * glm::vec4(r.direction, 0.0f));

    // 2. Perform Slab Test
    glm::vec3 invDir = 1.0f / localRay.direction;
    glm::vec3 near_t = (minCorner - localRay.origin) * invDir;
    glm::vec3 far_t = (maxCorner - localRay.origin) * invDir;

    glm::vec3 tmin = glm::min(near_t, far_t);
    glm::vec3 tmax = glm::max(near_t, far_t);

    float t0 = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
    float t1 = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

    // 3. Check for miss
    if (t0 >= t1) { // Note: GLSL used '>', '>=' is slightly more robust
        return -1;
    }

    float t = -1.0f;
    glm::vec3 objectNormal;

    // 4. Determine hit point and normal
    if (t0 > 0.0001f) // If we are outside the box
    {
        t = t0;
        outside = true; // This is an entry point

        // This clever trick creates a one-hot vector (1,0,0) or (0,1,0) etc.
        // that corresponds to the axis of the face we hit (the one with max tmin).
        // GLSL's tmin.yzx is manually created here.
        glm::vec3 yzx = glm::vec3(tmin.y, tmin.z, tmin.x);
        glm::vec3 zxy = glm::vec3(tmin.z, tmin.x, tmin.y);
        objectNormal = -sign(localRay.direction) * step(yzx, tmin) * step(zxy, tmin);
    }
    else if (t1 > 0.0001f) // If we are inside the box
    {
        t = t1;
        outside = false; // This is an exit point

        // Same trick, but for the exit point normal using tmax
        glm::vec3 yzx = glm::vec3(tmax.y, tmax.z, tmax.x);
        glm::vec3 zxy = glm::vec3(tmax.z, tmax.x, tmax.y);
        objectNormal = -sign(localRay.direction) * step(tmax, yzx) * step(tmax, zxy);
    }
    else {
        return -1.0; // Box is entirely behind the ray
    }

    // 5. Calculate final world-space results
    intersectionPoint = r.origin + r.direction * t;

    // Transform normal from object space back to world space
    // The inverse transpose correctly handles non-uniform scaling.
    normal = glm::normalize(glm::vec3(box.invTranspose * glm::vec4(objectNormal, 0.0f)));

    return t;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // 1. Transform the ray from world space into the sphere's object space
    Ray localRay;
    localRay.origin = glm::vec3(sphere.inverseTransform * glm::vec4(r.origin, 1.0f));
    localRay.direction = glm::vec3(sphere.inverseTransform * glm::vec4(r.direction, 0.0f));

    // The sphere is now at the origin (0,0,0) with radius 1.0 in its own space
    glm::vec3 objectSpaceCenter = glm::vec3(0.0f);
    float objectSpaceRadius = 0.5f;

    glm::vec3 oc = localRay.origin - objectSpaceCenter;

    // 2. Solve the quadratic equation in object space
    float a = glm::dot(localRay.direction, localRay.direction);
    float b = 2.0f * glm::dot(oc, localRay.direction);
    float c = glm::dot(oc, oc) - objectSpaceRadius * objectSpaceRadius;

    float t0, t1;
    if (!solveQuadratic(a, b, c, t0, t1)) {
        return FLT_MAX; // Ray misses the sphere
    }

    // 3. Find the correct intersection distance 't'
    // This 't' is valid for BOTH the local ray and the original world-space ray
    float t = -1.0f;
    if (t0 > 0.0001f) {
        t = t0;
    }
    else if (t1 > 0.0001f) {
        t = t1;
    }
    else {
        return FLT_MAX; // Both intersections are behind the ray's origin
    }

    // 4. Calculate world-space results
    intersectionPoint = r.origin + r.direction * t;

    // Calculate the normal in object space...
    glm::vec3 objectNormal = glm::normalize((localRay.origin + localRay.direction * t) - objectSpaceCenter);

    // ...and transform it back to world space using the inverse transpose matrix
    // This correctly handles non-uniform scaling.
    normal = glm::normalize(glm::vec3(sphere.invTranspose * glm::vec4(objectNormal, 0.0f)));

    // Determine if the original ray started inside or outside
    outside = (c >= 0.0f);

    return t;
}
