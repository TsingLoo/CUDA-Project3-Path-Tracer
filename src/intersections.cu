#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
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
    normal = glm::normalize(glm::vec3(glm::transpose(sphere.inverseTransform) * glm::vec4(objectNormal, 0.0f)));

    // Determine if the original ray started inside or outside
    outside = (c >= 0.0f);

    return t;
}
