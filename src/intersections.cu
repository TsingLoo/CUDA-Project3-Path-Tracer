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
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float squarePlaneIntersectionTest(
    Geom plane,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    glm::vec3 ro = multiplyMV(plane.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(plane.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 planeNormal_local = glm::vec3(0.0f, 0.0f, 1.0f);

    float denom = glm::dot(rd, planeNormal_local);

    if (fabs(denom) < EPSILON) {
        return -1.0f;
    }

    float t = -ro.z / denom;

    if (t < EPSILON) {
        return -1.0f;
    }


    glm::vec3 intersection_local = ro + t * rd;

    if (fabs(intersection_local.x) > 0.5f || fabs(intersection_local.y) > 0.5f) {
        return -1.0f; // The hit was outside the square's bounds.
    }

    // 7. We have a valid hit! Calculate the output parameters.
    // The world-space intersection point can be calculated most accurately
    // using the original ray and the calculated distance 't'.
    intersectionPoint = r.origin + r.direction * t;

    // The normal in local space is (0,0,1). Transform it back to world space.
    normal = glm::normalize(multiplyMV(plane.invTranspose, glm::vec4(planeNormal_local, 0.0f)));

    // Determine if we hit the front face ("outside").
    outside = (glm::dot(r.direction, normal) < 0.0f);
    // If we hit the back face, flip the normal for two-sided lighting.
    if (!outside) {
        normal = -normal;
    }

    // Return the true distance from the ray origin to the intersection point.
    // Note: this is equal to 't' if r.direction is normalized.
    return glm::length(r.origin - intersectionPoint);
}

__host__ 

//reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
__host__ __device__ float triangleIntersectionTest(
    Triangle triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    glm::vec3 ro = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 v0v1 = triangle.v1 - triangle.v0;
    glm::vec3 v0v2 = triangle.v2 - triangle.v0;

    glm::vec3 pvec = glm::cross(rd, v0v2);
    float det = glm::dot(v0v1, pvec);

    if (fabs(det) < EPSILON) {
        return -1.0f;
    }

    float invDet = 1.0f / det;

    glm::vec3 tvec = ro - triangle.v0;

    float u = glm::dot(tvec, pvec) * invDet;

    if (u < 0.0f || u > 1.0f) {
        return -1.0f;
    }

    glm::vec3 qvec = glm::cross(tvec, v0v1);
    float v = glm::dot(rd, qvec) * invDet;

    if (v < 0.0f || u + v > 1.0f) {
        return -1.0f;
    }

    float t = glm::dot(v0v2, qvec) * invDet;
    if (t > EPSILON) {

        intersectionPoint = r.origin + r.direction * t;

        normal = glm::normalize(multiplyMV(glm::transpose(triangle.inverseTransform), glm::vec4(triangle.normal, 0.0f)));

        outside = (glm::dot(r.direction, normal) < 0.0f);

        return t;
    }

    return -1.0f;
}
