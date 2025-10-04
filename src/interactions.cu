#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__
thrust::default_random_engine makeSeededRandomEngineKern(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}


__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
}

__global__ void kernShadeLambertian(int num_hit, LambertianHitWorkItem* queue, PathSegment* paths, Material* materials) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hit) return;

    LambertianHitWorkItem item = queue[idx];
    PathSegment& path = paths[item.path_idx];
    Material material = materials[item.material_id];

    thrust::default_random_engine rng = makeSeededRandomEngineKern(num_hit, path.pixelIndex, path.remainingBounces);
    thrust::uniform_real_distribution<float> u01(0, 1);
    const glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
    float rand = u01(rng);

    glm::vec3 nor = item.surface_normal;
    glm::vec3 wiLocal = squareToHemisphereCosine(xi);
    glm::mat3 mat = TangentSpaceToWorld(nor);
    glm::vec3 wiWorld = mat * wiLocal;

    path.color *= material.color;

    path.ray.origin = item.intersect_point + nor * EPSILON;
    path.ray.direction = wiWorld;

    float survival_prob = glm::max(path.color.r, glm::max(path.color.g, path.color.b));
    survival_prob = glm::min(survival_prob, 1.0f);

    if (rand > survival_prob){
        path.remainingBounces = 0;
    }
    else {
        path.color /= survival_prob;
    }

    path.remainingBounces--;
}
