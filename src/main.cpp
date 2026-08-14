#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX = 0.0;
static double lastY = 0.0;
static bool firstMouse = true;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;
std::vector<uchar4> hostDisplayBuffer;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
static bool mouseOverImGuiWinow = false;

// Pending file loads (set by ImGui, processed in runCuda)
static std::string pendingEnvMapPath = "";
static std::string pendingModelPath = "";

#ifdef _WIN32
static std::string openFileDialog(const char* filter, const char* title) {
    char filename[MAX_PATH] = { 0 };
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = title;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
    return "";
}
#endif

cudaEvent_t start_event, stop_event;
float elapsed_ms = 0.0f;
double total_time_s = 0.0;
long long total_iterations = 0;

// Forward declarations for window loop and interactivity
void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

void framebufferSizeCallback(GLFWwindow*, int framebufferWidth, int framebufferHeight)
{
    glViewport(0, 0, framebufferWidth, framebufferHeight);
}

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaError_t deviceResult = cudaSetDevice(0);
    if (deviceResult != cudaSuccess)
    {
        fprintf(stderr, "Failed to select CUDA device 0: %s\n",
            cudaGetErrorString(deviceResult));
        exit(EXIT_FAILURE);
    }

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    hostDisplayBuffer.resize(num_texels);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    GLenum glewResult = glewInit();
    const GLubyte* glVersion = glGetString(GL_VERSION);
    if (glewResult != GLEW_OK)
    {
        fprintf(stderr, "GLEW initialization warning: %s\n",
            reinterpret_cast<const char*>(glewGetErrorString(glewResult)));
        if (!glVersion)
        {
            return false;
        }
    }
    printf("OpenGL Version: %s\n", glVersion);
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    glViewport(0, 0, framebufferWidth, framebufferHeight);
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Path Tracer Analytics");                  // Create a window called "Hello, world!" and append into it.
    
    // LOOK: Un-Comment to check the output window and usage
    //ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    //ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
    //ImGui::Checkbox("Another Window", &show_another_window);

    //ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

    //if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
    //    counter++;
    //ImGui::SameLine();
    //ImGui::Text("counter = %d", counter);
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Camera Pos: %.2f, %.2f, %.2f", imguiData->CamPos.x, imguiData->CamPos.y, imguiData->CamPos.z);
    
    ImGui::Separator(); // Add a dividing line
    ImGui::Text("GPU Render Time: %.3f ms/iteration", elapsed_ms);
    if (total_time_s > 0) {
        ImGui::Text("Average Throughput: %.2f iterations/sec", total_iterations / total_time_s);
    }
    ImGui::Separator();

#if ENABLE_DENOISER
    ImGui::Checkbox("Denoiser (OptiX AI)", &guiData->denoiserEnabled);
#endif

#if ENABLE_NRC
    if (ImGui::Checkbox("Neural Radiance Cache", &guiData->nrcEnabled)) {
        camchanged = true;
    }
    if (guiData->nrcEnabled) {
        ImGui::SliderFloat("NRC Train Fraction", &guiData->nrcTrainFraction, 0.0f, 1.0f);
    }
#endif

#if ENABLE_RESTIR_DI
    if (ImGui::Checkbox("ReSTIR Direct Illumination", &guiData->restirEnabled)) {
        camchanged = true;
    }
    if (guiData->restirEnabled) {
        if (ImGui::SliderInt("M Initial Candidates", &guiData->restirM, 1, 32)) camchanged = true;
        if (ImGui::SliderInt("Spatial Taps", &guiData->restirSpatialTaps, 0, 10)) camchanged = true;
    }
#endif

    ImGui::Separator();
    ImGui::Text("File Loading");
#ifdef _WIN32
    if (ImGui::Button("Load Skybox (.hdr/.exr)")) {
        std::string path = openFileDialog(
            "HDR/EXR Files\0*.hdr;*.exr\0All Files\0*.*\0",
            "Select HDRI Environment Map");
        if (!path.empty()) {
            pendingEnvMapPath = path;
        }
    }
    if (ImGui::Button("Load Model (.gltf/.glb)")) {
        std::string path = openFileDialog(
            "glTF Files\0*.gltf;*.glb\0All Files\0*.*\0",
            "Select 3D Model");
        if (!path.empty()) {
            pendingModelPath = path;
        }
    }
#endif

    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if (!MouseOverImGuiWindow()) {
            float speed = 0.25f;
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
                speed *= 4.0f; // Move 4x faster when holding Shift
            }
            
            Camera& cam = scene->state.camera;
            
            // Use the true 3D view direction (includes vertical component)
            glm::vec3 forward = glm::normalize(cam.view);
            
            // Right stays horizontal to avoid drifting vertically when strafing
            glm::vec3 right = cam.right;
            right.y = 0.0f;
            if (glm::length(right) > 0.0f) right = glm::normalize(right);

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                cam.lookAt += forward * speed;  // move toward view direction
                camchanged = true;
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                cam.lookAt -= forward * speed;  // move away from view direction
                camchanged = true;
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                cam.lookAt -= right * speed;
                camchanged = true;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                cam.lookAt += right * speed;
                camchanged = true;
            }
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
                cam.lookAt.y += speed;
                camchanged = true;
            }
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
                cam.lookAt.y -= speed;
                camchanged = true;
            }
        }

        runCuda();

        std::string title = "CIS5650 Path Tracer | Wavefront | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);


    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    if (!init())
    {
        fprintf(stderr, "Failed to initialize the OpenGL/ImGui window\n");
        return EXIT_FAILURE;
    }

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage()
{
    float samples = iteration;
#if ENABLE_SPECTRAL_RENDERING
    samples *= SPECTRAL_N;
#endif
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;

        total_time_s = 0.0;
        total_iterations = 0;

        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0); // world up
        glm::vec3 r = glm::normalize(glm::cross(v, u));
        cam.up = glm::normalize(glm::cross(r, v));
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
    {
        pathtraceFree();
        pathtraceInit(scene);
    }

    // Process pending file loads
    if (!pendingEnvMapPath.empty()) {
        std::string path = pendingEnvMapPath;
        pendingEnvMapPath = "";
        if (scene->loadEnvironmentMap(path)) {
            pathtraceReloadEnvMap(scene->envMap);
            iteration = 0;  // reset to re-render
        }
    }
    if (!pendingModelPath.empty()) {
        std::string path = pendingModelPath;
        pendingModelPath = "";
        // Save env map state before scene reload
        EnvironmentMap savedEnvMap = scene->envMap;
        delete scene;
        scene = new Scene(path);
        scene->envMap = savedEnvMap;
        renderState = &scene->state;
        pathtraceReloadScene(scene);
        iteration = 0;
        camchanged = true;
    }

    if (iteration < renderState->iterations)
    {
        cudaEventRecord(start_event, 0);

        iteration++;

        // execute the kernel
        int frame = 0;
        pathtrace(nullptr, frame, iteration, guiData);


        cudaEventRecord(stop_event, 0);
        // unmap buffer object
        const float divisor = static_cast<float>(iteration * SPECTRAL_N);
        for (size_t i = 0; i < hostDisplayBuffer.size(); ++i)
        {
            glm::vec3 linear = renderState->image[i] / divisor;
            auto toSrgbByte = [](float value) -> unsigned char {
                value = glm::max(value, 0.0f);
                float srgb = value <= 0.0031308f
                    ? 12.92f * value
                    : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
                return static_cast<unsigned char>(glm::clamp(
                    static_cast<int>(srgb * 255.0f + 0.5f), 0, 255));
            };
            hostDisplayBuffer[i] = make_uchar4(
                toSrgbByte(linear.x), toSrgbByte(linear.y),
                toSrgbByte(linear.z), 0);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0,
            hostDisplayBuffer.size() * sizeof(uchar4), hostDisplayBuffer.data());
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
        total_time_s += elapsed_ms / 1000.0;
        total_iterations++;
        if (iteration % 10 == 0)
        {
            printf("Iteration %d: %.3f ms (average %.2f iterations/sec)\n",
                iteration, elapsed_ms,
                total_time_s > 0.0 ? total_iterations / total_time_s : 0.0);
        }
    }
    else
    {
        std::cout << "Maximum iterations reached. Average throughput is " << total_iterations / total_time_s << " iterations/sec"  << std::endl;


        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                saveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_P:  // P = Print/save (S is now used for backward movement)
                saveImage();
                break;
            case GLFW_KEY_SPACE:
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.lookAt = ogLookAt;
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        // Clear all pressed states so drags don't bleed into the viewport
        leftMousePressed = false;
        rightMousePressed = false;
        middleMousePressed = false;
        return;
    }

    // Track each button independently — don't let one button's event reset another
    if (button == GLFW_MOUSE_BUTTON_LEFT)
        leftMousePressed = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
        rightMousePressed = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        middleMousePressed = (action == GLFW_PRESS);

    // On any fresh press, capture current cursor position so the first drag
    // delta is zero (avoids a snap-jump from a stale lastX/lastY).
    if (action == GLFW_PRESS) {
        glfwGetCursorPos(window, &lastX, &lastY);
    }
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    if (xpos == lastX && ypos == lastY)
    {
        return; // skip if cursor truly hasn't moved
    }

    if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}
