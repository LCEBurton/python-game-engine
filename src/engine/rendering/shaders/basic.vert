#version 330 core

in vec3 position;
in vec3 color;

out vec3 fragColor;

uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    fragColor = color;
}
