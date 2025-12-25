// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <math.h>

#include "drone.h"
#include "dronelib.h"
#include "raylib.h"

#define R (Color){255, 0, 0, 255}
#define W (Color){255, 255, 255, 255}
#define B (Color){0, 0, 255, 255}
Color COLORS[64] = {
    W, B, B, R, R, B, B, W,
    B, W, B, R, R, B, W, B,
    B, B, W, R, R, W, B, B,
    R, R, R, R, R, R, R, R,
    R, R, R, R, R, R, R, R,
    B, B, W, R, R, W, B, B,
    B, W, B, R, R, B, W, B,
    W, B, B, R, R, B, B, W
};
#undef R
#undef W
#undef B

typedef struct Client Client;

struct Client {
  Camera3D camera;
  float width;
  float height;

  float camera_distance;
  float camera_azimuth;
  float camera_elevation;
  bool is_dragging;
  Vector2 last_mouse_pos;

  Trail *trails;

  int selected_drone;
  bool inspect_mode;
  int target_fps;
};

void c_close_client(Client *client) {
  CloseWindow();
  free(client->trails);
  free(client);
}

static void update_camera_position(Client *c) {
  float r = c->camera_distance;
  float az = c->camera_azimuth;
  float el = c->camera_elevation;

  float x = r * cosf(el) * cosf(az);
  float y = r * cosf(el) * sinf(az);
  float z = r * sinf(el);

  c->camera.position = (Vector3){x, y, z};
  c->camera.target = (Vector3){0, 0, 0};
}

void handle_camera_controls(Client *client) {
  Vector2 mouse_pos = GetMousePosition();

  if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
    client->is_dragging = true;
    client->last_mouse_pos = mouse_pos;
  }

  if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
    client->is_dragging = false;
  }

  if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                           mouse_pos.y - client->last_mouse_pos.y};

    float sensitivity = 0.005f;

    client->camera_azimuth -= mouse_delta.x * sensitivity;

    client->camera_elevation += mouse_delta.y * sensitivity;
    client->camera_elevation =
        clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

    client->last_mouse_pos = mouse_pos;

    update_camera_position(client);
  }

  float wheel = GetMouseWheelMove();
  if (wheel != 0) {
    client->camera_distance -= wheel * 2.0f;
    client->camera_distance = clampf(client->camera_distance, 5.0f, 100.0f);
    update_camera_position(client);
  }
}

void handle_drone_selection(Client *client, int num_agents) {
  if (IsKeyPressed(KEY_D)) {
    client->selected_drone = (client->selected_drone + 1) % num_agents;
  }
  if (IsKeyPressed(KEY_A)) {
    client->selected_drone = (client->selected_drone - 1 + num_agents) % num_agents;
  }
}

void handle_fps_control(Client *client) {
  if (IsKeyPressed(KEY_W)) {
    client->target_fps += 10;
    if (client->target_fps > 240) client->target_fps = 240;
    SetTargetFPS(client->target_fps);
  }
  if (IsKeyPressed(KEY_S)) {
    client->target_fps -= 10;
    if (client->target_fps < 10) client->target_fps = 10;
    SetTargetFPS(client->target_fps);
  }
}

Client *make_client(DroneEnv *env) {
  Client *client = (Client *)calloc(1, sizeof(Client));

  client->width = WIDTH;
  client->height = HEIGHT;

  SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
  InitWindow(WIDTH, HEIGHT, "PufferLib Drone");

#ifndef __EMSCRIPTEN__
  SetTargetFPS(60);
#endif

  if (!IsWindowReady()) {
    TraceLog(LOG_ERROR, "Window failed to initialize\n");
    free(client);
    return NULL;
  }

  client->camera_distance = 40.0f;
  client->camera_azimuth = 0.0f;
  client->camera_elevation = PI / 10.0f;
  client->is_dragging = false;
  client->last_mouse_pos = (Vector2){0.0f, 0.0f};

  client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
  client->camera.fovy = 45.0f;
  client->camera.projection = CAMERA_PERSPECTIVE;

  update_camera_position(client);

  // Initialize trail buffer
  client->trails = (Trail *)calloc(env->num_agents, sizeof(Trail));
  for (int i = 0; i < env->num_agents; i++) {
    Trail *trail = &client->trails[i];
    trail->index = 0;
    trail->count = 0;
    for (int j = 0; j < TRAIL_LENGTH; j++) {
      trail->pos[j] = env->agents[i].state.pos;
    }
  }

  client->selected_drone = 0;
  client->inspect_mode = false;
  client->target_fps = 60;

  return client;
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_GREEN = (Color){0, 220, 80, 255};

void DrawRing3D(Target ring, float thickness, Color entryColor, Color exitColor) {
  float half_thick = thickness / 2.0f;

  Vector3 center_pos = {ring.pos.x, ring.pos.y, ring.pos.z};

  Vector3 entry_start_pos = {center_pos.x - half_thick * ring.normal.x,
                             center_pos.y - half_thick * ring.normal.y,
                             center_pos.z - half_thick * ring.normal.z};

  DrawCylinderWiresEx(entry_start_pos, center_pos, ring.radius, ring.radius, 32,
                      entryColor);

  Vector3 exit_end_pos = {center_pos.x + half_thick * ring.normal.x,
                          center_pos.y + half_thick * ring.normal.y,
                          center_pos.z + half_thick * ring.normal.z};

  DrawCylinderWiresEx(center_pos, exit_end_pos, ring.radius, ring.radius, 32,
                      exitColor);
}

void c_render(DroneEnv *env) {
  if (env->client == NULL) {
    env->client = make_client(env);
    if (env->client == NULL) {
      TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
      return;
    }
  }

  if (WindowShouldClose()) {
    c_close(env);
    exit(0);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    c_close(env);
    exit(0);
  }

  if (IsKeyPressed(KEY_SPACE)) {
    env->task = (DroneTask)((env->task + 1) % TASK_N);

    if (env->task == RACE) {
      reset_rings(env->ring_buffer, env->max_rings);
    }

    for (int i = 0; i < env->num_agents; i++) {
      set_target(env->task, env->agents, i, env->num_agents);
    }
  }

  handle_camera_controls(env->client);
  handle_drone_selection(env->client, env->num_agents);
  handle_fps_control(env->client);

  if (IsKeyPressed(KEY_TAB)) {
    env->client->inspect_mode = !env->client->inspect_mode;
  }

  Client *client = env->client;
  bool inspect_mode = client->inspect_mode;

  for (int i = 0; i < env->num_agents; i++) {
    Drone *agent = &env->agents[i];
    Trail *trail = &client->trails[i];
    trail->pos[trail->index] = agent->state.pos;
    trail->index = (trail->index + 1) % TRAIL_LENGTH;
    if (trail->count < TRAIL_LENGTH) {
      trail->count++;
    }
    if (env->terminals[i]) {
      trail->index = 0;
      trail->count = 0;
    }
  }

  BeginDrawing();
  ClearBackground(PUFF_BACKGROUND);

  BeginMode3D(client->camera);

  // draws bounding cube
  DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_X * 2.0f, GRID_Y * 2.0f,
                GRID_Z * 2.0f, WHITE);

  for (int i = 0; i < env->num_agents; i++) {
    Drone *agent = &env->agents[i];
    bool is_selected = (i == client->selected_drone);

    // Determine drone color - green if selected in inspect mode
    Color body_color;
    if (inspect_mode && is_selected) {
      body_color = PUFF_GREEN;
    } else {
      body_color = COLORS[i % 64];
    }

    // draws drone body
    DrawSphere(
        (Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z},
        0.3f, body_color);

    // draws rotors according to thrust
    float T[4];
    for (int j = 0; j < 4; j++) {
      float rpm =
          (env->actions[4 * i + j] + 1.0f) * 0.5f * agent->params.max_rpm;
      T[j] = agent->params.k_thrust * rpm * rpm;
    }

    const float rotor_radius = 0.15f;
    const float visual_arm_len = 0.75f;

    Vec3 rotor_offsets_body[4] = {{+visual_arm_len, 0.0f, 0.0f},
                                  {-visual_arm_len, 0.0f, 0.0f},
                                  {0.0f, +visual_arm_len, 0.0f},
                                  {0.0f, -visual_arm_len, 0.0f}};

    // Color base_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};
    Color base_colors[4] = {body_color, body_color, body_color, body_color};

    for (int j = 0; j < 4; j++) {
      Vec3 world_off = quat_rotate(agent->state.quat, rotor_offsets_body[j]);

      Vector3 rotor_pos = {agent->state.pos.x + world_off.x,
                           agent->state.pos.y + world_off.y,
                           agent->state.pos.z + world_off.z};

      float rpm =
          (env->actions[4 * i + j] + 1.0f) * 0.5f * agent->params.max_rpm;
      float intensity = 0.75f + 0.25f * (rpm / agent->params.max_rpm);

      Color rotor_color =
          (Color){(unsigned char)(base_colors[j].r * intensity),
                  (unsigned char)(base_colors[j].g * intensity),
                  (unsigned char)(base_colors[j].b * intensity), 255};

      DrawSphere(rotor_pos, rotor_radius, rotor_color);

      DrawCylinderEx(
          (Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z},
          rotor_pos, 0.02f, 0.02f, 8, BLACK);
    }

    // draws line with direction and magnitude of velocity / 10
    if (norm3(agent->state.vel) > 0.1f) {
      DrawLine3D(
          (Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z},
          (Vector3){agent->state.pos.x + agent->state.vel.x * 0.1f,
                    agent->state.pos.y + agent->state.vel.y * 0.1f,
                    agent->state.pos.z + agent->state.vel.z * 0.1f},
          MAGENTA);
    }

    // Draw line to target for selected drone in inspect mode
    if (inspect_mode && is_selected) {
      DrawLine3D(
          (Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z},
          (Vector3){agent->target->pos.x, agent->target->pos.y, agent->target->pos.z},
          ColorAlpha(PUFF_GREEN, 0.5f));
    }

    // Draw trailing path
    Trail *trail = &client->trails[i];
    if (trail->count <= 2) {
      continue;
    }
    for (int j = 0; j < trail->count - 1; j++) {
      int idx0 = (trail->index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
      int idx1 = (trail->index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;
      float alpha =
          (float)(TRAIL_LENGTH - j) / (float)trail->count * 0.8f; // fade out
      
      Color trail_base = (inspect_mode && is_selected) ? PUFF_GREEN : (Color){0, 187, 187, 255};
      Color trail_color = ColorAlpha(trail_base, alpha);
      
      DrawLine3D(
          (Vector3){trail->pos[idx0].x, trail->pos[idx0].y, trail->pos[idx0].z},
          (Vector3){trail->pos[idx1].x, trail->pos[idx1].y, trail->pos[idx1].z},
          trail_color);
    }
  }

  // Rings
  if (env->task == RACE) {
    float ring_thickness = 0.2f;
    for (int i = 0; i < env->max_rings; i++) {
      Target ring = env->ring_buffer[i];
      DrawRing3D(ring, ring_thickness, GREEN, BLUE);
    }
  }

  // Draw targets when TAB is held
  if (inspect_mode) {
    for (int i = 0; i < env->num_agents; i++) {
      Drone *agent = &env->agents[i];
      Vec3 target_pos = agent->target->pos;
      
      if (i == client->selected_drone) {
        DrawSphere((Vector3){target_pos.x, target_pos.y, target_pos.z}, 0.5f,
                   (Color){0, 255, 100, 180});
      } else {
        DrawSphere((Vector3){target_pos.x, target_pos.y, target_pos.z}, 0.45f,
                   (Color){0, 255, 255, 100});
      }
    }
  }

  EndMode3D();

  // HUD - Left side (always visible)
  int y = 10;
  DrawText(TextFormat("Task: %s", TASK_NAMES[env->task]), 10, y, 20, WHITE); y += 25;
  DrawText(TextFormat("Tick: %d / %d", env->tick, HORIZON), 10, y, 20, WHITE); y += 25;
  DrawText(TextFormat("FPS: %d (W/S to adjust)", client->target_fps), 10, y, 18, WHITE); y += 30;

  // Drone stats - only when inspect mode is on
  if (inspect_mode) {
    int idx = client->selected_drone;
    Drone *agent = &env->agents[idx];

    // Drone selection info
    DrawText(TextFormat("Drone: %d / %d (A/D to switch)", idx, env->num_agents - 1), 10, y, 20, PUFF_GREEN); y += 30;

    // Position and velocity
    DrawText(TextFormat("Pos: (%.1f, %.1f, %.1f)", agent->state.pos.x, agent->state.pos.y, agent->state.pos.z), 10, y, 18, WHITE); y += 20;
    DrawText(TextFormat("Vel: %.2f m/s", norm3(agent->state.vel)), 10, y, 18, WHITE); y += 20;
    DrawText(TextFormat("Omega: (%.1f, %.1f, %.1f)", agent->state.omega.x, agent->state.omega.y, agent->state.omega.z), 10, y, 18, WHITE); y += 25;

    // Motor RPMs as bar charts
    DrawText("Motor RPMs:", 10, y, 18, WHITE); y += 22;
    
    int bar_width = 150;
    int bar_height = 14;
    Color motor_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};
    const char* motor_names[4] = {"M1", "M2", "M3", "M4"};
    
    for (int m = 0; m < 4; m++) {
      float rpm_pct = agent->state.rpms[m] / agent->params.max_rpm;
      if (rpm_pct > 1.0f) rpm_pct = 1.0f;
      if (rpm_pct < 0.0f) rpm_pct = 0.0f;
      
      int filled_width = (int)(rpm_pct * bar_width);
      
      // Label
      DrawText(motor_names[m], 10, y, 16, motor_colors[m]);
      
      // Bar background
      DrawRectangle(35, y, bar_width, bar_height, (Color){40, 40, 40, 255});
      
      // Bar fill
      DrawRectangle(35, y, filled_width, bar_height, motor_colors[m]);
      
      // Bar outline
      DrawRectangleLines(35, y, bar_width, bar_height, LIGHTGRAY);
      
      // RPM value text
      DrawText(TextFormat("%.0f", agent->state.rpms[m]), 35 + bar_width + 5, y, 14, WHITE);
      
      y += bar_height + 4;
    }
    y += 10;

    // Episode stats
    DrawText(TextFormat("Episode Return: %.4f", agent->episode_return), 10, y, 18, WHITE); y += 20;
    DrawText(TextFormat("Episode Length: %d", agent->episode_length), 10, y, 18, WHITE); y += 30;
  }

  // Controls (always visible)
  DrawText("Left click + drag: Rotate camera", 10, y, 16, LIGHTGRAY); y += 18;
  DrawText("Mouse wheel: Zoom in/out", 10, y, 16, LIGHTGRAY); y += 18;
  DrawText("Space: Change task", 10, y, 16, LIGHTGRAY); y += 18;
  DrawText(TextFormat("Tab: Inspect mode [%s]", inspect_mode ? "ON" : "OFF"), 10, y, 16, inspect_mode ? PUFF_GREEN : LIGHTGRAY);

  EndDrawing();
}