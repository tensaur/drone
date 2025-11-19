// Physical constants for the drone
#define BASE_MASS 1.0f       // kg
#define BASE_IXX 0.01f       // kgm^2
#define BASE_IYY 0.01f       // kgm^2
#define BASE_IZZ 0.02f       // kgm^2
#define BASE_ARM_LEN 0.1f    // m
#define BASE_K_THRUST 3e-5f  // thrust coefficient
#define BASE_K_ANG_DAMP 0.2f // angular damping coefficient
#define BASE_K_DRAG 1e-6f    // drag (torque) coefficient
#define BASE_B_DRAG 0.1f     // linear drag coefficient
#define BASE_GRAVITY 9.81f   // m/s^2
#define BASE_MAX_RPM 750.0f  // rad/s
#define BASE_MAX_VEL 50.0f   // m/s
#define BASE_MAX_OMEGA 50.0f // rad/s
#define BASE_K_MOT 0.1f      // s (Motor lag constant)
#define BASE_J_MOT 1e-5f     // kgm^2 (Motor rotational inertia)

// Simulation properties
#define GRID_X 30.0f
#define GRID_Y 30.0f
#define GRID_Z 10.0f
#define MARGIN_X (GRID_X - 1)
#define MARGIN_Y (GRID_Y - 1)
#define MARGIN_Z (GRID_Z - 1)
#define RING_RADIUS 2.0f
#define V_TARGET 0.05f
#define DT 0.05f
#define DT_RNG 0.0f

static float clip(float val, float min, float max) {
  if (val < min) return min;
  if (val > max) return max;
  return val;
}

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float x, y, z;
} Vec3;

static inline Quat quat_inverse(Quat q) { 
    return (Quat){q.w, -q.x, -q.y, -q.z}; 
}

static inline Quat quat_mul(Quat q1, Quat q2) {
    Quat out;
    out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return out;
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = {0.0f, v.x, v.y, v.z};
    Quat tmp = quat_mul(q, qv);
    Quat q_conj = {q.w, -q.x, -q.y, -q.z};
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}
