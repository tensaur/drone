#ifndef PUFFERNET_H
#define PUFFERNET_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== RNG ===== */

int rand(void);
double randn(double mean, double std);

/* ===== Arena allocator ===== */

typedef struct {
    void*  data;
    size_t capacity;
    size_t used;
} Arena;

Arena* make_allocator(size_t total_size);
void*  alloc(Arena* allocator, size_t size);

/* ===== Weights loading / management ===== */

typedef struct Weights Weights;
struct Weights {
    float* data;
    int    size;
    int    idx;
};

void     _load_weights(const char* filename, float* weights, size_t num_weights);
Weights* load_weights(const char* filename, size_t num_weights);
float*   get_weights(Weights* weights, int num_weights);

/* ===== Low-level tensor ops (PufferNet versions of PyTorch ops) ===== */

void  _relu(float* input, float* output, int size);
void  _gelu(float* input, float* output, int size);
float _sigmoid(float x);

void _linear(
    float* input, float* weights, float* bias, float* output,
    int batch_size, int input_dim, int output_dim
);

void _linear_accumulate(
    float* input, float* weights, float* bias, float* output,
    int batch_size, int input_dim, int output_dim
);

void _conv2d(
    float* input, float* weights, float* bias,
    float* output, int batch_size, int in_width, int in_height,
    int in_channels, int out_channels, int kernel_size, int stride
);

void _conv3d(
    float* input, float* weights, float* bias,
    float* output, int batch_size, int in_width, int in_height, int in_depth,
    int in_channels, int out_channels, int kernel_size, int stride
);

void _lstm(
    float* input, float* state_h, float* state_c, float* weights_input,
    float* weights_state, float* bias_input, float* bias_state,
    float* buffer, int batch_size, int input_size, int hidden_size
);

void _embedding(
    int* input, float* weights, float* output,
    int batch_size, int num_embeddings, int embedding_dim
);

void _layernorm(
    float* input, float* weights, float* bias, float* output,
    int batch_size, int input_dim
);

void _one_hot(
    int* input, int* output,
    int batch_size, int input_size, int num_classes
);

void _cat_dim1(
    float* x, float* y, float* output,
    int batch_size, int x_size, int y_size
);

void _argmax_multidiscrete(
    float* input, int* output,
    int batch_size, int logit_sizes[], int num_actions
);

void _softmax_multidiscrete(
    float* input, int* output,
    int batch_size, int logit_sizes[], int num_actions
);

void _max_dim1(
    float* input, float* output,
    int batch_size, int seq_len, int feature_dim
);

/* ===== Layer structs and APIs ===== */

/* --- Linear --- */

typedef struct Linear Linear;
struct Linear {
    float* output;
    float* weights;
    float* bias;
    int    batch_size;
    int    input_dim;
    int    output_dim;
};

Linear* make_linear(Weights* weights, int batch_size, int input_dim, int output_dim);
void    linear(Linear* layer, float* input);
void    linear_accumulate(Linear* layer, float* input);

/* --- ReLU --- */

typedef struct ReLU ReLU;
struct ReLU {
    float* output;
    int    batch_size;
    int    input_dim;
};

ReLU* make_relu(int batch_size, int input_dim);
void  relu(ReLU* layer, float* input);

/* --- GELU --- */

typedef struct GELU GELU;
struct GELU {
    float* output;
    int    batch_size;
    int    input_dim;
};

GELU* make_gelu(int batch_size, int input_dim);
void  gelu(GELU* layer, float* input);

/* --- Max over dim=1 --- */

typedef struct MaxDim1 MaxDim1;
struct MaxDim1 {
    float* output;
    int    batch_size;
    int    seq_len;
    int    feature_dim;
};

MaxDim1* make_max_dim1(int batch_size, int seq_len, int feature_dim);
void     max_dim1(MaxDim1* layer, float* input);

/* --- Conv2D --- */

typedef struct Conv2D Conv2D;
struct Conv2D {
    float* output;
    float* weights;
    float* bias;
    int    batch_size;
    int    in_width;
    int    in_height;
    int    in_channels;
    int    out_channels;
    int    kernel_size;
    int    stride;
};

Conv2D* make_conv2d(
    Weights* weights, int batch_size, int in_width, int in_height,
    int in_channels, int out_channels, int kernel_size, int stride
);
void    conv2d(Conv2D* layer, float* input);

/* --- Conv3D --- */

typedef struct Conv3D Conv3D;
struct Conv3D {
    float* output;
    float* weights;
    float* bias;
    int    batch_size;
    int    in_width;
    int    in_height;
    int    in_depth;
    int    in_channels;
    int    out_channels;
    int    kernel_size;
    int    stride;
};

Conv3D* make_conv3d(
    Weights* weights, int batch_size, int in_width, int in_height, int in_depth,
    int in_channels, int out_channels, int kernel_size, int stride
);
void    conv3d(Conv3D* layer, float* input);

/* --- LSTM --- */

typedef struct LSTM LSTM;
struct LSTM {
    float* state_h;
    float* state_c;
    float* weights_input;
    float* weights_state;
    float* bias_input;
    float* bias_state;
    float* buffer;
    int    batch_size;
    int    input_size;
    int    hidden_size;
};

LSTM* make_lstm(Weights* weights, int batch_size, int input_size, int hidden_size);
void  lstm(LSTM* layer, float* input);

/* --- Embedding --- */

typedef struct Embedding Embedding;
struct Embedding {
    float* output;
    float* weights;
    int    batch_size;
    int    num_embeddings;
    int    embedding_dim;
};

Embedding* make_embedding(Weights* weights, int batch_size, int num_embeddings, int embedding_dim);
void       embedding(Embedding* layer, int* input);

/* --- LayerNorm --- */

typedef struct LayerNorm LayerNorm;
struct LayerNorm {
    float* output;
    float* weights;
    float* bias;
    int    batch_size;
    int    input_dim;
};

LayerNorm* make_layernorm(Weights* weights, int batch_size, int input_dim);
void       layernorm(LayerNorm* layer, float* input);

/* --- OneHot --- */

typedef struct OneHot OneHot;
struct OneHot {
    int* output;
    int  batch_size;
    int  input_size;
    int  num_classes;
};

OneHot* make_one_hot(int batch_size, int input_size, int num_classes);
void    one_hot(OneHot* layer, int* input);

/* --- Cat along dim=1 --- */

typedef struct CatDim1 CatDim1;
struct CatDim1 {
    float* output;
    int    batch_size;
    int    x_size;
    int    y_size;
};

CatDim1* make_cat_dim1(int batch_size, int x_size, int y_size);
void     cat_dim1(CatDim1* layer, float* x, float* y);

/* --- Multidiscrete helper --- */

typedef struct Multidiscrete Multidiscrete;
struct Multidiscrete {
    int batch_size;
    int logit_sizes[32];
    int num_actions;
};

Multidiscrete* make_multidiscrete(int batch_size, int logit_sizes[], int num_actions);
void           argmax_multidiscrete(Multidiscrete* layer, float* input, int* output);
void           softmax_multidiscrete(Multidiscrete* layer, float* input, int* output);

/* ===== Default high-level models ===== */

/* --- Default --- */

typedef struct Default Default;
struct Default {
    int            num_agents;
    float*         obs;
    Linear*        encoder;
    ReLU*          relu1;
    Linear*        actor;
    Linear*        value_fn;
    Multidiscrete* multidiscrete;
};

Default* make_default(Weights* weights, int num_agents, int input_dim, int hidden_dim, int action_dim);
void     free_default(Default* net);
void     forward_default(Default* net, float* observations, int* actions);

/* --- LinearLSTM --- */

typedef struct LinearLSTM LinearLSTM;
struct LinearLSTM {
    int            num_agents;
    float*         obs;
    Linear*        encoder;
    GELU*          gelu1;
    LSTM*          lstm;
    Linear*        actor;
    Linear*        value_fn;
    Multidiscrete* multidiscrete;
};

LinearLSTM* make_linearlstm(Weights* weights, int num_agents, int input_dim,
                            int logit_sizes[], int num_actions);
void        free_linearlstm(LinearLSTM* net);
void        forward_linearlstm(LinearLSTM* net, float* observations, int* actions);

/* --- ConvLSTM --- */

typedef struct ConvLSTM ConvLSTM;
struct ConvLSTM {
    int            num_agents;
    float*         obs;
    Conv2D*        conv1;
    ReLU*          relu1;
    Conv2D*        conv2;
    ReLU*          relu2;
    Linear*        linear;
    LSTM*          lstm;
    Linear*        actor;
    Linear*        value_fn;
    Multidiscrete* multidiscrete;
};

ConvLSTM* make_convlstm(Weights* weights, int num_agents, int input_dim,
                        int input_channels, int cnn_channels, int hidden_dim, int action_dim);
void      free_convlstm(ConvLSTM* net);
void      forward_convlstm(ConvLSTM* net, float* observations, int* actions);

/* --- LinearContLSTM (continuous actions) --- */

typedef struct LinearContLSTM LinearContLSTM;
struct LinearContLSTM {
    int     num_agents;
    float*  obs;
    float*  log_std;
    Linear* encoder;
    GELU*   gelu1;
    LSTM*   lstm;
    Linear* actor;
    Linear* value_fn;
    int     num_actions;
};

LinearContLSTM* make_linearcontlstm(Weights* weights, int num_agents, int input_dim,
                                    int logit_sizes[], int num_actions);
void            free_linearcontlstm(LinearContLSTM* net);
void            forward_linearcontlstm(LinearContLSTM* net, float* observations, float* actions);

#ifdef __cplusplus
}
#endif

#endif /* PUFFERNET_H */
