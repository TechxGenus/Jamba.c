/* Inference for Jamba model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Jamba model

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling and top-k gate

typedef struct {
    int dim; // jamba dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    int num_experts; // number of experts in the expert model
    int top_k; // expert num per token
    int attn_offset; // offset for the attention layer
    int attn_period; // period for the attention layer
    int expert_offset; // offset for the expert layer
    int expert_period; // period for the expert layer
    int d_conv; // mamba conv1d dimension
    int d_state; // mamba state dimension
    int dt_rank; // time step rank
    int d_inner; // mamba intermediate size
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (attn_layer, dim, n_heads * head_size)
    float* wk; // (attn_layer, dim, n_kv_heads * head_size)
    float* wv; // (attn_layer, dim, n_kv_heads * head_size)
    float* wo; // (attn_layer, n_heads * head_size, dim)
    // weight for mamba
    float* in_proj; // (mamba_layer, 2*d_inner, dim)
    float* conv1d_weight; // (mamba_layer, d_inner, 1, d_conv)
    float* conv1d_bias; // (mamba_layer, d_inner)
    float* x_proj; // (mamba_layer, dt_rank+2*d_state, d_inner)
    float* dt_proj_weight; // (mamba_layer, d_inner, dt_rank)
    float* dt_proj_bias; // (mamba_layer, d_inner)
    float* A; // (mamba_layer, d_inner, d_state)
    float* D; // (mamba_layer, d_inner)
    float* out_proj; // (mamba_layer, dim, d_inner)
    float* dt_rms; // (mamba_layer, dt_rank)
    float* B_rms; // (mamba_layer, d_state)
    float* C_rms; // (mamba_layer, d_state)
    // weights for moe
    float* gate; // (moe_layer, dim, num_experts)
    float* moe_w1; // (moe_layer, num_experts, hidden_dim, dim)
    float* moe_w2; // (moe_layer, num_experts, dim, hidden_dim)
    float* moe_w3; // (moe_layer, num_experts, hidden_dim, dim)
    // weights for mlp
    float* mlp_w1; // (mlp_layer, num_experts, hidden_dim, dim)
    float* mlp_w2; // (mlp_layer, num_experts, dim, hidden_dim)
    float* mlp_w3; // (mlp_layer, num_experts, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} JambaWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb3; // buffer for hidden dimension in the ffn (dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *xz; // (2*d_inner) x and z are pointers into this buffer
    float *x_db; // (dt_rank+2*d_state) dt, B, C are pointers into this buffer
    float *dt; // (d_inner) later, dt is a pointer to this buffer
    float *dA; // (d_inner, d_state)
    float *dB; // (d_inner, d_state)
    float *temp; // (d_inner, d_state)
    float *y; // (d_inner)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (attn_layer, seq_len, dim)
    float* value_cache; // (attn_layer, seq_len, dim)
    // state cache
    float* conv_state; // (mamba_layer, d_inner, d_conv)
    float* ssm_state;  // (mamba_layer, d_inner, d_state)
    // MoE
    float *exp_logits; // expert logits (num_experts,)
    ProbIndex *exp_probindex; // buffer for expert probabilities
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    JambaWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Jamba;

void malloc_run_state(RunState* s, Config* p) {
    int attn_layer = (p->n_layers - p->attn_offset - 1) / p->attn_period + 1;
    int mamba_layer = p->n_layers - attn_layer;
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->hb3 = calloc(p->dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(attn_layer * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(attn_layer * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->exp_logits = calloc(p->num_experts, sizeof(float));
    s->exp_probindex = calloc(p->num_experts, sizeof(ProbIndex));
    s->xz = calloc(2 * p->d_inner, sizeof(float));
    s->x_db = calloc((p->dt_rank + 2 * p->d_state), sizeof(float));
    s->dt = calloc(p->d_inner, sizeof(float));
    s->dA = calloc(p->d_inner * p->d_state, sizeof(float));
    s->dB = calloc(p->d_inner * p->d_state, sizeof(float));
    s->temp = calloc(p->d_inner * p->d_state, sizeof(float));
    s->y = calloc(p->d_inner, sizeof(float));
    s->conv_state = calloc(mamba_layer * p->d_inner * p->d_conv, sizeof(float));
    s->ssm_state = calloc(mamba_layer * p->d_inner * p->d_state, sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->hb3 || !s->q || !s->key_cache || !s->value_cache
     || !s->att || !s->logits || !s->exp_logits || !s->exp_probindex || !s->xz || !s->x_db || !s->dt || !s->dA
     || !s->dB || !s->temp || !s->y || !s->logits || !s->conv_state || !s->ssm_state) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->hb3);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
    free(s->exp_logits);
    free(s->exp_probindex);
    free(s->xz);
    free(s->x_db);
    free(s->dt);
    free(s->dA);
    free(s->dB);
    free(s->temp);
    free(s->y);
    free(s->logits);
    free(s->conv_state);
    free(s->ssm_state);
}

void memory_map_weights(JambaWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    int attn_layer = (p->n_layers - p->attn_offset - 1) / p->attn_period + 1;
    int mamba_layer = p->n_layers - attn_layer;
    int moe_layer = (p->n_layers - p->expert_offset - 1) / p->expert_period + 1;
    int mlp_layer = p->n_layers - moe_layer;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += attn_layer * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += attn_layer * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += attn_layer * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += attn_layer * (p->n_heads * head_size) * p->dim;
    w->in_proj = ptr;
    ptr += mamba_layer * 2 * p->d_inner * p->dim;
    w->conv1d_weight = ptr;
    ptr += mamba_layer * p->d_inner * 1 * p->d_conv;
    w->conv1d_bias = ptr;
    ptr += mamba_layer * p->d_inner;
    w->x_proj = ptr;
    ptr += mamba_layer * (p->dt_rank + 2 * p->d_state) * p->d_inner;
    w->dt_proj_weight = ptr;
    ptr += mamba_layer * p->d_inner * p->dt_rank;
    w->dt_proj_bias = ptr;
    ptr += mamba_layer * p->d_inner;
    w->A = ptr;
    ptr += mamba_layer * p->d_inner * p->d_state;
    w->D = ptr;
    ptr += mamba_layer * p->d_inner;
    w->out_proj = ptr;
    ptr += mamba_layer * p->dim * p->d_inner;
    w->dt_rms = ptr;
    ptr += mamba_layer * p->dt_rank;
    w->B_rms = ptr;
    ptr += mamba_layer * p->d_state;
    w->C_rms = ptr;
    ptr += mamba_layer * p->d_state;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->gate = ptr;
    ptr += moe_layer * p->dim * p->num_experts;
    w->moe_w1 = ptr;
    ptr += moe_layer * p->num_experts * p->dim * p->hidden_dim;
    w->moe_w2 = ptr;
    ptr += moe_layer * p->num_experts * p->hidden_dim * p->dim;
    w->moe_w3 = ptr;
    ptr += moe_layer * p->num_experts * p->dim * p->hidden_dim;
    w->mlp_w1 = ptr;
    ptr += mlp_layer * p->dim * p->hidden_dim;
    w->mlp_w2 = ptr;
    ptr += mlp_layer * p->hidden_dim * p->dim;
    w->mlp_w3 = ptr;
    ptr += mlp_layer * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, JambaWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read the magic number
    unsigned int magic;
    if (fread(&magic, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic != 0x4a616d62) { fprintf(stderr, "Invalid magic number: %x\n", magic); exit(EXIT_FAILURE); }
    // read the version
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 1) { fprintf(stderr, "Invalid version: %d\n", version); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Jamba weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config) / sizeof(float) + 2;
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_jamba(Jamba *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_jamba(Jamba* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Jamba

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-6f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

float softplus(float x) {
    return logf(1.0f + expf(x));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float silu(float x) {
    return x * sigmoid(x);
}

void shift_matrix_left(float* matrix, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            matrix[i * cols + j] = matrix[i * cols + j + 1];
        }
    }
}

void update_last_column(float* matrix, float* x, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        matrix[i * cols + cols - 1] = x[i];
    }
}

void rowwise_dot_product(float* out, float* matrix, float* weights, int rows, int cols) {
    // matrix[rows,cols], weights[cols] -> out[rows]
    // this is a dot product of each row of the matrix with the weights
    // i.e. out[i] = matrix[i,:] @ weights
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float val = 0.0f;
        for (int j = 0; j < cols; j++) {
            val += matrix[i * cols + j] * weights[j];
        }
        out[i] = val;
    }
}

void linear(float* xout, float* x, float* w, float* b, int d, int n) {
    // w[d,n] @ x[n] + b[d] -> xout[d]
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val + b[i];
    }
}

void broadcast_multiply(float* out, float* x, float* y, int d, int n) {
    // x[d], y[d,n] -> out[d,n]
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            int index = i * n + j;
            out[index] = x[i] * y[index];
            //out[i * n + j] = x[i] * y[i * n + j];
        }
    }
}

void elementwise_multiply(float* result, float* matrix1, float* matrix2, int total_elements) {
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        result[i] = matrix1[i] * matrix2[i];
    }
}

void elementwise_add(float* result, float* matrix1, float* matrix2, int total_elements) {
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        result[i] = matrix1[i] + matrix2[i];
    }
}

void elementwise_multiply_and_add(float* result, float* matrix1, float* matrix2, float* matrix3, int total_elements) {
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        result[i] = matrix1[i] * matrix2[i] + matrix3[i];
    }
}

void outer_product(float* out, float* x, float* y, int d, int n) {
    // x[d], y[n] -> out[d,n]
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            out[i * n + j] = x[i] * y[j];
        }
    }
}

void sum_along_last_dim(float* result, float* matrix, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float val = 0.0f;
        for (int j = 0; j < cols; j++) {
            val += matrix[i * cols + j];
        }
        result[i] = val;
    }
}

float* forward(Jamba* jamba, int token, int pos) {

    // a few convenience variables
    Config* p = &jamba->config;
    JambaWeights* w = &jamba->weights;
    RunState* s = &jamba->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int num_experts = p->num_experts;
    int top_k = p->top_k;
    int attn_offset = p->attn_offset;
    int attn_period = p->attn_period;
    int expert_offset = p->expert_offset;
    int expert_period = p->expert_period;
    int d_conv = p->d_conv;
    int d_state = p->d_state;
    int dt_rank = p->dt_rank;
    int d_inner = p->d_inner;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        if (l >= attn_offset && (l - attn_offset) % attn_period == 0) {
            int attn_l = (l - attn_offset) / attn_period;

            // key and value point to the kv cache
            int loff = attn_l * p->seq_len * kv_dim; // kv cache layer offset for convenience
            s->k = s->key_cache + loff + pos * kv_dim;
            s->v = s->value_cache + loff + pos * kv_dim;

            // qkv matmuls for this position
            matmul(s->q, s->xb, w->wq + attn_l*dim*dim, dim, dim);
            matmul(s->k, s->xb, w->wk + attn_l*dim*kv_dim, dim, kv_dim);
            matmul(s->v, s->xb, w->wv + attn_l*dim*kv_dim, dim, kv_dim);

            // multihead attention. iterate over all heads
            int h;
            #pragma omp parallel for private(h)
            for (h = 0; h < p->n_heads; h++) {
                // get the query vector for this head
                float* q = s->q + h * head_size;
                // attention scores for this head
                float* att = s->att + h * p->seq_len;
                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
                    float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += q[i] * k[i];
                    }
                    score /= sqrtf(head_size);
                    // save the score to the attention buffer
                    att[t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(att, pos + 1);

                // weighted sum of the values, store back into xb
                float* xb = s->xb + h * head_size;
                memset(xb, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = att[t];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < head_size; i++) {
                        xb[i] += a * v[i];
                    }
                }
            }

            // final matmul to get the output of the attention
            matmul(s->xb2, s->xb, w->wo + attn_l*dim*dim, dim, dim);

            // residual connection back into x
            for (int i = 0; i < dim; i++) {
                x[i] += s->xb2[i];
            }
        }
        else {
            int mamba_l = l < attn_offset ? l : l - 1 - (l - attn_offset) / attn_period;
            float* dA = s->dA;  // (d_inner, d_state)
            float* dB = s->dB;  // (d_inner, d_state)
            float* y  = s->y;   // (d_inner)

            // conv_state, ssm_state = self._get_states_from_cache(inference_params)
            float* conv_state = s->conv_state + mamba_l * d_inner * d_conv;
            float* ssm_state  = s->ssm_state  + mamba_l * d_inner * d_state;

            // xz = self.in_proj(hidden_states)  # hidden_states: (dim), in_proj (2*d_inner, dim), xz (2*d_inner)
            matmul(s->xz, s->xb, w->in_proj + mamba_l * 2*d_inner*dim, dim, 2*d_inner);
            // hx, z = xz.chunk(2, dim=-1)
            float* hx = s->xz;            // hx (d_inner)
            float* z = s->xz + d_inner;  // z (d_inner)


            // Conv step

            // conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            shift_matrix_left(conv_state, d_inner, d_conv);
            // conv_state[:, -1] = hx
            update_last_column(conv_state, hx, d_inner, d_conv);
            // hx = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            elementwise_multiply(s->temp, conv_state, w->conv1d_weight + mamba_l*d_inner*d_conv, d_inner * d_conv);
            sum_along_last_dim(hx, s->temp, d_inner, d_conv);
            // hx = hx + self.conv1d.bias
            elementwise_add(hx, hx, w->conv1d_bias + mamba_l*d_inner, d_inner);
            // hx = F.silu(hx)
            for (int i = 0; i < d_inner; i++) {
                hx[i] = silu(hx[i]);
            }


            // SSM step

            // x_db = self.x_proj(hx)   # x_db (dt_rank+2*d_state)
            matmul(s->x_db, hx, w->x_proj + mamba_l*(dt_rank+2*d_state)*d_inner, d_inner, dt_rank+2*d_state);
            // dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            float *dt = s->x_db;                     // dt (dt_rank)
            float *B = s->x_db + dt_rank;            // B  (d_state)
            float *C = s->x_db + dt_rank + d_state;  // C  (d_state)

            rmsnorm(dt, dt, w->dt_rms + mamba_l*dt_rank, dt_rank);
            rmsnorm(B, B, w->B_rms + mamba_l*d_state, d_state);
            rmsnorm(C, C, w->C_rms + mamba_l*d_state, d_state);

            // dt = self.dt_proj(dt)   # dt (dt_rank), dt_proj_weight (d_inner, dt_rank), dt_proj_bias (d_inner)
            linear(s->dt, dt, w->dt_proj_weight + mamba_l*d_inner*dt_rank, w->dt_proj_bias + mamba_l*d_inner, d_inner, dt_rank);
            dt = s->dt;  // NOTE: dt is now bigger: (d_inner) instead of (dt_rank)
            // dt = F.softplus(dt)
            for (int i = 0; i < d_inner; i++) {
                dt[i] = softplus(dt[i]);
            }

            //  Discretize A and B
            // dA = torch.exp(torch.einsum("d,dn->dn", dt, self.A))   # A (d_inner, d_state), dA (d_inner, d_state)
            broadcast_multiply(dA, dt, w->A + mamba_l*d_inner*d_state, d_inner, d_state);
            for (int i = 0; i < d_inner * d_state; i++) {
                dA[i] = expf(dA[i]);
            }
            // dB = torch.einsum("d,n->dn", dt, B)    # dt (d_inner), B (d_state), dB (d_inner, d_state)
            outer_product(dB, dt, B, d_inner, d_state);

            //  Update ssm_state
            // ssm_state.copy_(ssm_state * dA + rearrange(hx, "d -> d 1") * dB)
            broadcast_multiply(s->temp, hx, dB, d_inner, d_state);
            elementwise_multiply_and_add(ssm_state, ssm_state, dA, s->temp, d_inner * d_state);

            //  Compute y
            // y = torch.einsum("dn,n->d", ssm_state, C) # ssm_state (d_inner, d_state), C (d_state), y (d_inner)
            rowwise_dot_product(y, ssm_state, C, d_inner, d_state);
            // y = y + self.D * hx
            elementwise_multiply_and_add(y, w->D + mamba_l*d_inner, hx, y, d_inner);
            // y = y * F.silu(z)  # (d_inner)
            for (int i = 0; i < d_inner; i++) {
                y[i] = y[i] * silu(z[i]);
            }

            // hidden_state = self.out_proj(y)  # out_proj (dim, d_inner), hidden_state (dim)
            linear(x, y, w->out_proj + mamba_l*dim*d_inner, x, dim, d_inner);
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        if (l >= expert_offset && (l - expert_offset) % expert_period == 0) {
            int moe_l = (l - expert_offset) / expert_period;
            matmul(s->exp_logits, s->xb, w->gate + moe_l*dim*num_experts, dim, num_experts);
            softmax(s->exp_logits, num_experts);
            for (int i = 0; i < num_experts; i++) {
                s->exp_probindex[i].index = i;
                s->exp_probindex[i].prob = s->exp_logits[i];
            }
            qsort(s->exp_probindex, num_experts, sizeof(ProbIndex), compare);

            for (int exp_id = 0; exp_id < top_k; exp_id++) {
                int index = s->exp_probindex[exp_id].index;
                // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
                // first calculate self.w1(x) and self.w3(x)
                matmul(s->hb, s->xb, w->moe_w1 + moe_l*num_experts*dim*hidden_dim + index*dim*hidden_dim, dim, hidden_dim);
                matmul(s->hb2, s->xb, w->moe_w3 + moe_l*num_experts*dim*hidden_dim + index*dim*hidden_dim, dim, hidden_dim);

                // SwiGLU non-linearity
                for (int i = 0; i < hidden_dim; i++) {
                    float val = s->hb[i];
                    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                    val *= (1.0f / (1.0f + expf(-val)));
                    // elementwise multiply with w3(x)
                    val *= s->hb2[i];
                    s->hb[i] = val;
                }

                // final matmul to get the output of the ffn
                matmul(s->hb3, s->hb, w->moe_w2 + moe_l*num_experts*dim*hidden_dim + index*dim*hidden_dim, hidden_dim, dim);

                // residual connection
                for (int i = 0; i < dim; i++) {
                    x[i] += s->hb3[i] * s->exp_probindex[exp_id].prob;
                }
            }
        }
        else {
            int mlp_l = l < expert_offset ? l : l - 1 - (l - expert_offset) / expert_period;
            matmul(s->hb, s->xb, w->mlp_w1 + mlp_l*dim*hidden_dim, dim, hidden_dim);
            matmul(s->hb2, s->xb, w->mlp_w3 + mlp_l*dim*hidden_dim, dim, hidden_dim);

            // SwiGLU non-linearity
            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= (1.0f / (1.0f + expf(-val)));
                // elementwise multiply with w3(x)
                val *= s->hb2[i];
                s->hb[i] = val;
            }

            // final matmul to get the output of the ffn
            matmul(s->hb3, s->hb, w->mlp_w2 + mlp_l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            for (int i = 0; i < dim; i++) {
                x[i] += s->hb3[i];
            }
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Jamba *jamba, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the jamba to get logits for the next token
        float* logits = forward(jamba, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Jamba *jamba, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the jamba
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the jamba next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the jamba to get logits for the next token
        float* logits = forward(jamba, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Jamba via the model .bin file
    Jamba jamba;
    build_jamba(&jamba, checkpoint_path);
    if (steps == 0 || steps > jamba.config.seq_len) steps = jamba.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, jamba.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, jamba.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&jamba, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&jamba, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_jamba(&jamba);
    return 0;
}
#endif
