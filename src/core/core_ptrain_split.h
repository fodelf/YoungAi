#ifndef DS4_PTRAIN_SPLIT_H
#define DS4_PTRAIN_SPLIT_H
#include "core_ptrain.h"
bool pt_splitcheck(ds4_engine *, const pt_cfg *, pt_run *, const pt_data *);
bool pt_margincheck(ds4_engine *, const pt_cfg *, pt_run *, const pt_data *);
bool pt_protectcheck(ds4_engine *, const pt_cfg *, pt_run *, const pt_data *);
const ds4_gpu_tensor *pt_protect_mask(void);
bool pt_split_mask(const pt_cfg *, pt_run *, ds4_gpu_tensor *, uint32_t, uint32_t);
bool pt_diag_weights(const pt_cfg *, const pt_run *);
bool pt_diag_gradient(const pt_cfg *, const pt_run *, FILE *);
#ifndef DS4_NO_GPU
bool pt_kp_restore(const pt_cfg *, pt_run *, ds4_gpu_tensor *[DS4_MAX_LAYER][2]);
bool pt_kp_frozen(const pt_cfg *, pt_run *, ds4_gpu_tensor *[DS4_MAX_LAYER][2]);
bool pt_kp_objective(ds4_engine *, const pt_cfg *, pt_run *, pt_sample *, ds4_gpu_tensor **, bool, int, double *);
bool pt_kp_direction_fixture(void);
bool pt_trace_run(ds4_engine *, const pt_cfg *, pt_run *, pt_sample *, ds4_gpu_tensor **, ds4_gpu_tensor *[DS4_MAX_LAYER][2], double);
uint32_t pt_train_lo(const pt_cfg *);
bool pt_cosine_init(uint32_t);
float pt_cosine_scale(uint32_t);
#endif
#endif
