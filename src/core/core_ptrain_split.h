#ifndef DS4_PTRAIN_SPLIT_H
#define DS4_PTRAIN_SPLIT_H
#include "core_ptrain.h"
bool pt_splitcheck(ds4_engine *, const pt_cfg *, pt_run *, const pt_data *);
bool pt_margincheck(ds4_engine *, const pt_cfg *, pt_run *, const pt_data *);
bool pt_split_mask(const pt_cfg *, pt_run *, ds4_gpu_tensor *, uint32_t, uint32_t);
bool pt_diag_weights(const pt_cfg *, const pt_run *);
bool pt_diag_gradient(const pt_cfg *, const pt_run *, FILE *);
#endif
