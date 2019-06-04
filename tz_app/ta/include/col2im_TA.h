#ifndef COL2IM_TA_H
#define COL2IM_TA_H

void col2im_cpu_TA(float* data_col,
                int channels, int height, int width,
                int ksize, int stride, int pad, float* data_im);
#endif
