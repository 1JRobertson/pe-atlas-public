#include <stdint.h>
#include <math.h>
/* Evaluate the released numeric histogram-gradient-boosting trees for finite features. */
void evaluate_forest(int rows, int columns, const float *input, int trees, int stride,
                     const int32_t *feature, const int32_t *left, const int32_t *right,
                     const double *threshold, const double *value, double bias, float *out) {
    for (int i=0;i<rows;i++) {
        double logit=bias;
        for (int t=0;t<trees;t++) {
            int idx=0, offset=t*stride;
            while (feature[offset+idx]>=0) {
                int node=offset+idx;
                idx=input[i*columns+feature[node]]<=threshold[node]?left[node]:right[node];
            }
            logit+=value[offset+idx];
        }
        out[i]=(float)(1./(1.+exp(-logit)));
    }
}
