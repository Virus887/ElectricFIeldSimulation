#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#include "stdio.h"
#include <cmath>
#include <cuda_runtime.h>

#include "kernel.h"

#define TX 16
#define TY 16
#define GRIDDIM 70


#define BIG_DISPLAY_COEF 0.01f
#define DIST_COEF 0.01f

#define Q 0.5f



__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }


__global__
void displayKernel(uchar4* d_out, float2* fieldPowers, int w, int h,
    float2* pos_prot, float2* pos_elec, float2* big_pos_prot, float2* big_pos_elec,
    int howManyBigProt, int howManyBigElec,
    unsigned int* proton_begins, unsigned int* proton_ends, unsigned int* proton_indexes,
    unsigned int* electron_begins, unsigned int* electron_ends, unsigned int* electron_indexes, float BIG_PARTICLE_POWER) {

    //check if the pixel is the center of particle
    bool particle_flag = false;
    bool big_particle_flag = false;

    //get appropiate index
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= w) || (j >= h)) return; 
    const int ii = i + j * w; 

    //find current bucket index
    unsigned int x = (i / (float)w) * GRIDDIM;
    unsigned int y = (j / (float)h) * GRIDDIM;
    if (y >= h) y = GRIDDIM - 1;
    if (x >= w) x = GRIDDIM - 1;
    int bucket = y * GRIDDIM + x;
    if (bucket >= GRIDDIM * GRIDDIM) bucket = GRIDDIM * GRIDDIM - 1;


    bool b2 = (bucket / GRIDDIM != GRIDDIM - 1);           
    bool b4 = (bucket % GRIDDIM != 0);                    
    bool b6 = (bucket % GRIDDIM != GRIDDIM - 1);           
    bool b8 = (bucket / GRIDDIM != 0);                     
    bool b1 = (b2 && b4);
    bool b3 = (b6 && b2);
    bool b7 = (b8 && b4);
    bool b9 = (b8 && b6);

#pragma region collect_data
    //collect neighbour particles data
    float fieldPower = 0;
    for (int p = 0; p < howManyBigProt; p++)
    {
        float distx = big_pos_prot[p].x - i - 0.5f;
        float disty = big_pos_prot[p].y - j - 0.5f;
        float dist = sqrtf(distx * distx + disty * disty);
        if (dist <= 3.6f)
        {
            big_particle_flag = true;
        }
        if (dist < 0.01)continue;
        float F = BIG_PARTICLE_POWER * Q / (dist * dist * DIST_COEF);
        fieldPowers[ii].x += F * distx / dist;
        fieldPowers[ii].y += F * disty / dist;

        fieldPower += F * BIG_DISPLAY_COEF;
    }
    for (int p = 0; p < howManyBigElec; p++)
    {
        float distx = big_pos_elec[p].x - i - 0.5f;
        float disty = big_pos_elec[p].y - j - 0.5f;
        float dist = sqrtf(distx * distx + disty * disty);
        if (dist <= 3.6f)
        {
            big_particle_flag = true;
        }
        if (dist < 0.01)continue;
        float F = BIG_PARTICLE_POWER * Q / (dist * dist * DIST_COEF);
        fieldPowers[ii].x -= F * distx / dist;
        fieldPowers[ii].y -= F * disty / dist;
        fieldPower -= F * BIG_DISPLAY_COEF;
    }

    for (int p = proton_begins[bucket]; p < proton_ends[bucket]; p++)
    {
        int pp = proton_indexes[p];
        float distx = pos_prot[pp].x - i - 0.5f;
        float disty = pos_prot[pp].y - j - 0.5f;
        float dist = sqrtf(distx * distx + disty * disty);
        if (dist <= 0.6f)
        {
            particle_flag = true;
        }
        if (dist < 0.01)continue;
        float F = Q / (dist * dist * DIST_COEF);
        fieldPowers[ii].x += (F * distx) / dist;
        fieldPowers[ii].y += (F * disty) / dist;
        fieldPower += F;
    }
    for (int p = electron_begins[bucket]; p < electron_ends[bucket]; p++)
    {
        int pp = electron_indexes[p];
        float distx = pos_elec[pp].x - i - 0.5f;
        float disty = pos_elec[pp].y - j - 0.5f;
        float dist = sqrtf(distx * distx + disty * disty);
        if (dist <= 0.6f)
        {
            particle_flag = true;
        }
        if (dist < 0.01)continue;
        float F = Q / (dist * dist * DIST_COEF);
        fieldPowers[ii].x -= (F * distx) / dist;
        fieldPowers[ii].y -= (F * disty) / dist;

        fieldPower -= F;
    }


    if (b7)
    {
        for (int p = proton_begins[bucket - GRIDDIM- 1]; p < proton_ends[bucket- GRIDDIM - 1]; p++)
        {
            int pp = proton_indexes[p];
            float distx = pos_prot[pp].x - i - 0.5f;
            float disty = pos_prot[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x += F * distx / dist;
            fieldPowers[ii].y += F * disty / dist;

            fieldPower += F;
        }

        for (int p = electron_begins[bucket - GRIDDIM - 1]; p < electron_ends[bucket - GRIDDIM - 1]; p++)
        {
            int pp = electron_indexes[p];
            float distx = pos_elec[pp].x - i - 0.5f;
            float disty = pos_elec[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x -= F * distx / dist;
            fieldPowers[ii].y -= F * disty / dist;

            fieldPower -= F;
        }
    }

    if (b8)
    {
        for (int p = proton_begins[bucket - GRIDDIM]; p < proton_ends[bucket - GRIDDIM]; p++)
        {
            int pp = proton_indexes[p];
            float distx = pos_prot[pp].x - i - 0.5f;
            float disty = pos_prot[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x += F * distx / dist;
            fieldPowers[ii].y += F * disty / dist;

            fieldPower += F;
        }

        for (int p = electron_begins[bucket - GRIDDIM]; p < electron_ends[bucket - GRIDDIM]; p++)
        {
            int pp = electron_indexes[p];
            float distx = pos_elec[pp].x - i - 0.5f;
            float disty = pos_elec[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x -= F * distx / dist;
            fieldPowers[ii].y -= F * disty / dist;

            fieldPower -= F;
        }
    }

    if (b9)
    {
        for (int p = proton_begins[bucket - GRIDDIM + 1]; p < proton_ends[bucket - GRIDDIM + 1]; p++)
        {
            int pp = proton_indexes[p];
            float distx = pos_prot[pp].x - i - 0.5f;
            float disty = pos_prot[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x += F * distx / dist;
            fieldPowers[ii].y += F * disty / dist;

            fieldPower += F;
        }

        for (int p = electron_begins[bucket - GRIDDIM + 1]; p < electron_ends[bucket - GRIDDIM + 1]; p++)
        {
            int pp = electron_indexes[p];
            float distx = pos_elec[pp].x - i - 0.5f;
            float disty = pos_elec[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x -= F * distx / dist;
            fieldPowers[ii].y -= F * disty / dist;

            fieldPower -= F;
        }
    }

    if (b4)
    {
        for (int p = proton_begins[bucket- 1]; p < proton_ends[bucket - 1]; p++)
        {
            int pp = proton_indexes[p];
            float distx = pos_prot[pp].x - i - 0.5f;
            float disty = pos_prot[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x += F * distx / dist;
            fieldPowers[ii].y += F * disty / dist;

            fieldPower += F;
        }

        for (int p = electron_begins[bucket - 1]; p < electron_ends[bucket - 1]; p++)
        {
            int pp = electron_indexes[p];
            float distx = pos_elec[pp].x - i - 0.5f;
            float disty = pos_elec[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x -= F * distx / dist;
            fieldPowers[ii].y -= F * disty / dist;

            fieldPower -= F;
        }
    }

    if (b6)
    {
        for (int p = proton_begins[bucket + 1]; p < proton_ends[bucket + 1]; p++)
        {
            int pp = proton_indexes[p];
            float distx = pos_prot[pp].x - i - 0.5f;
            float disty = pos_prot[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x += F * distx / dist;
            fieldPowers[ii].y += F * disty / dist;

            fieldPower += F;
        }

        for (int p = electron_begins[bucket + 1]; p < electron_ends[bucket + 1]; p++)
        {
            int pp = electron_indexes[p];
            float distx = pos_elec[pp].x - i - 0.5f;
            float disty = pos_elec[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x -= F * distx / dist;
            fieldPowers[ii].y -= F * disty / dist;

            fieldPower -= F;
        }
    }

    if (b1)
    {
        for (int p = proton_begins[bucket + GRIDDIM - 1]; p < proton_ends[bucket + GRIDDIM - 1]; p++)
        {
            int pp = proton_indexes[p];
            float distx = pos_prot[pp].x - i - 0.5f;
            float disty = pos_prot[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x += F * distx / dist;
            fieldPowers[ii].y += F * disty / dist;

            fieldPower += F;
        }

        for (int p = electron_begins[bucket + GRIDDIM - 1]; p < electron_ends[bucket + GRIDDIM - 1]; p++)
        {
            int pp = electron_indexes[p];
            float distx = pos_elec[pp].x - i - 0.5f;
            float disty = pos_elec[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x -= F * distx / dist;
            fieldPowers[ii].y -= F * disty / dist;

            fieldPower -= F;
        }
    }

    if (b2)
    {
        for (int p = proton_begins[bucket + GRIDDIM]; p < proton_ends[bucket + GRIDDIM]; p++)
        {
            int pp = proton_indexes[p];
            float distx = pos_prot[pp].x - i - 0.5f;
            float disty = pos_prot[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x += F * distx / dist;
            fieldPowers[ii].y += F * disty / dist;

            fieldPower += F;
        }

        for (int p = electron_begins[bucket + GRIDDIM]; p < electron_ends[bucket + GRIDDIM]; p++)
        {
            int pp = electron_indexes[p];
            float distx = pos_elec[pp].x - i - 0.5f;
            float disty = pos_elec[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x -= F * distx / dist;
            fieldPowers[ii].y -= F * disty / dist;

            fieldPower -= F;
        }
    }

    if (b3)
    {
        for (int p = proton_begins[bucket + GRIDDIM + 1]; p < proton_ends[bucket + GRIDDIM + 1]; p++)
        {
            int pp = proton_indexes[p];
            float distx = pos_prot[pp].x - i - 0.5f;
            float disty = pos_prot[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x += F * distx / dist;
            fieldPowers[ii].y += F * disty / dist;

            fieldPower += F;
        }

        for (int p = electron_begins[bucket + GRIDDIM + 1]; p < electron_ends[bucket + GRIDDIM + 1]; p++)
        {
            int pp = electron_indexes[p];
            float distx = pos_elec[pp].x - i - 0.5f;
            float disty = pos_elec[pp].y - j - 0.5f;
            float dist = sqrtf(distx * distx + disty * disty);
            if (dist < 0.01)continue;

            float F = Q / (dist * dist * DIST_COEF);
            fieldPowers[ii].x -= F * distx / dist;
            fieldPowers[ii].y -= F * disty / dist;

            fieldPower -= F;
        }
    }
#pragma endregion


    d_out[ii].y = 0;
    d_out[ii].w = 255;

    float disp_coef = 5.0f;
    if (fieldPower > 0)
    {
        const unsigned char intensity = clip(fieldPower * disp_coef);
        d_out[ii].x = intensity;//red
        d_out[ii].z = 0; //blue
    }
    else
    {
        const unsigned char intensity = clip(-fieldPower * disp_coef);
        d_out[ii].x = 0; //red
        d_out[ii].z = intensity; //blue
    }

    //put black dot inside small particle
    if (particle_flag)
    {
        d_out[ii].x = 0;
        d_out[ii].z = 0;
    }

    //put white dot inside big particle
    else if (big_particle_flag)
    {
        d_out[ii].y = 255;
        d_out[ii].w = 255;
        d_out[ii].x = 255;
        d_out[ii].z = 255;
    }
}


__global__
void movementKernel(int w, int h, float2* pos_prot, float2* pos_elec, float2* v_prot, float2* v_elec, float2* fieldPowers,
    int howManyProt, int howManyElec, int howManyBigProt, int howManyBigElec, float dt) {
    float ax = 0, ay = 0;
    
    int ii = blockIdx.x * blockDim.x + threadIdx.x; // 1D indexing
    if (ii >= howManyProt + howManyElec) return; // Check if within image bounds

    if (ii < howManyProt) // I am proton
    {   
        unsigned int x = floor(pos_prot[ii].x);
        unsigned int y = floor(pos_prot[ii].y);
        unsigned int pixel = y * w + x;


        ax -= Q * fieldPowers[pixel].x ;
        ay -= Q * fieldPowers[pixel].y ;

        v_prot[ii].x += ax * dt;
        v_prot[ii].y += ay * dt;


        pos_prot[ii].x += v_prot[ii].x * dt;
        pos_prot[ii].y += v_prot[ii].y * dt;

        //BOUNCE
        if (pos_prot[ii].x >= w)
        {
            v_prot[ii].x = -v_prot[ii].x / 2.0f;
            pos_prot[ii].x = w - 1.0f;
        }
        if (pos_prot[ii].y >= h)
        {
            v_prot[ii].y = -v_prot[ii].y / 2.0f;
            pos_prot[ii].y = h - 1.0f;
        }
        if (pos_prot[ii].x <= 0)
        {
            v_prot[ii].x = -v_prot[ii].x / 2.0f;
            pos_prot[ii].x = 1.0f;
        }
        if (pos_prot[ii].y <= 0)
        {
            v_prot[ii].y = -v_prot[ii].y / 2.0f;
            pos_prot[ii].y = 1.0f;
        }

    }
    else//I am electron
    {
        ii -= howManyProt;

        unsigned int x = floor(pos_elec[ii].x);
        unsigned int y = floor(pos_elec[ii].y);
        if (x >= w) x = w - 1;
        if (y >= h) y = h - 1;
        int pixel = y * w + x;

        ax += fieldPowers[pixel].x ;
        ay += fieldPowers[pixel].y ;


        v_elec[ii].x += ax * dt;
        v_elec[ii].y += ay * dt;

        pos_elec[ii].x += v_elec[ii].x * dt;
        pos_elec[ii].y += v_elec[ii].y * dt;

        //BOUNCE
        if (pos_elec[ii].x >= w)
        {
            v_elec[ii].x = -v_elec[ii].x / 2.0f;
            pos_elec[ii].x = w-1.0f;
        }
        if (pos_elec[ii].y >= h)
        {
            v_elec[ii].y = -v_elec[ii].y / 2.0f;
            pos_elec[ii].y = h-1.0f;
        }
        if (pos_elec[ii].x <= 0)
        {
            v_elec[ii].x = -v_elec[ii].x / 2.0f;
            pos_elec[ii].x = 1.0f;
        }
        if (pos_elec[ii].y <= 0)
        {
            v_elec[ii].y = -v_elec[ii].y / 2.0f;
            pos_elec[ii].y = 1.0f;
        }

    }
}


struct calculate_bucket_index
{
    float w, h;
    int gridDim;
    __host__ __device__
        calculate_bucket_index(float width, float height, int dim)
        :w(width), h(height), gridDim(dim) {}
    __host__ __device__
        unsigned int operator()(float2 p) const
    {
        // coordinates of the grid cell containing point p
        unsigned int x = ((p.x / w) * gridDim);
        unsigned int y = ((p.y / h) * gridDim);
        // return the bucket's linear index
        if (y * gridDim + x >= gridDim * gridDim) return gridDim * gridDim - 1;
        else return y * gridDim + x;
    }
};



thrust::device_vector<float2> d_p;// = particles->p_pos;
thrust::device_vector<float2> d_e;// = particles->e_pos;
thrust::device_vector<float2> d_big_p;// = particles->big_p_pos;
thrust::device_vector<float2> d_big_e;// = particles->big_e_pos;
thrust::device_vector<float2> d_pv;// = particles->p_v;
thrust::device_vector<float2> d_ev;// = particles->e_v;


float2* ptr_protons;//= thrust::raw_pointer_cast(&d_p[0]);
float2* ptr_electrons;//= thrust::raw_pointer_cast(&d_e[0]);
float2* ptr_big_protons;//= thrust::raw_pointer_cast(&d_big_p[0]);
float2* ptr_big_electrons;// = thrust::raw_pointer_cast(&d_big_e[0]);
float2* ptr_protons_v;//= thrust::raw_pointer_cast(&d_pv[0]);
float2* ptr_electrons_v;//= thrust::raw_pointer_cast(&d_ev[0]);


void prepareParticles(Particles* particles)
{
    d_p = particles->p_pos;
    d_e = particles->e_pos;
    d_big_p = particles->big_p_pos;
    d_big_e = particles->big_e_pos;
    d_pv = particles->p_v;
    d_ev = particles->e_v;

    ptr_protons = thrust::raw_pointer_cast(&d_p[0]);
    ptr_electrons = thrust::raw_pointer_cast(&d_e[0]);
    ptr_big_protons = thrust::raw_pointer_cast(&d_big_p[0]);
    ptr_big_electrons = thrust::raw_pointer_cast(&d_big_e[0]);
    ptr_protons_v = thrust::raw_pointer_cast(&d_pv[0]);
    ptr_electrons_v = thrust::raw_pointer_cast(&d_ev[0]);

}


void kernelLauncher(uchar4* d_out, Particles* particles, float DT, float BIG_PARTICLE_POWER) {
    const dim3 blockSize(TX, TY);
    const dim3 gridSize = dim3((particles->w + TX - 1) / TX, (particles->h + TY - 1) / TY);
    cudaError_t err;

#pragma region thrust

    //PROTONS
    //bucket sort using thrust library
    //give each proton its bucket index 
    thrust::device_vector<unsigned int> proton_sorted_indexes = particles->p_index;

    thrust::device_vector<unsigned int> proton_bucket_indices(particles->howManyProtons);
    thrust::transform(d_p.begin(), d_p.end(), proton_bucket_indices.begin(), calculate_bucket_index(particles->w, particles->h, GRIDDIM));

    //sort protons indexes by bucket
    thrust::sort_by_key(proton_bucket_indices.begin(), proton_bucket_indices.end(), proton_sorted_indexes.begin());

    //return each bucket begin and end
    thrust::device_vector<unsigned int> proton_bucket_start(GRIDDIM * GRIDDIM);
    thrust::device_vector<unsigned int> proton_bucket_end(GRIDDIM * GRIDDIM);

    thrust::counting_iterator<unsigned int> proton_search_begin(0);
    thrust::lower_bound(proton_bucket_indices.begin(), proton_bucket_indices.end(), proton_search_begin, proton_search_begin + GRIDDIM * GRIDDIM, proton_bucket_start.begin());  //returns iterator on first proton with bucket index >= i
    thrust::upper_bound(proton_bucket_indices.begin(), proton_bucket_indices.end(), proton_search_begin, proton_search_begin + GRIDDIM * GRIDDIM, proton_bucket_end.begin());  //returns iterator on first proton with bucket index > i 
    
    
    unsigned int* ptr_proton_starts = thrust::raw_pointer_cast(&proton_bucket_start[0]);
    unsigned int* ptr_proton_ends = thrust::raw_pointer_cast(&proton_bucket_end[0]);
    unsigned int* ptr_proton_sorted_indexes = thrust::raw_pointer_cast(&proton_sorted_indexes[0]);


    //ELECTRONS
    //repeat for electrons
    thrust::device_vector<unsigned int> electron_sorted_indexes = particles->e_index;

    thrust::device_vector<unsigned int> electron_bucket_indices(particles->howManyElectrons);
    thrust::transform(d_e.begin(), d_e.end(), electron_bucket_indices.begin(), calculate_bucket_index(particles->w, particles->h, GRIDDIM));

    //sort boid indexes by bucket
    thrust::sort_by_key(electron_bucket_indices.begin(), electron_bucket_indices.end(), electron_sorted_indexes.begin());

    //return each bucket begin and end
    thrust::device_vector<unsigned int> electron_bucket_start(GRIDDIM * GRIDDIM);
    thrust::device_vector<unsigned int> electron_bucket_end(GRIDDIM * GRIDDIM);

    thrust::counting_iterator<unsigned int> electron_search_begin(0);
    thrust::lower_bound(electron_bucket_indices.begin(), electron_bucket_indices.end(), electron_search_begin, electron_search_begin + GRIDDIM * GRIDDIM, electron_bucket_start.begin());  //returns iterator on first electron with bucket index >= i
    thrust::upper_bound(electron_bucket_indices.begin(), electron_bucket_indices.end(), electron_search_begin, electron_search_begin + GRIDDIM * GRIDDIM, electron_bucket_end.begin());  //returns iterator on first electron with bucket index > i 
    
    
    unsigned int* ptr_electron_starts = thrust::raw_pointer_cast(&electron_bucket_start[0]);
    unsigned int* ptr_electron_ends = thrust::raw_pointer_cast(&electron_bucket_end[0]);
    unsigned int* ptr_electron_sorted_indexes = thrust::raw_pointer_cast(&electron_sorted_indexes[0]);


    thrust::device_vector<float2> field_power_vec = thrust::device_vector<float2>(particles->w * particles->h);
    float2* ptr_field_powers = thrust::raw_pointer_cast(&field_power_vec[0]);

#pragma endregion

    displayKernel <<<gridSize, blockSize >> > 
        (d_out,ptr_field_powers ,particles->w, particles->h, 
         ptr_protons, ptr_electrons, ptr_big_protons, ptr_big_electrons,
         particles->howManyBigProtons, particles->howManyBigElectrons,
         ptr_proton_starts, ptr_proton_ends, ptr_proton_sorted_indexes, ptr_electron_starts, ptr_electron_ends, ptr_electron_sorted_indexes, BIG_PARTICLE_POWER);


    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed Display (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (particles->howManyProtons + particles->howManyElectrons + threadsPerBlock - 1) / threadsPerBlock;
    movementKernel << <blocksPerGrid, threadsPerBlock >> >
            (particles->w, particles->h, ptr_protons, ptr_electrons, ptr_protons_v, ptr_electrons_v, ptr_field_powers,
                particles->howManyProtons, particles->howManyElectrons, particles->howManyBigProtons, particles->howManyBigElectrons, DT);


    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed Movement (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    particles->p_pos = d_p;
    particles->e_pos = d_e;
    particles->p_v = d_pv;
    particles->e_v = d_ev;
}

