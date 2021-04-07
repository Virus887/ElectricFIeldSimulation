#ifndef KERNEL_H
#define KERNEL_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct uchar4;
struct int2;
struct float2;

class Particles
{
public:
    int w, h;
    int howManyProtons;
    int howManyElectrons;
    int howManyBigProtons;
    int howManyBigElectrons;
    thrust::host_vector<float2> p_pos;
    thrust::host_vector<float2> p_v;
    thrust::host_vector<float2> e_pos;
    thrust::host_vector<float2> e_v;
    thrust::host_vector<float2> big_p_pos;
    thrust::host_vector<float2> big_e_pos;
    thrust::host_vector<unsigned int> p_index;
    thrust::host_vector<unsigned int> e_index;

    Particles(int HowManyProtons, int HowManyElectrons,int HowManyBigProtons, int HowManyBigElectrons, int w, int h)
    {
        this->w = w;
        this->h = h;
        howManyProtons = HowManyProtons;
        howManyElectrons = HowManyElectrons;
        howManyBigProtons = HowManyBigProtons;
        howManyBigElectrons = HowManyBigElectrons;
        p_pos = thrust::host_vector<float2>(howManyProtons);
        e_pos = thrust::host_vector<float2>(howManyElectrons);
        p_v = thrust::host_vector<float2>(howManyProtons);
        e_v = thrust::host_vector<float2>(howManyElectrons);
        big_p_pos = thrust::host_vector<float2>(howManyBigProtons);
        big_e_pos = thrust::host_vector<float2>(howManyBigElectrons);

        p_index = thrust::host_vector<unsigned int>(howManyProtons);
        e_index = thrust::host_vector<unsigned int>(howManyElectrons);


        float speed = 1;
        for (int i = 0; i < howManyProtons; i++)
        {
            p_pos[i].x = (rand() % 400 / 400.0f) * w / 2.0f + w/4.0f;
            p_pos[i].y = (rand() % 400 / 400.0f) * h / 2.0f + h/4.0f;
            p_v[i].x = ((rand() % 400 / 200.0f) - 1.0f) * speed;
            p_v[i].y = ((rand() % 400 / 200.0f) - 1.0f) * speed;
            p_index[i] = i;
        }
        for (int i = 0; i < howManyElectrons; i++)
        {
            e_pos[i].x = (rand() % 400 / 400.0f) * w/2.0f + w/4.0f;
            e_pos[i].y = (rand() % 400 / 400.0f) * h / 2.0f + h/4.0f;
            e_v[i].x = ((rand() % 400 / 200.0f) - 1.0f) * speed;
            e_v[i].y = ((rand() % 400 / 200.0f) - 1.0f) * speed;
            e_index[i] = i;
        }

        for (int i = 0; i < howManyBigProtons; i++)
        {
            big_p_pos[i].x = (rand() % 400 / 400.0f) * w;
            big_p_pos[i].y = (rand() % 400 / 400.0f) * h;
        }
        for (int i = 0; i < howManyBigElectrons; i++)
        {
            big_e_pos[i].x = (rand() % 400 / 400.0f) * w;
            big_e_pos[i].y =(rand() % 400 / 400.0f) * h;
        }
    }
    Particles() {}
};

void prepareParticles(Particles* particles);
void kernelLauncher(uchar4* d_out, Particles* particles, float DT, float BIG_PARTICLE_POWER);

#endif