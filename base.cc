#include <cassert>
#include <iostream>
#include <fstream>
#include "CLDeviceManager.h"

using namespace std;

#if 1

#define P_CPU CL_DEVICE_TYPE_CPU
#define P_GPU CL_DEVICE_TYPE_GPU
int main()
{
        CLDeviceManager dm;
        auto& d = dm.GetDevices();
        cl_platform_id pid;
        auto p = P_CPU;
        for (auto& i: d) {
                for (auto &j: i.second) {
                        if (dm.IsGPU(j)) {
                                cout << "GPU found" << endl;
                                if (p == P_GPU)
                                        pid = i.first;
                        }
                        else if (dm.IsCPU(j)) {
                                cout << "CPU found" << endl;
                                if (p == P_CPU)
                                        pid = i.first;
                        }
                }
        }
        //auto c = dm.CreateContext(pid);
        auto c1 = dm.CreateSpecificContext(p);
        assert(c1.context);

        CLProgram pro(c1, "test.c", true);
        CLKernel kernel(pro, "x2");

        const int M = 1024*8;
        const int N = M*M;
        unsigned int *data = new unsigned int[N];
        unsigned int *odata = new unsigned int[N];
        for (auto i = 0; i < N; i++)
                data[i] = i;
        size_t sz = sizeof(unsigned int) * N;
        CLBuffer mem1(c1, sz, data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        CLBuffer mem2(c1, sz, NULL, CL_MEM_WRITE_ONLY);
        kernel.SetArg(0, sizeof(mem1.buff), &mem1.buff);
        kernel.SetArg(1, sizeof(mem2.buff), &mem2.buff);
        unsigned int width = M;
        kernel.SetArg(2, sizeof(uint), &width);
        cl_device_id dev;
        clGetDeviceIDs(pid, p, 1, &dev, NULL);
        auto q = CLQueue(c1, dev, CL_QUEUE_PROFILING_ENABLE);
        //auto q = CLQueue(c1, dev);
        const size_t gsz[] = {M, M};
        const size_t lsz[] = {16, 16};

        for (auto i = 0; i < N; i++)
                odata[i] = 0;

        CLEvent ev;
        kernel.Run(q, 2, gsz, lsz, &ev);
        //kernel.Run(q, 2, gsz, lsz);
///*
        ev.Wait();
        cl_uint start = ev.GetTime(CL_PROFILING_COMMAND_QUEUED);
        cl_uint end =   ev.GetTime(CL_PROFILING_COMMAND_END);
        cout << "exec time: " << end - start << endl;
//*/
        mem2.Read(q, true, 0, sz, odata);
        for (auto i = 0; i < 32; i++) {
                cout << odata[i] << " ";
        }
        cout << endl;
        c1.Release();

        return 0;
}

#else

int main()
{
        CLDeviceManager dm;
        auto& d = dm.GetDevices();
        cl_platform_id pid;
        for (auto& i: d) {
                for (auto &j: i.second) {
                        if (dm.IsGPU(j)) {
                                cout << "GPU found" << endl;
                                pid = i.first;
                        }
                        else if (dm.IsCPU(j))
                                cout << "CPU found" << endl;
                }
        }
        //auto c = dm.CreateContext(pid);
        auto c1 = dm.CreateSpecificContext(CL_DEVICE_TYPE_GPU);
        assert(c1.context);

        auto c2 = dm.CreateSpecificContext(CL_DEVICE_TYPE_CPU);
        assert(c2.context);

        //c1.Release();
        //c2.Release();
        return 0;
}

#endif
