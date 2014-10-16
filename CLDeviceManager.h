#pragma once

#include <CL/cl.h>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

typedef std::map<cl_platform_id, std::vector<cl_device_id> > mpd;

class CLContext {
public:
        cl_context context;
        CLContext(cl_platform_id pid) {
                cl_context_properties prop[] = {
                        CL_CONTEXT_PLATFORM,
                        (cl_context_properties)pid,
                        0
                };
                context = clCreateContextFromType(
                        prop,
                        CL_DEVICE_TYPE_ALL,
                        NULL,
                        NULL,
                        NULL);
        }
        void Release() {
                if (context) {
                        clReleaseContext(context);
                }
                context = NULL;
        }
        ~CLContext() {
                Release();
        }
};

class CLDeviceManager {
        mpd devices;
public:
        CLDeviceManager() {
                cl_uint n;
                clGetPlatformIDs(0, NULL, &n);
                auto pf = std::vector<cl_platform_id>(n);
                clGetPlatformIDs(n, &pf[0], NULL);
                for (auto i: pf) {
                        auto ret = devices.insert(
                                std::make_pair(
                                        i,
                                        std::vector<cl_device_id>())
                                );
                        auto& r = ret.first->second;
                        clGetDeviceIDs(i, CL_DEVICE_TYPE_ALL, 0, NULL, &n);
                        r = std::vector<cl_device_id>(n);
                        clGetDeviceIDs(i, CL_DEVICE_TYPE_ALL, n, &r[0], NULL);
                }
        }

        const mpd& GetDevices() {
                return devices;
        }

        cl_device_type GetDeviceType(cl_device_id did) {
                cl_device_type ret;
                clGetDeviceInfo(did,
                                CL_DEVICE_TYPE,
                                sizeof(cl_device_type),
                                &ret,
                                NULL);
                return ret;
        }

        bool IsGPU(cl_device_id did) {
                return GetDeviceType(did) == CL_DEVICE_TYPE_GPU;
        }

        bool IsCPU(cl_device_id did) {
                return GetDeviceType(did) == CL_DEVICE_TYPE_CPU;
        }

        cl_platform_id GetPlatformWithType(cl_device_type type=CL_DEVICE_TYPE_ALL) {
                cl_platform_id pid = 0;
                for (auto& p: devices) {
                        for (auto& dev: p.second) {
                                cl_device_type dev_type;
                                clGetDeviceInfo(
                                        dev,
                                        CL_DEVICE_TYPE,
                                        sizeof(cl_device_type),
                                        &dev_type,
                                        NULL);
                                if (dev_type == type) {
                                        pid = p.first;
                                        break;
                                }
                        }
                }
                return pid;
        }

        CLContext CreateSpecificContext(cl_device_type type=CL_DEVICE_TYPE_ALL) {
                cl_platform_id pid = GetPlatformWithType(type);
                return CLContext(pid);
        }
};


class CLProgram {
        void CreateProgramFromString(CLContext& context,
                                     const char src[],
                                     const char *options) {
                cl_int err;
                const char *pa[] = {src};
                program = clCreateProgramWithSource(
                        context.context,
                        1,
                        pa,
                        NULL,
                        &err);
                assert(program);
                err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
                if (err != CL_SUCCESS) {
                        char buildlog[1<<20];
                        buildlog[0] = 0;
                        size_t size;
                        clGetContextInfo(context.context,
                                         CL_CONTEXT_DEVICES,
                                         0, NULL, &size);
                        int cnt = size/sizeof(cl_device_id);
                        std::vector<cl_device_id> v(cnt);
                        clGetContextInfo(context.context,
                                         CL_CONTEXT_DEVICES,
                                         cnt, &v[0], NULL);
                        size_t copied;
                        for (auto& dev: v) {
                                clGetProgramBuildInfo(
                                        program,
                                        dev,
                                        CL_PROGRAM_BUILD_LOG,
                                        sizeof(buildlog),
                                        buildlog,
                                        &copied);
                                assert(copied);
                                std::cerr << "build error" << std::endl;
                                std::cerr << buildlog << std::endl;
                        }
                        abort();
                }
        }
public:
        cl_program program;
        CLProgram(CLContext& context,
                  const char src[],
                  bool is_file,
                  const char *options=NULL) {

                const char *src_str;
                char *buf = NULL;
                if (is_file) {
                        std::ifstream file(
                                src,
                                std::ios::in | std::ios::binary | std::ios::ate);
                        if (!file.is_open()) {
                                std::cout << "file can not be opened" << std::endl;
                                abort();
                        }
                        std::streampos pos = file.tellg();
                        size_t sz = pos;
                        buf = new char[sz+1];
                        file.seekg(0, std::ios::beg);
                        file.read(buf, sz);
                        buf[sz] = 0;
                        file.close();
                        src_str = buf;
                } else {
                        src_str = src;
                }
                CreateProgramFromString(context, src_str, options);
                delete buf;
        }
        void Release() {
                if (program) {
                        clReleaseProgram(program);
                }
                program = NULL;
        }
        ~CLProgram() {
                Release();
        }
};

class CLQueue {
public:
        cl_command_queue q;
        CLQueue(CLContext c,
                cl_device_id d,
                cl_command_queue_properties props=0) {

                cl_int err;
                q = clCreateCommandQueue(c.context, d, props, &err);
                assert(err == CL_SUCCESS);
        }
        void Release() {
                if (q) {
                        clReleaseCommandQueue(q);
                }
                q = NULL;
        }
        ~CLQueue() {
                Release();
        }
};

class CLEvent {
public:
        cl_event e;
        CLEvent() = default;
        cl_ulong GetTime(cl_profiling_info i) {
                size_t sz;
                cl_ulong v;
                clGetEventProfilingInfo(e, i, sizeof(cl_ulong), &v, &sz);
                assert(sz);
                return v;
        }
        void Wait() {
                clWaitForEvents(1, &e);
        }
};

class CLKernel {
public:
        cl_kernel kernel;
        CLKernel(CLProgram& prog, const char name[]) {
                cl_int err;
                kernel = clCreateKernel(prog.program, name, &err);
                assert(err == CL_SUCCESS);
        }
        void Release() {
                if (kernel) {
                        clReleaseKernel(kernel);
                }
                kernel = NULL;
        }
        ~CLKernel() {
                Release();
        }

        void SetArg(int pos, size_t size, void *arg) {
                auto ret = clSetKernelArg(kernel, pos, size, arg);
                assert(ret == CL_SUCCESS);
        }

        void Run(
                CLQueue& q,
                cl_uint dim,
                const size_t *gsize,
                const size_t *lsize,
                CLEvent *this_ev=NULL,
                const size_t *goffset=0,
                cl_uint nev=0,
                CLEvent *evl=NULL) {
                clEnqueueNDRangeKernel(
                        q.q,
                        kernel,
                        dim,
                        goffset, gsize, lsize,
                        nev, (cl_event *)evl, (cl_event *)this_ev);
        }
};

class CLBuffer {
public:
        cl_mem buff;
        CLBuffer(CLContext& context,
                 size_t size,
                 void *host_ptr=NULL,
                 cl_mem_flags flags=CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR) {
                cl_int err;
                buff = clCreateBuffer(
                        context.context,
                        flags,
                        size,
                        host_ptr,
                        &err);
                assert(err == CL_SUCCESS);
        }

        void Write(CLQueue& cq,
                   cl_bool should_block,
                   size_t offset,
                   size_t cnt,
                   void *src,
                   CLEvent *this_ev=NULL,
                   cl_uint nevents=0,
                   const CLEvent *evs=NULL) {
                auto r = clEnqueueWriteBuffer(
                        cq.q,
                        buff,
                        should_block,
                        offset,
                        cnt,
                        src,
                        nevents,
                        (cl_event *)evs,
                        (cl_event *)this_ev);
                assert(r == CL_SUCCESS);
        }

        void Read(CLQueue& cq,
                  cl_bool should_block,
                  size_t offset,
                  size_t cnt,
                  void *dst,
                  CLEvent *this_ev=NULL,
                  cl_uint nevents=0,
                  const CLEvent *evs=NULL) {

                auto r = clEnqueueReadBuffer(
                        cq.q,
                        buff,
                        should_block,
                        offset,
                        cnt,
                        dst,
                        nevents,
                        (cl_event *)evs,
                        (cl_event *)this_ev);
                assert(r == CL_SUCCESS);
        }

        void Release() {
                if (buff) {
                        clReleaseMemObject(buff);
                }
                buff = NULL;
        }
        ~CLBuffer() {
                Release();
        }
};
